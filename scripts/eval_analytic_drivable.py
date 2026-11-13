"""
How well the analytic drivable-area check (diffusion_planner.scorer.analytic_metrics) matches PDM's drivable_area
labels, and what it does to the verifier's selection when it replaces the verifier's drivable-area prediction.

    python scripts/eval_analytic_drivable.py --data $NUPLAN_HOME/labels/train-labels-2k \
        --ckpt results/verifier/w3_oracle/best.pt --out results/verifier/w3_oracle/analytic_drivable

Runs on the validation split of the training run (--val-frac / --split-seed as in training), on CPU by default.
Reports, per tolerance (a corner within that many metres of a lane polygon counts as inside):
  - agreement with the labels per candidate: accuracy, precision / recall of "off-road" (label 0), and per scene type
  - within-scene pairwise accuracy on drivable area (as in diagnose_verifier.py)
  - selection on the oracle combined score: verifier as trained, verifier with its drivable-area probability replaced
    by the analytic check, the same with the true label (upper bound), and PDM+SL
  - runtime of the check per scene
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from diffusion_planner.scorer.analytic_metrics import corner_lane_distance  # noqa: E402
from diffusion_planner.scorer.dataset import LabelDataset, find_samples, split_by_scenario  # noqa: E402
from diffusion_planner.scorer.labels import load_sample  # noqa: E402
from diffusion_planner.scorer.metrics import combined_score  # noqa: E402
from diffusion_planner.scorer.model import Verifier  # noqa: E402
from diffusion_planner.scorer.train import TIE_TOLERANCE, first_max  # noqa: E402

DRIVABLE = 1


def pairwise_accuracy(predicted: np.ndarray, labels: np.ndarray) -> float:
    gt = (labels[:, :, None] - labels[:, None, :]) > 1e-6
    diff = predicted[:, :, None] - predicted[:, None, :]
    correct = (diff > 0) + 0.5 * (diff == 0)
    return float((correct * gt).sum() / gt.sum()) if gt.sum() else float("nan")


def selection(scores: torch.Tensor, true: torch.Tensor):
    picks = first_max(scores)
    best = true.max(dim=1).values
    picked = true.gather(1, picks[:, None])[:, 0]
    return {"regret": (best - picked).mean().item(), "top1": (picked >= best - TIE_TOLERANCE).float().mean().item()}


@torch.no_grad()
def verifier_probs(ckpt, dp_args, files, device, batch_size, num_workers):
    model = Verifier.load(ckpt, dp_args, device=device)
    loader = DataLoader(LabelDataset(files), batch_size=batch_size, shuffle=False, num_workers=num_workers)
    out = []
    for batch in loader:
        inputs = {k: v.to(device) for k, v in batch["inputs"].items()}
        out.append(torch.sigmoid(model(inputs, batch["candidates"].to(device))).float().cpu())
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", default=str(ROOT / "results" / "verifier" / "w3_oracle" / "best.pt"))
    ap.add_argument("--dp-args", default=str(ROOT / "checkpoints" / "args.json"))
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--tolerances", default="0,0.1,0.2,0.3", help="comma-separated polygon tolerances [m]")
    ap.add_argument("--horizon", type=int, default=40)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=6, help="torch CPU threads (leave the machine to others)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    out = Path(args.out) if args.out else Path(args.ckpt).parent / "analytic_drivable"
    out.mkdir(parents=True, exist_ok=True)
    _, files = split_by_scenario(find_samples(args.data), args.val_frac, args.split_seed)
    if args.limit:
        files = files[:args.limit]
    print(f"[analytic] {len(files)} val samples, {len({f.parent for f in files})} scenarios", flush=True)

    # Labels, plans, lanes
    oracle, cv, analytic, types = [], [], {float(m): [] for m in args.tolerances.split(",")}, []
    seconds = 0.0
    for f in files:
        s = load_sample(f)
        oracle.append(s["metrics_oracle"])
        cv.append(s["metrics_cv"])
        types.append(f.parent.name.split("__", 2)[2])
        lanes, plans = s["inputs"]["lanes"], s["candidates"]
        start = time.perf_counter()
        distance = corner_lane_distance(lanes, plans, args.horizon)   # [K, T, 4], one call serves every tolerance
        seconds += time.perf_counter() - start
        for m in analytic:
            analytic[m].append((distance <= m).all(axis=(1, 2)).astype(np.float64))
    oracle, cv = np.stack(oracle), np.stack(cv)                 # [N, K, 7]
    analytic = {m: np.stack(v) for m, v in analytic.items()}    # [N, K]
    labels = oracle[..., DRIVABLE]
    types = np.array(types)
    print(f"[analytic] check: {1000 * seconds / len(files):.2f} ms per scene (K={labels.shape[1]})", flush=True)

    # Verifier probabilities
    probs = verifier_probs(args.ckpt, args.dp_args, files, args.device, args.batch_size, args.num_workers)
    true = combined_score(torch.from_numpy(oracle))

    def with_drivable(values):
        p = probs.clone()
        p[..., DRIVABLE] = torch.as_tensor(values, dtype=p.dtype)
        return combined_score(p)

    rows, per_type = [], []
    sel_base = {"verifier": selection(combined_score(probs), true),
                "verifier+true_drivable": selection(with_drivable(labels), true),
                "pdm_sl": selection(combined_score(torch.from_numpy(cv)), true)}
    for m, a in analytic.items():
        off_pred, off_true = a < 0.5, labels < 0.5
        mixed = off_true.any(axis=1) & (~off_true).any(axis=1)
        tp, fp, fn = (off_pred & off_true).sum(), (off_pred & ~off_true).sum(), (~off_pred & off_true).sum()
        sel = selection(with_drivable(a), true)
        rows.append({
            "tolerance": m,
            "accuracy": float((off_pred == off_true).mean()),
            "offroad_rate_label": float(off_true.mean()),
            "offroad_rate_analytic": float(off_pred.mean()),
            "offroad_precision": float(tp / max(tp + fp, 1)),
            "offroad_recall": float(tp / max(tp + fn, 1)),
            "scene_exact_match": float((off_pred == off_true).all(axis=1).mean()),
            # Scenes whose candidates differ on drivable area: the only ones where the metric changes the selection
            "mixed_scenes": int(mixed.sum()),
            "mixed_offroad_precision": float((off_pred & off_true)[mixed].sum() / max(off_pred[mixed].sum(), 1)),
            "mixed_offroad_recall": float((off_pred & off_true)[mixed].sum() / max(off_true[mixed].sum(), 1)),
            "pair_acc": pairwise_accuracy(a, labels),
            "verifier+analytic_regret": sel["regret"],
            "verifier+analytic_top1": sel["top1"],
        })
        for t in sorted(set(types)):
            r = types == t
            per_type.append({"tolerance": m, "scenario_type": t, "n": int(r.sum()),
                             "accuracy": float((off_pred[r] == off_true[r]).mean()),
                             "false_offroad": float((off_pred[r] & ~off_true[r]).mean()),
                             "missed_offroad": float((~off_pred[r] & off_true[r]).mean())})

    print("\n## Agreement with PDM drivable_area labels, and selection with the analytic check")
    keys = list(rows[0])
    print("  ".join(keys))
    for r in rows:
        print("  ".join(f"{r[k]:.4f}" if isinstance(r[k], float) else str(r[k]) for k in keys))
    print(f"\nverifier pair_acc on drivable_area: {pairwise_accuracy(probs[..., DRIVABLE].numpy(), labels):.4f}")
    print("\n## Selection reference (oracle combined score)")
    for k, v in sel_base.items():
        print(f"  {k:24s} regret {v['regret']:.4f}  top1 {v['top1']:.4f}")
    print("\n## Per scenario type, tolerance 0.2 m (share of candidates)")
    print("scenario_type  n  accuracy  false_offroad  missed_offroad")
    for r in per_type:
        if r["tolerance"] == 0.2:
            print(f"{r['scenario_type']}  {r['n']}  {r['accuracy']:.4f}  {r['false_offroad']:.4f}  {r['missed_offroad']:.4f}")

    (out / "summary.json").write_text(json.dumps({
        "n": len(files), "ms_per_scene": 1000 * seconds / len(files), "rows": rows, "reference": sel_base}, indent=2))
    with open(out / "per_type.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_type[0]))
        w.writeheader()
        w.writerows(per_type)
    np.savez_compressed(out / "analytic.npz", files=np.array([str(f) for f in files]),
                        **{f"tolerance_{m}": a for m, a in analytic.items()}, labels=labels)
    print(f"[analytic] wrote {out}")


if __name__ == "__main__":
    main()
