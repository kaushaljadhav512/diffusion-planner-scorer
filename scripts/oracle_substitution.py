"""
How much selection would improve if the verifier got one metric right: per-metric oracle substitution.

Runs a trained verifier (metric heads, e.g. w6_timestep) on the validation split of its training run, then replaces
one metric at a time with the label (--labels, default cv = what PDM+SL scores) and keeps the model's prediction for
the other six. Every variant is scored with the same combined score (CLS weights) and first-max selection, and judged
by regret and top-1 against the labels' combined score.

Progress in the timestep arch is gated by the predicted multiplicative metrics (collision x drivable x direction) and
normalized by the best candidate. When a multiplicative metric is substituted, the "regated" rows recompute progress
from the model's raw metres with the substituted safety (as if the model had predicted that metric correctly); the
"no_regate" rows keep the model's progress as it was. Progress has two rows: the normalized label, which leaks the true
multiplicative metrics (PDM zeroes progress for unsafe candidates), and the clean one, true raw metres gated by the
model's own safety predictions.

    python scripts/oracle_substitution.py --data $NUPLAN_HOME/labels/train-labels-2k \
        --ckpts runs/w6_timestep/seed0/best.pt runs/w6_timestep/seed1/best.pt runs/w6_timestep/seed2/best.pt \
        --out runs/w6_timestep/oracle_substitution

Writes <out>/results.csv (one row per checkpoint and variant), <out>/summary.csv (mean and std over checkpoints)
and prints the summary.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from diffusion_planner.scorer.dataset import LabelDataset, find_samples, split_by_scenario  # noqa: E402
from diffusion_planner.scorer.metrics import METRICS, MULTIPLICATIVE, combined_score  # noqa: E402
from diffusion_planner.scorer.model import PROGRESS, PROGRESS_THRESHOLD, Verifier  # noqa: E402
from diffusion_planner.scorer.train import TIE_TOLERANCE, first_max, to_device  # noqa: E402


def normalized_progress(raw: torch.Tensor, safe: torch.Tensor) -> torch.Tensor:
    """Verifier._timestep_forward's progress: raw [N, K] metres gated by safe [N, K], over the best candidate."""
    gated = raw * safe
    best = gated.max(dim=1, keepdim=True).values
    return torch.where(best > PROGRESS_THRESHOLD, gated / best.clamp_min(PROGRESS_THRESHOLD), safe)


def selection(scores: torch.Tensor, true: torch.Tensor) -> dict:
    pick = first_max(scores)
    picked = true.gather(1, pick[:, None])[:, 0]
    best = true.max(dim=1).values
    return {"regret": (best - picked).mean().item(), "top1": (picked >= best - TIE_TOLERANCE).float().mean().item()}


def variants(probs: torch.Tensor, raw: torch.Tensor, labels: torch.Tensor, raw_label: torch.Tensor):
    """Yields (name, metric probabilities [N, K, 7]) for every substitution."""
    yield "none (model)", probs
    for i, metric in enumerate(METRICS):
        sub = probs.clone()
        sub[..., i] = labels[..., i]
        if i in MULTIPLICATIVE and raw is not None:
            yield f"{metric} (no_regate)", sub.clone()
            sub[..., PROGRESS] = normalized_progress(raw, sub[..., MULTIPLICATIVE].prod(dim=-1))
            yield f"{metric} (regated)", sub
        elif i == PROGRESS:
            # The normalized label is zero for unsafe candidates (PDM gates it by the true multiplicative metrics),
            # so it leaks safety. The clean variant: true raw metres, gated by the model's own safety
            yield "progress (label, leaks safety)", sub
            if raw is not None:
                clean = probs.clone()
                clean[..., PROGRESS] = normalized_progress(raw_label, probs[..., MULTIPLICATIVE].prod(dim=-1))
                yield "progress (true metres, model gating)", clean
        else:
            yield metric, sub
    sub = probs.clone()
    sub[..., MULTIPLICATIVE] = labels[..., MULTIPLICATIVE]
    if raw is not None:
        sub[..., PROGRESS] = normalized_progress(raw, sub[..., MULTIPLICATIVE].prod(dim=-1))
    yield "all multiplicative (regated)", sub
    yield "all 7 (labels)", labels.clone()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="label dir (one subdirectory per scenario)")
    ap.add_argument("--ckpts", nargs="+", required=True, help="verifier checkpoints (train.py best.pt)")
    ap.add_argument("--dp-args", default=str(ROOT / "checkpoints" / "args.json"))
    ap.add_argument("--labels", choices=["oracle", "cv"], default="cv")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="only the first N val samples (smoke test)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    _, val_files = split_by_scenario(find_samples(args.data), args.val_frac, args.split_seed)
    if args.limit:
        val_files = val_files[:args.limit]
    print(f"{len(val_files)} val samples, {len({f.parent for f in val_files})} scenarios", flush=True)
    loader = DataLoader(LabelDataset(val_files), batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=args.device.startswith("cuda"))

    results = []
    for ckpt in args.ckpts:
        model = Verifier.load(ckpt, args.dp_args, device=args.device)
        assert model.hparams["num_outputs"] == len(METRICS), "needs a model with metric heads (not a classifier)"
        logits, raws, labels, raw_labels = [], [], [], []
        with torch.no_grad():
            for batch in loader:
                batch = to_device(batch, args.device)
                out, aux = model(batch["inputs"], batch["candidates"], return_aux=True)
                logits.append(out.float().cpu())
                raws.append(aux["progress_raw"].float().cpu() if "progress_raw" in aux else None)
                labels.append(batch[f"metrics_{args.labels}"].float().cpu())
                raw_labels.append(batch[f"progress_{args.labels}"].float().cpu())
        probs, labels, raw_label = torch.sigmoid(torch.cat(logits)), torch.cat(labels), torch.cat(raw_labels)
        raw = None if raws[0] is None else torch.cat(raws)
        true = combined_score(labels)
        for name, metric_probs in variants(probs, raw, labels, raw_label):
            row = {"ckpt": ckpt, "variant": name, **selection(combined_score(metric_probs), true)}
            results.append(row)
            print(f"{ckpt}  {name:32s} regret {row['regret']:.4f}  top1 {row['top1']:.3f}", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["ckpt", "variant", "regret", "top1"])
        writer.writeheader()
        writer.writerows(results)

    names = list(dict.fromkeys(r["variant"] for r in results))
    base = np.mean([r["regret"] for r in results if r["variant"] == "none (model)"])
    summary = []
    for name in names:
        regret = np.array([r["regret"] for r in results if r["variant"] == name])
        top1 = np.array([r["top1"] for r in results if r["variant"] == name])
        summary.append({"variant": name, "regret_mean": regret.mean(), "regret_std": regret.std(),
                        "regret_removed": 1 - regret.mean() / base, "top1_mean": top1.mean()})
    with open(out / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(f"\n{'variant':34s} {'regret':>17s} {'removed':>8s} {'top1':>6s}")
    for s in summary:
        print(f"{s['variant']:34s} {s['regret_mean']:.4f} +- {s['regret_std']:.4f} {s['regret_removed']:8.1%} "
              f"{s['top1_mean']:6.3f}")


if __name__ == "__main__":
    main()
