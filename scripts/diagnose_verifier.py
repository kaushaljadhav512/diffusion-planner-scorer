"""
Where the verifier's selection errors come from, compared with PDM+SL, on the held-out scenarios of a training run.

    python scripts/diagnose_verifier.py --data $NUPLAN_HOME/labels/train-labels-2k \
        --ckpt results/verifier/w3_oracle/best.pt --out results/verifier/w3_oracle/diagnostic

Use the same --val-frac / --split-seed as the training run (w3_oracle: 0.1 / 0) so the samples are the run's
validation set. Reports:

1. Per-metric prediction quality over all candidates of the val samples: mean label, MAE and Brier score of the
   predicted probability, global AUC for failure (label < 1), and within-scene pairwise accuracy (of the candidate
   pairs in one sample whose labels differ, the share the verifier orders correctly). Selection depends only on the
   within-scene ordering.
2. Attribution of the misses (pick not among the best by the oracle combined score), for the verifier and PDM+SL.
   Each miss gets one primary cause, the metric that explains the gap between the pick and the best candidate:
   - a multiplicative metric (collision, drivable area, driving direction) on which the pick is worse, the largest
     such drop first;
   - otherwise the weighted metric with the largest weighted deficit (progress, ttc, speed limit, comfort).
   For each cause the table gives the number of misses, its share of the total regret, and how often the verifier
   predicted the pick at least as good as the best candidate on that metric ("misjudged").
3. Regret and top-1 per scenario type.

Writes summary.json, metric_quality.csv, attribution.csv, per_type.csv to --out.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from diffusion_planner.scorer.dataset import LabelDataset, find_samples, split_by_scenario  # noqa: E402
from diffusion_planner.scorer.metrics import CLS_WEIGHTS, METRICS, MULTIPLICATIVE, WEIGHTED, combined_score  # noqa: E402
from diffusion_planner.scorer.model import Verifier  # noqa: E402
from diffusion_planner.scorer.train import TIE_TOLERANCE, first_max, to_device  # noqa: E402

LABEL_TOLERANCE = 1e-6


def auc(scores: np.ndarray, positives: np.ndarray) -> float:
    """ROC AUC with ties counted as half (Mann-Whitney U); NaN if one class is empty."""
    n_pos, n_neg = positives.sum(), (~positives).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    # Average ranks over ties
    boundaries = np.flatnonzero(np.diff(sorted_scores)) + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [len(scores)]])
    for s, e in zip(starts, ends):
        ranks[order[s:e]] = (s + e + 1) / 2.0
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def pairwise_accuracy(predicted: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Args:
        predicted, labels: [N, K]

    Returns:
        among pairs (i, j) of one sample with labels[i] > labels[j], the share with predicted[i] > predicted[j]
        (predicted ties count half); NaN if there is no such pair
    """
    label_gt = (labels[:, :, None] - labels[:, None, :]) > LABEL_TOLERANCE    # [N, K, K]
    diff = predicted[:, :, None] - predicted[:, None, :]
    correct = (diff > 0).float() + 0.5 * (diff == 0).float()
    n = label_gt.sum().item()
    return float((correct * label_gt).sum().item() / n) if n else float("nan")


def primary_cause(pick: np.ndarray, best: np.ndarray) -> int:
    """
    Args:
        pick, best: [7] oracle metrics of the picked and the best candidate

    Returns:
        index into METRICS of the metric that explains the gap (see the module docstring)
    """
    drops = [(best[i] - pick[i], i) for i in MULTIPLICATIVE if best[i] - pick[i] > LABEL_TOLERANCE]
    if drops:
        return max(drops)[1]
    deficits = [(w * (best[i] - pick[i]), i) for w, i in zip(CLS_WEIGHTS, WEIGHTED)]
    return max(deficits)[1]


def attribute(name: str, picks: torch.Tensor, true: torch.Tensor, oracle: torch.Tensor, probs: torch.Tensor):
    """Rows of the attribution table for one selector (see the module docstring)."""
    best_idx = first_max(true)
    rows_n = np.arange(len(true))
    picked_score = true[rows_n, picks]
    regret = (true.max(dim=1).values - picked_score).numpy()
    misses = np.flatnonzero(regret > TIE_TOLERANCE)
    total_regret = regret.sum()

    per_cause = {m: {"misses": 0, "regret": 0.0, "misjudged": 0} for m in METRICS}
    for n in misses:
        p, b = int(picks[n]), int(best_idx[n])
        cause = primary_cause(oracle[n, p].numpy(), oracle[n, b].numpy())
        entry = per_cause[METRICS[cause]]
        entry["misses"] += 1
        entry["regret"] += float(regret[n])
        if probs is not None:
            entry["misjudged"] += int(probs[n, p, cause] >= probs[n, b, cause])

    rows = []
    for metric, entry in per_cause.items():
        if entry["misses"] == 0:
            continue
        rows.append({
            "selector": name, "cause": metric, "misses": entry["misses"],
            "share_of_misses": entry["misses"] / max(len(misses), 1),
            "share_of_regret": entry["regret"] / total_regret if total_regret > 0 else float("nan"),
            "regret_contribution": entry["regret"] / len(true),
            "verifier_misjudged": entry["misjudged"] / entry["misses"] if probs is not None else float("nan"),
        })
    rows.sort(key=lambda r: -r["share_of_regret"])
    return rows, regret


def write_csv(path: Path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fmt(v):
    return f"{v:.4f}" if isinstance(v, float) else str(v)


def print_table(title: str, rows):
    print(f"\n## {title}")
    if not rows:
        print("(none)")
        return
    keys = list(rows[0])
    widths = [max(len(k), *(len(fmt(r[k])) for r in rows)) for k in keys]
    print("  ".join(k.ljust(w) for k, w in zip(keys, widths)))
    for r in rows:
        print("  ".join(fmt(r[k]).ljust(w) for k, w in zip(keys, widths)))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="label dir (one subdirectory per scenario)")
    ap.add_argument("--ckpt", default=str(ROOT / "results" / "verifier" / "w3_oracle" / "best.pt"))
    ap.add_argument("--dp-args", default=str(ROOT / "checkpoints" / "args.json"))
    ap.add_argument("--labels", choices=["oracle", "cv"], default="oracle", help="target the selection is judged by")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="only the first N val samples (smoke test)")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None, help="output dir (default: <ckpt dir>/diagnostic)")
    args = ap.parse_args()

    out = Path(args.out) if args.out else Path(args.ckpt).parent / "diagnostic"
    out.mkdir(parents=True, exist_ok=True)

    _, val_files = split_by_scenario(find_samples(args.data), args.val_frac, args.split_seed)
    if args.limit:
        val_files = val_files[:args.limit]
    print(f"[diagnose] {len(val_files)} val samples, {len({f.parent for f in val_files})} scenarios")

    model = Verifier.load(args.ckpt, args.dp_args, device=args.device)
    loader = DataLoader(LabelDataset(val_files), batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=args.device.startswith("cuda"))
    outputs, oracle, cv = [], [], []
    for batch in loader:
        batch = to_device(batch, args.device)
        outputs.append(model(batch["inputs"], batch["candidates"]).float().cpu())
        oracle.append(batch[f"metrics_{args.labels}"].cpu())
        cv.append(batch["metrics_cv"].cpu())
    outputs, oracle, cv = torch.cat(outputs), torch.cat(oracle), torch.cat(cv)     # [N, K, 7 or 1], [N, K, 7]
    scenario_types = [f.parent.name.split("__", 2)[2] for f in val_files]

    # A classifier (one score per candidate, train.py --objective classify) predicts no metrics: its picks are
    # attributed to metrics through the labels, but the per-metric prediction stats do not apply
    classifier = outputs.shape[-1] == 1
    probs = None if classifier else torch.sigmoid(outputs)
    verifier_scores = outputs[..., 0] if classifier else combined_score(probs)

    true = combined_score(oracle)
    picks = {"verifier": first_max(verifier_scores), "pdm_sl": first_max(combined_score(cv))}

    # 1. Per-metric prediction quality
    quality = []
    for i, metric in enumerate([] if classifier else METRICS):
        p, y = probs[..., i], oracle[..., i]
        quality.append({
            "metric": metric,
            "mean_label": y.mean().item(),
            "failure_rate": (y < 1 - LABEL_TOLERANCE).float().mean().item(),
            "mae": (p - y).abs().mean().item(),
            "brier": ((p - y) ** 2).mean().item(),
            "failure_auc": auc(-p.flatten().numpy(), (y < 1 - LABEL_TOLERANCE).flatten().numpy()),
            "within_scene_pair_acc": pairwise_accuracy(p, y),
            "cv_within_scene_pair_acc": pairwise_accuracy(cv[..., i], y),
        })
    quality.append({
        "metric": "COMBINED", "mean_label": true.mean().item(), "failure_rate": float("nan"),
        "mae": float("nan") if classifier else (verifier_scores - true).abs().mean().item(),
        "brier": float("nan") if classifier else ((verifier_scores - true) ** 2).mean().item(),
        "failure_auc": float("nan"),
        "within_scene_pair_acc": pairwise_accuracy(verifier_scores, true),
        "cv_within_scene_pair_acc": pairwise_accuracy(combined_score(cv), true),
    })

    # 2. Attribution of the misses
    attribution, regrets = [], {}
    for name, pick in picks.items():
        rows, regrets[name] = attribute(name, pick, true, oracle, probs)
        attribution += rows

    # Where the two selectors' misses overlap
    v_miss, p_miss = regrets["verifier"] > TIE_TOLERANCE, regrets["pdm_sl"] > TIE_TOLERANCE
    overlap = {
        "both_hit": float((~v_miss & ~p_miss).mean()),
        "verifier_miss_pdm_hit": float((v_miss & ~p_miss).mean()),
        "verifier_hit_pdm_miss": float((~v_miss & p_miss).mean()),
        "both_miss": float((v_miss & p_miss).mean()),
        "regret_from_verifier_only_misses": float(regrets["verifier"][v_miss & ~p_miss].sum() / len(true)),
    }

    # 3. Per scenario type
    per_type = []
    types = np.array(scenario_types)
    for t in sorted(set(scenario_types)):
        rows = types == t
        per_type.append({
            "scenario_type": t, "n": int(rows.sum()),
            "verifier_regret": float(regrets["verifier"][rows].mean()),
            "pdm_sl_regret": float(regrets["pdm_sl"][rows].mean()),
            "gap": float(regrets["verifier"][rows].mean() - regrets["pdm_sl"][rows].mean()),
            "verifier_top1": float((regrets["verifier"][rows] <= TIE_TOLERANCE).mean()),
            "pdm_sl_top1": float((regrets["pdm_sl"][rows] <= TIE_TOLERANCE).mean()),
        })
    per_type.sort(key=lambda r: -r["gap"] * r["n"])

    summary = {
        "ckpt": args.ckpt, "labels": args.labels, "n": len(true),
        "regret": {k: float(v.mean()) for k, v in regrets.items()},
        "top1": {k: float((v <= TIE_TOLERANCE).mean()) for k, v in regrets.items()},
        "overlap": overlap,
    }
    print_table("Per-metric prediction quality (all val candidates)", quality)
    print_table("Miss attribution (primary cause per miss)", attribution)
    print("\n## Overlap of misses (share of samples)")
    for k, v in overlap.items():
        print(f"  {k}: {v:.4f}")
    print_table("Per scenario type (sorted by total regret gap)", per_type)
    print(f"\nregret {summary['regret']}  top1 {summary['top1']}")

    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    write_csv(out / "metric_quality.csv", quality)
    write_csv(out / "attribution.csv", attribution)
    write_csv(out / "per_type.csv", per_type)
    print(f"[diagnose] wrote {out}")


if __name__ == "__main__":
    main()
