"""Sanity statistics of recorded verifier labels (task 2.9).

    python scripts/label_stats.py $NUPLAN_HOME/labels/train-labels-2k

Prints, over all samples (one sample = one recorded step with K candidates), with the oracle labels:

    per-metric pass rate        mean of each metric over all candidates
    informative                 fraction of samples whose K combined scores are not all tied
    best_not_first              fraction where candidate 0 is not among the best (where selection can help)
    headroom                    mean of best combined score - candidate 0's (offline oracle gain over FirstSample)
    random_regret               mean of best - mean over candidates (expected loss of a random pick)
    cv_top1 / cv_regret         the constant-velocity labels' argmax (PDM selection) judged by the oracle labels
    cv_agreement                fraction where the oracle's and constant-velocity labels pick the same candidate

and the same by scenario type.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from diffusion_planner.scorer.dataset import find_samples  # noqa: E402
from diffusion_planner.scorer.labels import load_sample  # noqa: E402
from diffusion_planner.scorer.metrics import METRICS, combined_score  # noqa: E402

TIE = 1e-6


def sample_stats(sample) -> dict:
    oracle, cv = combined_score(sample["metrics_oracle"]), combined_score(sample["metrics_cv"])
    best = oracle.max()
    cv_pick = int(np.argmax(cv >= cv.max() - TIE)) # ties to the lowest index, as PDMSelector
    oracle_pick = int(np.argmax(oracle >= best - TIE))
    row = {
        "scenario_type": sample["scenario_type"],
        "informative": float(best - oracle.min() > TIE),
        "best_not_first": float(oracle[0] < best - TIE),
        "headroom": float(best - oracle[0]),
        "random_regret": float(best - oracle.mean()),
        "cv_top1": float(oracle[cv_pick] >= best - TIE),
        "cv_regret": float(best - oracle[cv_pick]),
        "cv_agreement": float(cv_pick == oracle_pick),
    }
    row.update({f"pass_{m}": float(v) for m, v in zip(METRICS, sample["metrics_oracle"].mean(axis=0))})
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("label_dir")
    ap.add_argument("--csv", help="also write the per-type table here")
    args = ap.parse_args()

    files = find_samples(args.label_dir)
    assert files, f"no label samples under {args.label_dir}"
    df = pd.DataFrame([sample_stats(load_sample(f)) for f in files])
    size_mb = sum(f.stat().st_size for f in files) / 1e6
    print(f"{len(files)} samples from {len({f.parent for f in files})} scenarios, {size_mb:.0f} MB")

    pd.set_option("display.width", 200, "display.max_columns", 30)
    print("\n=== Per-metric pass rate (oracle labels, all candidates) ===")
    print(df[[c for c in df if c.startswith("pass_")]].mean().round(3).to_string())
    selection = ["informative", "best_not_first", "headroom", "random_regret", "cv_top1", "cv_regret", "cv_agreement"]
    print("\n=== Selection statistics (combined score, CLS weights) ===")
    print(df[selection].mean().round(4).to_string())
    by_type = df.groupby("scenario_type")[selection].mean().assign(samples=df.groupby("scenario_type").size())
    print("\n=== By scenario type ===")
    print(by_type.round(3).to_string())
    if args.csv:
        by_type.to_csv(args.csv)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
