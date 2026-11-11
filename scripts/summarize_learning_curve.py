"""
Table of the verifier learning curve (scripts/run_learning_curve.sh): held-out regret and top-1 per training
fraction, mean and std over seeds, with the PDM+SL / Random references from the same validation set.

    python scripts/summarize_learning_curve.py runs/learning_curve --csv results/verifier/learning_curve.csv
"""
import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", help="directory with one frac<f>_seed<s> run per subdirectory")
    ap.add_argument("--csv", help="write one row per run here")
    args = ap.parse_args()

    rows = []
    for d in sorted(Path(args.runs).glob("frac*_seed*")):
        m = re.fullmatch(r"frac([\d.]+)_seed(\d+)", d.name)
        best = d / "best_metrics.json"
        if not m or not best.exists():
            continue
        metrics = json.loads(best.read_text())
        log = (d / "train.log").read_text() if (d / "train.log").exists() else ""
        n = re.search(r"(\d+) train / \d+ val samples, (\d+) / \d+ scenarios", log)
        rows.append({
            "frac": float(m.group(1)), "seed": int(m.group(2)),
            "train_samples": int(n.group(1)) if n else "", "train_scenarios": int(n.group(2)) if n else "",
            "best_epoch": metrics["epoch"],
            "regret": metrics["all/verifier/regret"], "top1": metrics["all/verifier/top1"],
            "regret_informative": metrics["informative/verifier/regret"],
            "pdm_regret": metrics["all/pdm/regret"], "random_regret": metrics["all/random/regret"],
        })
    if not rows:
        print(f"no finished runs under {args.runs}")
        return

    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(sorted(rows, key=lambda r: (r["frac"], r["seed"])))

    by_frac = defaultdict(list)
    for r in rows:
        by_frac[r["frac"]].append(r)

    def mean_std(values):
        return f"{statistics.mean(values):.4f} ± {statistics.stdev(values):.4f}" if len(values) > 1 \
            else f"{values[0]:.4f}"

    print(f"{'frac':>5} {'scenarios':>9} {'seeds':>5} {'regret':>17} {'top-1':>17}")
    for frac in sorted(by_frac):
        rs = by_frac[frac]
        print(f"{frac:>5} {str(rs[0]['train_scenarios']):>9} {len(rs):>5} "
              f"{mean_std([r['regret'] for r in rs]):>17} {mean_std([r['top1'] for r in rs]):>17}")
    print(f"reference on the same val set: PDM+SL regret {rows[0]['pdm_regret']:.4f}, "
          f"Random {rows[0]['random_regret']:.4f}")


if __name__ == "__main__":
    main()
