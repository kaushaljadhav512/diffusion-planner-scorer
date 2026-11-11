"""
Summary of the reactive agreement check (diffusion_planner/scorer/agreement.py): how well picks made with
constant-velocity forecasts (PDM+SL) or logged futures do when the candidates are scored against reactive agents.

    python scripts/agreement_report.py $NUPLAN_HOME/agreement/train-agree-100 [--csv results/agreement.csv]

All scores are the labels' combined score (CLS weights). "agree" = the pick is among the best candidates under the
target (ties within 1e-6 count, as top-1 in train.py); regret = best score minus the pick's score. 95% intervals
bootstrap over scenarios (samples of one scenario are correlated). For reference, in non-reactive label data the cv
pick agrees with the logged-future pick on 92% of steps (docs/verifier_heldout_report.md).
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diffusion_planner.scorer.metrics import combined_score  # noqa: E402 (numpy only)

TIE = 1e-6
KEYS = ["metrics_cv", "metrics_log", "metrics_reactive", "selected", "reactive_seconds", "scenario_type", "iteration"]


def load(root: Path):
    rows = []
    for path in sorted(root.glob("*/*.npz")):
        if path.name.endswith(".tmp.npz"):
            continue
        with np.load(path) as data:
            row = {k: data[k] for k in KEYS}
        row["scenario"] = path.parent.name
        rows.append(row)
    return rows


def pick_stats(target: np.ndarray, pick: np.ndarray):
    """target [N, K] scores, pick [N] indices -> agree [N] bool, regret [N]."""
    best = target.max(axis=1)
    picked = target[np.arange(len(target)), pick]
    return picked >= best - TIE, best - picked


def first_max(scores: np.ndarray) -> np.ndarray:
    """argmax with ties to the lowest index (PDMSelector's rule)."""
    return (scores >= scores.max(axis=1, keepdims=True) - TIE).argmax(axis=1)


def bootstrap(values: np.ndarray, groups: np.ndarray, n: int = 2000, seed: int = 0):
    """95% interval of the mean, resampling groups (scenarios)."""
    rng = np.random.default_rng(seed)
    names = np.unique(groups)
    by_group = [values[groups == g] for g in names]
    means = [np.concatenate([by_group[i] for i in rng.integers(0, len(names), len(names))]).mean() for _ in range(n)]
    return np.percentile(means, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, help="ReactiveAgreementRecorder out_dir")
    ap.add_argument("--csv", type=Path, help="per-sample picks and regrets")
    args = ap.parse_args()

    rows = load(args.root)
    if not rows:
        sys.exit(f"no samples under {args.root}")
    scores = {name: np.stack([combined_score(r[f"metrics_{name}"].astype(np.float64)) for r in rows])
              for name in ("cv", "log", "reactive")}
    groups = np.array([r["scenario"] for r in rows])
    types = np.array([str(r["scenario_type"]) for r in rows])
    K = scores["reactive"].shape[1]
    reactive = scores["reactive"]
    informative = reactive.max(axis=1) - reactive.min(axis=1) > TIE

    picks = {
        "PDM+SL (cv)": first_max(scores["cv"]),
        "logged futures": first_max(scores["log"]),
        "first sample": np.zeros(len(rows), dtype=int),
    }
    executed = np.array([int(r["selected"]) for r in rows])
    assert (executed == picks["PDM+SL (cv)"]).all(), "recorded selection differs from the cv argmax"

    print(f"{len(rows)} samples from {len(np.unique(groups))} scenarios, K = {K}; "
          f"candidates not all tied under reactive scoring: {informative.mean():.1%}")
    print(f"reactive scoring: {np.mean([float(r['reactive_seconds']) for r in rows]):.1f} s per step (K candidates)\n")

    print("Picks scored against reactive agents (all samples)")
    print(f"{'pick':<16} {'agree':>7} {'95% CI':>15} {'regret':>8} {'95% CI':>17}")
    for name, pick in picks.items():
        agree, regret = pick_stats(reactive, pick)
        a_lo, a_hi = bootstrap(agree.astype(float), groups)
        r_lo, r_hi = bootstrap(regret, groups)
        print(f"{name:<16} {agree.mean():>7.1%} {a_lo:>7.1%}-{a_hi:<6.1%} {regret.mean():>8.4f} {r_lo:>8.4f}-{r_hi:.4f}")
    random_agree = (reactive >= reactive.max(axis=1, keepdims=True) - TIE).mean()
    random_regret = (reactive.max(axis=1, keepdims=True) - reactive).mean()
    print(f"{'random':<16} {random_agree:>7.1%} {'':>15} {random_regret:>8.4f}")

    if informative.any():
        print("\nSame, only samples whose candidates are not all tied under reactive scoring")
        for name, pick in picks.items():
            agree, regret = pick_stats(reactive[informative], pick[informative])
            print(f"{name:<16} {agree.mean():>7.1%} {regret.mean():>24.4f}")

    agree_cv_log, _ = pick_stats(scores["log"], picks["PDM+SL (cv)"])
    print(f"\nReference, same states: PDM+SL's pick among the best under logged futures: {agree_cv_log.mean():.1%} "
          f"(non-reactive label data: 92%)")

    print("\nBy scenario type (PDM+SL pick against reactive agents)")
    print(f"{'type':<45} {'n':>4} {'agree':>7} {'regret':>8}")
    agree, regret = pick_stats(reactive, picks["PDM+SL (cv)"])
    for t in sorted(set(types)):
        m = types == t
        print(f"{t:<45} {m.sum():>4} {agree[m].mean():>7.1%} {regret[m].mean():>8.4f}")

    check_path = args.root / "fork_check.jsonl"
    if check_path.exists():
        checks = [json.loads(line) for line in check_path.read_text().splitlines() if line.strip()]
        deviations = np.array([c["max_deviation_m"] for c in checks])
        mismatched = sum(c["only_predicted"] + c["only_actual"] > 0 for c in checks)
        print(f"\nFork fidelity ({len(checks)} checks): max agent deviation one step ahead "
              f"{deviations.max():.2e} m, median {np.median(deviations):.2e} m; steps with differing agent sets: "
              f"{mismatched}")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["scenario", "scenario_type", "iteration", "pick_cv", "pick_log", "best_reactive",
                             "regret_cv", "regret_log", "informative"])
            _, regret_log = pick_stats(reactive, picks["logged futures"])
            for i, r in enumerate(rows):
                writer.writerow([r["scenario"], str(r["scenario_type"]), int(r["iteration"]), picks["PDM+SL (cv)"][i],
                                 picks["logged futures"][i], int(first_max(reactive[i:i + 1])[0]),
                                 f"{regret[i]:.6f}", f"{regret_log[i]:.6f}", bool(informative[i])])


if __name__ == "__main__":
    main()
