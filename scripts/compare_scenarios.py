"""Per-scenario comparison of best-of-K selectors against FirstSample (task 2.14).

For every dp_k run of a split, takes nuPlan's per-scenario scores (aggregator parquet) and compares each scenario with
the FirstSample run of the same K / noise / seed / challenge:

    per (config, challenge):   scenarios improved / worse / unchanged per seed, and how many move the same way in
                               every seed (a gain that repeats across seeds is real, one that flips is sampling noise)
    per scenario:              mean change over seeds and the number of seeds with a gain / a loss

Scenarios where no selector changes the score at any seed are tagged "insensitive": there the K candidates are
effectively one plan (near-zero sample spread), so no selector, learned or not, can help.

    python scripts/compare_scenarios.py --split test14-hard-100
    python scripts/compare_scenarios.py --split test14-hard-100 --config dp_k16_pdm_noise1.5 --challenge nr --top 15
"""
import argparse
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_results import CHALLENGE_SHORT, DP_K_RUN  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CHANGE = 1e-6  # scores are fractions in [0, 1]; anything above this is a change


def scenario_scores(exp_root: Path, split: str) -> pd.DataFrame:
    """
    Latest run per (planner, challenge) of the split, one row per scenario.

    Returns:
        columns planner, challenge, scenario, scenario_type, score (0-100)
    """
    runs = {}
    for parquet in sorted(Path(exp_root).rglob("aggregator_metric/*weighted_average_metrics*.parquet")):
        run_dir = parquet.parent.parent # .../<challenge>/<planner>/<split>/<timestamp>/aggregator_metric/x.parquet
        if run_dir.parent.name != split:
            continue
        planner, challenge = run_dir.parent.parent.name, run_dir.parent.parent.parent.name
        runs[(planner, CHALLENGE_SHORT.get(challenge, challenge))] = parquet # sorted: the latest timestamp wins

    frames = []
    for (planner, challenge), parquet in sorted(runs.items()):
        df = pd.read_parquet(parquet)
        df = df[df["log_name"].notna() & (df["scenario"] != "final_score")]
        frames.append(df[["scenario", "scenario_type"]].assign(planner=planner, challenge=challenge,
                                                               score=100 * df["score"].astype(float)))
    if not frames:
        return pd.DataFrame(columns=["planner", "challenge", "scenario", "scenario_type", "score"])
    return pd.concat(frames, ignore_index=True)


def deltas_vs_first(scores: pd.DataFrame) -> pd.DataFrame:
    """
    Per-scenario change against FirstSample of the same K / noise / seed / challenge.

    Returns:
        columns config (run name without seed and selector suffixes kept), selector, seed, challenge, scenario,
        scenario_type, score, first, delta
    """
    parsed = scores.planner.str.extract(DP_K_RUN.pattern)
    scores = scores.assign(k=parsed[0], selector=parsed[1], noise=parsed[2].fillna(""), seed=parsed[3].fillna(""))
    scores = scores[scores.k.notna()]
    first = scores[scores.selector == "first"][["k", "noise", "seed", "challenge", "scenario", "score"]]
    out = scores[scores.selector != "first"].merge(
        first.rename(columns={"score": "first"}), on=["k", "noise", "seed", "challenge", "scenario"])
    out["config"] = out.k + "_" + out.selector + out.noise
    out["seed"] = out.seed.str.replace("_seed", "").replace("", "0").astype(int)
    out["delta"] = out.score - out["first"]
    return out[["config", "selector", "seed", "challenge", "scenario", "scenario_type", "score", "first", "delta"]]


def per_scenario(deltas: pd.DataFrame) -> pd.DataFrame:
    """Mean change over seeds and gain / loss counts per (config, challenge, scenario)."""
    grouped = deltas.groupby(["config", "challenge", "scenario", "scenario_type"])["delta"]
    out = grouped.agg(seeds="size", mean_delta="mean",
                      gains=lambda d: int((d > CHANGE).sum()), losses=lambda d: int((d < -CHANGE).sum())).reset_index()
    out["consistent"] = ((out.gains == out.seeds) | (out.losses == out.seeds)) & (out.seeds > 1)

    # Insensitive: no selector changes the score at any seed of this challenge (given the noise setting in config)
    noise = out.config.str.extract(r"(_noise[\d.]+(?:-[\d.]+)?)?$")[0].fillna("")
    changed = (out.gains + out.losses > 0).groupby([noise, out.challenge, out.scenario]).transform("any")
    out["insensitive"] = ~changed
    return out


def per_config(deltas: pd.DataFrame, scenarios: pd.DataFrame) -> pd.DataFrame:
    """Counts per (config, challenge): improved / worse / unchanged per seed (mean), and consistent gains / losses."""
    per_seed = deltas.assign(improved=deltas.delta > CHANGE, worse=deltas.delta < -CHANGE).groupby(
        ["config", "challenge", "seed"]).agg(improved=("improved", "sum"), worse=("worse", "sum"), n=("delta", "size"))
    out = per_seed.groupby(["config", "challenge"]).agg(
        seeds=("n", "size"), scenarios=("n", "mean"), improved=("improved", "mean"), worse=("worse", "mean"))
    consistent = scenarios[scenarios.consistent]
    out["consistent_gains"] = consistent[consistent.gains > 0].groupby(["config", "challenge"]).size()
    out["consistent_losses"] = consistent[consistent.losses > 0].groupby(["config", "challenge"]).size()
    out["insensitive"] = scenarios[scenarios.insensitive].groupby(["config", "challenge"]).size()
    return out.fillna(0).reset_index()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp-root", default=os.environ.get("NUPLAN_EXP_ROOT", "."))
    ap.add_argument("--split", default="test14-hard-100")
    ap.add_argument("--config", help="print the top movers of this config (e.g. dp_k16_pdm_noise1.5)")
    ap.add_argument("--challenge", default="nr")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--out", default=None, help="per-scenario CSV (default: results/per_scenario_<split>.csv)")
    args = ap.parse_args()

    deltas = deltas_vs_first(scenario_scores(Path(args.exp_root), args.split))
    if deltas.empty:
        print(f"No best-of-K runs with a matching FirstSample run on split {args.split}")
        return
    scenarios = per_scenario(deltas)
    summary = per_config(deltas, scenarios)

    pd.set_option("display.width", 200, "display.max_columns", 30)
    print("=== Scenarios that change vs FirstSample (improved / worse: mean over seeds) ===")
    print(summary.round(1).to_string(index=False))

    if args.config:
        rows = scenarios[(scenarios.config == args.config) & (scenarios.challenge == args.challenge)]
        rows = rows.sort_values("mean_delta")
        cols = ["scenario", "scenario_type", "seeds", "mean_delta", "gains", "losses", "consistent"]
        print(f"\n=== {args.config} / {args.challenge}: top {args.top} gains ===")
        print(rows[::-1].head(args.top)[cols].round(2).to_string(index=False))
        print(f"\n=== {args.config} / {args.challenge}: top {args.top} losses ===")
        print(rows.head(args.top)[cols].round(2).to_string(index=False))
        by_type = rows.groupby("scenario_type").agg(mean_delta=("mean_delta", "mean"), insensitive=("insensitive", "sum"))
        print(f"\n=== {args.config} / {args.challenge}: by scenario type ===")
        print(by_type.round(2).to_string())

    out = args.out or ROOT / "results" / f"per_scenario_{args.split}.csv"
    scenarios.to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
