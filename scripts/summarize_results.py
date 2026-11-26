"""Collect nuPlan closed-loop scores from every simulation run into one table.

Reads the metric aggregator parquet that nuPlan writes per run
(<run>/aggregator_metric/*weighted_average_metrics*.parquet) and prints the
final CLS score plus the per-metric averages, with the paper's numbers alongside.

    python scripts/summarize_results.py                 # latest run per (planner, split, challenge)
    python scripts/summarize_results.py --all           # every run
    python scripts/summarize_results.py --by-type       # also break scores down by scenario type
    python scripts/summarize_results.py --csv out.csv
    python scripts/summarize_results.py --split test14-hard-100   # only that split (the headroom table)
    python scripts/summarize_results.py --split test14-hard-100 --seeds-csv seeds.csv   # also mean / std over seeds

Best-of-K runs (dp_k<K>_<selector>[_noise<f>][_seed<n>]) also get the gain over FirstSample at the same K / noise /
seed, and the per-step sampling / selection time that the planner reports to nuPlan (<run>/runner_report.parquet).
Configurations run with several seeds are also averaged: mean and std of CLS and of the gain over FirstSample.
"""
import argparse
import os
import re
from pathlib import Path

import pandas as pd

# Published numbers (Diffusion Planner README / paper) for sanity-checking reproduction.
PAPER = {
    ("dp", "val14", "nr"): 89.87, ("dp", "val14", "r"): 82.80,
    ("dp", "test14-hard", "nr"): 75.99, ("dp", "test14-hard", "r"): 69.22,
    ("dp", "test14-random", "nr"): 89.19, ("dp", "test14-random", "r"): 82.93,
    ("pdm_closed", "val14", "nr"): 92.84, ("pdm_closed", "val14", "r"): 92.12,
    ("pdm_closed", "test14-hard", "nr"): 65.08, ("pdm_closed", "test14-hard", "r"): 75.19,
    ("pdm_closed", "test14-random", "nr"): 90.05, ("pdm_closed", "test14-random", "r"): 91.63,
}
CHALLENGE_SHORT = {"closed_loop_nonreactive_agents": "nr", "closed_loop_reactive_agents": "r", "open_loop_boxes": "ol"}
METRICS = [
    "no_ego_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance",
    "ego_is_making_progress", "time_to_collision_within_bound", "speed_limit_compliance",
    "ego_progress_along_expert_route", "ego_is_comfortable",
]
# Per-step runtimes [ms]: mean over scenarios of each scenario's mean (runner_report.parquet columns)
RUNTIMES = {
    "step_ms": "compute_trajectory_runtimes_mean",
    "sample_ms": "sampling_runtimes_mean",   # dp_k only
    "select_ms": "selection_runtimes_mean",  # dp_k only
}
# dp_k<K>_<selector>[_noise<f>[-<f>]][_seed<n>], see run_sim.sh; selectors may contain "_" (pdm_sl)
DP_K_RUN = re.compile(r"^(dp_k\d+)_([a-z]+(?:_[a-z]+)*?)(_noise[\d.]+(?:-[\d.]+)?)?(_seed\d+)?$")


def parse_run(parquet: Path):
    """Path layout from run_sim.sh: .../simulation/<challenge>/<planner>/<split>/<timestamp>/aggregator_metric/x.parquet"""
    run_dir = parquet.parent.parent
    timestamp, split, planner, challenge = run_dir.name, run_dir.parent.name, run_dir.parent.parent.name, run_dir.parent.parent.parent.name
    df = pd.read_parquet(parquet)
    final = df[df["scenario"] == "final_score"]
    scen = df[df["log_name"].notna() & (df["scenario"] != "final_score")]
    row = final.iloc[0] if len(final) else scen.mean(numeric_only=True)
    out = {
        "planner": planner, "split": split, "challenge": CHALLENGE_SHORT.get(challenge, challenge),
        "timestamp": timestamp, "n": len(scen), "CLS": 100 * float(row["score"]),
    }
    out.update({m: 100 * float(row[m]) for m in METRICS if m in row and pd.notna(row[m])})
    out.update(parse_runner_report(run_dir / "runner_report.parquet"))
    by_type = None
    types = df[df["log_name"].isna() & (df["scenario"] != "final_score")]
    if len(types):
        by_type = types[["scenario_type", "num_scenarios", "score"]].assign(
            score=lambda d: 100 * d["score"], planner=planner, split=split, challenge=out["challenge"])
    return out, by_type


def parse_runner_report(path: Path):
    """Failed scenarios and per-step runtimes [ms] of a run; empty if nuPlan did not write the report."""
    if not path.exists():
        return {}
    df = pd.read_parquet(path)
    out = {"failed": int((~df["succeeded"].astype(bool)).sum())} if "succeeded" in df else {}
    out.update({name: 1000 * df[col].mean() for name, col in RUNTIMES.items() if col in df})
    return out


def gain_over_first(df: pd.DataFrame) -> pd.Series:
    """CLS minus the CLS of dp_k<K>_first[_noise][_seed] on the same split and challenge (NaN when not a dp_k run or missing)."""
    cls = {(p, s, c): v for p, s, c, v in zip(df.planner, df.split, df.challenge, df.CLS)}
    gains = []
    for p, s, c, v in zip(df.planner, df.split, df.challenge, df.CLS):
        m = DP_K_RUN.match(p)
        first = cls.get((f"{m.group(1)}_first{m.group(3) or ''}{m.group(4) or ''}", s, c)) if m else None
        gains.append(v - first if first is not None else float("nan"))
    return pd.Series(gains, index=df.index)


def over_seeds(df: pd.DataFrame) -> pd.DataFrame:
    """Mean / std over seeds of CLS and vs_first per dp_k configuration (the run name without _seed<n>)."""
    runs = df[df.planner.map(lambda p: bool(DP_K_RUN.match(p)))]
    out = runs.assign(config=runs.planner.str.replace(r"_seed\d+$", "", regex=True)).groupby(["config", "split", "challenge"]).agg(
        seeds=("CLS", "size"), CLS_mean=("CLS", "mean"), CLS_std=("CLS", "std"),
        vs_first_mean=("vs_first", "mean"), vs_first_std=("vs_first", "std"))
    return out.reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-root", default=os.environ.get("NUPLAN_EXP_ROOT", "."))
    ap.add_argument("--all", action="store_true", help="show every run, not just the latest per config")
    ap.add_argument("--by-type", action="store_true")
    ap.add_argument("--split", help="only runs of this split (as named in the run path, e.g. test14-hard-100_smoke5)")
    ap.add_argument("--csv")
    ap.add_argument("--seeds-csv", help="write the mean / std over seeds table here")
    args = ap.parse_args()

    files = sorted(Path(args.exp_root).rglob("aggregator_metric/*weighted_average_metrics*.parquet"))
    if not files:
        print(f"No aggregator parquet files under {args.exp_root}")
        return
    rows, types = [], []
    for f in files:
        try:
            r, t = parse_run(f)
        except Exception as e:  # half-finished runs etc.
            print(f"[skip] {f}: {e}")
            continue
        rows.append(r)
        if t is not None:
            types.append(t.assign(timestamp=r["timestamp"]))

    df = pd.DataFrame(rows).sort_values(["planner", "split", "challenge", "timestamp"])
    if args.split:
        df = df[df.split == args.split]
        if df.empty:
            print(f"No runs of split {args.split}")
            return
    if not args.all:
        df = df.groupby(["planner", "split", "challenge"], as_index=False).last()
    # Smoke runs cover a handful of scenarios, so a paper comparison would be meaningless.
    df["paper"] = [PAPER.get((p, s, c)) for p, s, c in zip(df.planner, df.split, df.challenge)]
    df["delta"] = df["CLS"] - df["paper"]
    df["vs_first"] = gain_over_first(df)

    pd.set_option("display.width", 200, "display.max_columns", 30)
    print("\n=== Closed-loop score (CLS, 0-100) ===")
    print(df[["planner", "split", "challenge", "n", "CLS", "paper", "delta", "timestamp"]].round(2).to_string(index=False))
    print("\n=== Per-metric averages (%) ===")
    print(df[["planner", "split", "challenge"] + [m for m in METRICS if m in df]].round(1).to_string(index=False))
    if df["vs_first"].notna().any():
        print("\n=== Best-of-K: gain over FirstSample (CLS points) and per-step runtime (ms) ===")
        cols = ["planner", "split", "challenge", "n", "CLS", "vs_first"] + [c for c in ["failed", *RUNTIMES] if c in df]
        print(df[df["vs_first"].notna()][cols].round(2).to_string(index=False))

    seeds = over_seeds(df)
    if (seeds.seeds > 1).any():
        print("\n=== Best-of-K over seeds: mean / std of CLS and gain over FirstSample ===")
        print(seeds.round(2).to_string(index=False))

    if args.by_type and types:
        t = pd.concat(types)
        t = t[t.timestamp.isin(df.timestamp)]
        pivot = t.pivot_table(index="scenario_type", columns=["planner", "split", "challenge"], values="score")
        print("\n=== CLS by scenario type ===")
        print(pivot.round(1).to_string())
    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"\nwrote {args.csv}")
    if args.seeds_csv:
        seeds.to_csv(args.seeds_csv, index=False)
        print(f"wrote {args.seeds_csv}")


if __name__ == "__main__":
    main()
