"""Collect nuPlan closed-loop scores from every simulation run into one table.

Reads the metric aggregator parquet that nuPlan writes per run
(<run>/aggregator_metric/*weighted_average_metrics*.parquet) and prints the
final CLS score plus the per-metric averages, with the paper's numbers alongside.

    python scripts/summarize_results.py                 # latest run per (planner, split, challenge)
    python scripts/summarize_results.py --all           # every run
    python scripts/summarize_results.py --by-type       # also break scores down by scenario type
    python scripts/summarize_results.py --csv out.csv
"""
import argparse
import os
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
CHALLENGE_SHORT = {"closed_loop_nonreactive_agents": "nr", "closed_loop_reactive_agents": "r"}
METRICS = [
    "no_ego_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance",
    "ego_is_making_progress", "time_to_collision_within_bound", "speed_limit_compliance",
    "ego_progress_along_expert_route", "ego_is_comfortable",
]


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
    by_type = None
    types = df[df["log_name"].isna() & (df["scenario"] != "final_score")]
    if len(types):
        by_type = types[["scenario_type", "num_scenarios", "score"]].assign(
            score=lambda d: 100 * d["score"], planner=planner, split=split, challenge=out["challenge"])
    return out, by_type


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-root", default=os.environ.get("NUPLAN_EXP_ROOT", "."))
    ap.add_argument("--all", action="store_true", help="show every run, not just the latest per config")
    ap.add_argument("--by-type", action="store_true")
    ap.add_argument("--csv")
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
    if not args.all:
        df = df.groupby(["planner", "split", "challenge"], as_index=False).last()
    # Smoke runs cover a handful of scenarios, so a paper comparison would be meaningless.
    df["paper"] = [PAPER.get((p, s, c)) for p, s, c in zip(df.planner, df.split, df.challenge)]
    df["delta"] = df["CLS"] - df["paper"]

    pd.set_option("display.width", 200, "display.max_columns", 30)
    print("\n=== Closed-loop score (CLS, 0-100) ===")
    print(df[["planner", "split", "challenge", "n", "CLS", "paper", "delta", "timestamp"]].round(2).to_string(index=False))
    print("\n=== Per-metric averages (%) ===")
    print(df[["planner", "split", "challenge"] + [m for m in METRICS if m in df]].round(1).to_string(index=False))

    if args.by_type and types:
        t = pd.concat(types)
        t = t[t.timestamp.isin(df.timestamp)]
        pivot = t.pivot_table(index="scenario_type", columns=["planner", "split", "challenge"], values="score")
        print("\n=== CLS by scenario type ===")
        print(pivot.round(1).to_string())
    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
