"""Per-scenario diagnosis: where S1 loses CLS against pdm_sl (and first), and the open-loop miss-rate failures.
Reads aggregator_metric parquets only (no simulation logs). Run on aegis with the nuPlan env active."""
import glob
import os
import sys

import numpy as np
import pandas as pd

EXP = os.path.expandvars("$NUPLAN_EXP_ROOT/exp/simulation")
OUT = sys.argv[1] if len(sys.argv) > 1 else "diag_out"
os.makedirs(OUT, exist_ok=True)
CFGS = {"first": "dp_k16_first_noise1.5", "pdm_sl": "dp_k16_pdm_sl_noise1.5",
        "s1": "dp_k16_verifier_s1_noise1.5", "w6": "dp_k16_verifier_w6_noise1.5"}
GATES = ["no_ego_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance", "ego_is_making_progress"]
WEIGHTED = {"ego_progress_along_expert_route": 5, "time_to_collision_within_bound": 5,
            "speed_limit_compliance": 4, "ego_is_comfortable": 2}
SHORT = {"no_ego_at_fault_collisions": "collision", "drivable_area_compliance": "drivable",
         "driving_direction_compliance": "direction", "ego_is_making_progress": "making_progress",
         "ego_progress_along_expert_route": "progress", "time_to_collision_within_bound": "ttc",
         "speed_limit_compliance": "speed_limit", "ego_is_comfortable": "comfort"}


def load(challenge, cols):
    rows = []
    for cfg, base in CFGS.items():
        for seed in (0, 1, 2):
            name = base + ("" if seed == 0 else f"_seed{seed}")
            runs = sorted(glob.glob(f"{EXP}/{challenge}/{name}/test14-hard-100/*/aggregator_metric/*.parquet"))
            if not runs:
                print(f"missing {challenge} {name}")
                continue
            d = pd.read_parquet(runs[-1])
            d = d[d.log_name.notna()]  # per-scenario rows only
            assert len(d) == 100, (name, len(d))
            d = d[["scenario", "log_name", "scenario_type"] + cols].copy()
            d["cfg"], d["seed"] = cfg, seed
            rows.append(d)
    return pd.concat(rows, ignore_index=True)


# ---------------- closed loop ----------------
cl = load("closed_loop_nonreactive_agents", GATES + list(WEIGHTED) + ["score"])
print("closed-loop CLS check (mean over scenarios and seeds):")
print((100 * cl.groupby("cfg").score.mean()).round(2).to_string(), "\n")

wide = cl.pivot_table(index=["scenario", "scenario_type", "seed"], columns="cfg", values="score").reset_index()


def attribute(a, b):
    """Why config a scored below config b on one (scenario, seed): the first gate a fails and b passes, else the
    weighted metric with the largest weighted shortfall. Returns (cause, points lost in CLS units)."""
    for g in GATES:
        if a[g] < b[g]:
            return SHORT[g] + " (gate)"
    gaps = {SHORT[m]: w * (b[m] - a[m]) for m, w in WEIGHTED.items()}
    return max(gaps, key=gaps.get) if max(gaps.values()) > 0 else "other"


def compare(a_cfg, b_cfg):
    A = cl[cl.cfg == a_cfg].set_index(["scenario", "seed"])
    B = cl[cl.cfg == b_cfg].set_index(["scenario", "seed"])
    diff = (A.score - B.score)  # per (scenario, seed), in [−1, 1]
    rec = []
    for key, dlt in diff.items():
        if abs(dlt) < 1e-9:
            continue
        lo, hi = (A.loc[key], B.loc[key]) if dlt < 0 else (B.loc[key], A.loc[key])
        rec.append({"scenario": key[0], "seed": key[1], "scenario_type": A.loc[key].scenario_type,
                    "delta": dlt, "cause": attribute(lo, hi), "loser": a_cfg if dlt < 0 else b_cfg})
    r = pd.DataFrame(rec)
    print(f"===== {a_cfg} vs {b_cfg}: mean CLS difference {100 * diff.mean():+.2f} "
          f"(per seed: {', '.join(f'{100 * diff.xs(s, level=1).mean():+.2f}' for s in (0, 1, 2))})")
    print(f"(scenario, seed) pairs: {len(diff)}; {a_cfg} lower on {(diff < -1e-9).sum()}, higher on {(diff > 1e-9).sum()}, "
          f"equal on {(diff.abs() <= 1e-9).sum()}")
    # Net CLS points by cause: losses of a (negative) and wins of a (positive), divided by 300 pairs, in CLS points
    n = len(diff)
    t = r.assign(pts=100 * r.delta / n).groupby(["cause", "loser"]).pts.agg(["sum", "count"]).round(2)
    print(f"CLS points by cause (sum over pairs / {n}); loser = config that scored lower on that pair:")
    print(t.to_string(), "\n")
    # Concentration: per scenario, mean over seeds
    per = diff.groupby(level=0).mean().sort_values()
    losses = per[per < 0]
    print(f"scenarios with mean loss: {len(losses)}; worst 10 carry {100 * losses.head(10).sum() / losses.sum():.0f}% "
          f"of total loss; scenarios lost in all 3 seeds: "
          f"{int((diff.unstack() < -1e-9).all(axis=1).sum())}, won in all 3: {int((diff.unstack() > 1e-9).all(axis=1).sum())}")
    types = A.reset_index()[["scenario", "scenario_type"]].drop_duplicates().set_index("scenario").scenario_type
    bt = (100 * per).groupby(types).agg(["mean", "count"]).sort_values("mean").round(2)
    print("mean CLS difference by scenario type (per scenario, mean over seeds; 100 = whole score):")
    print(bt.to_string(), "\n")
    return r, per


r_sp, per_sp = compare("s1", "pdm_sl")
r_sf, per_sf = compare("s1", "first")
r_wp, per_wp = compare("w6", "pdm_sl")
r_fp, per_fp = compare("first", "pdm_sl")

# Worst S1-vs-pdm_sl scenarios, with every config's per-seed scores and S1's causes
worst = per_sp.head(15).index
tab = wide[wide.scenario.isin(worst)]
piv = tab.pivot_table(index=["scenario", "scenario_type"], columns="seed", values=["s1", "pdm_sl", "first", "w6"])
causes = r_sp[(r_sp.loser == "s1") & r_sp.scenario.isin(worst)].groupby("scenario").cause.agg(lambda c: ", ".join(sorted(set(c))))
print("===== 15 scenarios where S1 loses most to pdm_sl (scores 0-100 per seed):")
summ = pd.DataFrame({
    "type": tab.groupby("scenario").scenario_type.first(),
    "s1": tab.groupby("scenario").s1.apply(lambda s: " ".join(f"{100 * v:3.0f}" for v in s)),
    "pdm_sl": tab.groupby("scenario").pdm_sl.apply(lambda s: " ".join(f"{100 * v:3.0f}" for v in s)),
    "first": tab.groupby("scenario")["first"].apply(lambda s: " ".join(f"{100 * v:3.0f}" for v in s)),
    "w6": tab.groupby("scenario").w6.apply(lambda s: " ".join(f"{100 * v:3.0f}" for v in s)),
    "mean_loss": (100 * per_sp[worst]).round(1),
    "s1_causes": causes}).sort_values("mean_loss")
print(summ.to_string(), "\n")
summ.to_csv(f"{OUT}/cl_worst_s1_vs_pdm_sl.csv")
r_sp.to_csv(f"{OUT}/cl_pairs_s1_vs_pdm_sl.csv", index=False)
wide.to_csv(f"{OUT}/cl_scores_wide.csv", index=False)

# Gate failure rates per config (share of the 300 scenario-seed runs where the gate is < 1)
print("===== gate failures and weighted metrics, closed loop (% of 300 scenario-seed runs; weighted = mean score):")
g = cl.groupby("cfg")
gate = pd.DataFrame({SHORT[c]: 100 * g[c].apply(lambda s: (s < 1).mean()) for c in GATES})
wt = pd.DataFrame({SHORT[c]: 100 * g[c].mean() for c in WEIGHTED})
print(pd.concat([gate.add_suffix(" fail%"), wt.add_suffix(" mean")], axis=1).round(1).loc[list(CFGS)].to_string(), "\n")

# ---------------- open loop ----------------
OLC = ["planner_miss_rate_within_bound", "planner_expert_average_l2_error_within_bound",
       "planner_expert_final_l2_error_within_bound", "planner_expert_average_heading_error_within_bound",
       "planner_expert_final_heading_error_within_bound", "score"]
ol = load("open_loop_boxes", OLC)
ol["miss_fail"] = ol.planner_miss_rate_within_bound < 1
print("===== open loop: miss-rate gate failures per seed (of 100 scenarios):")
print(ol.groupby(["cfg", "seed"]).miss_fail.sum().unstack().loc[list(CFGS)].to_string(), "\n")
mf = ol.pivot_table(index=["scenario", "scenario_type", "seed"], columns="cfg", values="miss_fail").reset_index()
only_v = mf[(mf.s1 == 1) & (mf.pdm_sl == 0) & (mf["first"] == 0)]
print(f"(scenario, seed) where S1 fails the miss gate but pdm_sl and first pass: {len(only_v)}; "
      f"distinct scenarios {only_v.scenario.nunique()}; also failed by w6 in the same seed: {int((only_v.w6 == 1).sum())}")
print(only_v.groupby("scenario_type").size().sort_values(ascending=False).to_string(), "\n")
cnt = mf.groupby(["scenario", "scenario_type"])[["s1", "w6", "pdm_sl", "first"]].sum()
rec = cnt[(cnt.s1 + cnt.w6) > (cnt.pdm_sl + cnt["first"])].sort_values(["s1", "w6"], ascending=False)
print("scenarios failing the miss gate more often with the verifiers (seeds failed, of 3):")
print(rec.head(15).to_string(), "\n")
rec.to_csv(f"{OUT}/ol_miss_verifier_scenarios.csv")

# Overlap with closed-loop losses
cl_loss = set(per_sp[per_sp < -0.05].index)
ol_bad = set(rec[(rec.s1 >= 1)].index.get_level_values(0))
print(f"overlap: scenarios with S1 closed-loop mean loss > 5 points: {len(cl_loss)}; with extra verifier miss "
      f"failures: {len(ol_bad)}; in both: {len(cl_loss & ol_bad)}")
