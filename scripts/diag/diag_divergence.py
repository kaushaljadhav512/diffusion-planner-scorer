"""(1) At-fault collision types per config and seed. (2) For the worst S1-vs-pdm_sl scenarios with seed-0 simulation
logs: the first planning step where S1's executed plan differs from pdm_sl's, what each plan did there, and when S1
first touches a pedestrian. Run on aegis with the nuPlan env active."""
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.planning.simulation.simulation_log import SimulationLog

E = os.path.expandvars("$NUPLAN_EXP_ROOT/exp/simulation/closed_loop_nonreactive_agents")
CFGS = {"first": "dp_k16_first_noise1.5", "pdm_sl": "dp_k16_pdm_sl_noise1.5",
        "s1": "dp_k16_verifier_s1_noise1.5", "w6": "dp_k16_verifier_w6_noise1.5"}

# ---- (1) collision types
rows = []
for cfg, base in CFGS.items():
    for seed in (0, 1, 2):
        name = base + ("" if seed == 0 else f"_seed{seed}")
        f = sorted(glob.glob(f"{E}/{name}/test14-hard-100/*/metrics/no_ego_at_fault_collisions.parquet"))[-1]
        d = pd.read_parquet(f)
        for _, r in d.iterrows():
            rows.append({"cfg": cfg, "seed": seed, "scenario": r.scenario_name, "type": r.scenario_type,
                         "vru": r.number_of_at_fault_collisions_with_VRUs_stat_value,
                         "vehicle": r.number_of_at_fault_collisions_with_vehicles_stat_value,
                         "object": r.number_of_at_fault_collisions_with_objects_stat_value,
                         "score": r.metric_score})
c = pd.DataFrame(rows)
c["what"] = np.select([c.vru > 0, c.vehicle > 0, c.object > 0], ["VRU", "vehicle", "object"], "none")
print("runs with an at-fault collision (of 100 per seed), by what was hit; gate score < 1:")
fail = c[c.score < 1]
print(fail.groupby(["cfg", "what"]).size().unstack(fill_value=0).loc[list(CFGS)].to_string())
print("\nscenario-seed runs with a VRU collision:")
v = fail[fail.what == "VRU"].pivot_table(index=["scenario", "type"], columns="cfg", values="seed",
                                         aggfunc=lambda s: ",".join(map(str, sorted(s))))
print(v.reindex(columns=list(CFGS)).fillna("").to_string(), "\n")


# ---- (2) divergence on seed-0 logs
def logpath(cfg, token):
    return Path(glob.glob(f"{E}/{CFGS[cfg]}/test14-hard-100/*/simulation_log/*/*/*/{token}/{token}.msgpack.xz")[0])


def plan_xy(traj):
    states = traj.get_sampled_trajectory()
    return np.array([[s.rear_axle.x, s.rear_axle.y] for s in states]), states


def plan_length(xy):
    return float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())


def peds(obs):
    return [o for o in obs.tracked_objects.get_tracked_objects_of_type(TrackedObjectType.PEDESTRIAN)]


def min_ped_dist(xy, ped_list):
    if not ped_list:
        return np.inf
    P = np.array([[p.center.x, p.center.y] for p in ped_list])
    return float(np.min(np.linalg.norm(xy[:, None, :] - P[None], axis=-1)))


for token in ["e94b99e48b355de6", "159942fd13675580", "40e1f34c92255a0e", "2754fbd9c2445dce"]:
    logs = {cfg: SimulationLog.load_data(logpath(cfg, token)).simulation_history.data for cfg in ("s1", "pdm_sl")}
    a, b = logs["s1"], logs["pdm_sl"]
    n = min(len(a), len(b))
    first_div = None
    for i in range(n):
        xa, _ = plan_xy(a[i].trajectory)
        xb, _ = plan_xy(b[i].trajectory)
        m = min(len(xa), len(xb))
        if np.max(np.linalg.norm(xa[:m] - xb[:m], axis=1)) > 0.05:
            first_div = i
            break
    print(f"===== {token} ({c[(c.scenario == token)].type.iloc[0]}), {n} steps at 0.1 s")
    if first_div is None:
        print("  plans never differ (seed 0)")
        continue
    i = first_div
    xa, sa = plan_xy(a[i].trajectory)
    xb, sb = plan_xy(b[i].trajectory)
    ego = a[i].ego_state
    pl = peds(a[i].observation)
    # plan speed at 2 s and 4 s ahead (index by time), and planned travel over the horizon
    def speed_at(states, t):
        t0 = states[0].time_point.time_s
        k = min(range(len(states)), key=lambda j: abs(states[j].time_point.time_s - t0 - t))
        return states[k].dynamic_car_state.speed if hasattr(states[k], "dynamic_car_state") else float("nan")
    print(f"  first different pick at step {i} (t = {i * 0.1:.1f} s); ego speed {ego.dynamic_car_state.speed:.2f} m/s; "
          f"pedestrians in view: {len(pl)}")
    for name, xy, st in (("s1", xa, sa), ("pdm_sl", xb, sb)):
        print(f"   {name:7s} plan: {plan_length(xy):5.1f} m over {st[-1].time_point.time_s - st[0].time_point.time_s:.1f} s, "
              f"speed at +2 s {speed_at(st, 2.0):.2f}, +4 s {speed_at(st, 4.0):.2f} m/s, "
              f"closest approach to a pedestrian's current position {min_ped_dist(xy, pl):.2f} m")
    # executed paths afterwards: S1's first pedestrian contact (ego footprint within 0.5 m of a pedestrian centre)
    contact = None
    for j in range(i, len(a)):
        e = a[j].ego_state
        poly = e.car_footprint.oriented_box.geometry
        for p in peds(a[j].observation):
            if poly.distance(p.box.geometry) < 1e-6:
                contact = (j, p)
                break
        if contact:
            break
    if contact:
        j, p = contact
        print(f"  S1 touches a pedestrian at step {j} (t = {j * 0.1:.1f} s, {(j - i) * 0.1:.1f} s after the first "
              f"different pick); ego speed {a[j].ego_state.dynamic_car_state.speed:.2f} m/s; pdm_sl ego speed at "
              f"that step {b[min(j, len(b) - 1)].ego_state.dynamic_car_state.speed:.2f} m/s")
    else:
        print("  no pedestrian contact by S1 found by footprint overlap")
    # ego speed profiles, every 2 s
    print("  ego speed every 2 s  s1:", " ".join(f"{a[k].ego_state.dynamic_car_state.speed:4.1f}" for k in range(0, len(a), 20)))
    print("                   pdm_sl:", " ".join(f"{b[k].ego_state.dynamic_car_state.speed:4.1f}" for k in range(0, len(b), 20)))
