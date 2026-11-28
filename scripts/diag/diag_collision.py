"""Why S1 hit a pedestrian (seed-0 closed-loop logs). For the pedestrian S1 hits, at each planning step before contact:
(a) was it in the verifier's agent input (DP's rule: 32 nearest agents, pedestrians and cyclists capped at the nearest 10
    when more than 32 agents are present, leftover slots refilled with more of them; agent_process.agent_past_process)?
(b) does S1's chosen plan, as planned, overlap that pedestrian's logged future (non-reactive, so exact), and when?
(c) would a constant-velocity forecast of the pedestrian (what PDM's scorer assumes) show the overlap, and when?
Run on aegis with the nuPlan env active and PYTHONPATH at the repo."""
import glob
import os
import sys
from pathlib import Path

import numpy as np
from shapely import affinity
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.planning.simulation.simulation_log import SimulationLog

E = os.path.expandvars("$NUPLAN_EXP_ROOT/exp/simulation/closed_loop_nonreactive_agents")
RUN = sys.argv[1] if len(sys.argv) > 1 else "dp_k16_verifier_s1_noise1.5"
TOKENS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["e94b99e48b355de6", "159942fd13675580"]
AGENT_TYPES = (TrackedObjectType.VEHICLE, TrackedObjectType.PEDESTRIAN, TrackedObjectType.BICYCLE)
VRU = (TrackedObjectType.PEDESTRIAN, TrackedObjectType.BICYCLE)
NUM_AGENTS, MAX_PED_BIKE = 32, 10


def selected_tokens(sample):
    """DP's agent selection at one step, by distance from ego's rear axle to each agent's current centre."""
    ego = sample.ego_state.rear_axle
    objs = [o for o in sample.observation.tracked_objects.tracked_objects if o.tracked_object_type in AGENT_TYPES]
    dist = np.array([np.hypot(o.center.x - ego.x, o.center.y - ego.y) for o in objs])
    order = list(np.argsort(dist))
    vru = [i for i in order if objs[i].tracked_object_type in VRU]
    veh = [i for i in order if objs[i].tracked_object_type == TrackedObjectType.VEHICLE]
    if len(objs) <= NUM_AGENTS:
        sel = order[:NUM_AGENTS]
    else:
        sel = vru[:MAX_PED_BIKE] + veh
        room = NUM_AGENTS - len(sel)
        if room > 0:
            sel += vru[MAX_PED_BIKE:MAX_PED_BIKE + room]
        sel = sorted(sel, key=lambda i: dist[i])[:NUM_AGENTS]
    tok = {objs[i].track_token for i in sel}
    vru_rank = {objs[i].track_token: r for r, i in enumerate(vru)}
    return tok, vru_rank, len(vru), len(veh), dist, objs


def find(sample, token):
    for o in sample.observation.tracked_objects.tracked_objects:
        if o.track_token == token:
            return o
    return None


def first_overlap(plan_states, t0_us, hist, k, ped_token, cv=None):
    """First time [s] after step k at which the planned ego footprint overlaps the pedestrian.
    cv = (box geometry at k, vx, vy) → constant-velocity forecast instead of the logged future."""
    for st in plan_states[1:]:
        dt = (st.time_point.time_us - t0_us) * 1e-6
        m = int(round(dt / 0.1))
        if m <= 0:
            continue
        foot = st.car_footprint.oriented_box.geometry
        if cv is None:
            if k + m >= len(hist):
                return None
            p = find(hist[k + m], ped_token)
            if p is None:
                continue
            geom = p.box.geometry
        else:
            g0, vx, vy = cv
            geom = affinity.translate(g0, vx * dt, vy * dt)
        if foot.intersects(geom):
            return dt
    return None


for token in TOKENS:
    path = Path(glob.glob(f"{E}/{RUN}/test14-hard-100/*/simulation_log/*/*/*/{token}/{token}.msgpack.xz")[0])
    hist = SimulationLog.load_data(path).simulation_history.data
    # contact: first step where ego's footprint overlaps a pedestrian or cyclist
    contact = None
    for j, s in enumerate(hist):
        foot = s.ego_state.car_footprint.oriented_box.geometry
        for o in s.observation.tracked_objects.get_tracked_objects_of_types(list(VRU)):
            if foot.intersects(o.box.geometry):
                contact = (j, o)
                break
        if contact:
            break
    if contact is None:
        print(f"===== {token}: no pedestrian/cyclist overlap found in {RUN}")
        continue
    j, ped = contact
    pt = ped.track_token
    print(f"===== {token} ({RUN}): contact at step {j} (t = {j * 0.1:.1f} s) with a {ped.tracked_object_type.name.lower()} "
          f"(speed {np.hypot(ped.velocity.x, ped.velocity.y):.2f} m/s); ego speed {hist[j].ego_state.dynamic_car_state.speed:.2f} m/s")
    first_seen = next((k for k in range(len(hist)) if find(hist[k], pt) is not None), None)
    print(f"  pedestrian first in the observation at step {first_seen} (t = {first_seen * 0.1:.1f} s)")
    print("  step   t[s]  ego v  dist  in_input  ped rank (VRUs, vehicles)  plan hits it (log future)  "
          "plan hits it (CV forecast)")
    rows = []
    for k in range(max(0, j - 60), j + 1):
        s = hist[k]
        p = find(s, pt)
        if p is None:
            continue
        tok, rank, nv, nveh, dist, objs = selected_tokens(s)
        e = s.ego_state.rear_axle
        d = float(np.hypot(p.center.x - e.x, p.center.y - e.y))
        states = s.trajectory.get_sampled_trajectory()
        t0 = s.ego_state.time_point.time_us
        hit_log = first_overlap(states, t0, hist, k, pt)
        hit_cv = first_overlap(states, t0, hist, k, pt, cv=(p.box.geometry, p.velocity.x, p.velocity.y))
        rows.append((k, pt in tok, hit_log, hit_cv))
        if (k - j) % 5 == 0 or k == j:
            fmt = lambda h: "no (8 s)" if h is None else f"+{h:.1f} s" + (" (beyond 4 s)" if h > 4.0 else "")
            print(f"  {k:4d}  {k * 0.1:5.1f}  {s.ego_state.dynamic_car_state.speed:5.2f}  {d:4.1f}  "
                  f"{'yes' if pt in tok else 'NO':8s}  {rank.get(pt, -1) + 1:3d} of {nv:3d} ({nveh} veh)       "
                  f"{fmt(hit_log):24s}  {fmt(hit_cv)}")
    r = np.array([(a, b, -1 if c is None else c, -1 if d is None else d) for a, b, c, d in rows], dtype=float)
    vis = r[:, 1] == 1
    hit4 = (r[:, 2] >= 0) & (r[:, 2] <= 4.0)
    hit8 = r[:, 2] >= 0
    cv4 = (r[:, 3] >= 0) & (r[:, 3] <= 4.0)
    print(f"  over the {len(r)} steps before contact with the pedestrian observed: in S1's input on {int(vis.sum())}; "
          f"S1's plan hits it within 4 s on {int(hit4.sum())} (within 8 s on {int(hit8.sum())}); "
          f"a CV forecast shows the hit within 4 s on {int(cv4.sum())}")
    if hit4.any():
        k4 = int(r[hit4][0, 0])
        print(f"  first step whose chosen plan hits it within 4 s: {k4} (t = {k4 * 0.1:.1f} s, {(j - k4) * 0.1:.1f} s before "
              f"contact); in input then: {'yes' if r[r[:, 0] == k4][0, 1] else 'NO'}; steps from then to contact with the "
              f"pedestrian in the input: {int(vis[r[:, 0] >= k4].sum())} of {int((r[:, 0] >= k4).sum())}")
