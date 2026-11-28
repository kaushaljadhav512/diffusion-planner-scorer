"""Analyse the probe run: (1) does it reproduce the original S1 seed-0 run; (2) at the steps where S1's pick fails
PDM+SL's collision or TTC check, which predicted metric made S1 prefer it over PDM+SL's pick.
Run on aegis with the nuPlan env active."""
import glob
import os
from pathlib import Path

import numpy as np
from nuplan.planning.simulation.simulation_log import SimulationLog

E = os.path.expandvars("$NUPLAN_EXP_ROOT/exp/simulation/closed_loop_nonreactive_agents")
PROBE = Path(os.path.expanduser("~/IDL_project/dps-closed-loop/runs/diag/probe_out"))
NAMES = ["collision", "drivable", "direction", "progress", "ttc", "speed_limit", "comfort"]
MULT, WT = [0, 1, 2], [3, 4, 5, 6]
W = np.array([5.0, 5.0, 4.0, 2.0])
CONTACT = {"e94b99e48b355de6": 58, "159942fd13675580": 51}  # from diag_collision.py on the original run


def log_of(run, token):
    p = Path(glob.glob(f"{E}/{run}/test14-hard-100/*/simulation_log/*/*/*/{token}/{token}.msgpack.xz")[0])
    return SimulationLog.load_data(p).simulation_history.data


def decompose(p_a, p_b):
    """log S(a) − log S(b) of the combined score, split into one term per multiplicative metric (log ratio) and the
    weighted-mean term; the weighted term is further split linearly by metric (its share of W(a) − W(b))."""
    eps = 1e-6
    mult = {NAMES[i]: float(np.log(p_a[i] + eps) - np.log(p_b[i] + eps)) for i in MULT}
    wa, wb = float(W @ p_a[WT] / W.sum()), float(W @ p_b[WT] / W.sum())
    wterm = float(np.log(wa + eps) - np.log(wb + eps))
    dW = wa - wb
    split = {NAMES[i]: (wterm * (W[j] * (p_a[i] - p_b[i]) / W.sum()) / dW if abs(dW) > 1e-9 else 0.0)
             for j, i in enumerate(WT)}
    return {**mult, **split}


for token, contact in CONTACT.items():
    orig = log_of("dp_k16_verifier_s1_noise1.5", token)
    probe = log_of("dp_k16_verifier_s1probe_noise1.5", token)
    n = min(len(orig), len(probe))
    dev = [np.hypot(orig[i].ego_state.rear_axle.x - probe[i].ego_state.rear_axle.x,
                    orig[i].ego_state.rear_axle.y - probe[i].ego_state.rear_axle.y) for i in range(n)]
    print(f"===== {token}: probe vs original S1 seed 0 — max ego position gap {max(dev):.3f} m over {n} steps "
          f"(first gap > 1 cm at step {next((i for i, d in enumerate(dev) if d > 0.01), 'none')})")

    steps = [dict(np.load(f)) for f in sorted((PROBE / token).glob("*.npz"))]
    rows = []
    for s in steps:
        it, a, b = int(s["iteration"]), int(s["pick"]), int(s["pdm_pick"])
        if it > contact:
            break
        pm, vp = s["pdm_metrics"], s["verifier_probs"]
        rows.append({"it": it, "a": a, "b": b, "pdm_a": pm[a], "pdm_b": pm[b], "v_a": vp[a], "v_b": vp[b],
                     "n_safe": int(((pm[:, 0] == 1) & (pm[:, 4] == 1)).sum()), "n_col_free": int((pm[:, 0] == 1).sum()),
                     "vs_a": float(s["verifier_score"][a]), "vs_b": float(s["verifier_score"][b]),
                     "ps_a": float(s["pdm_score"][a]), "ps_b": float(s["pdm_score"][b]),
                     "prog_a": float(s["progress_raw"][a]), "prog_b": float(s["progress_raw"][b]),
                     "v_col_all": vp[:, 0], "pdm_col_all": pm[:, 0]})
    same = sum(r["a"] == r["b"] for r in rows)
    bad = [r for r in rows if r["pdm_a"][0] < 1 or r["pdm_a"][4] < 1]
    badc = [r for r in rows if r["pdm_a"][0] < 1]
    print(f"  steps 0..{contact}: {len(rows)}; S1 picks PDM+SL's candidate on {same}")
    print(f"  S1's pick fails PDM+SL's collision check on {len(badc)} steps, collision or TTC on {len(bad)}; on those "
          f"steps a collision-free candidate existed on {sum(r['n_col_free'] > 0 for r in badc)} of {len(badc)}, "
          f"one passing both on {sum(r['n_safe'] > 0 for r in bad)} of {len(bad)}")

    # every 5th failing step: both candidates' PDM label and verifier probabilities
    print("  step  pick  PDM+SL  | PDM label of S1's pick (col ttc prog) | S1's probs for its pick (col ttc prog) "
          "| S1's probs for PDM+SL's pick (col ttc prog) | S1 score pick vs PDM+SL's | progress m pick vs PDM+SL's")
    for r in bad[::max(1, len(bad) // 12)]:
        f = lambda v: f"{v[0]:.2f} {v[4]:.2f} {v[3]:.2f}"
        print(f"  {r['it']:4d}  {r['a']:4d}  {r['b']:6d}  | {f(r['pdm_a']):>26s}             | {f(r['v_a']):>24s}"
              f"               | {f(r['v_b']):>24s}                     | {r['vs_a']:.3f} vs {r['vs_b']:.3f}"
              f"            | {r['prog_a']:5.1f} vs {r['prog_b']:5.1f}")

    # decomposition of S1's preference, averaged over failing steps where S1 and PDM+SL disagree
    dis = [r for r in bad if r["a"] != r["b"]]
    if dis:
        dec = np.array([[decompose(r["v_a"], r["v_b"])[k] for k in NAMES] for r in dis])
        lab = np.array([[decompose(r["pdm_a"], r["pdm_b"])[k] for k in NAMES] for r in dis])
        print(f"  why S1 preferred its pick over PDM+SL's, mean over {len(dis)} failing steps where they differ "
              f"(log score difference, + favours S1's pick):")
        print("    metric        S1's prediction   PDM+SL label (what S1 was trained to predict)")
        for k, name in enumerate(NAMES):
            print(f"    {name:12s}  {dec[:, k].mean():+8.3f}          {lab[:, k].mean():+8.3f}")
        print(f"    total         {dec.sum(axis=1).mean():+8.3f}          {np.clip(lab.sum(axis=1), -14, 14).mean():+8.3f}")

    # calibration of the collision head in this scene: predicted pass probability for candidates PDM marks colliding
    vc = np.concatenate([r["v_col_all"] for r in rows])
    pc = np.concatenate([r["pdm_col_all"] for r in rows])
    if (pc < 1).any() and (pc == 1).any():
        print(f"  collision head over all {len(vc)} candidate-steps: mean P(no collision) {vc[pc < 1].mean():.3f} for the "
              f"{int((pc < 1).sum())} PDM marks colliding vs {vc[pc == 1].mean():.3f} for the {int((pc == 1).sum())} it "
              f"marks collision-free")
        # within-step ranking: share of (colliding, free) pairs where S1 rates the free one safer
        acc, tot = 0, 0
        for r in rows:
            col, free = r["v_col_all"][r["pdm_col_all"] < 1], r["v_col_all"][r["pdm_col_all"] == 1]
            if len(col) and len(free):
                acc += (free[None, :] > col[:, None]).sum()
                tot += len(col) * len(free)
        print(f"  within-step pair accuracy of the collision head (free rated safer than colliding): {acc / tot:.3f} "
              f"over {tot} pairs")
