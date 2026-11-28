"""Offline check of training-free estimates of the human driver's 4 s progress (E), against the true E.

True E: path length of the human's logged ego poses (nuPlan DB) from the sample's time to 4 s later; for a
lane-following driver this is close to the progress PDM's expert_progress projects onto its centerline.
Estimates (what the pdm_sl_ep_<x> selectors use, plus one that only uses the simulated ego):
    median    median of the K candidates' PDM raw progress (progress_cv)
    first     candidate 0's progress_cv (label runs use noise 1.5 for every candidate: NOT vanilla DP's plan)
    cv, ca    constant velocity / acceleration from the HUMAN's logged speed and acceleration at that time
              (the same DB fields patch_ego_state.py writes into train-labels-2k-ego), so optimistic
    cv_plan   simulated ego's speed read off the plans (mean over candidates of |first planned pose| / 0.1 s) x 4 s
Selection effect: pick by combined_score_expert(metrics_cv, progress_cv, E_estimate) vs by the same with the true E;
agreement and regret against the true-E score.

    python check_e_estimates.py ~/nuplan/labels/train-labels-2k 3000
"""
import glob
import os
import random
import sqlite3
import sys
from pathlib import Path

import numpy as np

from diffusion_planner.scorer.labels import load_sample
from diffusion_planner.scorer.metrics import combined_score_expert

ROOT, N = Path(sys.argv[1]).expanduser(), int(sys.argv[2]) if len(sys.argv) > 2 else 3000
H = 4.0  # [s] scored horizon
DBS = {os.path.basename(p)[:-3]: p for p in glob.glob(os.path.expanduser("~/nuplan/dataset/nuplan-v1.1/*/*.db"))}
_poses, _conns = {}, {}


def poses(log):
    """[n, 5] timestamp [us], x, y, vx, ax of the log's ego poses, sorted by time (cached)."""
    if log not in _poses:
        con = sqlite3.connect(DBS[log])
        a = np.array(con.execute("select timestamp, x, y, vx, acceleration_x from ego_pose").fetchall(), dtype=np.float64)
        _poses[log] = a[np.argsort(a[:, 0])]
        _conns[log] = con
    return _poses[log]


def start_time(log, token):
    r = _conns[log].execute("select timestamp from lidar_pc where token=?", (bytes.fromhex(token),)).fetchone()
    return None if r is None else float(r[0])


def ca(v, a, t=H):
    v = max(v, 0.0)
    if a >= 0 or v / -a >= t:
        return max(v * t + 0.5 * a * t * t, 0.0)
    return v * v / (2 * -a)


files = sorted(ROOT.glob("*/*.npz"))
random.Random(0).shuffle(files)
rows, skipped = [], 0
for f in files:
    if len(rows) >= N:
        break
    s = load_sample(f)
    log, tok, it = str(s["log_name"]), str(s["scenario_token"]), int(s["iteration"])
    if log not in DBS:
        skipped += 1
        continue
    P = poses(log)
    t0 = start_time(log, tok)
    if t0 is None:
        skipped += 1
        continue
    t = t0 + it * 1e5
    i0, i1 = np.searchsorted(P[:, 0], [t, t + H * 1e6])
    if i1 >= len(P) or i1 - i0 < 10:
        skipped += 1
        continue
    seg = P[i0:i1 + 1:5, 1:3]                         # ~every 0.05 s at 100 Hz logging
    E = float(np.linalg.norm(np.diff(seg, axis=0), axis=1).sum())
    v, a = float(P[i0, 3]), float(P[i0, 4])
    prog = np.asarray(s["progress_cv"], dtype=np.float64)
    cand = np.asarray(s["candidates"], dtype=np.float64)
    v_plan = float(np.linalg.norm(cand[:, 0, :2], axis=1).mean() / 0.1)
    rows.append({"E": E, "v": v, "metrics": np.asarray(s["metrics_cv"], dtype=np.float64), "prog": prog,
                 "median": float(np.median(prog)), "first": float(prog[0]), "cv": max(v, 0) * H, "ca": ca(v, a),
                 "cv_plan": v_plan * H, "noise0": float(np.asarray(s["noise_scale"])[0])})

E = np.array([r["E"] for r in rows]); V = np.array([r["v"] for r in rows])
print(f"samples {len(rows)} (skipped {skipped}); true E: mean {E.mean():.2f} m, median {np.median(E):.2f}, "
      f"range {E.min():.1f} to {E.max():.1f}; candidate 0 noise std {np.unique([r['noise0'] for r in rows])}")
# starts and stops: the human's progress differs from constant velocity at its current speed by more than 4 m
change = np.abs(E - np.maximum(V, 0) * H) > 4.0
print(f"samples where the human speeds up or slows down a lot (|E - v*4 s| > 4 m): {change.sum()}")

EST = ["median", "first", "cv", "ca", "cv_plan"]
true_scores = [combined_score_expert(r["metrics"], r["prog"], r["E"]) for r in rows]
true_pick = [int(np.argmax(ts)) for ts in true_scores]
print(f"\n{'estimate':8s} {'MAE':>6s} {'median AE':>9s} {'bias':>6s} {'corr':>5s} {'MAE starts/stops':>16s} "
      f"{'same pick as true E':>19s} {'regret vs true-E target':>23s}")
for k in EST:
    x = np.array([r[k] for r in rows]); err = x - E
    same, reg = [], []
    for r, ts, tp in zip(rows, true_scores, true_pick):
        ps = combined_score_expert(r["metrics"], r["prog"], r[k])
        p = int(np.argmax(ps))
        same.append(ts[p] >= ts[tp] - 1e-9)
        reg.append(ts[tp] - ts[p])
    print(f"{k:8s} {np.abs(err).mean():6.2f} {np.median(np.abs(err)):9.2f} {err.mean():+6.2f} {np.corrcoef(x, E)[0, 1]:5.3f} "
          f"{np.abs(err[change]).mean():16.2f} {np.mean(same):19.3f} {np.mean(reg):23.4f}")

# reference: the current progress definition (normalized by the best candidate, pdm_sl's labels) against the true-E target
from diffusion_planner.scorer.metrics import combined_score
reg_sl, same_sl = [], []
for r, ts, tp in zip(rows, true_scores, true_pick):
    p = int(np.argmax(combined_score(r["metrics"])))
    same_sl.append(ts[p] >= ts[tp] - 1e-9); reg_sl.append(ts[tp] - ts[p])
print(f"{'pdm_sl':8s} {'(current progress definition, no E)':>45s} {np.mean(same_sl):19.3f} {np.mean(reg_sl):23.4f}")

print("\nMAE by the human's speed [m/s] (logged):")
bands = [(-1, 0.5), (0.5, 3), (3, 8), (8, 100)]
print("  " + " ".join(f"{k:>8s}" for k in ["band", "n"] + EST))
for lo, hi in bands:
    m = (V >= lo) & (V < hi)
    print(f"  {f'[{lo},{hi})':>8s} {m.sum():8d} " +
          " ".join(f"{np.abs(np.array([r[k] for r in rows])[m] - E[m]).mean():8.2f}" for k in EST))
