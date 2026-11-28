"""How well does progress computed from the plan's 40th pose match PDM's raw progress label (progress_cv)?
PDM: LQR-track the plan, project the tracked pose at 4 s and the current pose onto its route centerline, take the
difference. Here, no tracking: (route) arc length along a centerline chained from DP's route-lane inputs, (euclid)
straight-line distance, (forward) x of the pose in the ego frame. CPU only.

    python check_progress.py ~/nuplan/labels/train-labels-2k 3000
"""
import random
import sys
from pathlib import Path

import numpy as np
import torch

from diffusion_planner.scorer.labels import load_sample
from diffusion_planner.scorer.lane_features import polylines

ROOT = Path(sys.argv[1]).expanduser()
N = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
POSE = 39          # 40th planned pose = 4 s (the 41 scored states are the current state plus poses 0..39)
LINK = 1.5         # [m] a route lane follows another if it starts this close to the other's end


def route_centerline(route_lanes: np.ndarray):
    """Chain DP's route lanes into one polyline [P, 2] (ego frame) starting at the lane under ego, or None."""
    c, _, _, valid = polylines(torch.from_numpy(route_lanes)[None])
    c, valid = c[0].numpy(), valid[0].numpy()
    lanes = [c[i] for i in range(len(c)) if valid[i]]
    lanes = [l[np.abs(l).sum(-1) > 0] if (np.abs(l).sum(-1) > 0).sum() >= 2 else l for l in lanes]
    if not lanes:
        return None

    def dist_to_origin(l):
        seg_a, seg_b = l[:-1], l[1:]
        ab = seg_b - seg_a
        t = np.clip(-(seg_a * ab).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-9), 0, 1)
        p = seg_a + t[:, None] * ab
        k = int(np.argmin(np.linalg.norm(p, axis=-1)))
        heading = ab[k] / max(np.linalg.norm(ab[k]), 1e-9)
        return float(np.linalg.norm(p[k])) - 0.5 * float(heading[0])  # nearest, and aligned with ego's +x

    start = min(range(len(lanes)), key=lambda i: dist_to_origin(lanes[i]))
    chain, used = [lanes[start]], {start}
    # predecessors too: the lane under ego may start ahead of its rear axle
    while True:
        head = chain[0]
        prev = [i for i in range(len(lanes)) if i not in used and np.linalg.norm(lanes[i][-1] - head[0]) < LINK]
        if not prev:
            break
        i = prev[0]
        chain.insert(0, lanes[i]); used.add(i)
    while True:
        tail = chain[-1]
        d_tail = tail[-1] - tail[-2]
        nxt = [i for i in range(len(lanes)) if i not in used and np.linalg.norm(lanes[i][0] - tail[-1]) < LINK]
        if not nxt:
            break
        def align(i):
            d = lanes[i][1] - lanes[i][0]
            return -float(np.dot(d, d_tail) / max(np.linalg.norm(d) * np.linalg.norm(d_tail), 1e-9))
        i = min(nxt, key=align)
        chain.append(lanes[i]); used.add(i)
    return np.concatenate(chain, axis=0)


def arc_position(line: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Arc length [m] along the polyline of each point's projection onto it (nearest segment)."""
    a, b = line[:-1], line[1:]
    ab = b - a
    seg_len = np.linalg.norm(ab, axis=-1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])[:-1]
    ap = pts[:, None, :] - a[None]
    t = np.clip((ap * ab[None]).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-9)[None], 0, 1)
    proj = a[None] + t[..., None] * ab[None]
    d = np.linalg.norm(pts[:, None, :] - proj, axis=-1)
    k = np.argmin(d, axis=1)
    return cum[k] + t[np.arange(len(pts)), k] * seg_len[k]


files = sorted(ROOT.glob("*/*.npz"))
random.Random(0).shuffle(files)
files = files[:N]
est = {"route": [], "euclid": [], "forward": []}
label, scene, no_route = [], [], 0
for si, f in enumerate(files):
    s = load_sample(f)
    cand = np.asarray(s["candidates"], dtype=np.float64)          # [K, 80, 4] ego frame, unnormalized
    end = cand[:, POSE, :2]
    line = route_centerline(np.asarray(s["inputs"]["route_lanes"]))
    if line is None or len(line) < 2:
        no_route += 1
        continue
    pos = arc_position(line, np.vstack([np.zeros((1, 2)), end]))
    est["route"].append(pos[1:] - pos[0])
    est["euclid"].append(np.linalg.norm(end, axis=-1))
    est["forward"].append(end[:, 0])
    label.append(np.asarray(s["progress_cv"], dtype=np.float64))
    scene.append(np.full(len(end), si))

y = np.concatenate(label)
print(f"samples {len(label)} (skipped without route lanes: {no_route}), candidates {len(y)}; "
      f"PDM progress_cv: mean {y.mean():.2f} m, median {np.median(y):.2f}, range {y.min():.1f} to {y.max():.1f}")
print(f"{'estimate':8s}  {'MAE [m]':>8s}  {'median AE':>9s}  {'p90 AE':>7s}  {'bias':>6s}  {'corr':>5s}  "
      f"{'same best':>9s}  {'best within 0.5 m':>17s}  {'within-scene rank corr':>22s}")
for name, v in est.items():
    x = np.concatenate(v)
    err = x - y
    same, close, rc = [], [], []
    for a, b in zip(v, label):
        same.append(int(np.argmax(a)) == int(np.argmax(b)))
        close.append(b[int(np.argmax(a))] >= b.max() - 0.5)       # the estimate's pick is within 0.5 m of PDM's best
        if np.std(a) > 1e-6 and np.std(b) > 1e-6:
            ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
            rc.append(np.corrcoef(ra, rb)[0, 1])
    print(f"{name:8s}  {np.abs(err).mean():8.2f}  {np.median(np.abs(err)):9.2f}  {np.percentile(np.abs(err), 90):7.2f}  "
          f"{err.mean():+6.2f}  {np.corrcoef(x, y)[0, 1]:5.3f}  {np.mean(same):9.3f}  {np.mean(close):17.3f}  "
          f"{np.mean(rc):22.3f}")

# where the route estimate misses most: speed bands of PDM's label
x = np.concatenate(est["route"])
for lo, hi in [(-5, 2), (2, 10), (10, 25), (25, 80)]:
    m = (y >= lo) & (y < hi)
    if m.any():
        print(f"  route, PDM progress in [{lo}, {hi}) m: {m.sum()} candidates, MAE {np.abs(x[m] - y[m]).mean():.2f} m, "
              f"bias {(x[m] - y[m]).mean():+.2f} m")
