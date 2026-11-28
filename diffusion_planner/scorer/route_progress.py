"""
Analytic progress of a plan along the route, from Diffusion Planner's own inputs: the route lanes are chained into one
centerline starting at the lane under ego (following each lane's best-aligned successor), and progress is the arc
length between the projections of the current pose (the ego-frame origin) and the plan's pose at the scored horizon.
On 3,000 label samples this matches PDM's LQR-tracked raw progress (progress_cv) to 0.77 m on average (median 0.31 m).

Also the CLS-style progress score against an estimate of the human driver's progress (combined_score_expert's
definition) and the constant-velocity estimate of it.
"""
from typing import Optional

import numpy as np
import torch

from diffusion_planner.scorer.lane_features import polylines
from diffusion_planner.scorer.metrics import EXPERT_PROGRESS_THRESHOLD, MIN_PROGRESS_RATIO

HORIZON_POSE = 39   # the 40th planned pose = 4 s (the 41 scored states are the current state plus poses 0..39)
HORIZON_S = 4.0
LINK = 1.5          # [m] a route lane follows another if it starts this close to the other's end


def route_centerline(route_lanes: np.ndarray) -> Optional[np.ndarray]:
    """Normalized DP route-lane input [R, N, 12] -> one chained centerline [P, 2] in the ego frame, or None."""
    c, _, _, valid = polylines(torch.from_numpy(np.asarray(route_lanes))[None])
    c, valid = c[0].numpy(), valid[0].numpy()
    lanes = [c[i] for i in range(len(c)) if valid[i]]
    lanes = [l[np.abs(l).sum(-1) > 0] if (np.abs(l).sum(-1) > 0).sum() >= 2 else l for l in lanes]
    if not lanes:
        return None

    def dist_to_origin(l):
        a, b = l[:-1], l[1:]
        ab = b - a
        t = np.clip(-(a * ab).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-9), 0, 1)
        p = a + t[:, None] * ab
        k = int(np.argmin(np.linalg.norm(p, axis=-1)))
        h = ab[k] / max(np.linalg.norm(ab[k]), 1e-9)
        return float(np.linalg.norm(p[k])) - 0.5 * float(h[0])   # nearest, and aligned with ego's +x

    start = min(range(len(lanes)), key=lambda i: dist_to_origin(lanes[i]))
    chain, used = [lanes[start]], {start}
    while True:   # predecessors: the lane under ego may start ahead of its rear axle
        prev = [i for i in range(len(lanes)) if i not in used and np.linalg.norm(lanes[i][-1] - chain[0][0]) < LINK]
        if not prev:
            break
        chain.insert(0, lanes[prev[0]])
        used.add(prev[0])
    while True:   # successors, best aligned with the current end
        tail = chain[-1]
        d_tail = tail[-1] - tail[-2]
        nxt = [i for i in range(len(lanes)) if i not in used and np.linalg.norm(lanes[i][0] - tail[-1]) < LINK]
        if not nxt:
            break

        def align(i):
            d = lanes[i][1] - lanes[i][0]
            return -float(np.dot(d, d_tail) / max(np.linalg.norm(d) * np.linalg.norm(d_tail), 1e-9))

        i = min(nxt, key=align)
        chain.append(lanes[i])
        used.add(i)
    line = np.concatenate(chain, axis=0)
    return line if len(line) >= 2 else None


def arc_position(line: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Arc length [m] along the polyline [P, 2] of each point's [n, 2] projection onto it (nearest segment)."""
    a, b = line[:-1], line[1:]
    ab = b - a
    seg = np.linalg.norm(ab, axis=-1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])[:-1]
    t = np.clip(((pts[:, None] - a[None]) * ab[None]).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-9)[None], 0, 1)
    proj = a[None] + t[..., None] * ab[None]
    k = np.argmin(np.linalg.norm(pts[:, None] - proj, axis=-1), axis=1)
    return cum[k] + t[np.arange(len(pts)), k] * seg[k]


def route_progress(route_lanes: np.ndarray, plans: np.ndarray) -> np.ndarray:
    """
    Args:
        route_lanes: [R, N, 12] normalized DP route-lane input of one scene
        plans: [K, T, >=2] ego plans in the ego frame (unnormalized)
    Returns:
        [K] progress [m] from the current pose to the plan's pose at 4 s along the chained route centerline, clipped at
        0 as PDM's label is; the straight-line distance when the scene has no usable route lanes
    """
    end = np.asarray(plans, dtype=np.float64)[:, HORIZON_POSE, :2]
    line = route_centerline(route_lanes)
    if line is None:
        return np.linalg.norm(end, axis=-1)
    pos = arc_position(line, np.vstack([np.zeros((1, 2)), end]))
    return np.clip(pos[1:] - pos[0], 0.0, None)


def constant_velocity_expert(speed: torch.Tensor) -> torch.Tensor:
    """Constant-velocity estimate of the human driver's progress over the horizon [m] from ego's speed [m/s]."""
    return speed.clamp(min=0.0) * HORIZON_S


def expert_progress_score(raw: torch.Tensor, expert: torch.Tensor):
    """
    CLS-style progress (combined_score_expert's definition), batched.

    Args:
        raw: [B, K] progress of each candidate [m]
        expert: [B] (estimated) progress of the human driver [m]
    Returns:
        ratio [B, K] = min(1, max(raw, 2 m) / max(expert, 2 m)), 0 where raw < -2 m;
        making [B, K] = 1 where ratio >= 0.2 (ego_is_making_progress), else 0
    """
    thr = EXPERT_PROGRESS_THRESHOLD
    ratio = torch.clamp(torch.clamp(raw, min=thr) / torch.clamp(expert, min=thr)[:, None], max=1.0)
    ratio = torch.where(raw < -thr, torch.zeros_like(ratio), ratio)
    return ratio, (ratio >= MIN_PROGRESS_RATIO).to(ratio.dtype)
