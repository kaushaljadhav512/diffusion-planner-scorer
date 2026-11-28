"""
Tests for diffusion_planner.scorer.route_progress and train.apply_progress_mode (--progress-mode ep_cv). Synthetic
inputs only.

    pytest tests/test_route_progress.py
"""
import numpy as np
import torch

from diffusion_planner.scorer.lane_features import LANE_MEAN_X, LANE_STD
from diffusion_planner.scorer.route_progress import (constant_velocity_expert, expert_progress_score, route_centerline,
                                                     route_progress)


def _lane(x0, x1, y=0.0, n=20):
    """One normalized DP lane [n, 12] along +x from x0 to x1 at lateral offset y, 1.5 m half-width."""
    lane = np.zeros((n, 12), dtype=np.float32)
    xs = np.linspace(x0, x1, n)
    lane[:, 0] = (xs - LANE_MEAN_X) / LANE_STD
    lane[:, 1] = y / LANE_STD
    lane[:, 5] = 1.5 / LANE_STD     # left boundary offset (y)
    lane[:, 7] = -1.5 / LANE_STD    # right boundary offset (y)
    return lane


def _route(*lanes, total=25):
    r = np.zeros((total, 20, 12), dtype=np.float32)
    for i, l in enumerate(lanes):
        r[i] = l
    return r


def _plans(ends):
    """[K, 80, 4] plans whose pose at 4 s (index 39) is at the given (x, y)."""
    p = np.zeros((len(ends), 80, 4), dtype=np.float32)
    for k, (x, y) in enumerate(ends):
        p[k, :, 0] = np.linspace(0, x * 2, 80)
        p[k, :, 1] = y
        p[k, 39, :2] = (x, y)
    return p


def test_route_progress_straight_and_chained():
    # two chained lanes along x: -5..20 m, then 20..60 m; a parallel lane at y = 3.5 m that must not be chained
    route = _route(_lane(-5, 20), _lane(20, 60), _lane(-5, 60, y=3.5))
    line = route_centerline(route)
    assert line is not None and np.allclose(line[:, 1], 0, atol=1e-4) and line[-1, 0] > 59
    prog = route_progress(route, _plans([(10, 0), (30, 0), (12, 0.8), (-3, 0)]))
    assert np.allclose(prog[:3], [10, 30, 12], atol=1e-3)   # lateral offset does not count
    assert prog[3] == 0.0                                   # backwards clips at 0, as PDM's label


def test_route_progress_without_route_falls_back_to_distance():
    prog = route_progress(np.zeros((25, 20, 12), dtype=np.float32), _plans([(3, 4)]))
    assert np.allclose(prog, [5.0])


def test_expert_progress_score():
    raw = torch.tensor([[0.5, 2.0, 5.0, 12.0], [0.0, 0.3, 1.0, 3.0]])
    expert = constant_velocity_expert(torch.tensor([1.25, -0.2]))   # 5 m, and 0 m for a reversing ego
    ratio, making = expert_progress_score(raw, expert)
    assert torch.allclose(ratio[0], torch.tensor([0.4, 0.4, 1.0, 1.0]))       # max(raw, 2) / 5, capped at 1
    assert torch.allclose(ratio[1], torch.ones(4))                           # stopped: everything ties at 1
    assert torch.equal(making, torch.ones(2, 4))


def test_apply_progress_mode_ep_cv():
    from diffusion_planner.scorer.model import PROGRESS
    from diffusion_planner.scorer.train import apply_progress_mode

    B, K = 2, 3
    metrics = torch.rand(B, K, 7)
    logits = torch.randn(B, K, 7, requires_grad=True)
    batch = {"metrics_cv": metrics, "progress_route": torch.tensor([[1.0, 4.0, 9.0], [0.1, 0.2, 9.0]]),
             "ego_speed": torch.tensor([1.0, 10.0])}                # E = 4 m and 40 m
    lg, target, gate = apply_progress_mode(logits, batch, "cv", "ep_cv")
    assert torch.allclose(target[..., PROGRESS], torch.tensor([[0.5, 1.0, 1.0], [0.05, 0.05, 0.225]]))
    assert torch.allclose(torch.sigmoid(lg[..., PROGRESS]), target[..., PROGRESS], atol=1e-3)
    assert torch.equal(gate, torch.tensor([[1.0, 1.0, 1.0], [0.0, 0.0, 1.0]]))
    other = [i for i in range(7) if i != PROGRESS]
    assert torch.equal(target[..., other], metrics[..., other]) and torch.equal(lg[..., other], logits[..., other])
    # pdm mode leaves everything as recorded
    lg2, target2, gate2 = apply_progress_mode(logits, batch, "cv", "pdm")
    assert lg2 is logits and target2 is metrics and gate2 is None
