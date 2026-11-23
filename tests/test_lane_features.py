"""
Tests for the per-step local lane features (diffusion_planner/scorer/lane_features.py) and the verifier's step_lane
option. Hand-built straight lanes, so every feature has a known value.

    pytest tests/test_lane_features.py
"""
import math

import pytest
import torch

from diffusion_planner.scorer import train as train_module
from diffusion_planner.scorer.analytic_metrics import HALF_WIDTH, REAR_AXLE_TO_CENTER, corner_lane_distance_torch
from diffusion_planner.scorer.lane_features import (
    RED, SEGMENT_FEATURES, STEP_LANE_SCALARS, on_route, step_lane_features,
)
from diffusion_planner.scorer.metrics import METRICS
from diffusion_planner.scorer.model import HORIZON, Verifier

from test_sampling import _random_config, _scene_inputs
from test_verifier import _straight_lanes, _write_dataset

WIDTH = 3.5
LANE_Y = (0.0, 3.5)          # two adjacent lanes along x, sharing the boundary at y = 1.75


@pytest.fixture(scope="module")
def config():
    return _random_config()


def _inputs(batch=1, route=(1,)):
    """Two straight lanes (lane 0 red light, 0.5 speed limit; lane 1 0.25), route = the given lanes."""
    lanes = _straight_lanes(batch, 70, 20, centers_y=LANE_Y, width=WIDTH)
    lanes[:, 0, :, 8 + RED] = 1.0
    lanes[:, 1, :, 8] = 1.0                                        # green
    speed = torch.zeros(batch, 70, 1)
    speed[:, 0], speed[:, 1] = 0.5, 0.25
    has = torch.zeros(batch, 70, 1, dtype=torch.bool)
    has[:, :2] = True
    route_lanes = torch.zeros(batch, 25, 20, 12)
    for i, j in enumerate(route):
        route_lanes[:, i] = lanes[:, j]
    return {"lanes": lanes, "lanes_speed_limit": speed, "lanes_has_speed_limit": has, "route_lanes": route_lanes}


def _plans(ys, yaw=0.0, batch=1):
    """[batch, len(ys), 80, 4] straight plans at constant lateral offset y (the current state sits at the origin)."""
    plans = torch.zeros(batch, len(ys), 80, 4)
    plans[..., 0] = torch.linspace(0.5, 40.0, 80)
    plans[..., 1] = torch.tensor(ys, dtype=torch.float32)[None, :, None]
    plans[..., 2], plans[..., 3] = math.cos(yaw), math.sin(yaw)
    return plans


def test_scalar_features_on_straight_lanes():
    inputs = _inputs()
    plans = _plans([0.0, 0.5, 3.5, 1.75, 9.0])
    f = step_lane_features(inputs, plans, HORIZON)["scalars"]
    assert f.shape == (1, 5, HORIZON + 1, STEP_LANE_SCALARS)
    s = f[0, :, 5]                                                 # a state mid-plan (x = 2.5 m)

    # Centred in lane 0: every corner 1.75 - 1.1485 m deep, one lane, off route, lane 0's limit and red light
    depth = WIDTH / 2 - HALF_WIDTH
    assert torch.allclose(s[0, 0:4], torch.full((4,), -depth / 2), atol=1e-4)
    assert s[0, 4] == pytest.approx(1 / 3)
    assert s[0, 5] == 0 and s[0, 6] == pytest.approx(1.75 / 5, abs=1e-4)
    assert s[0, 7] == pytest.approx(1.0) and s[0, 8] == pytest.approx(0.0, abs=1e-6)
    assert (s[0, 9], s[0, 10], s[0, 11]) == (pytest.approx(0.5), 1.0, 1.0)
    assert s[0, 12] == pytest.approx(0.0, abs=1e-5) and s[0, 13] == pytest.approx(WIDTH / 4, abs=1e-4)

    # 0.5 m left of lane 0's middle: shallower left corners, lateral offset +0.5 m
    assert s[1, 0] == pytest.approx(-(depth - 0.5) / 2, abs=1e-4) and s[1, 2] == pytest.approx(s[1, 0], abs=1e-6)
    assert s[1, 1] == pytest.approx(-(depth + 0.5) / 2, abs=1e-4)
    assert s[1, 12] == pytest.approx(0.5 / 2, abs=1e-4)

    # In lane 1, the route lane: on route, lane 1's attributes, no red light
    assert s[2, 5] == 1 and s[2, 6] < 0
    assert (s[2, 9], s[2, 11]) == (pytest.approx(0.25), 0.0)

    # On the shared boundary: the footprint covers both lanes
    assert s[3, 4] == pytest.approx(2 / 3)

    # Off the road (y = 9 m, 2.25 m past the outer edge of lane 1): positive corner distances
    assert (s[4, 0:4] > 0).all()
    assert s[4, 1] == pytest.approx((9.0 - HALF_WIDTH - 5.25) / 2, abs=1e-4)


def test_outside_distances_match_analytic():
    """Outside every lane, the signed corner distance is corner_lane_distance_torch's (exact) distance."""
    g = torch.Generator().manual_seed(0)
    inputs = _inputs(batch=2)
    inputs["lanes"] = inputs["lanes"] + 0.01 * torch.randn(inputs["lanes"].shape, generator=g) * (inputs["lanes"] != 0)
    plans = torch.zeros(2, 5, 80, 4)
    t = torch.linspace(0.1, 8.0, 80)
    plans[..., 0] = t * 8.0
    plans[..., 1] = t[None, None] * torch.tensor([-1.0, -0.3, 0.0, 0.5, 1.5])[None, :, None]
    plans[..., 2] = 1.0
    signed = step_lane_features(inputs, plans, HORIZON)["scalars"][..., 0:4] * 2.0
    reference = corner_lane_distance_torch(inputs["lanes"], plans, horizon=HORIZON)
    outside = (reference > 1e-3) & (reference < 3.9)
    assert outside.sum() > 100
    assert torch.allclose(signed[outside], reference[outside], atol=1e-3)
    assert (signed[reference == 0] <= 1e-4).all()


def test_heading_relative_to_lane():
    yaw = math.radians(10.0)
    f = step_lane_features(_inputs(), _plans([0.0], yaw=yaw), HORIZON)["scalars"]
    assert f[0, 0, 5, 7] == pytest.approx(math.cos(yaw), abs=1e-4)
    assert f[0, 0, 5, 8] == pytest.approx(math.sin(yaw), abs=1e-4)


def test_on_route_matching():
    inputs = _inputs(route=(1,))
    routed = on_route(inputs["lanes"], inputs["route_lanes"])
    assert routed[0, :3].tolist() == [False, True, False]


def test_segment_features():
    inputs = _inputs()
    out = step_lane_features(inputs, _plans([0.0, 60.0]), HORIZON, segments=True, n_boundary=8, n_center=2)
    seg, mask = out["segments"], out["segment_mask"]
    assert seg.shape == (1, 2, HORIZON + 1, 10, SEGMENT_FEATURES) and mask.shape == (1, 2, HORIZON + 1, 10)

    # Far from every lane (y = 60 m): every segment empty, features zero (after the current state, on the lanes)
    assert not mask[0, 1, 0].all()
    assert mask[0, 1, 1:].all() and (seg[0, 1, 1:] == 0).all()

    s, m = seg[0, 0, 5], mask[0, 0, 5]
    assert not m.any()
    assert (s[:8, 4:7].argmax(-1) < 2).all() and (s[8:, 6] == 1).all()     # boundaries first, then centerlines
    assert torch.allclose(s[:, 11], torch.ones(10), atol=1e-5)               # straight lanes along ego's heading
    # Lane 0's left boundary (y = 1.75, lane side below it): every corner on the lane's side
    left0 = (s[:, 4] == 1) & (s[:, 13] == 0.5) & (s[:, 1] * 10 > 1.7)
    assert left0.any() and (s[left0][:, 7:11] > 0).all()
    # Lane 1's right boundary is the same line, seen from lane 1: every corner outside it
    right1 = (s[:, 5] == 1) & (s[:, 13] == 0.25) & (s[:, 1] * 10 > 1.7)
    assert right1.any() and (s[right1][:, 7:11] < 0).all()
    # Lane attributes ride on the segments: lane 0 red and off route, lane 1 green and on route
    assert (s[left0][:, 15 + RED] == 1).all() and (s[left0][:, 19] == 0).all()
    assert (s[right1][:, 15] == 1).all() and (s[right1][:, 19] == 1).all()
    # Endpoints in the state's frame: the boundary sits 1.75 m to the left of ego's centre
    assert torch.allclose(s[left0][:, 1] * 10, torch.full((int(left0.sum()),), 1.75), atol=1e-4)


@pytest.mark.parametrize("mode", ["scalars", "segments"])
def test_verifier_step_lane_shapes_and_equivariance(config, mode):
    torch.manual_seed(0)
    model = Verifier(config, arch="timestep", step_lane=mode).eval()
    inputs = _scene_inputs(config, batch=2)
    candidates = torch.randn(2, 4, 80, 4) * 10
    logits = model(inputs, candidates)
    assert logits.shape == (2, 4, len(METRICS)) and torch.isfinite(logits).all()
    perm = torch.tensor([2, 0, 3, 1])
    assert torch.allclose(model(inputs, candidates[:, perm]), logits[:, perm], atol=1e-5)

    # A scene without lanes: no segment in reach, still finite
    inputs["lanes"] = torch.zeros_like(inputs["lanes"])
    assert torch.isfinite(model(inputs, candidates)).all()


def test_step_lane_needs_timestep(config):
    with pytest.raises(AssertionError):
        Verifier(config, arch="plan", step_lane="scalars")


@pytest.mark.parametrize("mode", ["scalars", "segments"])
def test_train_with_step_lane(tmp_path, config, mode):
    _write_dataset(tmp_path / "labels", config, n_scenarios=4, steps=4)
    args = train_module.parse_args([
        "--data", str(tmp_path / "labels"), "--out", str(tmp_path / mode), "--overfit", "16", "--arch", "timestep",
        "--step-lane", mode, "--epochs", "3", "--batch-size", "8", "--lr", "1e-3", "--num-workers", "0",
        "--device", "cpu"])
    best = train_module.run(args, config=config)
    assert best["all/verifier/regret"] >= 0
    ckpt = torch.load(tmp_path / mode / "best.pt", weights_only=False)
    assert ckpt["hparams"]["step_lane"] == mode
    # Old checkpoints have no step_lane: they load as w6
    model = Verifier(config, **{k: v for k, v in ckpt["hparams"].items() if k != "step_lane"})
    assert model.step_lane == "none"
