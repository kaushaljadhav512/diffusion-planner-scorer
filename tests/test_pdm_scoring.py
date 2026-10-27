"""
Tests for diffusion_planner.scorer.pdm_scoring and PDMSelector. Need nuplan-devkit and tuplan_garage
(skipped otherwise), no map or nuPlan data.

    pytest tests/test_pdm_scoring.py
"""
import numpy as np
import pytest
import torch

pytest.importorskip("nuplan")
pytest.importorskip("tuplan_garage")

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.common.geometry.convert import relative_to_absolute_poses

from diffusion_planner.scorer.pdm_scoring import relative_to_global_poses
from diffusion_planner.scorer.selectors import PDMSelector, SelectionContext


def _ego_state(x=12.0, y=-3.0, heading=2.4):
    return EgoState.build_from_rear_axle(
        rear_axle_pose=StateSE2(x, y, heading),
        rear_axle_velocity_2d=StateVector2D(5.0, 0.0),
        rear_axle_acceleration_2d=StateVector2D(0.0, 0.0),
        tire_steering_angle=0.0,
        time_point=TimePoint(0),
        vehicle_parameters=get_pacifica_parameters(),
    )


def test_relative_to_global_matches_nuplan():
    """Same result as the conversion nuPlan applies to the trajectory DP executes."""
    rng = np.random.default_rng(0)
    K, T = 3, 80
    rel_heading = rng.uniform(-np.pi, np.pi, (K, T))
    predictions = np.concatenate([rng.normal(0, 20, (K, T, 2)),
                                  np.stack([np.cos(rel_heading), np.sin(rel_heading)], axis=-1)], axis=-1)
    ego_state = _ego_state()

    poses = relative_to_global_poses(ego_state, predictions)

    for k in range(K):
        expected = relative_to_absolute_poses(
            ego_state.rear_axle, [StateSE2(px, py, h) for (px, py), h in zip(predictions[k, :, :2], rel_heading[k])])
        expected = np.array([p.serialize() for p in expected])
        np.testing.assert_allclose(poses[k, :, :2], expected[:, :2], atol=1e-9)
        # headings equal up to wrapping
        np.testing.assert_allclose(np.cos(poses[k, :, 2] - expected[:, 2]), 1.0, atol=1e-9)


class _FakeScorer:
    def __init__(self, scores):
        self.scores = np.asarray(scores, dtype=np.float64)

    def score(self, current_input, predictions):
        assert predictions.shape == (len(self.scores), 80, 4)
        return self.scores


def _selector(scores):
    selector = PDMSelector.__new__(PDMSelector) # skip building the real scorer (needs a map)
    selector._scorer = _FakeScorer(scores)
    return selector


@pytest.mark.parametrize("scores, expected", [
    ([0.2, 0.9, 0.5, 0.9], 1), # best, first of the tied maxima
    ([0.0, 0.0, 0.0, 0.0], 0), # all fail: fall back to candidate 0
    ([0.7, 0.7, 0.7, 0.7], 0), # indistinguishable: candidate 0 (vanilla DP)
])
def test_pdm_selector_argmax_and_ties(scores, expected):
    candidates = torch.zeros(1, len(scores), 11, 80, 4)
    indices = _selector(scores).select(candidates, SelectionContext(planner_input=object()))

    assert indices.shape == (1,) and indices.dtype == torch.long
    assert indices.item() == expected
