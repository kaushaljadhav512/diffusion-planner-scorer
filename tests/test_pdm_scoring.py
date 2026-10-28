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


# ---------------------------------------------------------------------------------------------------------------
# Oracle: logged-future occupancy maps

from types import SimpleNamespace

from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.scene_object import SceneObjectMetadata
from nuplan.common.actor_state.tracked_objects import TrackedObjects
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks

from diffusion_planner.scorer.pdm_scoring import PDMOracleCandidateScorer
from diffusion_planner.scorer.selectors import OracleSelector


def _car(x, y, token="car", vx=0.0):
    return Agent(
        tracked_object_type=TrackedObjectType.VEHICLE,
        oriented_box=OrientedBox(StateSE2(x, y, 0.0), 4.0, 2.0, 1.5),
        velocity=StateVector2D(vx, 0.0),
        metadata=SceneObjectMetadata(timestamp_us=0, token=token, track_id=None, track_token=token),
    )


def _detections(*agents):
    return DetectionsTracks(TrackedObjects(list(agents)))


class _FakeScenario:
    """Logged future: a car that is reported stopped but drives at 2.5 m/s along x, starting at x=15."""

    def __init__(self, available_frames=None):
        self.available_frames = available_frames
        self.calls = []

    def get_future_tracked_objects(self, iteration, time_horizon, num_samples):
        self.calls.append((iteration, time_horizon, num_samples))
        n = num_samples if self.available_frames is None else self.available_frames
        for k in range(1, n + 1):
            agents = [_car(15.0 + 0.5 * k, 0.0)] # 0.5 m per 0.2 s frame, stays inside the 50 m map radius
            if k >= 10:
                agents.append(_car(-20.0, 5.0, token="late")) # appears 2 s from now
            yield _detections(*agents)


def _scorer_and_input(scenario, iteration=7):
    scorer = PDMOracleCandidateScorer()
    scorer.scenario = scenario
    scorer._route_lane_dict = {}
    ego_state = _ego_state(x=0.0, y=0.0, heading=0.0)
    current_input = SimpleNamespace(
        history=SimpleNamespace(current_state=(ego_state, _detections(_car(15.0, 0.0)))),
        iteration=SimpleNamespace(index=iteration),
        traffic_light_data=[],
    )
    return scorer, ego_state, current_input


def test_oracle_observation_follows_the_log():
    scenario = _FakeScenario()
    scorer, ego_state, current_input = _scorer_and_input(scenario)
    scorer._update_observation(ego_state, current_input)
    observation = scorer._observation

    # One query per step: 40 frames over 8 s, i.e. one per occupancy map after the current one
    assert observation.num_maps == 41
    (iteration, horizon, num_samples), = scenario.calls
    assert iteration == 7 and num_samples == 40 and horizon == pytest.approx(8.0)

    # time_idx is in 0.1 s simulation steps, one occupancy map per 2 steps. Constant velocity would keep the
    # (reported stopped) car at x=15; the oracle puts it where the log has it.
    for time_idx in [0, 1, 2, 9, 40, 80]:
        assert observation[time_idx]["car"].centroid.x == pytest.approx(15.0 + 0.5 * (time_idx // 2))

    # Objects that only appear in the future are in the maps from then on, and known to the scorer
    assert "late" not in observation[19].tokens and "late" in observation[20].tokens
    assert set(observation.unique_objects) == {"car", "late"}


def test_oracle_observation_holds_last_frame_at_log_end():
    scorer, ego_state, current_input = _scorer_and_input(_FakeScenario(available_frames=3))
    scorer._update_observation(ego_state, current_input)

    xs = [scorer._observation[2 * k]["car"].centroid.x for k in range(41)]
    assert xs[:4] == pytest.approx([15.0, 15.5, 16.0, 16.5])
    assert xs[4:] == pytest.approx([16.5] * 37)


def test_oracle_selector_passes_scenario():
    selector = OracleSelector.__new__(OracleSelector) # skip building the real scorer
    selector._scorer = _FakeScorer([0.1, 0.8])
    candidates = torch.zeros(1, 2, 11, 80, 4)

    with pytest.raises(AssertionError):
        selector.select(candidates, SelectionContext(planner_input=object()))

    scenario = object()
    assert selector.select(candidates, SelectionContext(planner_input=object(), scenario=scenario)).item() == 1
    assert selector._scorer.scenario is scenario
