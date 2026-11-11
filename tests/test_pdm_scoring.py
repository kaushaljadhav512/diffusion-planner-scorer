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


# ---------------------------------------------------------------------------------------------------------------
# Speed limit (task 2.15)

from shapely.geometry import box
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from tuplan_garage.planning.simulation.planner.pdm_planner.observation.pdm_occupancy_map import PDMOccupancyMap
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_enums import BBCoordsIndex, StateIndex

from diffusion_planner.scorer.pdm_scoring import PDMCandidateScorer
from diffusion_planner.scorer.selectors import PDMSpeedLimitSelector


class _FakeMap:
    """lane_a: 10 m/s; conn: a connector from lane_a (10) to lane_b (15); park: a carpark (no limit)."""

    def __init__(self):
        lane_a, lane_b = SimpleNamespace(speed_limit_mps=10.0), SimpleNamespace(speed_limit_mps=15.0)
        self.objects = {
            (SemanticMapLayer.LANE, "lane_a"): lane_a,
            (SemanticMapLayer.LANE, "lane_b"): lane_b,
            (SemanticMapLayer.LANE_CONNECTOR, "conn"): SimpleNamespace(incoming_edges=[lane_a], outgoing_edges=[lane_b]),
        }
        self.calls = 0

    def get_map_object(self, token, layer):
        self.calls += 1
        return self.objects.get((layer, token))


def _speed_limit_scorer(xs, speeds, H=40):
    """Scorer whose last score() left K candidates standing at x = xs[k] (y = 0) with constant speed speeds[k]."""
    scorer = PDMCandidateScorer()
    K = len(xs)
    states = np.zeros((K, H + 1, StateIndex.size()))
    states[..., StateIndex.VELOCITY_X] = np.asarray(speeds, dtype=np.float64)[:, None]
    coords = np.zeros((K, H + 1, len(BBCoordsIndex), 2))
    coords[..., BBCoordsIndex.CENTER, 0] = np.asarray(xs, dtype=np.float64)[:, None]
    scorer._scorer._states, scorer._scorer._ego_coords = states, coords
    scorer._map_api = _FakeMap()
    scorer._drivable_area_map = PDMOccupancyMap(
        ["lane_a", "conn", "park"], [box(0, -2, 50, 2), box(50, -2, 100, 2), box(100, -2, 150, 2)])
    return scorer


def test_speed_limit_compliance():
    scorer = _speed_limit_scorer(
        xs=[25, 25, 75, 125, 500],      # lane_a, lane_a, connector, carpark, off the map
        speeds=[8, 12, 12, 30, 30])

    scores = scorer.speed_limit_compliance()

    # 2 m/s over a 10 m/s limit for the whole horizon: 1 - 2 / 2.23; the connector takes its fastest lane (15)
    np.testing.assert_allclose(scores, [1.0, 1 - 2 / 2.23, 1.0, 1.0, 1.0])

    # Speed limits are looked up once per map object and scenario
    calls = scorer._map_api.calls
    scorer.speed_limit_compliance()
    assert scorer._map_api.calls == calls


def test_speed_limit_clipped_at_zero():
    scorer = _speed_limit_scorer(xs=[25], speeds=[20])
    np.testing.assert_allclose(scorer.speed_limit_compliance(), [0.0])


class _FakeMetricScorer(_FakeScorer):
    """PDM metrics where candidate 1 is faster (more progress) but 3 m/s over the limit."""

    def last_metrics(self):
        ones = np.ones(2)
        return {"multi_metrics": np.stack([ones, ones, ones]),
                "weighted_metrics": np.stack([np.array([0.8, 1.0]), ones, ones])}

    def speed_limit_compliance(self):
        return np.array([1.0, 0.0])


def test_pdm_speed_limit_selector_prefers_compliant_plan():
    candidates = torch.zeros(1, 2, 11, 80, 4)
    scorer = _FakeMetricScorer([0.9, 1.0]) # PDM's own score: candidate 1

    plain = _selector([0.9, 1.0])
    assert plain.select(candidates, SelectionContext(planner_input=object())).item() == 1

    selector = PDMSpeedLimitSelector.__new__(PDMSpeedLimitSelector)
    selector._scorer = scorer
    assert selector.select(candidates, SelectionContext(planner_input=object())).item() == 0
    # CLS weights: (5 * 0.8 + 5 + 4 + 2) / 16 vs (5 + 5 + 0 + 2) / 16
    np.testing.assert_allclose(selector.last_scores, [15 / 16, 12 / 16])


# ---------------------------------------------------------------------------------------------------------------
# Expert progress

from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_path import PDMPath


class _EgoLog:
    """Logged ego driving along x at 5 m/s from x=10; n_future frames (0.1 s apart) exist after `iteration`."""

    def __init__(self, n_future=40):
        self.n_future = n_future

    def _ego(self, k):
        state = _ego_state(x=10.0 + 0.5 * k, y=0.0, heading=0.0)
        return EgoState.build_from_rear_axle(
            rear_axle_pose=state.rear_axle, rear_axle_velocity_2d=StateVector2D(5.0, 0.0),
            rear_axle_acceleration_2d=StateVector2D(0.0, 0.0), tire_steering_angle=0.0,
            time_point=TimePoint(int(k * 1e5)), vehicle_parameters=get_pacifica_parameters(),
        )

    def get_ego_state_at_iteration(self, iteration):
        return self._ego(0)

    def get_ego_future_trajectory(self, iteration, time_horizon, num_samples):
        assert time_horizon == pytest.approx(4.0) and num_samples == 40
        for k in range(1, min(num_samples, self.n_future) + 1):
            yield self._ego(k)


def _straight_centerline_scorer():
    scorer = PDMCandidateScorer()
    scorer._centerline = PDMPath([StateSE2(x, 0.0, 0.0) for x in np.arange(0.0, 100.0, 1.0)])
    return scorer


def test_expert_progress_along_centerline():
    # 4 s at 5 m/s along the centerline
    assert _straight_centerline_scorer().expert_progress(_EgoLog(), iteration=3) == pytest.approx(20.0)


def test_expert_progress_scaled_at_log_end():
    # Only 1 s logged (5 m): scaled to the 4 s horizon
    assert _straight_centerline_scorer().expert_progress(_EgoLog(n_future=10), iteration=3) == pytest.approx(20.0)
    assert _straight_centerline_scorer().expert_progress(_EgoLog(n_future=0), iteration=3) == 0.0
