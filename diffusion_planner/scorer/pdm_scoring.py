from typing import Dict, List

import numpy as np
import numpy.typing as npt
import shapely.creation

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.tracked_objects import TrackedObject
from nuplan.common.maps.abstract_map_objects import LaneGraphEdgeMapObject
from nuplan.common.maps.maps_datatypes import TrafficLightStatusData
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.observation.observation_type import Observation
from nuplan.planning.simulation.planner.abstract_planner import PlannerInput
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from tuplan_garage.planning.simulation.planner.pdm_planner.observation.pdm_observation import PDMObservation
from tuplan_garage.planning.simulation.planner.pdm_planner.observation.pdm_observation_utils import (
    get_drivable_area_map,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.observation.pdm_occupancy_map import PDMOccupancyMap
from tuplan_garage.planning.simulation.planner.pdm_planner.pdm_closed_planner import PDMClosedPlanner
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import (
    ego_state_to_state_array,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_enums import BBCoordsIndex, StateIndex
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_path import PDMPath


def relative_to_global_poses(ego_state: EgoState, predictions: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Ego-frame DP predictions to global rear-axle poses, as transform_predictions_to_states does.

    Args:
        ego_state: current ego state (DP's frame is its rear axle)
        predictions: [K, T, 4] ego-frame (x, y, cos, sin)

    Returns:
        poses: [K, T, 3] global (x, y, heading)
    """
    x0, y0, h0 = ego_state.rear_axle.serialize()
    cos0, sin0 = np.cos(h0), np.sin(h0)
    x, y = predictions[..., 0], predictions[..., 1]
    heading = np.arctan2(predictions[..., 3], predictions[..., 2]) + h0
    return np.stack([
        x0 + cos0 * x - sin0 * y,
        y0 + sin0 * x + cos0 * y,
        np.arctan2(np.sin(heading), np.cos(heading)),
    ], axis=-1)


class PDMCandidateScorer(PDMClosedPlanner):
    """
    PDM-Closed's map handling, constant-velocity forecasts, LQR simulation and scoring, applied to externally
    given proposals instead of PDM's own centerline proposals. Mirrors PDMClosedPlanner.compute_planner_trajectory
    and _get_closed_loop_trajectory with steps 3 (generate) and 6 (brake / extend) replaced.
    """

    def __init__(self, map_radius: float = 50.0, proposal_poses: int = 40, interval_length: float = 0.1):
        # Same sampling as tuplan's pdm_closed_planner.yaml; PDM's proposal generator is never used
        super().__init__(
            trajectory_sampling=TrajectorySampling(num_poses=80, interval_length=interval_length),
            proposal_sampling=TrajectorySampling(num_poses=proposal_poses, interval_length=interval_length),
            idm_policies=None,
            lateral_offsets=None,
            map_radius=map_radius,
        )

    @property
    def horizon_poses(self) -> int:
        return self._proposal_sampling.num_poses

    def _update_observation(self, ego_state: EgoState, current_input: PlannerInput) -> None:
        """Constant-velocity forecasts of the current detections (overridden by the oracle)."""
        _, observation = current_input.history.current_state
        self._observation.update(ego_state, observation, current_input.traffic_light_data, self._route_lane_dict)

    def score(self, current_input: PlannerInput, predictions: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Args:
            current_input: planner input of the current step
            predictions: [K, T, 4] ego-frame DP plans (x, y, cos, sin) at 0.1 s, T >= horizon_poses

        Returns:
            scores: [K] PDM scores in [0, 1] of the first horizon_poses steps (after LQR tracking)
        """
        ego_state, _ = current_input.history.current_state

        # As in PDMClosedPlanner.compute_planner_trajectory
        if self._iteration == 0:
            self._route_roadblock_correction(ego_state)
        self._drivable_area_map = get_drivable_area_map(self._map_api, ego_state, self._map_radius)

        # 1. Environment forecast and observation update
        self._update_observation(ego_state, current_input)

        # 2. Centerline, built once per scenario like PDM's proposal paths (needed for progress)
        if self._iteration == 0:
            self._centerline = PDMPath(self._get_discrete_centerline(self._get_starting_lane(ego_state)))

        # 3. Candidates as PDM state arrays [K, 1 + horizon, StateIndex.size()]: current state, then the plan's
        #    poses. The LQR tracker only reads the poses of the proposals it follows.
        poses = relative_to_global_poses(ego_state, predictions[:, :self.horizon_poses])
        states = np.zeros((len(poses), self.horizon_poses + 1, StateIndex.size()), dtype=np.float64)
        states[:, 0] = ego_state_to_state_array(ego_state)
        states[:, 1:, StateIndex.STATE_SE2] = poses

        # 4. Simulate, 5. score
        simulated_states = self._simulator.simulate_proposals(states, ego_state)
        scores = self._scorer.score_proposals(
            simulated_states,
            ego_state,
            self._observation,
            self._centerline,
            self._route_lane_dict,
            self._drivable_area_map,
            self._map_api,
        )

        self._iteration += 1
        return scores

    def last_metrics(self) -> Dict[str, npt.NDArray[np.float64]]:
        """Per-candidate metrics of the last score() call (MultiMetricIndex / WeightedMetricIndex order)."""
        return {
            "multi_metrics": self._scorer._multi_metrics.copy(),       # [3, K] no collision, drivable area, direction
            "weighted_metrics": self._scorer._weighted_metrics.copy(), # [3, K] progress (normalized), ttc, comfort
            "progress_raw": self._scorer._progress_raw.copy(),         # [K] metres along the centerline
        }


class PDMLogFutureObservation(PDMObservation):
    """
    PDMObservation whose occupancy maps place every agent at its logged pose at that time, instead of moving the
    current detections at constant velocity. Exact in non-reactive simulation, where agents replay the log.
    Same filtering as PDM (map radius around the current ego, nearest objects per type, past collisions).
    """

    @property
    def num_maps(self) -> int:
        """Occupancy maps per update, one every observation_sample_res steps starting at the current one."""
        return len(range(0, self._observation_samples + self._observation_sample_res, self._observation_sample_res))

    @property
    def map_interval(self) -> float:
        """[s] between consecutive occupancy maps."""
        return self._observation_sample_res * self._sample_interval

    @property
    def unique_objects(self) -> Dict[str, TrackedObject]:
        """Every object of any frame, as first seen (the current detection if it has one)."""
        assert self._initialized, "PDMLogFutureObservation: Has not been updated yet!"
        return self._unique_objects

    def update_from_log(
        self,
        ego_state: EgoState,
        observations: List[Observation],
        traffic_light_data: List[TrafficLightStatusData],
        route_lane_dict: Dict[str, LaneGraphEdgeMapObject],
    ) -> None:
        """
        Args:
            ego_state: current ego state
            observations: num_maps detections, the current ones then the logged ones every map_interval
            traffic_light_data: current traffic lights, held for the whole horizon as PDM does
            route_lane_dict: on-route lanes
        """
        assert len(observations) == self.num_maps, f"expected {self.num_maps} observations, got {len(observations)}"

        traffic_light_tokens, traffic_light_polygons = self._get_traffic_light_geometries(
            traffic_light_data, route_lane_dict
        )
        traffic_light_polygons = np.array(traffic_light_polygons, dtype=np.object_)

        self._occupancy_maps = []
        self._unique_objects = {}
        for observation in observations:
            object_manager = self._get_object_manager(ego_state, observation)
            for token, tracked_object in object_manager.unique_objects.items():
                self._unique_objects.setdefault(token, tracked_object)

            static_tokens, static_coords, dynamic_tokens, dynamic_coords, _ = object_manager.get_nearest_objects(
                ego_state.center.point
            )

            tokens, polygons = [], []
            for object_tokens, coords in ((static_tokens, static_coords), (dynamic_tokens, dynamic_coords)):
                if len(object_tokens) == 0:
                    continue
                # [N, 5, 2] corners + center; overwrite the center with the first corner to close the ring, as PDM does
                coords = coords.reshape(len(object_tokens), len(BBCoordsIndex), 2).copy()
                coords[:, BBCoordsIndex.CENTER] = coords[:, BBCoordsIndex.FRONT_LEFT]
                tokens += list(object_tokens)
                polygons.append(shapely.creation.polygons(coords))

            self._occupancy_maps.append(PDMOccupancyMap(
                tokens + traffic_light_tokens,
                np.concatenate(polygons + [traffic_light_polygons], axis=0),
            ))

        # As PDMObservation.update: remember what ego overlaps now and ignore it from here on
        ego_polygon = ego_state.car_footprint.geometry
        new_collided_track_ids = []
        for token in self._occupancy_maps[0].intersects(ego_polygon):
            if self._red_light_token in token and not ego_polygon.within(self._occupancy_maps[0][token]):
                continue
            new_collided_track_ids.append(token)

        self._collided_track_ids = self._collided_track_ids + new_collided_track_ids
        self._initialized = True


class PDMOracleCandidateScorer(PDMCandidateScorer):
    """
    PDMCandidateScorer with the other agents' logged futures instead of constant-velocity forecasts. Only valid
    in non-reactive simulation; in reactive simulation agents leave the log and the forecast is wrong.
    """

    def __init__(self, map_radius: float = 50.0, proposal_poses: int = 40, interval_length: float = 0.1):
        super().__init__(map_radius=map_radius, proposal_poses=proposal_poses, interval_length=interval_length)
        self._observation = PDMLogFutureObservation(self._trajectory_sampling, self._proposal_sampling, map_radius)
        self.scenario: AbstractScenario = None # set by OracleSelector before each score()

    def _update_observation(self, ego_state: EgoState, current_input: PlannerInput) -> None:
        assert self.scenario is not None, "PDMOracleCandidateScorer needs the scenario"
        _, observation = current_input.history.current_state

        num_future = self._observation.num_maps - 1
        future = list(self.scenario.get_future_tracked_objects(
            iteration=current_input.iteration.index,
            time_horizon=num_future * self._observation.map_interval,
            num_samples=num_future,
        ))
        # Near the end of a log fewer frames exist: hold the last one
        future += [future[-1] if future else observation] * (num_future - len(future))

        self._observation.update_from_log(
            ego_state, [observation] + future, current_input.traffic_light_data, self._route_lane_dict
        )
