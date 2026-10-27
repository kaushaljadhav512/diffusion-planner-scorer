from typing import Dict

import numpy as np
import numpy.typing as npt

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.planning.simulation.planner.abstract_planner import PlannerInput
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from tuplan_garage.planning.simulation.planner.pdm_planner.observation.pdm_observation_utils import (
    get_drivable_area_map,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.pdm_closed_planner import PDMClosedPlanner
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import (
    ego_state_to_state_array,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_enums import StateIndex
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
