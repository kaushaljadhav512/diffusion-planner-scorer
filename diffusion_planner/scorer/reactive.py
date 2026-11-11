"""
Candidate scoring against reactive agents, for the agreement check (does PDM+SL pick what reactive traffic rewards?).

In reactive simulation (closed_loop_reactive_agents) the other vehicles are nuPlan's IDMAgents: they follow lanes and
adjust their speed to whatever is ahead of them, ego included. PDM forecasts them at constant velocity and the oracle
replays the log, so neither lets them react to the candidate. PDMReactiveCandidateScorer forks the simulation's live
IDM agents once per candidate, rolls them forward while ego follows that candidate (after the LQR tracking PDM uses),
and scores the candidate with PDM's scorer against the agents it provoked. The metric definitions are those of the
labels (PDM's metrics, speed limit, progress normalized by the best of the K), so only the agent model differs.

The live agents come from the running nuPlan Simulation, found on the call stack (the planner runs inside
SimulationRunner.run, in the same process). They are never modified: a fork copies every piece of state an IDM step
mutates, and shares the rest (map, scenario, agent shapes).
"""
import copy
import sys
from collections import deque
from functools import lru_cache
from typing import Dict, List, Tuple

import numpy as np
import numpy.typing as npt

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.observation.idm_agents import IDMAgents
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.planner.abstract_planner import PlannerInput
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration

from tuplan_garage.planning.simulation.planner.pdm_planner.observation.pdm_observation_utils import (
    get_drivable_area_map,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import (
    ego_state_to_state_array,
    state_array_to_ego_state,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_enums import StateIndex
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_path import PDMPath

from diffusion_planner.scorer.pdm_scoring import (
    PDMCandidateScorer,
    PDMLogFutureObservation,
    relative_to_global_poses,
)

# PDM's TTC looks up to 0.9 s past the proposal horizon (tuplan's pdm_scorer._calculate_ttc)
TTC_LOOKAHEAD_STEPS = 10


def find_simulation():
    """The nuPlan Simulation whose runner called the planner (same process, on the call stack)."""
    from nuplan.planning.simulation.runner.simulations_runner import SimulationRunner

    frame = sys._getframe(1)
    while frame is not None:
        owner = frame.f_locals.get("self")
        if isinstance(owner, SimulationRunner):
            return owner.simulation
        frame = frame.f_back
    raise RuntimeError("no nuPlan SimulationRunner on the call stack: reactive scoring needs closed-loop simulation")


def fork_idm_agents(observation: IDMAgents) -> IDMAgents:
    """
    An IDMAgents whose agents can be propagated without touching the original. Copies what an IDM step mutates
    (each agent's progress / velocity, route deque, policy target velocity; the occupancy map's geometry dict; the
    agent dict, which out-of-range filtering pops from); shares the rest.
    """
    manager = observation._get_idm_agent_manager()
    agents = {}
    for token, agent in manager.agents.items():
        forked = copy.copy(agent)
        forked._state = copy.copy(agent._state)
        forked._route = deque(agent._route, maxlen=agent._route.maxlen)
        forked._policy = copy.copy(agent._policy)
        agents[token] = forked

    occupancy = copy.copy(manager.agent_occupancy)
    occupancy._geom_map = dict(manager.agent_occupancy._geom_map)

    forked_manager = copy.copy(manager)
    forked_manager.agents = agents
    forked_manager.agent_occupancy = occupancy

    fork = copy.copy(observation)
    fork._idm_agent_manager = forked_manager
    return fork


def idm_state_snapshot(observation: IDMAgents) -> Tuple:
    """Everything an IDM step changes, to check that scoring left the live simulation untouched."""
    manager = observation._get_idm_agent_manager()
    agents = tuple(sorted(
        (token, agent._state.progress, agent._state.velocity, len(agent._route), agent._policy.target_velocity)
        for token, agent in manager.agents.items()
    ))
    return observation.current_iteration, agents, tuple(sorted(manager.agent_occupancy._geom_map))


class CachedScenario:
    """
    The scenario lookups IDMAgents makes every step (traffic lights, open-loop detections), memoized per iteration:
    the K rollouts of one step query the same iterations. Everything else is delegated.
    """

    def __init__(self, scenario: AbstractScenario):
        self._scenario = scenario
        self.get_traffic_light_status_at_iteration = lru_cache(maxsize=None)(
            lambda iteration: list(scenario.get_traffic_light_status_at_iteration(iteration)))
        self.get_tracked_objects_at_iteration = lru_cache(maxsize=None)(scenario.get_tracked_objects_at_iteration)
        self.get_time_point = lru_cache(maxsize=None)(scenario.get_time_point)

    def __getattr__(self, name):
        return getattr(self._scenario, name)


class _History:
    """The part of SimulationHistoryBuffer that IDMAgents.update_observation reads."""

    def __init__(self, ego_state: EgoState):
        self.current_state = (ego_state, None)


def extend_constant_velocity(states: npt.NDArray[np.float64], num_states: int, dt: float) -> npt.NDArray[np.float64]:
    """
    Args:
        states: [T, StateIndex.size()] PDM state arrays, 0.1 s apart
        num_states: length wanted; past T, the last state moves on at its velocity (no acceleration)

    Returns:
        [num_states, StateIndex.size()]
    """
    if num_states <= len(states):
        return states[:num_states]
    last = states[-1].copy()
    last[StateIndex.ACCELERATION_2D] = 0.0
    extra = []
    for _ in range(num_states - len(states)):
        heading = last[StateIndex.HEADING]
        vx, vy = last[StateIndex.VELOCITY_X], last[StateIndex.VELOCITY_Y] # rear-axle velocity, ego frame
        last = last.copy()
        last[StateIndex.X] += (vx * np.cos(heading) - vy * np.sin(heading)) * dt
        last[StateIndex.Y] += (vx * np.sin(heading) + vy * np.cos(heading)) * dt
        extra.append(last)
    return np.concatenate([states, np.stack(extra)], axis=0)


def detections_centers(observation: DetectionsTracks) -> Dict[str, Tuple[float, float]]:
    """Track token -> (x, y) of every tracked object."""
    return {obj.track_token: (obj.center.x, obj.center.y) for obj in observation.tracked_objects.tracked_objects}


class PDMReactiveCandidateScorer(PDMCandidateScorer):
    """
    PDMCandidateScorer whose other agents are forked from the simulation's IDM agents and react to each candidate.
    Use score_reactive (score() would fall back to constant-velocity forecasts).
    """

    def __init__(self, map_radius: float = 50.0, proposal_poses: int = 40, interval_length: float = 0.1):
        super().__init__(map_radius=map_radius, proposal_poses=proposal_poses, interval_length=interval_length)
        self._observation = PDMLogFutureObservation(self._trajectory_sampling, self._proposal_sampling, map_radius)
        self.first_frames: List[DetectionsTracks] = [] # per candidate: the forked agents one step ahead (fidelity)

    @property
    def rollout_steps(self) -> int:
        """Simulation steps rolled out per candidate: the scored horizon plus PDM's TTC look-ahead."""
        return self.horizon_poses + TTC_LOOKAHEAD_STEPS

    def score_reactive(
        self, current_input: PlannerInput, predictions: npt.NDArray[np.float64], idm_agents: IDMAgents,
        scenario: AbstractScenario,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Args:
            current_input: planner input of the current step
            predictions: [K, T, 4] ego-frame DP plans (x, y, cos, sin) at 0.1 s, T >= horizon_poses
            idm_agents: the simulation's live IDMAgents observation (not modified)
            scenario: the simulated scenario (traffic lights, open-loop detections)

        Returns:
            metrics: [K, 7] in diffusion_planner.scorer.metrics order, progress normalized by the best candidate
            progress_raw: [K] metres along the centerline
        """
        ego_state, observation = current_input.history.current_state
        iteration = current_input.iteration.index
        dt = self._proposal_sampling.interval_length

        # As PDMCandidateScorer.score
        if self._iteration == 0:
            self._route_roadblock_correction(ego_state)
        self._drivable_area_map = get_drivable_area_map(self._map_api, ego_state, self._map_radius)
        if self._iteration == 0:
            self._centerline = PDMPath(self._get_discrete_centerline(self._get_starting_lane(ego_state)))

        poses = relative_to_global_poses(ego_state, predictions[:, :self.horizon_poses])
        states = np.zeros((len(poses), self.horizon_poses + 1, StateIndex.size()), dtype=np.float64)
        states[:, 0] = ego_state_to_state_array(ego_state)
        states[:, 1:, StateIndex.STATE_SE2] = poses
        simulated_states = self._simulator.simulate_proposals(states, ego_state) # [K, 1 + H, S]

        cached = CachedScenario(scenario)
        n = self.rollout_steps
        res = self._observation._observation_sample_res
        iterations = [SimulationIteration(cached.get_time_point(iteration + s), iteration + s) for s in range(n + 1)]
        vehicle = ego_state.car_footprint.vehicle_parameters

        K = len(simulated_states)
        multi, ttc, comfort, progress_raw, speed_limit = [], [], [], [], []
        self.first_frames = []
        for k in range(K):
            ego_arrays = extend_constant_velocity(simulated_states[k], n, dt)
            fork = fork_idm_agents(idm_agents)
            fork._scenario = cached
            frames = [observation]
            for s in range(n):
                ego = state_array_to_ego_state(ego_arrays[s], iterations[s].time_point, vehicle)
                fork.update_observation(iterations[s], iterations[s + 1], _History(ego))
                if s == 0:
                    self.first_frames.append(fork.get_observation())
                if (s + 1) % res == 0:
                    frames.append(fork.get_observation())
            # Occupancy maps past the rolled-out steps are never read by the scorer; hold the last frame
            frames += [frames[-1]] * (self._observation.num_maps - len(frames))

            self._observation.update_from_log(
                ego_state, frames, current_input.traffic_light_data, self._route_lane_dict
            )
            self._scorer.score_proposals(
                simulated_states[k:k + 1], ego_state, self._observation, self._centerline,
                self._route_lane_dict, self._drivable_area_map, self._map_api,
            )
            last = self.last_metrics()
            multi.append(last["multi_metrics"][:, 0])
            ttc.append(last["weighted_metrics"][1, 0])
            comfort.append(last["weighted_metrics"][2, 0])
            progress_raw.append(last["progress_raw"][0])
            speed_limit.append(self.speed_limit_compliance()[0])

        self._iteration += 1
        multi = np.stack(multi)                    # [K, 3]
        progress_raw = np.asarray(progress_raw)
        progress = normalized_progress(progress_raw, multi.prod(axis=1))
        metrics = np.column_stack([multi, progress, ttc, speed_limit, comfort])
        return metrics, progress_raw


def normalized_progress(progress_raw: npt.NDArray[np.float64], multiplicative: npt.NDArray[np.float64],
                        threshold: float = 0.1) -> npt.NDArray[np.float64]:
    """PDM's progress normalization across the K candidates (tuplan's PDMScorer._aggregate_scores)."""
    raw = progress_raw * multiplicative
    best = raw.max()
    if best > threshold:
        return raw / best
    progress = np.ones(len(raw), dtype=np.float64)
    progress[multiplicative == 0.0] = 0.0
    return progress
