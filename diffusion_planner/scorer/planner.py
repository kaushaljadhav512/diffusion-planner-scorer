from typing import Optional

import torch

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization, PlannerInput

from diffusion_planner.planner.planner import DiffusionPlanner
from diffusion_planner.scorer.sampling import sample_candidates
from diffusion_planner.scorer.selectors import AbstractSelector, SelectionContext, gather_selected
from diffusion_planner.utils.config import Config


class MultiSampleDiffusionPlanner(DiffusionPlanner):
    """
    Best-of-K Diffusion Planner: samples K candidate plans per step and executes the one the selector picks.
    With K=1 (or FirstSample) it behaves like DiffusionPlanner.
    """

    # nuPlan's planner builder then passes the scenario; only OracleSelector reads it (logged future)
    requires_scenario: bool = True

    def __init__(
            self,
            config: Config,
            ckpt_path: str,

            past_trajectory_sampling: TrajectorySampling,
            future_trajectory_sampling: TrajectorySampling,

            selector: AbstractSelector,
            K: int = 16,
            noise_scale: float = 0.5,
            steps: int = 10,
            seed: Optional[int] = 0,

            enable_ema: bool = True,
            device: str = "cpu",

            scenario: Optional[AbstractScenario] = None,
        ):
        super().__init__(config, ckpt_path, past_trajectory_sampling, future_trajectory_sampling, enable_ema, device)

        self._scenario = scenario
        self._selector = selector
        self._K = K
        self._noise_scale = noise_scale
        self._steps = steps
        self._seed = seed

    def name(self) -> str:
        """
        Inherited.
        """
        return "diffusion_planner_k"

    def initialize(self, initialization: PlannerInitialization) -> None:
        """
        Inherited.
        """
        super().initialize(initialization)
        self._selector.initialize(initialization)
        # Seed per scenario so a scenario's samples do not depend on what the worker ran before it
        if self._seed is not None:
            torch.manual_seed(self._seed)

    def compute_planner_trajectory(self, current_input: PlannerInput) -> AbstractTrajectory:
        """
        Inherited.
        """
        inputs = self.planner_input_to_model_inputs(current_input)

        inputs = self.observation_normalizer(inputs)
        candidates = sample_candidates(self._planner, inputs, self._K, self._noise_scale, self._steps) # [1, K, P, T, 4]

        context = SelectionContext(inputs=inputs, planner_input=current_input, scenario=self._scenario)
        indices = self._selector.select(candidates, context)
        outputs = {"prediction": gather_selected(candidates, indices)} # [1, P, T, 4]

        trajectory = InterpolatedTrajectory(
            trajectory=self.outputs_to_trajectory(outputs, current_input.history.ego_states)
        )

        return trajectory
