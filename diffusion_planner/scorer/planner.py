import time
from dataclasses import dataclass
from typing import List, Optional

import torch

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization, PlannerInput
from nuplan.planning.simulation.planner.planner_report import PlannerReport

from diffusion_planner.planner.planner import DiffusionPlanner
from diffusion_planner.scorer.sampling import sample_candidates
from diffusion_planner.scorer.selectors import AbstractSelector, SelectionContext, gather_selected
from diffusion_planner.utils.config import Config


@dataclass(frozen=True)
class MultiSamplePlannerReport(PlannerReport):
    """
    Per-step runtimes [s] of the two stages of a step. nuPlan stores their per-scenario mean / median / std in the
    run's runner_report.parquet as <field>_mean etc. (see scripts/summarize_results.py).
    """
    sampling_runtimes: List[float]  # K candidates from the diffusion model
    selection_runtimes: List[float] # selector.select


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
            noise_scale_first: Optional[float] = None,
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
        # Per-candidate noise std; noise_scale_first sets candidate 0's (0.5 keeps vanilla DP's own plan in the set)
        self._noise_scale = [noise_scale] * K if noise_scale_first is None else [noise_scale_first] + [noise_scale] * (K - 1)
        self._steps = steps
        self._seed = seed

        self._sampling_runtimes: List[float] = []
        self._selection_runtimes: List[float] = []

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

        start = time.perf_counter()
        candidates = sample_candidates(self._planner, inputs, self._K, self._noise_scale, self._steps) # [1, K, P, T, 4]
        self._sampling_runtimes.append(_elapsed(start, candidates))

        start = time.perf_counter()
        context = SelectionContext(
            inputs=inputs, planner_input=current_input, scenario=self._scenario, noise_scale=self._noise_scale
        )
        indices = self._selector.select(candidates, context)
        self._selection_runtimes.append(_elapsed(start, indices))

        outputs = {"prediction": gather_selected(candidates, indices)} # [1, P, T, 4]

        trajectory = InterpolatedTrajectory(
            trajectory=self.outputs_to_trajectory(outputs, current_input.history.ego_states)
        )

        return trajectory

    def generate_planner_report(self, clear_stats: bool = True) -> PlannerReport:
        """
        Inherited.
        """
        report = MultiSamplePlannerReport(
            compute_trajectory_runtimes=self._compute_trajectory_runtimes,
            sampling_runtimes=self._sampling_runtimes,
            selection_runtimes=self._selection_runtimes,
        )
        if clear_stats:
            self._compute_trajectory_runtimes = []
            self._sampling_runtimes = []
            self._selection_runtimes = []
        return report


def _elapsed(start: float, output: torch.Tensor) -> float:
    """Seconds since start, after waiting for the GPU work that produced output (CUDA kernels run asynchronously)."""
    if output.is_cuda:
        torch.cuda.synchronize(output.device)
    return time.perf_counter() - start
