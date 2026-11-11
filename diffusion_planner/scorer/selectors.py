from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional

import torch

if TYPE_CHECKING:
    import numpy as np
    from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
    from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization, PlannerInput


@dataclass
class SelectionContext:
    """
    Everything a selector may look at besides the candidates. Selectors read only what they need.

        inputs:         normalized model inputs, each [B, ...] (learned selector)
        planner_input:  nuPlan PlannerInput of the current step (PDM / oracle selectors)
        scenario:       the nuPlan scenario, i.e. the logged future (oracle selector only)
        noise_scale:    initial-noise std of each of the K candidates (recorded with the labels)
    """
    inputs: Optional[Dict[str, torch.Tensor]] = None
    planner_input: Optional["PlannerInput"] = None
    scenario: Optional["AbstractScenario"] = None
    noise_scale: Optional[List[float]] = None


class AbstractSelector(ABC):
    """Picks one of K trajectory candidates per scene."""

    def initialize(self, initialization: "PlannerInitialization") -> None:
        """Called once per scenario, from the planner's initialize (map, route, ...)."""
        pass

    @abstractmethod
    def select(self, candidates: torch.Tensor, context: SelectionContext) -> torch.Tensor:
        """
        Args:
            candidates: [B, K, P, T, 4] unnormalized future states from sample_candidates
            context: SelectionContext of the current step

        Returns:
            indices: [B] long, the chosen candidate per scene
        """
        raise NotImplementedError


class FirstSample(AbstractSelector):
    """Always candidate 0. With the same seed this is the vanilla planner."""

    def select(self, candidates: torch.Tensor, context: SelectionContext) -> torch.Tensor:
        return torch.zeros(candidates.shape[0], dtype=torch.long, device=candidates.device)


class Random(AbstractSelector):
    """Uniformly random candidate, from its own generator so it does not shift the sampler's noise."""

    def __init__(self, seed: int = 0):
        self._seed = seed
        self._generator = torch.Generator().manual_seed(seed)
        #self._generator = None

    def initialize(self, initialization: "PlannerInitialization") -> None:
        #self._generator = None
        # Reseed per scenario so results do not depend on which scenarios a worker ran before
        self._generator.manual_seed(self._seed)

    def select(self, candidates: torch.Tensor, context: SelectionContext) -> torch.Tensor:
        # if self._generator is None:
        #     self._generator = torch.Generator().manual_seed(self._seed)
            
        B, K = candidates.shape[:2]
        return torch.randint(0, K, (B,), generator=self._generator).to(candidates.device)

    def __getstate__(self):
        state = self.__dict__.copy()
        generator = state.pop("_generator")
        state["_generator_state"] = generator.get_state()
        return state

    def __setstate__(self, state):
        state = state.copy()
        generator_state = state.pop("_generator_state")
        self.__dict__.update(state)
        self._generator = torch.Generator()
        self._generator.set_state(generator_state)


class PDMSelector(AbstractSelector):
    """
    Baseline (2): PDM-Closed's scorer over the K DP candidates, with constant-velocity forecasts of the other agents.
    Scores the first 4 s of each ego plan after LQR tracking; ties (including all-zero) go to the lowest index,
    so it falls back to candidate 0 (vanilla DP) when it cannot tell candidates apart. No emergency brake.
    """

    def __init__(self, map_radius: float = 50.0, proposal_poses: int = 40):
        # Lazy import: nuplan / tuplan_garage are only needed when this selector is used
        from diffusion_planner.scorer.pdm_scoring import PDMCandidateScorer

        self._scorer = PDMCandidateScorer(map_radius=map_radius, proposal_poses=proposal_poses)
        self.last_scores = None

    def initialize(self, initialization: "PlannerInitialization") -> None:
        self._scorer.initialize(initialization)

    def select(self, candidates: torch.Tensor, context: SelectionContext) -> torch.Tensor:
        assert candidates.shape[0] == 1, "PDMSelector scores one scene (closed-loop simulation)"
        assert context.planner_input is not None, "PDMSelector needs the PlannerInput"

        ego_plans = candidates[0, :, 0].detach().cpu().numpy().astype("float64") # [K, T, 4]
        self.last_scores = self._score(context.planner_input, ego_plans)         # [K]

        # np.argmax returns the first maximum, i.e. the lowest index on ties
        return torch.tensor([int(self.last_scores.argmax())], dtype=torch.long, device=candidates.device)

    def _score(self, planner_input: "PlannerInput", ego_plans) -> "np.ndarray":
        """[K] score of each ego plan ([K, T, 4] float64), higher is better."""
        return self._scorer.score(planner_input, ego_plans)

    def last_metrics(self):
        """Per-metric PDM scores of the last select() call, see PDMCandidateScorer.last_metrics."""
        return self._scorer.last_metrics()


class PDMSpeedLimitSelector(PDMSelector):
    """
    PDMSelector plus nuPlan's speed-limit metric, which PDM's score leaves out: the weighted metrics are combined with
    the closed-loop score's weights (progress 5, TTC 5, speed limit 4, comfort 2) instead of PDM's (5, 5, 2). Week 1's
    PDM-selected runs lost 2-3 points of speed-limit compliance (task 2.15).
    """

    def _score(self, planner_input: "PlannerInput", ego_plans) -> "np.ndarray":
        from diffusion_planner.scorer.metrics import combined_score, from_pdm

        self._scorer.score(planner_input, ego_plans)
        return combined_score(from_pdm(self._scorer.last_metrics(), self._scorer.speed_limit_compliance()))


class OracleSelector(PDMSelector):
    """
    Upper bound for the headroom run: PDMSelector, but the other agents follow their logged futures instead of
    constant-velocity forecasts. Non-reactive simulation only (in reactive simulation agents leave the log).
    """

    def __init__(self, map_radius: float = 50.0, proposal_poses: int = 40):
        from diffusion_planner.scorer.pdm_scoring import PDMOracleCandidateScorer

        self._scorer = PDMOracleCandidateScorer(map_radius=map_radius, proposal_poses=proposal_poses)
        self.last_scores = None

    def select(self, candidates: torch.Tensor, context: SelectionContext) -> torch.Tensor:
        assert context.scenario is not None, "OracleSelector needs the scenario (logged future)"
        self._scorer.scenario = context.scenario
        return super().select(candidates, context)


class OracleSpeedLimitSelector(OracleSelector):
    """
    OracleSelector with PDMSpeedLimitSelector's combined score (CLS weights, speed limit included), i.e. the argmax
    of the verifier's current training target (metrics_oracle of the label files). OracleSelector itself uses PDM's
    own score, without speed limit. Non-reactive simulation only.
    """

    def _score(self, planner_input: "PlannerInput", ego_plans) -> "np.ndarray":
        from diffusion_planner.scorer.metrics import combined_score, from_pdm

        self._scorer.score(planner_input, ego_plans)
        return combined_score(from_pdm(self._scorer.last_metrics(), self._scorer.speed_limit_compliance()))


class OracleExpertProgressSelector(OracleSelector):
    """
    OracleSpeedLimitSelector with nuPlan's progress definition instead of PDM's: each candidate's progress relative
    to the human driver's logged progress over the same horizon, plus ego_is_making_progress as a multiplicative
    metric (diffusion_planner.scorer.metrics.combined_score_expert). Tests whether this label definition tracks the
    closed-loop score better than the current one. Non-reactive simulation only.

    stats_dir: if set, one CSV per scenario with, per step, the pick under both definitions and the progress values.
    """

    def __init__(self, map_radius: float = 50.0, proposal_poses: int = 40, stats_dir: Optional[str] = None):
        super().__init__(map_radius=map_radius, proposal_poses=proposal_poses)
        self._stats_dir = stats_dir
        self._scenario = None

    def select(self, candidates: torch.Tensor, context: SelectionContext) -> torch.Tensor:
        self._scenario = context.scenario
        return super().select(candidates, context)

    def _score(self, planner_input: "PlannerInput", ego_plans) -> "np.ndarray":
        import numpy as np
        from diffusion_planner.scorer.metrics import combined_score, combined_score_expert, from_pdm

        self._scorer.score(planner_input, ego_plans)
        metrics = from_pdm(self._scorer.last_metrics(), self._scorer.speed_limit_compliance())
        progress = self._scorer.last_metrics()["progress_raw"]
        expert = self._scorer.expert_progress(self._scenario, planner_input.iteration.index)
        scores = combined_score_expert(metrics, progress, expert)

        if self._stats_dir is not None:
            old = combined_score(metrics)
            self._write_stats(planner_input.iteration.index, int(np.argmax(old)), int(np.argmax(scores)),
                              expert, progress)
        return scores

    def _write_stats(self, iteration: int, old_pick: int, new_pick: int, expert: float, progress) -> None:
        import os
        scenario = self._scenario
        path = os.path.join(self._stats_dir, f"{scenario.log_name}__{scenario.token}__{scenario.scenario_type}.csv")
        new_file = not os.path.exists(path)
        os.makedirs(self._stats_dir, exist_ok=True)
        with open(path, "a") as f:
            if new_file:
                f.write("iteration,old_pick,new_pick,expert_progress,max_progress,min_progress\n")
            f.write(f"{iteration},{old_pick},{new_pick},{expert:.3f},{max(progress):.3f},{min(progress):.3f}\n")


def gather_selected(candidates: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """
    Args:
        candidates: [B, K, P, T, 4]
        indices: [B]

    Returns:
        selected: [B, P, T, 4], same layout as model(inputs)[1]["prediction"]
    """
    return candidates[torch.arange(candidates.shape[0], device=candidates.device), indices]
