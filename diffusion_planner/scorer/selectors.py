from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Optional

import torch

if TYPE_CHECKING:
    from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
    from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization, PlannerInput


@dataclass
class SelectionContext:
    """
    Everything a selector may look at besides the candidates. Selectors read only what they need.

        inputs:         normalized model inputs, each [B, ...] (learned selector)
        planner_input:  nuPlan PlannerInput of the current step (PDM / oracle selectors)
        scenario:       the nuPlan scenario, i.e. the logged future (oracle selector only)
    """
    inputs: Optional[Dict[str, torch.Tensor]] = None
    planner_input: Optional["PlannerInput"] = None
    scenario: Optional["AbstractScenario"] = None


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

    def initialize(self, initialization: "PlannerInitialization") -> None:
        # Reseed per scenario so results do not depend on which scenarios a worker ran before
        self._generator.manual_seed(self._seed)

    def select(self, candidates: torch.Tensor, context: SelectionContext) -> torch.Tensor:
        B, K = candidates.shape[:2]
        return torch.randint(0, K, (B,), generator=self._generator).to(candidates.device)


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
        self.last_scores = self._scorer.score(context.planner_input, ego_plans)  # [K]

        # np.argmax returns the first maximum, i.e. the lowest index on ties
        return torch.tensor([int(self.last_scores.argmax())], dtype=torch.long, device=candidates.device)

    def last_metrics(self):
        """Per-metric PDM scores of the last select() call, see PDMCandidateScorer.last_metrics."""
        return self._scorer.last_metrics()


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


def gather_selected(candidates: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """
    Args:
        candidates: [B, K, P, T, 4]
        indices: [B]

    Returns:
        selected: [B, P, T, 4], same layout as model(inputs)[1]["prediction"]
    """
    return candidates[torch.arange(candidates.shape[0], device=candidates.device), indices]
