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


def gather_selected(candidates: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """
    Args:
        candidates: [B, K, P, T, 4]
        indices: [B]

    Returns:
        selected: [B, P, T, 4], same layout as model(inputs)[1]["prediction"]
    """
    return candidates[torch.arange(candidates.shape[0], device=candidates.device), indices]


class VerifierSelector(AbstractSelector):
    """
    Learned verifier (diffusion_planner.scorer.model.Verifier): the candidate with the highest predicted combined
    score; ties go to the lowest index, as in PDMSelector. Loads the checkpoint once.
    """

    def __init__(self, ckpt_path: str, args_file: str, device: str = "cuda"):
        from diffusion_planner.scorer.model import Verifier

        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self._device = device
        self._model = Verifier.load(ckpt_path, args_file, device=device)

    @torch.no_grad()
    def select(self, candidates: torch.Tensor, context: SelectionContext) -> torch.Tensor:
        assert context.inputs is not None, "VerifierSelector needs the normalized model inputs"
        inputs = {k: v.to(self._device) for k, v in context.inputs.items()}
        scores = self._model.score(inputs, candidates[:, :, 0].to(self._device).float())   # [B, K]
        best = scores.max(dim=1, keepdim=True).values
        return (scores >= best - 1e-6).float().argmax(dim=1).to(candidates.device)
