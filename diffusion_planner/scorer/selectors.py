from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Optional

import torch

if TYPE_CHECKING:
    from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization, PlannerInput


@dataclass
class SelectionContext:
    """
    Everything a selector may look at besides the candidates. Selectors read only what they need.

        inputs:         normalized model inputs, each [B, ...] (learned selector)
        planner_input:  nuPlan PlannerInput of the current step (PDM / oracle selectors)
    """
    inputs: Optional[Dict[str, torch.Tensor]] = None
    planner_input: Optional["PlannerInput"] = None


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


def gather_selected(candidates: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """
    Args:
        candidates: [B, K, P, T, 4]
        indices: [B]

    Returns:
        selected: [B, P, T, 4], same layout as model(inputs)[1]["prediction"]
    """
    return candidates[torch.arange(candidates.shape[0], device=candidates.device), indices]
