"""
Per-candidate metrics and the combined score, shared by the PDM selectors, the label pipeline and the verifier.
No nuPlan imports, so the dataset / model / training code can use it on any machine.

Metric order (the last axis of every metrics array):

    0 no_at_fault_collision   multiplicative  {0, 0.5, 1}
    1 drivable_area           multiplicative  {0, 1}
    2 driving_direction       multiplicative  {0, 0.5, 1}
    3 progress                weighted 5      [0, 1], raw progress / best raw progress among the K (PDM's normalization)
    4 ttc                     weighted 5      {0, 1}
    5 speed_limit             weighted 4      [0, 1], nuPlan's speed_limit_compliance over the scored horizon
    6 comfort                 weighted 2      {0, 1}

PDM-Closed's own score leaves out speed limit (weights 5 / 5 / 2). The closed-loop score (CLS) counts it with
weight 4, and week 1's PDM-selected runs lost 2-3 points of speed-limit compliance, so the combined score here uses
CLS's weights.
"""
from typing import Dict

import numpy as np

METRICS = [
    "no_at_fault_collision", "drivable_area", "driving_direction",
    "progress", "ttc", "speed_limit", "comfort",
]
MULTIPLICATIVE = [0, 1, 2]
WEIGHTED = [3, 4, 5, 6]
CLS_WEIGHTS = [5.0, 5.0, 4.0, 2.0]  # progress, ttc, speed_limit, comfort (nuPlan's CLS)
PDM_WEIGHTS = [5.0, 5.0, 0.0, 2.0]  # PDM-Closed: no speed-limit term

# nuPlan's speed_limit_compliance_statistics.yaml: max_overspeed_value_threshold [m/s]
MAX_OVERSPEED = 2.23


def combined_score(metrics, weights=CLS_WEIGHTS):
    """
    Args:
        metrics: [..., 7] numpy array or torch tensor, metric order of this module
        weights: weights of the weighted metrics (progress, ttc, speed_limit, comfort)

    Returns:
        scores: [...] product of the multiplicative metrics times the weighted mean of the others
    """
    multiplicative = metrics[..., 0] * metrics[..., 1] * metrics[..., 2]
    weighted = sum(w * metrics[..., i] for w, i in zip(weights, WEIGHTED)) / sum(weights)
    return multiplicative * weighted


def from_pdm(last_metrics: Dict[str, np.ndarray], speed_limit: np.ndarray) -> np.ndarray:
    """
    Args:
        last_metrics: PDMCandidateScorer.last_metrics() of a score() call
        speed_limit: [K] PDMCandidateScorer.speed_limit_compliance() of the same call

    Returns:
        metrics: [K, 7] in this module's order
    """
    multi, weighted = last_metrics["multi_metrics"], last_metrics["weighted_metrics"] # [3, K] each
    return np.stack([multi[0], multi[1], multi[2], weighted[0], weighted[1], speed_limit, weighted[2]], axis=-1)


def speed_limit_compliance(speeds: np.ndarray, limits: np.ndarray) -> np.ndarray:
    """
    nuPlan's speed_limit_compliance score of each trajectory: 1 - mean overspeed / MAX_OVERSPEED, clipped at 0.

    Args:
        speeds: [K, T] ego speed [m/s]
        limits: [K, T] speed limit at ego's position [m/s], NaN where there is none (no violation counted)

    Returns:
        scores: [K] in [0, 1]
    """
    overspeed = np.where(np.isnan(limits), 0.0, np.maximum(speeds - np.nan_to_num(limits), 0.0))
    return np.maximum(0.0, 1.0 - overspeed.mean(axis=-1) / MAX_OVERSPEED)
