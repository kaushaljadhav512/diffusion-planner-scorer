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

# nuPlan's ego_progress_along_expert_route_statistics.yaml (score_progress_threshold [m]) and
# ego_is_making_progress_statistics.yaml (min_progress_threshold)
EXPERT_PROGRESS_THRESHOLD = 2.0
MIN_PROGRESS_RATIO = 0.2


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


def expert_progress_ratio(progress: np.ndarray, expert_progress: float,
                          threshold: float = EXPERT_PROGRESS_THRESHOLD) -> np.ndarray:
    """
    nuPlan's ego_progress_along_expert_route score of each candidate, applied to one scored horizon instead of a
    whole scenario: min(1, max(ego, threshold) / max(expert, threshold)), and 0 if ego goes back by more than
    threshold. Unlike PDM's normalization (by the best of the K candidates) it does not depend on the other
    candidates, and it does not reward driving faster than the human driver.

    Args:
        progress: [K] raw progress of each candidate along the centerline [m]
        expert_progress: progress of the logged ego (human driver) over the same horizon [m]

    Returns:
        ratio: [K] in [0, 1]
    """
    progress = np.asarray(progress, dtype=np.float64)
    ratio = np.minimum(1.0, np.maximum(progress, threshold) / max(float(expert_progress), threshold))
    return np.where(progress < -threshold, 0.0, ratio)


def with_expert_progress(metrics: np.ndarray, progress: np.ndarray, expert_progress: float):
    """
    Metrics with nuPlan's expert-relative progress instead of PDM's normalized progress.

    Args:
        metrics: [K, 7] in this module's order (progress normalized by the best candidate, as from_pdm gives)
        progress: [K] raw progress of each candidate [m]
        expert_progress: progress of the logged ego over the same horizon [m]

    Returns:
        metrics: [K, 7] copy with column 3 replaced by expert_progress_ratio
        making_progress: [K] {0, 1}, nuPlan's ego_is_making_progress (ratio >= MIN_PROGRESS_RATIO); CLS multiplies by it
    """
    ratio = expert_progress_ratio(progress, expert_progress)
    out = np.array(metrics, dtype=np.float64, copy=True)
    out[..., 3] = ratio
    return out, (ratio >= MIN_PROGRESS_RATIO).astype(np.float64)


def combined_score_expert(metrics: np.ndarray, progress: np.ndarray, expert_progress: float,
                          weights=CLS_WEIGHTS) -> np.ndarray:
    """
    combined_score with nuPlan's progress definition: expert-relative progress in the weighted part and
    ego_is_making_progress as a fourth multiplicative metric, as in CLS.

    Returns:
        scores: [K]
    """
    expert_metrics, making_progress = with_expert_progress(metrics, progress, expert_progress)
    return combined_score(expert_metrics, weights) * making_progress


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
