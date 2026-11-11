"""
Tests for diffusion_planner.scorer.metrics. No model or nuPlan data.

    pytest tests/test_metrics.py
"""
import numpy as np
import torch

from diffusion_planner.scorer.metrics import METRICS, PDM_WEIGHTS, combined_score, from_pdm, speed_limit_compliance


def test_combined_score_numpy_and_torch():
    metrics = np.array([
        [1, 1, 1, 1.0, 1, 1, 1],    # perfect
        [0.5, 1, 1, 1.0, 1, 1, 1],  # at-fault collision with a static object: halves everything
        [1, 1, 1, 0.5, 0, 1, 0],    # half progress, TTC and comfort fail
        [1, 0, 1, 1.0, 1, 1, 1],    # off the drivable area: 0
    ])
    expected = [1.0, 0.5, (5 * 0.5 + 4) / 16, 0.0]

    np.testing.assert_allclose(combined_score(metrics), expected)
    np.testing.assert_allclose(combined_score(torch.tensor(metrics)).numpy(), expected)
    # PDM's weights ignore speed limit
    assert combined_score(np.array([1, 1, 1, 1, 1, 0, 1.0]), PDM_WEIGHTS) == 1.0


def test_from_pdm_order():
    last_metrics = {
        "multi_metrics": np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),    # collision, drivable, direction
        "weighted_metrics": np.array([[0.7, 0.8], [0.9, 1.0], [0.0, 1.0]]), # progress, ttc, comfort
    }
    metrics = from_pdm(last_metrics, np.array([0.25, 0.75]))

    assert metrics.shape == (2, len(METRICS))
    np.testing.assert_allclose(metrics[0], [0.1, 0.3, 0.5, 0.7, 0.9, 0.25, 0.0])


def test_speed_limit_compliance_nan_means_no_limit():
    speeds = np.array([[12.0, 12.0], [12.0, 12.0]])
    limits = np.array([[10.0, np.nan], [np.nan, np.nan]])
    np.testing.assert_allclose(speed_limit_compliance(speeds, limits), [1 - 1 / 2.23, 1.0])


# ---------------------------------------------------------------------------------------------------------------
# nuPlan's expert-relative progress

from diffusion_planner.scorer.metrics import combined_score_expert, expert_progress_ratio, with_expert_progress


def test_expert_progress_ratio_matches_nuplan():
    # min(1, max(ego, 2) / max(expert, 2)); 0 below -2 m
    np.testing.assert_allclose(expert_progress_ratio([0.0, 5.0, 10.0, 20.0, -3.0], 10.0), [0.2, 0.5, 1.0, 1.0, 0.0])
    # Expert (nearly) stopped: anything up to 2 m scores 1, as nuPlan's threshold intends
    np.testing.assert_allclose(expert_progress_ratio([0.0, 1.5, 6.0], 0.5), [1.0, 1.0, 1.0])


def test_combined_score_expert():
    metrics = np.ones((3, 7))
    metrics[:, 3] = [1.0, 0.5, 0.05]  # PDM's normalized progress, ignored
    progress = np.array([30.0, 10.0, 1.0])

    expert_metrics, making = with_expert_progress(metrics, progress, expert_progress=20.0)
    np.testing.assert_allclose(expert_metrics[:, 3], [1.0, 0.5, 0.1])
    np.testing.assert_allclose(making, [1, 1, 0])      # 0.1 < 0.2: not making progress
    assert metrics[2, 3] == 0.05                      # input untouched

    np.testing.assert_allclose(combined_score_expert(metrics, progress, 20.0), [1.0, (5 * 0.5 + 11) / 16, 0.0])
    # Faster than the human driver earns nothing extra: 30 m and 20 m both score 1
    assert combined_score_expert(metrics, np.array([30.0, 20.0, 20.0]), 20.0)[0] == 1.0
