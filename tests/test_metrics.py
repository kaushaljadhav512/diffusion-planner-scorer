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
