"""
Tests for diffusion_planner.scorer.selectors. Synthetic candidates only, no model or nuPlan data.

    pytest tests/test_selectors.py
"""
import torch

from diffusion_planner.scorer.selectors import FirstSample, Random, SelectionContext, gather_selected

B, K, P, T = 3, 8, 11, 80


def _candidates():
    # Candidate k of scene b is filled with 100 * b + k so the selected one is easy to identify
    values = 100 * torch.arange(B)[:, None] + torch.arange(K)[None]
    return values[:, :, None, None, None].float().expand(B, K, P, T, 4)


def test_first_sample():
    candidates = _candidates()
    indices = FirstSample().select(candidates, SelectionContext())

    assert indices.shape == (B,) and indices.dtype == torch.long
    assert (indices == 0).all()


def test_random_in_range_and_reproducible():
    candidates = _candidates()
    selector = Random(seed=1)
    selector.initialize(None)
    first = torch.stack([selector.select(candidates, SelectionContext()) for _ in range(50)])

    assert first.shape == (50, B) and first.dtype == torch.long
    assert ((first >= 0) & (first < K)).all()
    assert len(first.unique()) > 1

    # Reseeded per scenario: the same sequence again
    selector.initialize(None)
    second = torch.stack([selector.select(candidates, SelectionContext()) for _ in range(50)])
    assert torch.equal(first, second)


def test_random_does_not_touch_global_rng():
    torch.manual_seed(0)
    expected = torch.randn(4)

    torch.manual_seed(0)
    Random(seed=1).select(_candidates(), SelectionContext())
    assert torch.equal(torch.randn(4), expected)


def test_gather_selected():
    candidates = _candidates()
    indices = torch.tensor([0, 5, 7])
    selected = gather_selected(candidates, indices)

    assert selected.shape == (B, P, T, 4)
    assert torch.equal(selected[:, 0, 0, 0], torch.tensor([0., 105., 207.]))
