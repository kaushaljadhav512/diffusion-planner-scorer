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


class _StubScorer:
    """Stands in for PDMCandidateScorer: 3 candidates, all safe, raw progress 2, 5 and 12 m."""

    def score(self, planner_input, ego_plans):
        self.scored = len(ego_plans)

    def last_metrics(self):
        import numpy as np
        return {"multi_metrics": np.ones((3, 3)),
                "weighted_metrics": np.array([[2 / 12, 5 / 12, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
                "progress_raw": np.array([2.0, 5.0, 12.0])}

    def speed_limit_compliance(self):
        import numpy as np
        return np.ones(3)

    def expert_progress(self, scenario, iteration):
        return 5.0


def test_pdm_expert_progress_selector_caps_progress_at_the_human():
    from types import SimpleNamespace

    from diffusion_planner.scorer.selectors import PDMExpertProgressSelector

    # Bypass __init__ (it builds PDM's nuPlan scorer); the stub supplies what _score reads
    selector = object.__new__(PDMExpertProgressSelector)
    selector._scorer, selector.last_scores = _StubScorer(), None
    candidates = torch.zeros(1, 3, P, T, 4)
    context = SelectionContext(planner_input=SimpleNamespace(iteration=SimpleNamespace(index=0)), scenario=object())
    pick = selector.select(candidates, context)

    # PDM's normalization would prefer the 12 m candidate. Against a human progress of 5 m, the 5 m and 12 m
    # candidates both score progress 1 and tie, so the lowest index of the two wins; the 2 m one scores 0.4.
    scores = selector.last_scores
    assert abs(scores[1] - scores[2]) < 1e-9 and scores[0] < scores[1]
    assert int(pick[0]) == 1


def _estimated(estimate, v, a):
    from types import SimpleNamespace

    from diffusion_planner.scorer.selectors import PDMEstimatedProgressSelector

    selector = object.__new__(PDMEstimatedProgressSelector)
    scorer = _StubScorer()
    scorer.horizon_poses, scorer._proposal_sampling = 40, SimpleNamespace(interval_length=0.1)
    selector._scorer, selector.last_scores, selector._estimate, selector.last_estimate = scorer, None, estimate, None
    car = SimpleNamespace(rear_axle_velocity_2d=SimpleNamespace(x=v), rear_axle_acceleration_2d=SimpleNamespace(x=a))
    planner_input = SimpleNamespace(iteration=SimpleNamespace(index=0),
                                    history=SimpleNamespace(current_state=[SimpleNamespace(dynamic_car_state=car)]))
    selector.select(torch.zeros(1, 3, P, T, 4), SelectionContext(planner_input=planner_input))
    return selector.last_estimate


def test_estimated_progress():
    # stub candidates progress 2, 5 and 12 m over a 4 s horizon
    assert _estimated("median", 0.0, 0.0) == 5.0
    assert _estimated("first", 0.0, 0.0) == 2.0
    assert abs(_estimated("cv", 3.0, -1.0) - 12.0) < 1e-9             # 3 m/s x 4 s, acceleration ignored
    assert abs(_estimated("ca", 3.0, 0.5) - (12.0 + 4.0)) < 1e-9      # + a t^2 / 2
    assert abs(_estimated("ca", 3.0, -0.5) - (12.0 - 4.0)) < 1e-9     # still moving at 4 s (stops at 6 s)
    assert abs(_estimated("ca", 2.0, -2.0) - 1.0) < 1e-9              # stops after 1 s: v^2 / (2 |a|)
    assert _estimated("cv", -0.3, 0.0) == 0.0                         # reversing counts as standing still
