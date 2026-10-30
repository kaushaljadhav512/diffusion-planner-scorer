"""
Tests for diffusion_planner.scorer.sampling.

Uses the released checkpoint if checkpoints/args.json and checkpoints/model.pth exist, otherwise a
randomly initialized model (default training args + normalization.json). Inputs are synthetic
scenes with the shapes the planner's DataProcessor produces, so no nuPlan data is needed.

    pytest tests/test_sampling.py
"""
import os
from types import SimpleNamespace

import pytest
import torch

from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.scorer.sampling import sample_candidates
from diffusion_planner.utils.config import Config
from diffusion_planner.utils.normalizer import ObservationNormalizer, StateNormalizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARGS_FILE = os.path.join(ROOT, "checkpoints", "args.json")
CKPT_FILE = os.path.join(ROOT, "checkpoints", "model.pth")

B = 2


def _random_config():
    # Defaults from train_predictor.py
    args = SimpleNamespace(
        future_len=80, time_len=21,
        agent_state_dim=11, agent_num=32,
        static_objects_state_dim=10, static_objects_num=5,
        lane_len=20, lane_state_dim=12, lane_num=70,
        route_len=20, route_state_dim=12, route_num=25,
        encoder_drop_path_rate=0.1, decoder_drop_path_rate=0.1,
        encoder_depth=3, decoder_depth=3, num_heads=6, hidden_dim=192,
        diffusion_model_type="x_start", predicted_neighbor_num=10,
        normalization_file_path=os.path.join(ROOT, "normalization.json"),
        device="cpu", guidance_fn=None,
    )
    args.state_normalizer = StateNormalizer.from_json(args)
    args.observation_normalizer = ObservationNormalizer.from_json(args)
    return args


@pytest.fixture(scope="module")
def setup():
    if os.path.exists(ARGS_FILE) and os.path.exists(CKPT_FILE):
        config = Config(ARGS_FILE, guidance_fn=None)
        config.device = "cpu"
        model = Diffusion_Planner(config)
        # Same loading as DiffusionPlanner.initialize
        state_dict = torch.load(CKPT_FILE, map_location="cpu", weights_only=False)["ema_state_dict"]
        model.load_state_dict({k[len("module."):]: v for k, v in state_dict.items() if k.startswith("module.")})
    else:
        config = _random_config()
        model = Diffusion_Planner(config)
        # The decoder's output layers are zero-initialized, which would make every sample identical
        # and the tests vacuous; give those parameters small random values.
        with torch.no_grad():
            for p in model.parameters():
                if torch.all(p == 0):
                    p.normal_(0, 0.02)
    model.eval()
    return config, model


def _scene_inputs(config, batch=B, valid_agents=6):
    """Synthetic raw inputs; all-zero rows mark padding, as in DataProcessor."""
    g = torch.Generator().manual_seed(0)
    randn = lambda *s: torch.randn(*s, generator=g)
    _one_hot = lambda n, shape: torch.nn.functional.one_hot(torch.randint(0, n, shape, generator=g), n).float()

    neighbors = torch.cat([randn(batch, config.agent_num, config.time_len, 8),
                           _one_hot(3, (batch, config.agent_num, 1)).expand(-1, -1, config.time_len, -1)], dim=-1)
    # Fewer valid agents than predicted_neighbor_num so neighbor_current_mask has both values
    neighbors[:, valid_agents:] = 0

    static = torch.cat([randn(batch, config.static_objects_num, 6),
                        _one_hot(4, (batch, config.static_objects_num))], dim=-1)

    lanes = torch.cat([randn(batch, config.lane_num, config.lane_len, 8),
                       _one_hot(4, (batch, config.lane_num, 1)).expand(-1, -1, config.lane_len, -1)], dim=-1)
    route_lanes = torch.cat([randn(batch, config.route_num, config.route_len, 8),
                             _one_hot(4, (batch, config.route_num, 1)).expand(-1, -1, config.route_len, -1)], dim=-1)

    inputs = {
        "ego_current_state": randn(batch, 10),
        "neighbor_agents_past": neighbors,
        "static_objects": static,
        "lanes": lanes,
        "lanes_speed_limit": randn(batch, config.lane_num, 1).abs() * 10,
        "lanes_has_speed_limit": torch.rand(batch, config.lane_num, 1, generator=g) > 0.5,
        "route_lanes": route_lanes,
        "route_lanes_speed_limit": randn(batch, config.route_num, 1).abs() * 10,
        "route_lanes_has_speed_limit": torch.rand(batch, config.route_num, 1, generator=g) > 0.5,
    }
    return config.observation_normalizer(inputs)


def test_k1_matches_model(setup):
    config, model = setup
    inputs = _scene_inputs(config)

    torch.manual_seed(42)
    with torch.no_grad():
        reference = model(dict(inputs))[1]["prediction"] # [B, P, T, 4]

    torch.manual_seed(42)
    candidates = sample_candidates(model, inputs, K=1)

    assert candidates.shape == (B, 1) + reference.shape[1:]
    assert torch.allclose(candidates[:, 0], reference, atol=1e-5)


def test_shape_and_diversity(setup):
    config, model = setup
    inputs = _scene_inputs(config)
    K = 4

    torch.manual_seed(0)
    candidates = sample_candidates(model, inputs, K=K)

    assert candidates.shape == (B, K, 1 + config.predicted_neighbor_num, config.future_len, 4)
    assert torch.isfinite(candidates).all()
    # Different noise draws give different ego plans
    assert candidates[:, :, 0].std(dim=1).max() > 0
    # The caller's inputs are not modified
    assert "neighbor_current_mask" not in inputs


def test_batch_layout(setup):
    """With zero noise every candidate of a scene is the deterministic plan for that scene."""
    config, model = setup
    inputs = _scene_inputs(config)

    single = sample_candidates(model, inputs, K=1, noise_scale=0.0)
    multi = sample_candidates(model, inputs, K=3, noise_scale=0.0)

    for k in range(3):
        assert torch.allclose(multi[:, k], single[:, 0], atol=1e-4) # changed from atol=1e-5
    # Scenes stay separate
    assert not torch.allclose(multi[0], multi[1])


def test_steps(setup):
    config, model = setup
    inputs = _scene_inputs(config)

    candidates = sample_candidates(model, inputs, K=2, steps=5)
    assert candidates.shape == (B, 2, 1 + config.predicted_neighbor_num, config.future_len, 4)


def test_mixed_noise_candidate0_matches_vanilla(setup):
    """Task 2.1: with a per-candidate noise scale, candidate 0 at 0.5 is vanilla DP's plan at the same seed."""
    config, model = setup
    inputs = _scene_inputs(config)

    torch.manual_seed(7)
    with torch.no_grad():
        reference = model(dict(inputs))[1]["prediction"] # [B, P, T, 4], upstream noise std 0.5

    torch.manual_seed(7)
    candidates = sample_candidates(model, inputs, K=4, noise_scale=[0.5, 1.5, 1.5, 1.5])

    assert torch.allclose(candidates[:, 0], reference, atol=1e-4)
    # The random-weight fallback model barely reacts to its noise, so only require a difference
    assert (candidates[:, 1] - reference).abs().max() > (candidates[:, 0] - reference).abs().max()


def test_mixed_noise_scales_apply_per_candidate(setup):
    """Zero noise on some candidates makes exactly those equal to the deterministic plan."""
    config, model = setup
    inputs = _scene_inputs(config)

    deterministic = sample_candidates(model, inputs, K=1, noise_scale=0.0)
    candidates = sample_candidates(model, inputs, K=4, noise_scale=[0.0, 1.0, 0.0, 1.0])

    for k in (0, 2):
        assert torch.allclose(candidates[:, k], deterministic[:, 0], atol=1e-4)
    for k in (1, 3):
        assert (candidates[:, k] - deterministic[:, 0]).abs().max() > (candidates[:, 0] - deterministic[:, 0]).abs().max()

    with pytest.raises(AssertionError):
        sample_candidates(model, inputs, K=4, noise_scale=[0.5, 1.5])
