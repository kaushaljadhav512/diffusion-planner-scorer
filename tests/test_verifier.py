"""
Tests for the verifier pipeline: label files (labels.py), dataset.py, model.py and train.py. Synthetic scenes and
labels with a randomly initialized DP config, no nuPlan data or checkpoint.

    pytest tests/test_verifier.py
"""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.scorer import train as train_module
from diffusion_planner.scorer.dataset import (
    INPUT_KEYS, LabelDataset, find_samples, split_by_scenario, subsample_scenarios,
)
from diffusion_planner.scorer.labels import LabelRecorder, load_sample, sample_dir, save_sample
from diffusion_planner.scorer.metrics import METRICS
from diffusion_planner.scorer.model import Verifier, scene_padding_mask
from diffusion_planner.scorer.selectors import SelectionContext
from diffusion_planner.scorer.train import first_max, random_metrics, selection_metrics

from test_sampling import _random_config, _scene_inputs

K, T = 4, 80


@pytest.fixture(scope="module")
def config():
    return _random_config()


def _metrics_for(plans: np.ndarray) -> np.ndarray:
    """Synthetic labels that depend on the plan: progress = endpoint x, at-fault collision if the endpoint y > 0."""
    end_x, end_y = plans[:, -1, 0], plans[:, -1, 1]
    metrics = np.ones((len(plans), len(METRICS)), dtype=np.float32)
    metrics[:, 0] = (end_y <= 0).astype(np.float32)
    metrics[:, 3] = np.clip(end_x, 0, None) / max(np.clip(end_x, 0, None).max(), 1e-6)
    return metrics


def _write_dataset(root: Path, config, n_scenarios=4, steps=4, seed=0):
    rng = np.random.default_rng(seed)
    inputs = _scene_inputs(config, batch=n_scenarios * steps)
    for i in range(n_scenarios * steps):
        scenario, iteration = divmod(i, steps)
        plans = np.zeros((K, T, 4), dtype=np.float32)
        plans[..., 0] = np.linspace(0, 1, T)[None] * rng.uniform(5, 40, (K, 1))
        plans[..., 1] = np.linspace(0, 1, T)[None] * rng.normal(0, 2, (K, 1))
        plans[..., 2] = 1.0
        metrics = _metrics_for(plans)
        save_sample(
            sample_dir(root, "log", f"tok{scenario}", "type") / f"{5 * iteration:04d}.npz",
            inputs={k: v[i].numpy() for k, v in inputs.items()},
            candidates=plans, metrics_oracle=metrics, metrics_cv=metrics[::-1].copy(),
            progress_oracle=plans[:, -1, 0], progress_cv=plans[:, -1, 0], selected=0,
            noise_scale=np.full(K, 1.5), scenario_token=f"tok{scenario}", scenario_type="type", log_name="log",
            iteration=5 * iteration,
        )


# ---------------------------------------------------------------------------------------------------------------
# Label files and dataset


def test_save_load_roundtrip(tmp_path, config):
    _write_dataset(tmp_path, config, n_scenarios=1, steps=1)
    files = find_samples(tmp_path)
    assert len(files) == 1 and files[0].parent.name == "log__tok0__type"
    assert not list(tmp_path.rglob("*.tmp.npz"))

    sample = load_sample(files[0])
    assert set(sample["inputs"]) == set(INPUT_KEYS)
    assert sample["candidates"].shape == (K, T, 4)
    assert sample["metrics_oracle"].shape == (K, len(METRICS))
    assert sample["scenario_token"] == "tok0" and sample["iteration"] == 0 and sample["selected"] == 0


def test_split_by_scenario(tmp_path, config):
    _write_dataset(tmp_path, config, n_scenarios=10, steps=2)
    files = find_samples(tmp_path)

    train, val = split_by_scenario(files, val_frac=0.2, seed=0)
    assert len(train) + len(val) == len(files)
    assert {f.parent for f in train}.isdisjoint({f.parent for f in val})
    assert len({f.parent for f in val}) == 2
    assert split_by_scenario(files, val_frac=0.2, seed=0) == (train, val)


def test_subsample_scenarios(tmp_path, config):
    _write_dataset(tmp_path, config, n_scenarios=10, steps=2)
    files = find_samples(tmp_path)

    half = subsample_scenarios(files, 0.5, seed=1)
    fifth = subsample_scenarios(files, 0.2, seed=1)
    assert len({f.parent for f in half}) == 5 and len(half) == 10  # whole scenarios, all their samples
    assert len({f.parent for f in fifth}) == 2
    assert set(fifth) <= set(half)                                  # nested for one seed
    assert len({f.parent for f in subsample_scenarios(files, 0.01)}) == 1


def test_dataset_batches(tmp_path, config):
    _write_dataset(tmp_path, config, n_scenarios=2, steps=3)
    batch = next(iter(DataLoader(LabelDataset(find_samples(tmp_path)), batch_size=4)))

    assert batch["candidates"].shape == (4, K, T, 4)
    assert batch["metrics_oracle"].shape == batch["metrics_cv"].shape == (4, K, len(METRICS))
    assert batch["inputs"]["lanes"].shape == (4, config.lane_num, config.lane_len, 12)
    assert batch["inputs"]["lanes_has_speed_limit"].dtype == torch.bool


class _FakeScorer:
    def __init__(self, progress):
        self.progress = np.asarray(progress, dtype=np.float64)
        self.scored = []

    def score(self, planner_input, plans):
        self.scored.append(plans.shape)
        return self.progress

    def last_metrics(self):
        ones = np.ones(len(self.progress))
        return {"multi_metrics": np.stack([ones] * 3), "weighted_metrics": np.stack([self.progress, ones, ones]),
                "progress_raw": 10 * self.progress}

    def speed_limit_compliance(self):
        return np.ones(len(self.progress))


def test_label_recorder_records_every_nth_step(tmp_path):
    from diffusion_planner.scorer.selectors import PDMSelector

    recorder = LabelRecorder.__new__(LabelRecorder) # skip building the real scorers (need a map)
    recorder._driver = PDMSelector.__new__(PDMSelector)
    recorder._driver._scorer = _FakeScorer([0.2, 1.0, 0.5])  # constant-velocity view: candidate 1
    recorder._oracle = _FakeScorer([1.0, 0.0, 0.5])           # logged future: candidate 0
    recorder._out_dir, recorder._record_every, recorder._save_neighbors, recorder._step = tmp_path, 2, False, 0

    scenario = SimpleNamespace(token="abc", scenario_type="turn", log_name="log1")
    candidates = torch.randn(1, 3, 11, T, 4)
    for step in range(5):
        context = SelectionContext(
            inputs={"ego_current_state": torch.ones(1, 10)}, scenario=scenario, noise_scale=[0.5, 1.5, 1.5],
            planner_input=SimpleNamespace(iteration=SimpleNamespace(index=step)))
        assert recorder.select(candidates, context).item() == 1 # drives with the constant-velocity PDM pick

    files = find_samples(tmp_path)
    assert [f.name for f in files] == ["0000.npz", "0002.npz", "0004.npz"]
    assert files[0].parent.name == "log1__abc__turn"
    sample = load_sample(files[1])
    assert sample["selected"] == 1 and sample["iteration"] == 2
    np.testing.assert_allclose(sample["metrics_oracle"][:, 3], [1.0, 0.0, 0.5])
    np.testing.assert_allclose(sample["metrics_cv"][:, 3], [0.2, 1.0, 0.5])
    np.testing.assert_allclose(sample["progress_oracle"], [10.0, 0.0, 5.0])
    np.testing.assert_allclose(sample["noise_scale"], [0.5, 1.5, 1.5])
    np.testing.assert_allclose(sample["candidates"], candidates[0, :, 0].numpy(), atol=1e-6)
    assert recorder._oracle.scenario is scenario and len(recorder._oracle.scored) == 3


# ---------------------------------------------------------------------------------------------------------------
# Model


def test_scene_padding_mask_matches_encoder(config):
    inputs = _scene_inputs(config, batch=2)
    encoder = Diffusion_Planner(config).encoder.encoder
    _, agents, _ = encoder.neighbor_encoder(inputs["neighbor_agents_past"])
    _, static, _ = encoder.static_encoder(inputs["static_objects"])
    _, lanes, _ = encoder.lane_encoder(inputs["lanes"], inputs["lanes_speed_limit"], inputs["lanes_has_speed_limit"])
    expected = torch.cat([agents, static, lanes], dim=1)
    expected[:, 0] = False

    assert torch.equal(scene_padding_mask(inputs), expected)
    assert expected.any() and not expected.all()


@pytest.mark.parametrize("arch", ["plan", "timestep"])
@pytest.mark.parametrize("use_scene", [True, False])
def test_verifier_shapes_and_equivariance(config, use_scene, arch):
    torch.manual_seed(0)
    model = Verifier(config, depth=2, use_scene=use_scene, arch=arch).eval()
    inputs = _scene_inputs(config, batch=2)
    candidates = torch.randn(2, K, T, 4) * 10

    logits = model(inputs, candidates)
    assert logits.shape == (2, K, len(METRICS))
    assert model.score(inputs, candidates).shape == (2, K)

    # No candidate position embedding: permuting the candidates permutes the outputs
    perm = torch.tensor([2, 0, 3, 1])
    assert torch.allclose(model(inputs, candidates[:, perm]), logits[:, perm], atol=1e-5)


@pytest.mark.parametrize("arch", ["plan", "timestep"])
def test_verifier_uses_precomputed_encoding(config, arch):
    model = Verifier(config, arch=arch).eval()
    inputs = _scene_inputs(config, batch=2)
    candidates = torch.randn(2, K, T, 4)
    encoding = model.encoder(inputs)["encoding"]
    assert torch.allclose(model(inputs, candidates, encoding=encoding), model(inputs, candidates), atol=1e-5)


def test_timestep_step_features(config):
    from diffusion_planner.scorer.model import HORIZON, STEP_FEATURES
    model = Verifier(config, arch="timestep")
    plans = torch.zeros(1, 1, T, 4)
    plans[..., 0] = 0.5 * torch.arange(1, T + 1)          # 5 m/s straight ahead, poses every 0.1 s
    plans[..., 2] = 1.0
    f = model.step_features(plans)
    assert f.shape == (1, 1, HORIZON + 1, STEP_FEATURES)
    assert torch.allclose(f[0, 0, :, 4], torch.full((HORIZON + 1,), 0.5))  # vx / 10 at every state
    assert torch.allclose(f[0, 0, :, 6:8], torch.zeros(HORIZON + 1, 2))   # no acceleration
    # Only the first HORIZON poses are read
    plans[:, :, HORIZON:] = 1e3
    assert torch.allclose(model.step_features(plans), f)


def test_timestep_any_step_and_progress(config):
    """One failing step fails an any-step metric; progress is gated and normalized across candidates."""
    from diffusion_planner.scorer.model import ANY_STEP, HORIZON, PROGRESS, PROGRESS_SCALE
    torch.manual_seed(0)
    model = Verifier(config, arch="timestep").eval()
    inputs = _scene_inputs(config, batch=1)
    candidates = torch.randn(1, 3, T, 4)
    with torch.no_grad():
        model.step_head.weight.zero_()
        model.step_head.bias.fill_(-20.0)                  # every step passes, speed-limit compliance ~0
        logits = model(inputs, candidates)
        assert (torch.sigmoid(logits[0, :, ANY_STEP]) > 0.99).all()

        # Step outputs that depend on the token: make step 7 of every candidate fail on drivable area (index 1)
        model.time_emb.zero_()
        model.time_emb[7, 0] = 1.0
        model.layers = torch.nn.ModuleList()               # tokens reach the head unchanged (after the norm)
        model.step_proj = torch.nn.Identity()
        model.step_features = lambda c: torch.zeros(*c.shape[:2], HORIZON + 1, model.time_emb.shape[1])
        model.norm = torch.nn.Identity()
        model.step_head.weight[1, 0] = 40.0
        logits = model(inputs, candidates)
        assert (torch.sigmoid(logits[0, :, 1]) < 0.01).all()   # drivable fails
        assert (torch.sigmoid(logits[0, :, 0]) > 0.99).all()   # collision still passes

        # Progress: gated by the (now failing) drivable metric, so every candidate is unsafe -> 0 progress
        model.progress_head.bias.fill_(1.0)
        _, aux = model(inputs, candidates, return_aux=True)
        assert torch.allclose(aux["progress_raw"], torch.full((1, 3), PROGRESS_SCALE * float(torch.nn.functional.softplus(torch.tensor(1.0)))))
        assert (torch.sigmoid(model(inputs, candidates)[0, :, PROGRESS]) < 0.01).all()


def test_load_dp_weights_and_freeze(config):
    dp = Diffusion_Planner(config)
    model = Verifier(config)
    model.load_dp_weights(dp.state_dict())
    for name, p in dp.encoder.state_dict().items():
        assert torch.equal(model.encoder.state_dict()[name], p)
    for name, p in dp.decoder.decoder.dit.route_encoder.state_dict().items():
        assert torch.equal(model.route_encoder.state_dict()[name], p)

    # Frozen modules don't train and stay in eval mode
    assert not any(p.requires_grad for p in model.encoder.parameters())
    assert not any(p.requires_grad for p in model.route_encoder.parameters())
    model.train()
    assert not model.encoder.training and not model.route_encoder.training and model.layers[0].training


# ---------------------------------------------------------------------------------------------------------------
# Training


def test_selection_metrics():
    true = torch.tensor([[0.2, 0.9, 0.9], [0.5, 0.5, 0.5], [1.0, 0.0, 0.4]])
    out = selection_metrics(true, torch.tensor([2, 1, 1]))
    assert out["top1"] == pytest.approx(2 / 3)
    assert out["regret"] == pytest.approx(1.0 / 3)

    rand = random_metrics(true)
    assert rand["top1"] == pytest.approx((2 / 3 + 1 + 1 / 3) / 3)
    assert rand["regret"] == pytest.approx((0.7 / 3 + 0 + 1.6 / 3) / 3)

    assert first_max(true).tolist() == [1, 0, 0]


def test_ranking_loss_prefers_correct_order():
    metrics = torch.ones(1, 2, len(METRICS))
    metrics[0, 1, 3] = 0.0 # candidate 1 has no progress
    right = torch.full((1, 2, len(METRICS)), 5.0)
    right[0, 1, 3] = -5.0
    wrong = right.flip(1)
    assert train_module.loss_fn(right, metrics, 1.0, 0.05) < train_module.loss_fn(wrong, metrics, 1.0, 0.05)


def test_train_overfits_small_set(tmp_path, config):
    """Done-when of 2.12, on synthetic labels: the verifier fits a few samples and beats First and Random."""
    _write_dataset(tmp_path / "labels", config, n_scenarios=4, steps=4)
    args = train_module.parse_args([
        "--data", str(tmp_path / "labels"), "--out", str(tmp_path / "run"), "--overfit", "16",
        "--epochs", "40", "--batch-size", "8", "--lr", "1e-3", "--num-workers", "0", "--device", "cpu",
        "--rank-weight", "0.5",
    ])
    best = train_module.run(args, config=config)

    assert best["all/verifier/regret"] < 0.02
    assert best["all/verifier/regret"] < best["all/first/regret"]
    assert best["all/verifier/regret"] < best["all/random/regret"]
    assert (tmp_path / "run" / "best.pt").exists() and (tmp_path / "run" / "log.jsonl").exists()

    # The best checkpoint reloads into the same validation result
    checkpoint = torch.load(tmp_path / "run" / "best.pt", weights_only=False)
    model = Verifier(config, **checkpoint["hparams"])
    model.load_state_dict(checkpoint["state_dict"])
    loader = DataLoader(LabelDataset(find_samples(tmp_path / "labels")[:16]), batch_size=8)
    reloaded = train_module.evaluate(model, loader, "oracle", "cpu")
    assert reloaded["all/verifier/regret"] == pytest.approx(best["all/verifier/regret"], abs=1e-6)


def test_train_timestep_overfits_small_set(tmp_path, config):
    _write_dataset(tmp_path / "labels", config, n_scenarios=4, steps=4)
    args = train_module.parse_args([
        "--data", str(tmp_path / "labels"), "--out", str(tmp_path / "run"), "--overfit", "16", "--arch", "timestep",
        "--epochs", "40", "--batch-size", "8", "--lr", "1e-3", "--num-workers", "0", "--device", "cpu",
        "--rank-weight", "0.5",
    ])
    best = train_module.run(args, config=config)
    assert best["all/verifier/regret"] < best["all/first/regret"]
    assert best["all/verifier/regret"] < best["all/random/regret"]

    checkpoint = torch.load(tmp_path / "run" / "best.pt", weights_only=False)
    assert checkpoint["hparams"]["arch"] == "timestep"
    model = Verifier(config, **checkpoint["hparams"])
    model.load_state_dict(checkpoint["state_dict"])
    loader = DataLoader(LabelDataset(find_samples(tmp_path / "labels")[:16]), batch_size=8)
    reloaded = train_module.evaluate(model, loader, "oracle", "cpu")
    assert reloaded["all/verifier/regret"] == pytest.approx(best["all/verifier/regret"], abs=1e-6)


def test_timestep_rejects_classifier(config):
    with pytest.raises(AssertionError):
        Verifier(config, arch="timestep", num_outputs=1)


def test_label_stats(tmp_path, config):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "label_stats", Path(__file__).resolve().parent.parent / "scripts" / "label_stats.py")
    ls = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ls)

    ones = np.ones((3, len(METRICS)), dtype=np.float32)
    oracle, cv = ones.copy(), ones.copy()
    oracle[0, 0] = 0.0 # candidate 0 collides in the logged future: best are 1 and 2, headroom 1
    cv[1, 4] = 0.0     # constant velocity misses the collision and flags candidate 1's TTC instead
    row = ls.sample_stats({"metrics_oracle": oracle, "metrics_cv": cv, "scenario_type": "t"})

    assert row["informative"] == 1.0 and row["best_not_first"] == 1.0
    assert row["headroom"] == pytest.approx(1.0)
    assert row["random_regret"] == pytest.approx(1 / 3)
    assert row["cv_top1"] == 0.0 and row["cv_agreement"] == 0.0 # PDM's view ties 0 and 2 and keeps 0
    assert row["pass_no_at_fault_collision"] == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------------------------------------------
# Dense analytic drivable-area labels


def _straight_lanes(batch: int, lane_num: int, lane_len: int, centers_y=(0.0, 3.5), width=3.5) -> torch.Tensor:
    """Normalized DP lane inputs: straight lanes along x from -20 m to 80 m at the given y, the rest padding."""
    lanes = torch.zeros(batch, lane_num, lane_len, 12)
    x = torch.linspace(-20.0, 80.0, lane_len)
    for i, y in enumerate(centers_y):
        lanes[:, i, :, 0] = (x - 10.0) / 20.0
        lanes[:, i, :, 1] = y / 20.0
        lanes[:, i, :, 2] = (x[1] - x[0]) / 20.0
        lanes[:, i, :, 5] = width / 2 / 20.0
        lanes[:, i, :, 7] = -width / 2 / 20.0
    return lanes


def test_corner_lane_distance_torch_matches_numpy():
    from diffusion_planner.scorer.analytic_metrics import corner_lane_distance, corner_lane_distance_torch

    g = torch.Generator().manual_seed(0)
    lanes = _straight_lanes(3, 10, 20)
    lanes[1, 1] = 0.0                                          # sample 1: one lane only
    lanes[2] = lanes[2] + 0.01 * torch.randn(lanes[2].shape, generator=g) * (lanes[2] != 0)  # wobbly boundaries
    plans = torch.zeros(3, 5, 80, 4)
    t = torch.linspace(0.1, 8.0, 80)
    plans[..., 0] = t * 8.0
    plans[..., 1] = t[None, None] * torch.tensor([-1.0, -0.3, 0.0, 0.5, 1.5])[None, :, None]  # drift sideways
    plans[..., 2] = 1.0

    dist_t = corner_lane_distance_torch(lanes, plans, horizon=40)
    assert dist_t.shape == (3, 5, 41, 4)
    for b in range(3):
        dist_np = corner_lane_distance(lanes[b].numpy(), plans[b].numpy(), horizon=40)
        np.testing.assert_allclose(dist_t[b].numpy(), dist_np, atol=1e-3)
    # The straight candidate stays inside; the hardest drift leaves the two-lane road
    assert (dist_t[0, 2] == 0).all() and (dist_t[0, 4] > 0).any()


def test_analytic_targets_and_loss():
    from diffusion_planner.scorer.model import ANY_STEP, GEOM_OUTPUTS

    lanes = _straight_lanes(2, 10, 20)
    lanes[1] = 0.0                                             # a scene without lanes is skipped
    plans = torch.zeros(2, 3, 80, 4)
    plans[..., 0] = torch.linspace(0.8, 64.0, 80)
    plans[:, 1, :, 1] = torch.linspace(0.0, 12.0, 80)          # leaves the road
    plans[..., 2] = 1.0
    targets = train_module.analytic_targets({"inputs": {"lanes": lanes}, "candidates": plans})
    assert targets["valid"].tolist() == [True, False]
    assert targets["step_off"][0, 0].sum() == 0 and targets["step_off"][0, 1, -1] == 1
    assert (targets["margin"][0] <= train_module.MARGIN_CLIP).all()

    aux = {"step_logits": torch.zeros(2, 3, 41, len(ANY_STEP) + 1, requires_grad=True),
           "geom": torch.zeros(2, 3, 41, GEOM_OUTPUTS, requires_grad=True)}
    assert train_module.analytic_loss({"step_logits": aux["step_logits"]}, targets) > 0   # step term alone
    loss = train_module.analytic_loss(aux, targets, corner_weight=1.0, margin_weight=0.5)
    loss.backward()
    assert torch.isfinite(loss) and aux["step_logits"].grad[1].abs().sum() == 0   # invalid scene: no gradient


def test_analytic_pretrain_then_finetune(tmp_path, config):
    _write_dataset(tmp_path / "labels", config, n_scenarios=4, steps=4)
    common = ["--data", str(tmp_path / "labels"), "--overfit", "16", "--arch", "timestep", "--batch-size", "8",
              "--lr", "1e-3", "--num-workers", "0", "--device", "cpu"]

    pre = train_module.run(train_module.parse_args(
        common + ["--out", str(tmp_path / "pre"), "--analytic-only", "--epochs", "6"]), config=config)
    log = [json.loads(line) for line in (tmp_path / "pre" / "log.jsonl").read_text().splitlines()]
    assert log[-1]["analytic/loss"] < log[0]["analytic/loss"]
    assert not torch.load(tmp_path / "pre" / "best.pt", weights_only=False)["hparams"]["geom_head"]

    # Plain fine-tuning from the pretrained weights, and the multi-task variant with the optional geom_head terms
    fine = train_module.run(train_module.parse_args(
        common + ["--out", str(tmp_path / "fine"), "--epochs", "2", "--init-from", str(tmp_path / "pre" / "best.pt")]),
        config=config)
    assert "drivable/pair_acc" in fine and "analytic/loss" not in fine
    multi = train_module.run(train_module.parse_args(
        common + ["--out", str(tmp_path / "multi"), "--epochs", "2", "--analytic-weight", "0.3",
                  "--corner-weight", "0.5", "--margin-weight", "0.5"]), config=config)
    assert "analytic/loss" in multi
    assert torch.load(tmp_path / "multi" / "best.pt", weights_only=False)["hparams"]["geom_head"]
