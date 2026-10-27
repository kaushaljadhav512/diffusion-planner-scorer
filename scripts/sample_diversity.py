"""Measure how diverse Diffusion Planner's K samples are (task 1.4).

Caches ~100 nuPlan mini scenes with DP's DataProcessor (first run only), then draws K ego plans per
scene with diffusion_planner.scorer.sampling.sample_candidates at several initial-noise scales and
reports, averaged over scenes (ego frame, metres):

    endpoint_spread   RMS distance of the K endpoints (t = 8 s) from their mean
    lon_std, lat_std  std of the endpoints along / across the ego heading
    pairwise_ade      mean over candidate pairs of their average point-wise distance
    fde_first         endpoint error of candidate 0 vs the logged ego future (= vanilla DP)
    min_fde           best endpoint error over the K candidates (an open-loop headroom hint)

    source setup/config.env
    python scripts/sample_diversity.py                          # K=16, noise 0.5 / 1.0 / 1.5
    python scripts/sample_diversity.py --noise_scales 0.5 2.0 --K 32
"""
import argparse
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.scorer.sampling import sample_candidates
from diffusion_planner.utils.config import Config

ROOT = Path(__file__).resolve().parents[1]
NUPLAN_DATA_ROOT = os.environ.get("NUPLAN_DATA_ROOT", os.path.expanduser("~/nuplan/dataset"))
NUPLAN_HOME = os.environ.get("NUPLAN_HOME", os.path.expanduser("~/nuplan"))

# Same keys and order as DiffusionPlannerData / train_epoch
INPUT_KEYS = [
    "ego_current_state", "neighbor_agents_past", "static_objects",
    "lanes", "lanes_speed_limit", "lanes_has_speed_limit",
    "route_lanes", "route_lanes_speed_limit", "route_lanes_has_speed_limit",
]


def load_model(args_file, ckpt_path, device):
    config = Config(args_file, guidance_fn=None)
    config.device = device
    model = Diffusion_Planner(config)
    # Same loading as DiffusionPlanner.initialize (EMA weights, strip the DDP prefix)
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)["ema_state_dict"]
    model.load_state_dict({k[len("module."):]: v for k, v in state_dict.items() if k.startswith("module.")})
    return config, model.eval().to(device)


def build_cache(config, data_path, map_path, cache_dir, num_scenes, seed):
    """Process num_scenes random mini scenarios into DP's npz format (one file per scenario)."""
    from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
    from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
    from nuplan.planning.utils.multithreading.worker_parallel import SingleMachineParallelExecutor
    from diffusion_planner.data_process.data_processor import DataProcessor

    os.makedirs(cache_dir, exist_ok=True)
    random.seed(seed)  # the filter's shuffle uses python's random

    builder = NuPlanScenarioBuilder(data_path, map_path, None, None, os.environ.get("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0"))
    # Same fields and order as get_filter_parameters in data_process.py, without the train-log filter
    scenario_filter = ScenarioFilter(
        None, None, None, None,      # scenario_types, scenario_tokens, log_names, map_names
        None, num_scenes,            # num_scenarios_per_type, limit_total_scenarios
        None, None,                  # timestamp_threshold_s, ego_displacement_minimum_m
        True, False, True,           # expand_scenarios, remove_invalid_goals, shuffle
        None, None, None,            # ego start / stop speed thresholds, speed_noise_tolerance
    )
    worker = SingleMachineParallelExecutor(use_process_pool=True)
    scenarios = builder.get_scenarios(scenario_filter, worker)
    print(f"[cache] processing {len(scenarios)} mini scenarios into {cache_dir}")

    config.save_path = cache_dir
    DataProcessor(config).work(scenarios)


def load_scenes(cache_dir, config):
    files = sorted(Path(cache_dir).glob("*.npz"))
    assert files, f"no npz files in {cache_dir}"
    scenes, futures, names = [], [], []
    for f in files:
        data = np.load(f)
        scene = {k: data[k] for k in INPUT_KEYS}
        scene["neighbor_agents_past"] = scene["neighbor_agents_past"][:config.agent_num]
        scenes.append(scene)
        futures.append(data["ego_agent_future"][:, :2]) # [T, 2], ego frame
        names.append(f.stem)
    return scenes, np.stack(futures), names


def to_inputs(scenes, device):
    inputs = {}
    for k in INPUT_KEYS:
        v = np.stack([s[k] for s in scenes])
        dtype = torch.bool if v.dtype == np.bool_ else torch.float32
        inputs[k] = torch.as_tensor(v, dtype=dtype, device=device)
    return inputs


def diversity_metrics(ego, gt):
    """
    ego: [K, T, 2] candidate ego positions, gt: [T, 2] logged ego future.
    """
    K = ego.shape[0]
    end = ego[:, -1]                                            # [K, 2]
    dev = end - end.mean(axis=0)
    pair_dist = np.linalg.norm(ego[:, None] - ego[None], axis=-1).mean(axis=-1) # [K, K]
    fde = np.linalg.norm(end - gt[-1], axis=-1)                 # [K]
    return {
        "endpoint_spread": np.sqrt((dev ** 2).sum(axis=-1).mean()),
        "lon_std": end[:, 0].std(),
        "lat_std": end[:, 1].std(),
        "pairwise_ade": pair_dist[np.triu_indices(K, k=1)].mean(),
        "fde_first": fde[0],
        "min_fde": fde.min(),
    }


def markdown_table(df):
    # DataFrame.to_markdown needs the optional tabulate package
    header = [df.index.name] + list(df.columns)
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(v) for v in [idx] + list(row)) + " |" for idx, row in df.iterrows()]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--args_file", default=str(ROOT / "checkpoints" / "args.json"))
    parser.add_argument("--ckpt", default=str(ROOT / "checkpoints" / "model.pth"))
    parser.add_argument("--data_path", default=os.path.join(NUPLAN_DATA_ROOT, "nuplan-v1.1", "mini"))
    parser.add_argument("--map_path", default=os.environ.get("NUPLAN_MAPS_ROOT", os.path.join(NUPLAN_DATA_ROOT, "maps")))
    parser.add_argument("--cache_dir", default=os.path.join(NUPLAN_HOME, "cache", "mini_diversity"))
    parser.add_argument("--num_scenes", type=int, default=100)
    parser.add_argument("--K", type=int, default=16)
    parser.add_argument("--noise_scales", type=float, nargs="+", default=[0.5, 1.0, 1.5])
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8, help="scenes per sampler call (B * K rows)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=str(ROOT / "results" / "sample_diversity"),
                        help="writes <out>.csv (per scene) and <out>.md (summary table)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config, model = load_model(args.args_file, args.ckpt, args.device)

    if not any(Path(args.cache_dir).glob("*.npz")):
        build_cache(config, args.data_path, args.map_path, args.cache_dir, args.num_scenes, args.seed)
    scenes, futures, names = load_scenes(args.cache_dir, config)
    scenes, futures, names = scenes[:args.num_scenes], futures[:args.num_scenes], names[:args.num_scenes]
    print(f"[diversity] {len(scenes)} scenes, K={args.K}, steps={args.steps}, device={args.device}")

    rows = []
    for noise_scale in args.noise_scales:
        torch.manual_seed(args.seed)
        elapsed = 0.0
        for i in tqdm(range(0, len(scenes), args.batch_size), desc=f"noise {noise_scale}"):
            inputs = config.observation_normalizer(to_inputs(scenes[i:i + args.batch_size], args.device))

            if args.device == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            candidates = sample_candidates(model, inputs, K=args.K, noise_scale=noise_scale, steps=args.steps)
            if args.device == "cuda":
                torch.cuda.synchronize()
            elapsed += time.perf_counter() - start

            ego = candidates[:, :, 0, :, :2].cpu().numpy() # [B, K, T, 2]
            for b in range(ego.shape[0]):
                rows.append({"noise_scale": noise_scale, "scene": names[i + b],
                             **diversity_metrics(ego[b], futures[i + b])})
        print(f"[diversity] noise {noise_scale}: {1000 * elapsed / len(scenes):.1f} ms per scene (K={args.K})")

    per_scene = pd.DataFrame(rows)
    summary = per_scene.drop(columns="scene").groupby("noise_scale").mean().round(2)
    summary.insert(0, "scenes", per_scene.groupby("noise_scale").size())

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    per_scene.to_csv(f"{args.out}.csv", index=False)
    with open(f"{args.out}.md", "w") as f:
        f.write(f"Ego-plan diversity, K={args.K}, steps={args.steps}, nuPlan mini, metres, mean over scenes\n\n")
        f.write(markdown_table(summary) + "\n")
    print(markdown_table(summary))
    print(f"[diversity] wrote {args.out}.csv and {args.out}.md")


if __name__ == "__main__":
    main()
