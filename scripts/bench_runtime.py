"""Single-process runtime of the best-of-K pieces on the GPU, outside simulation (task 2.13).

In-simulation sample_ms varies 39-71 ms with GPU contention between ray workers, so the final runtime table comes
from here. One scene per call (B=1, as in simulation), median over the cached scenes of scripts/sample_diversity.py
(run that first), for each K:

    encoder_ms        DP's scene encoder alone (part of sampling)
    sample_ms         sample_candidates: encoder + K-sample DPM-Solver
    verifier_ms       verifier given the sampler's encoding (what a LearnedSelector would add per step)
    verifier_full_ms  verifier including its own encoder pass

PDM selection needs a map and the simulator: its per-K cost comes from scripts/run_runtime_sweep.sh instead.

    source setup/config.env
    python scripts/bench_runtime.py                                # K = 4 8 16 32 64, random verifier weights
    python scripts/bench_runtime.py --verifier runs/verifier/best.pt
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sample_diversity import NUPLAN_HOME, ROOT, load_model, load_scenes, to_inputs  # noqa: E402

from diffusion_planner.scorer.model import Verifier
from diffusion_planner.scorer.sampling import sample_candidates


def timed(fn, device):
    """(result, ms) of fn(), waiting for the GPU before and after."""
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    result = fn()
    if device == "cuda":
        torch.cuda.synchronize()
    return result, 1000 * (time.perf_counter() - start)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--args_file", default=str(ROOT / "checkpoints" / "args.json"))
    parser.add_argument("--ckpt", default=str(ROOT / "checkpoints" / "model.pth"))
    parser.add_argument("--verifier", default=None, help="trained verifier checkpoint (default: random weights)")
    parser.add_argument("--cache_dir", default=os.path.join(NUPLAN_HOME, "cache", "test_diversity"))
    parser.add_argument("--num_scenes", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--K", type=int, nargs="+", default=[4, 8, 16, 32, 64])
    parser.add_argument("--noise_scale", type=float, default=1.5)
    parser.add_argument("--out", default=str(ROOT / "results" / "runtime_bench.csv"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config, model = load_model(args.args_file, args.ckpt, args.device)
    if args.verifier:
        verifier = Verifier.load(args.verifier, args.args_file, device=args.device)
    else:
        verifier = Verifier.from_dp_checkpoint(args.args_file, args.ckpt, device=args.device).eval()
    scenes, _, _ = load_scenes(args.cache_dir, config)
    scenes = scenes[:args.num_scenes]
    gpu = torch.cuda.get_device_name() if args.device == "cuda" else "cpu"
    print(f"[bench] {len(scenes)} scenes, K={args.K}, device={args.device} ({gpu})")

    rows = []
    with torch.no_grad():
        for K in args.K:
            times = {"encoder_ms": [], "sample_ms": [], "verifier_ms": [], "verifier_full_ms": []}
            for i, scene in enumerate([scenes[0]] * args.warmup + scenes):
                inputs = config.observation_normalizer(to_inputs([scene], args.device))
                encoding, t_enc = timed(lambda: model.encoder(inputs)["encoding"], args.device)
                candidates, t_sample = timed(
                    lambda: sample_candidates(model, inputs, K, args.noise_scale), args.device)
                plans = candidates[:, :, 0]                                  # [1, K, T, 4]
                _, t_verifier = timed(lambda: verifier.score(inputs, plans, encoding=encoding), args.device)
                _, t_full = timed(lambda: verifier.score(inputs, plans), args.device)
                if i >= args.warmup:
                    for name, t in zip(times, (t_enc, t_sample, t_verifier, t_full)):
                        times[name].append(t)
            row = {"K": K, **{name: float(np.median(v)) for name, v in times.items()}, "device": gpu}
            rows.append(row)
            print(f"[bench] K={K:3d}: " + "  ".join(f"{k} {v:.2f}" for k, v in row.items() if k.endswith("_ms")))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"[bench] wrote {args.out}")


if __name__ == "__main__":
    main()
