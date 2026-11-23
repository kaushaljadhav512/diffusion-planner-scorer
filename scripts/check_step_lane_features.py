"""
Sanity checks of the per-step lane features (lane_features.py) on real label samples, plus cost at training batch size.

    python scripts/check_step_lane_features.py --data $NUPLAN_HOME/labels/train-labels-2k --batches 20

Prints:
  route match    share of the (valid) route lanes whose centerline matches one of the 70 input lanes
  boundary side  share of lanes whose centerline lies right of the left boundary and left of the right boundary
                 (the sign convention of the segment corner distances assumes both)
  drivable       agreement of "every corner within 0.2 m of the lanes at every step" (from the signed corner
                 distances) with the cv drivable-area label, per candidate
  cost           seconds per batch for the features, and peak GPU memory of a forward + backward of the
                 verifier with each step_lane mode
"""
import argparse
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from diffusion_planner.scorer.dataset import LabelDataset, find_samples, split_by_scenario
from diffusion_planner.scorer.lane_features import _segment_distance, on_route, polylines, step_lane_features
from diffusion_planner.scorer.metrics import METRICS
from diffusion_planner.scorer.model import HORIZON, Verifier
from diffusion_planner.scorer.train import loss_fn, to_device

DRIVABLE = METRICS.index("drivable_area")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--dp-args", default="checkpoints/args.json")
    ap.add_argument("--dp-ckpt", default="checkpoints/model.pth")
    ap.add_argument("--batches", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    _, val_files = split_by_scenario(find_samples(args.data), 0.1, 0)
    loader = DataLoader(LabelDataset(val_files), batch_size=args.batch_size, shuffle=True, num_workers=4,
                        generator=torch.Generator().manual_seed(0))
    route_hit = route_n = side_ok = side_n = agree = n_cand = 0
    feat_time = []
    first = None
    for i, batch in enumerate(loader):
        if i == args.batches:
            break
        batch = to_device(batch, args.device)
        first = first or batch
        inputs = batch["inputs"]

        routed = on_route(inputs["route_lanes"], inputs["lanes"])            # route lane matched by an input lane
        route_valid = polylines(inputs["route_lanes"])[3]
        route_hit += (routed & route_valid).sum().item()
        route_n += route_valid.sum().item()

        center, left, right, valid = polylines(inputs["lanes"])
        mid = center[:, :, 10]
        for line, want in ((left, -1), (right, 1)):
            d, cross = _segment_distance(mid[:, :, None], line[:, :, :-1], line[:, :, 1:])
            nearest = cross.gather(2, d.argmin(-1, keepdim=True))[..., 0]
            side_ok += ((torch.sign(nearest) == want) & valid).sum().item()
        side_n += 2 * valid.sum().item()

        torch.cuda.synchronize()
        start = time.time()
        scalars = step_lane_features(inputs, batch["candidates"], HORIZON, segments=True)["scalars"]
        torch.cuda.synchronize()
        feat_time.append(time.time() - start)
        on_lanes = (scalars[..., 0:4] * 2.0 <= 0.2).flatten(2).all(-1).float()   # [B, K]
        label = batch["metrics_cv"][..., DRIVABLE]
        agree += (on_lanes == label).sum().item()
        n_cand += label.numel()

    print(f"route match    {route_hit / route_n:.4f} of {route_n} route lanes")
    print(f"boundary side  {side_ok / side_n:.4f} of {side_n} boundaries")
    print(f"drivable       analytic (0.2 m) agrees with the cv label on {agree / n_cand:.4f} of {n_cand} candidates")
    print(f"features       {sum(feat_time[1:]) / max(len(feat_time) - 1, 1):.3f} s per batch of {args.batch_size} "
          f"(scalars + segments)")

    for mode in ("none", "scalars", "segments"):
        model = Verifier.from_dp_checkpoint(args.dp_args, args.dp_ckpt, device=args.device, arch="timestep",
                                            step_lane=mode).train()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(3):
            logits = model(first["inputs"], first["candidates"])
            loss_fn(logits, first["metrics_cv"], 3.0, 0.02).backward()
        torch.cuda.synchronize()
        print(f"{mode:9s} forward+backward {(time.time() - start) / 3:.3f} s per batch, "
              f"peak memory {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GiB")
        del model


if __name__ == "__main__":
    main()
