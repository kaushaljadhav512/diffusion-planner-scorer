"""
Train the verifier on recorded labels (task 2.12).

    python -m diffusion_planner.scorer.train --data $NUPLAN_HOME/labels/train-labels-2k --out runs/verifier
    python -m diffusion_planner.scorer.train --data ... --overfit 100      # sanity check: regret should go to ~0
    python -m diffusion_planner.scorer.train --data ... --no-scene         # trajectory-only ablation (3.3)
    python -m diffusion_planner.scorer.train --data ... --labels cv        # constant-velocity labels (3.2)
    python -m diffusion_planner.scorer.train --data ... --train-frac 0.25 --split-seed 0   # learning curve point

Loss: BCE per metric against the PDM labels (soft targets), optionally plus a listwise ranking loss on the combined
score. Every epoch reports, on the held-out scenarios, against the oracle's combined score (the selection target):

    top1    fraction of samples where the pick is among the best candidates (ties count)
    regret  mean of (best combined score - combined score of the pick)

for the verifier, First (candidate 0), Random (expected value), and PDM (argmax of the constant-velocity labels,
i.e. what PDMSpeedLimitSelector picks). "informative" restricts to samples whose candidates are not all tied.
"""
import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from diffusion_planner.scorer.dataset import LabelDataset, find_samples, split_by_scenario, subsample_scenarios
from diffusion_planner.scorer.metrics import combined_score
from diffusion_planner.scorer.model import Verifier

ROOT = Path(__file__).resolve().parents[2]
TIE_TOLERANCE = 1e-6


def selection_metrics(true_scores: torch.Tensor, picks: torch.Tensor) -> Dict[str, float]:
    """
    Args:
        true_scores: [N, K] target combined scores
        picks: [N] chosen candidate per sample

    Returns:
        top1, regret (means over samples)
    """
    best = true_scores.max(dim=1).values
    picked = true_scores.gather(1, picks[:, None].long())[:, 0]
    return {"top1": (picked >= best - TIE_TOLERANCE).float().mean().item(), "regret": (best - picked).mean().item()}


def random_metrics(true_scores: torch.Tensor) -> Dict[str, float]:
    """Expected top1 / regret of a uniformly random pick."""
    best = true_scores.max(dim=1, keepdim=True).values
    return {"top1": (true_scores >= best - TIE_TOLERANCE).float().mean().item(),
            "regret": (best - true_scores).mean().item()}


def first_max(scores: torch.Tensor) -> torch.Tensor:
    """Argmax with ties to the lowest index (candidate 0 when all tie), as the PDM selectors do."""
    best = scores.max(dim=1, keepdim=True).values
    return (scores >= best - TIE_TOLERANCE).float().argmax(dim=1)


def loss_fn(logits: torch.Tensor, metrics: torch.Tensor, rank_weight: float, rank_temperature: float) -> torch.Tensor:
    """
    Args:
        logits: [B, K, M] verifier output
        metrics: [B, K, M] target metric values in [0, 1]
    """
    loss = F.binary_cross_entropy_with_logits(logits, metrics)
    if rank_weight > 0:
        # Listwise: the softmax over candidates of the predicted combined score should match the target's
        predicted = combined_score(torch.sigmoid(logits)) / rank_temperature
        target = F.softmax(combined_score(metrics) / rank_temperature, dim=1)
        loss = loss + rank_weight * -(target * F.log_softmax(predicted, dim=1)).sum(dim=1).mean()
    return loss


def to_device(batch, device):
    if isinstance(batch, dict):
        return {k: to_device(v, device) for k, v in batch.items()}
    return batch.to(device) if isinstance(batch, torch.Tensor) else batch


@torch.no_grad()
def evaluate(model: Verifier, loader: DataLoader, labels: str, device: str) -> Dict[str, float]:
    model.eval()
    true, predicted, pdm = [], [], []
    for batch in loader:
        batch = to_device(batch, device)
        predicted.append(model.score(batch["inputs"], batch["candidates"]).cpu())
        true.append(combined_score(batch[f"metrics_{labels}"]).cpu())
        pdm.append(combined_score(batch["metrics_cv"]).cpu())
    true, predicted, pdm = torch.cat(true), torch.cat(predicted), torch.cat(pdm)

    informative = (true.max(dim=1).values - true.min(dim=1).values) > TIE_TOLERANCE
    out = {"n": len(true), "informative": informative.float().mean().item()}
    picks = {"verifier": first_max(predicted), "first": torch.zeros(len(true), dtype=torch.long), "pdm": first_max(pdm)}
    for subset, rows in [("all", slice(None)), ("informative", informative)]:
        if subset == "informative" and not informative.any():
            continue
        for name, pick in picks.items():
            for k, v in selection_metrics(true[rows], pick[rows]).items():
                out[f"{subset}/{name}/{k}"] = v
        for k, v in random_metrics(true[rows]).items():
            out[f"{subset}/random/{k}"] = v
    return out


def run(args, config=None) -> Dict[str, float]:
    """Trains and returns the best validation metrics. config: a DP config to build the verifier from (tests)."""
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    files = find_samples(args.data)
    assert files, f"no label samples under {args.data}"
    if args.overfit:
        # Sanity check: train and evaluate on the same few samples
        train_files = val_files = files[:args.overfit]
    else:
        split_seed = args.seed if args.split_seed is None else args.split_seed
        train_files, val_files = split_by_scenario(files, args.val_frac, split_seed)
        if args.train_frac < 1.0:
            # Learning curve: fewer training scenarios, same validation scenarios
            train_files = subsample_scenarios(train_files, args.train_frac, args.seed)
    print(f"[train] {len(train_files)} train / {len(val_files)} val samples, "
          f"{len({f.parent for f in train_files})} / {len({f.parent for f in val_files})} scenarios")

    hparams = dict(depth=args.depth, num_heads=args.num_heads, dropout=0.0 if args.overfit else args.dropout,
                   use_scene=not args.no_scene)
    if config is None:
        model = Verifier.from_dp_checkpoint(args.dp_args, args.dp_ckpt, device=args.device, **hparams)
    else:
        model = Verifier(config, **hparams).to(args.device)

    loader_kwargs = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                         pin_memory=args.device.startswith("cuda"))
    train_loader = DataLoader(LabelDataset(train_files), shuffle=True, drop_last=False, **loader_kwargs)
    val_loader = DataLoader(LabelDataset(val_files), shuffle=False, **loader_kwargs)

    params = [p for p in model.parameters() if p.requires_grad]
    print(f"[train] trainable parameters: {sum(p.numel() for p in params) / 1e6:.2f} M")
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=args.epochs * len(train_loader), pct_start=0.1)

    best, log = None, open(out / "log.jsonl", "a")
    for epoch in range(args.epochs):
        model.train()
        start, losses = time.time(), []
        for batch in train_loader:
            batch = to_device(batch, args.device)
            logits = model(batch["inputs"], batch["candidates"])
            loss = loss_fn(logits, batch[f"metrics_{args.labels}"], args.rank_weight, args.rank_temperature)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())

        metrics = {"epoch": epoch, "loss": float(np.mean(losses)), "seconds": time.time() - start,
                   **evaluate(model, val_loader, args.labels, args.device)}
        log.write(json.dumps(metrics) + "\n")
        log.flush()
        print(f"[epoch {epoch:3d}] loss {metrics['loss']:.4f} | regret verifier {metrics['all/verifier/regret']:.4f} "
              f"pdm {metrics['all/pdm/regret']:.4f} first {metrics['all/first/regret']:.4f} "
              f"random {metrics['all/random/regret']:.4f} | top1 verifier {metrics['all/verifier/top1']:.3f} "
              f"pdm {metrics['all/pdm/top1']:.3f} first {metrics['all/first/top1']:.3f}")

        if best is None or metrics["all/verifier/regret"] < best["all/verifier/regret"]:
            best = metrics
            model.save(out / "best.pt", epoch=epoch, labels=args.labels, val_metrics=metrics)
    model.save(out / "last.pt", epoch=args.epochs - 1, labels=args.labels)
    log.close()
    (out / "best_metrics.json").write_text(json.dumps(best, indent=2))
    print(f"[train] best epoch {best['epoch']}: regret {best['all/verifier/regret']:.4f}, wrote {out}")
    return best


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="label dir (one subdirectory per scenario)")
    ap.add_argument("--out", default="runs/verifier")
    ap.add_argument("--dp-args", default=str(ROOT / "checkpoints" / "args.json"))
    ap.add_argument("--dp-ckpt", default=str(ROOT / "checkpoints" / "model.pth"))
    ap.add_argument("--labels", choices=["oracle", "cv"], default="oracle", help="training target")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--num-heads", type=int, default=6)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--no-scene", action="store_true", help="trajectory-only verifier (ablation 3.3)")
    ap.add_argument("--rank-weight", type=float, default=3.0, help="weight of the listwise ranking loss")
    ap.add_argument("--rank-temperature", type=float, default=0.02)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--split-seed", type=int, default=None,
                    help="seed of the train/val split (default: --seed); fix it to compare runs on one val set")
    ap.add_argument("--train-frac", type=float, default=1.0,
                    help="learning curve: train on this fraction of the training scenarios (subset chosen by --seed)")
    ap.add_argument("--overfit", type=int, default=0, help="train and evaluate on the first N samples")
    ap.add_argument("--num-workers", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
