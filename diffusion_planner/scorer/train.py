"""
Train the verifier on recorded labels (task 2.12).

    python -m diffusion_planner.scorer.train --data $NUPLAN_HOME/labels/train-labels-2k --out runs/verifier
    python -m diffusion_planner.scorer.train --data ... --overfit 100      # sanity check: regret should go to ~0
    python -m diffusion_planner.scorer.train --data ... --no-scene         # trajectory-only ablation (3.3)
    python -m diffusion_planner.scorer.train --data ... --labels cv        # constant-velocity labels (3.2)
    python -m diffusion_planner.scorer.train --data ... --objective classify   # one score per candidate, no metrics
    python -m diffusion_planner.scorer.train --data ... --train-frac 0.25 --split-seed 0   # learning curve point
    python -m diffusion_planner.scorer.train --data ... --arch timestep    # one query token per scored state

Loss: BCE per metric against the PDM labels (soft targets), optionally plus a listwise ranking loss on the combined
score. The timestep arch adds a smooth-L1 loss on its raw progress (in units of PROGRESS_SCALE metres) against the
labels' raw progress, weighted by --progress-weight. Every epoch reports, on the held-out scenarios, against the oracle's combined score (the selection target):

    top1    fraction of samples where the pick is among the best candidates (ties count)
    regret  mean of (best combined score - combined score of the pick)

Dense analytic drivable-area labels (timestep arch, --analytic-weight / --analytic-only): for every scored state
and every corner of ego's box, the distance outside DP's input lane polygons (analytic_metrics), computed on the
GPU per batch. They train the per-step drivable-area failure logit (the one the smooth max reads); optionally also a
per-corner off-lane logit and the step's overshoot in metres (geom_head, --corner-weight / --margin-weight). --analytic-only trains on them alone (pretraining);
--init-from then fine-tunes on the PDM labels.

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

from diffusion_planner.scorer.analytic_metrics import corner_lane_distance_torch
from diffusion_planner.scorer.dataset import LabelDataset, find_samples, split_by_scenario, subsample_scenarios
from diffusion_planner.scorer.metrics import METRICS, combined_score
from diffusion_planner.scorer.model import ANY_STEP, ARCHS, HORIZON, PROGRESS_SCALE, Verifier

ROOT = Path(__file__).resolve().parents[2]
TIE_TOLERANCE = 1e-6
DRIVABLE = METRICS.index("drivable_area")
DRIVABLE_STEP = ANY_STEP.index(DRIVABLE)     # column of the drivable-area failure logit in the per-step logits
ANALYTIC_TOLERANCE = 0.2  # [m] a corner this close to a lane polygon counts as inside (closes the slivers between
                          # rebuilt neighbouring lane polygons, see analytic_metrics.drivable_area_compliance)
MARGIN_CLIP = 2.0         # [m] overshoot target clipped here


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


def classify_loss(scores: torch.Tensor, metrics: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Classifier objective (--objective classify): no metric heads, one score per candidate.

    Args:
        scores: [B, K] predicted score per candidate
        metrics: [B, K, M] target metric values; only their combined score is used

    Returns:
        cross-entropy between softmax(scores) and softmax(target combined score / temperature) over the K candidates,
        averaged over samples whose candidates are not all tied (a tied sample says nothing about which is best)
    """
    target_scores = combined_score(metrics)
    informative = (target_scores.max(dim=1).values - target_scores.min(dim=1).values) > TIE_TOLERANCE
    if not informative.any():
        return scores.sum() * 0.0
    target = F.softmax(target_scores[informative] / temperature, dim=1)
    return -(target * F.log_softmax(scores[informative], dim=1)).sum(dim=1).mean()


def analytic_targets(batch) -> Dict[str, torch.Tensor]:
    """Dense analytic drivable-area targets of a batch: corner_off [B, K, T, 4], step_off [B, K, T], margin [B, K, T],
    valid [B] (scenes with at least one lane)."""
    distance = corner_lane_distance_torch(batch["inputs"]["lanes"], batch["candidates"], horizon=HORIZON)
    valid = torch.isfinite(distance).flatten(1).all(dim=1)
    corner_off = (distance > ANALYTIC_TOLERANCE).float()
    margin = distance.clamp(0.0, MARGIN_CLIP).amax(dim=-1)
    return {"corner_off": corner_off, "step_off": corner_off.amax(dim=-1), "margin": margin, "valid": valid}


def analytic_loss(aux: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], corner_weight: float = 0.0,
                  margin_weight: float = 0.0) -> torch.Tensor:
    """
    BCE of the per-step drivable-area failure logit (the one the smooth max reads) against "some corner off the lanes
    at this step", over scenes with lanes. Optional geom_head terms: per-corner off-lane BCE (corner_weight) and smooth
    L1 on the step's overshoot outside the lanes [m] (margin_weight; 0 for every on-lane step).
    """
    v = targets["valid"]
    if not v.any():
        return aux["step_logits"].sum() * 0.0
    loss = F.binary_cross_entropy_with_logits(aux["step_logits"][v][..., DRIVABLE_STEP], targets["step_off"][v])
    if corner_weight > 0:
        loss = loss + corner_weight * F.binary_cross_entropy_with_logits(aux["geom"][v][..., :4],
                                                                         targets["corner_off"][v])
    if margin_weight > 0:
        loss = loss + margin_weight * F.smooth_l1_loss(aux["geom"][v][..., 4], targets["margin"][v])
    return loss


def to_device(batch, device):
    if isinstance(batch, dict):
        return {k: to_device(v, device) for k, v in batch.items()}
    return batch.to(device) if isinstance(batch, torch.Tensor) else batch


def pairwise_accuracy(predicted: torch.Tensor, labels: torch.Tensor) -> float:
    """Among pairs (i, j) of one sample [N, K] with labels[i] > labels[j], the share with predicted[i] > predicted[j]
    (ties count half), as in scripts/diagnose_verifier.py; NaN if there is no such pair."""
    label_gt = (labels[:, :, None] - labels[:, None, :]) > TIE_TOLERANCE
    diff = predicted[:, :, None] - predicted[:, None, :]
    correct = (diff > 0).float() + 0.5 * (diff == 0).float()
    n = label_gt.sum().item()
    return float((correct * label_gt).sum().item() / n) if n else float("nan")


@torch.no_grad()
def evaluate(model: Verifier, loader: DataLoader, labels: str, device: str, analytic: bool = False,
             corner_weight: float = 0.0, margin_weight: float = 0.0) -> Dict[str, float]:
    model.eval()
    true, predicted, pdm, drivable_pred, drivable_label = [], [], [], [], []
    step_hits = step_n = cand_hits = cand_n = 0
    analytic_losses = []
    for batch in loader:
        batch = to_device(batch, device)
        out, aux = model(batch["inputs"], batch["candidates"], return_aux=True)
        predicted.append((out[..., 0] if out.shape[-1] == 1 else combined_score(torch.sigmoid(out))).cpu())
        true.append(combined_score(batch[f"metrics_{labels}"]).cpu())
        pdm.append(combined_score(batch["metrics_cv"]).cpu())
        if out.shape[-1] > 1:
            drivable_pred.append(torch.sigmoid(out[..., DRIVABLE]).cpu())
            drivable_label.append(batch[f"metrics_{labels}"][..., DRIVABLE].cpu())
        if analytic:
            targets = analytic_targets(batch)
            v = targets["valid"]
            analytic_losses.append(analytic_loss(aux, targets, corner_weight, margin_weight).item())
            step_logit = aux["step_logits"][v][..., DRIVABLE_STEP]
            step_hits += ((step_logit > 0).float() == targets["step_off"][v]).sum().item()
            step_n += step_logit.numel()
            cand_off = targets["step_off"][v].amax(dim=-1)
            cand_hits += ((out[v][..., DRIVABLE] < 0).float() == cand_off).sum().item()
            cand_n += cand_off.numel()
    true, predicted, pdm = torch.cat(true), torch.cat(predicted), torch.cat(pdm)
    extra = {}
    if drivable_pred:
        extra["drivable/pair_acc"] = pairwise_accuracy(torch.cat(drivable_pred), torch.cat(drivable_label))
    if analytic:
        extra.update({"analytic/loss": float(np.mean(analytic_losses)), "analytic/step_acc": step_hits / step_n,
                      "analytic/candidate_acc": cand_hits / cand_n})

    informative = (true.max(dim=1).values - true.min(dim=1).values) > TIE_TOLERANCE
    out = {"n": len(true), "informative": informative.float().mean().item(), **extra}
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

    assert args.arch == "plan" or args.objective == "metrics", "--arch timestep needs --objective metrics"
    analytic = args.analytic_only or args.analytic_weight > 0
    init = torch.load(args.init_from, map_location="cpu", weights_only=False) if args.init_from else None
    assert not analytic or args.arch == "timestep", "dense analytic labels need --arch timestep"
    assert analytic or (args.corner_weight == 0 and args.margin_weight == 0), \
        "--corner-weight / --margin-weight need --analytic-weight or --analytic-only"
    hparams = dict(depth=args.depth, num_heads=args.num_heads, dropout=0.0 if args.overfit else args.dropout,
                   use_scene=not args.no_scene, num_outputs=1 if args.objective == "classify" else len(METRICS),
                   arch=args.arch, geom_head=(analytic and (args.corner_weight > 0 or args.margin_weight > 0))
                   or bool(init and init["hparams"].get("geom_head")), lane_segments=args.lane_segments)
    if config is None:
        model = Verifier.from_dp_checkpoint(args.dp_args, args.dp_ckpt, device=args.device, **hparams)
    else:
        model = Verifier(config, **hparams).to(args.device)
    if init is not None:
        # Fine-tuning: every weight from a previous run (e.g. --analytic-only pretraining). Its geom_head is kept
        # (unused without --analytic-weight) so the state dicts match
        model.load_state_dict(init["state_dict"])
        print(f"[train] initialized from {args.init_from} (epoch {init.get('epoch')})")

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
            logits, aux = model(batch["inputs"], batch["candidates"], return_aux=True)
            if args.analytic_only:
                loss = analytic_loss(aux, analytic_targets(batch), args.corner_weight, args.margin_weight)
            elif args.objective == "classify":
                loss = classify_loss(logits[..., 0], batch[f"metrics_{args.labels}"], args.rank_temperature)
            else:
                loss = loss_fn(logits, batch[f"metrics_{args.labels}"], args.rank_weight, args.rank_temperature)
            if "progress_raw" in aux and args.progress_weight > 0 and not args.analytic_only:
                loss = loss + args.progress_weight * F.smooth_l1_loss(
                    aux["progress_raw"] / PROGRESS_SCALE, batch[f"progress_{args.labels}"] / PROGRESS_SCALE)
            if args.analytic_weight > 0 and not args.analytic_only:
                loss = loss + args.analytic_weight * analytic_loss(aux, analytic_targets(batch), args.corner_weight,
                                                                   args.margin_weight)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())

        metrics = {"epoch": epoch, "loss": float(np.mean(losses)), "seconds": time.time() - start,
                   **evaluate(model, val_loader, args.labels, args.device, analytic, args.corner_weight,
                              args.margin_weight)}
        log.write(json.dumps(metrics) + "\n")
        log.flush()
        print(f"[epoch {epoch:3d}] loss {metrics['loss']:.4f} | regret verifier {metrics['all/verifier/regret']:.4f} "
              f"pdm {metrics['all/pdm/regret']:.4f} first {metrics['all/first/regret']:.4f} "
              f"random {metrics['all/random/regret']:.4f} | top1 verifier {metrics['all/verifier/top1']:.3f} "
              f"pdm {metrics['all/pdm/top1']:.3f} first {metrics['all/first/top1']:.3f}"
              + (f" | drivable pair acc {metrics['drivable/pair_acc']:.3f}" if "drivable/pair_acc" in metrics else "")
              + (f" | analytic loss {metrics['analytic/loss']:.4f} step acc {metrics['analytic/step_acc']:.4f}"
                 f" cand acc {metrics['analytic/candidate_acc']:.4f}" if analytic else ""))

        # Pretraining keeps the checkpoint with the lowest analytic loss; everything else the lowest regret
        key = "analytic/loss" if args.analytic_only else "all/verifier/regret"
        if best is None or metrics[key] < best[key]:
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
    ap.add_argument("--arch", choices=ARCHS, default="plan",
                    help="plan: one token per candidate; timestep: one token per scored state (see model.py)")
    ap.add_argument("--progress-weight", type=float, default=1.0,
                    help="timestep arch: weight of the smooth-L1 loss on raw progress")
    ap.add_argument("--rank-weight", type=float, default=3.0, help="weight of the listwise ranking loss")
    ap.add_argument("--rank-temperature", type=float, default=0.02,
                    help="softmax temperature of the target combined scores (ranking loss and classifier)")
    ap.add_argument("--objective", choices=["metrics", "classify"], default="metrics",
                    help="metrics: 7 metric logits, BCE + ranking loss; classify: one score per candidate, "
                         "cross-entropy against the best candidate (see classify_loss)")
    ap.add_argument("--analytic-weight", type=float, default=0.0,
                    help="timestep arch: weight of the dense analytic drivable-area loss added to the PDM-label loss")
    ap.add_argument("--analytic-only", action="store_true",
                    help="timestep arch: train on the dense analytic labels alone (pretraining; best = lowest analytic loss)")
    ap.add_argument("--corner-weight", type=float, default=0.0,
                    help="analytic loss: weight of the per-corner off-lane BCE (adds geom_head; 0 = off)")
    ap.add_argument("--margin-weight", type=float, default=0.0,
                    help="analytic loss: weight of the overshoot regression [m] (adds geom_head; 0 = off)")
    ap.add_argument("--lane-segments", type=int, default=0,
                    help="> 0: add trainable lane segment tokens, each lane cut into this many pieces (0 = w6)")
    ap.add_argument("--init-from", default=None,
                    help="checkpoint (train.py best.pt / last.pt) to start from, e.g. an --analytic-only run")
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
