"""S1 with its learned raw progress replaced by progress computed from the plan (40th pose projected on a centerline
chained from DP's route lanes), in the same PDM normalization (gated by S1's predicted safety, over the best of the
K candidates). Validation regret on cv labels, split seed 0, against S1 as is and against PDM's own raw progress.

    python eval_computed_progress.py ~/nuplan/labels/train-labels-2k checkpoints/args.json ckpt_seed0.pt [ckpt ...]
"""
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from diffusion_planner.scorer.dataset import LabelDataset, find_samples, split_by_scenario
from diffusion_planner.scorer.lane_features import polylines
from diffusion_planner.scorer.metrics import combined_score
from diffusion_planner.scorer.model import PROGRESS_THRESHOLD, Verifier
from diffusion_planner.scorer.train import first_max, selection_metrics, to_device

DATA, ARGS, CKPTS = sys.argv[1], sys.argv[2], sys.argv[3:]
POSE, LINK = 39, 1.5
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def route_centerline(route_lanes: np.ndarray):
    """As in check_progress.py: chain DP's route lanes into one polyline, starting at the lane under ego."""
    c, _, _, valid = polylines(torch.from_numpy(route_lanes)[None])
    c, valid = c[0].numpy(), valid[0].numpy()
    lanes = [c[i] for i in range(len(c)) if valid[i]]
    lanes = [l[np.abs(l).sum(-1) > 0] if (np.abs(l).sum(-1) > 0).sum() >= 2 else l for l in lanes]
    if not lanes:
        return None

    def dist_to_origin(l):
        a, b = l[:-1], l[1:]
        ab = b - a
        t = np.clip(-(a * ab).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-9), 0, 1)
        p = a + t[:, None] * ab
        k = int(np.argmin(np.linalg.norm(p, axis=-1)))
        h = ab[k] / max(np.linalg.norm(ab[k]), 1e-9)
        return float(np.linalg.norm(p[k])) - 0.5 * float(h[0])

    start = min(range(len(lanes)), key=lambda i: dist_to_origin(lanes[i]))
    chain, used = [lanes[start]], {start}
    while True:
        prev = [i for i in range(len(lanes)) if i not in used and np.linalg.norm(lanes[i][-1] - chain[0][0]) < LINK]
        if not prev:
            break
        chain.insert(0, lanes[prev[0]]); used.add(prev[0])
    while True:
        tail = chain[-1]
        d_tail = tail[-1] - tail[-2]
        nxt = [i for i in range(len(lanes)) if i not in used and np.linalg.norm(lanes[i][0] - tail[-1]) < LINK]
        if not nxt:
            break
        def align(i):
            d = lanes[i][1] - lanes[i][0]
            return -float(np.dot(d, d_tail) / max(np.linalg.norm(d) * np.linalg.norm(d_tail), 1e-9))
        i = min(nxt, key=align)
        chain.append(lanes[i]); used.add(i)
    return np.concatenate(chain, axis=0)


def arc_position(line, pts):
    a, b = line[:-1], line[1:]
    ab = b - a
    seg = np.linalg.norm(ab, axis=-1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])[:-1]
    t = np.clip(((pts[:, None] - a[None]) * ab[None]).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-9)[None], 0, 1)
    proj = a[None] + t[..., None] * ab[None]
    k = np.argmin(np.linalg.norm(pts[:, None] - proj, axis=-1), axis=1)
    return cum[k] + t[np.arange(len(pts)), k] * seg[k]


def computed_progress(route_lanes: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
    """[B, K] metres along the chained route centerline, current pose to the 40th planned pose, clipped at 0 as PDM's
    label is; falls back to the straight-line distance when a sample has no route lanes."""
    rl, cand = route_lanes.cpu().numpy(), candidates[:, :, POSE, :2].cpu().double().numpy()
    out = np.zeros(cand.shape[:2])
    for i in range(len(cand)):
        line = route_centerline(rl[i])
        if line is None or len(line) < 2:
            out[i] = np.linalg.norm(cand[i], axis=-1)
            continue
        pos = arc_position(line, np.vstack([np.zeros((1, 2)), cand[i]]))
        out[i] = pos[1:] - pos[0]
    return torch.from_numpy(np.clip(out, 0, None)).float()


def renormalized(out: torch.Tensor, raw: torch.Tensor) -> torch.Tensor:
    """S1's combined score with its progress recomputed from `raw` [B, K] metres, exactly as _timestep_forward does."""
    probs = torch.sigmoid(out).clone()
    safe = probs[..., :3].prod(dim=-1)
    gated = raw * safe
    best = gated.max(dim=1, keepdim=True).values
    probs[..., 3] = torch.where(best > PROGRESS_THRESHOLD, gated / best.clamp_min(PROGRESS_THRESHOLD), safe)
    return combined_score(probs)


files = find_samples(DATA)
_, val = split_by_scenario(files, 0.1, 0)
print(f"validation samples: {len(val)} ({len({f.parent for f in val})} scenarios)")
loader = DataLoader(LabelDataset(val), batch_size=64, shuffle=False, num_workers=4)

# computed progress depends only on the data: once
comp, pdm_raw, true, cands = [], [], [], []
batches = []
for batch in loader:
    comp.append(computed_progress(batch["inputs"]["route_lanes"], batch["candidates"]))
    pdm_raw.append(batch["progress_cv"].float())
    true.append(combined_score(batch["metrics_cv"]).float())
comp, pdm_raw, true = torch.cat(comp), torch.cat(pdm_raw), torch.cat(true)
print(f"computed vs PDM raw progress on validation: MAE {(comp - pdm_raw).abs().mean():.2f} m, "
      f"median {(comp - pdm_raw).abs().median():.2f} m")

res = {"as is": [], "computed raw": [], "PDM raw (ceiling)": []}
for ck in CKPTS:
    model = Verifier.load(ck, ARGS, device=DEV)
    model.eval()
    preds = {k: [] for k in res}
    learned = []
    with torch.no_grad():
        n = 0
        for batch in loader:
            b = to_device(batch, DEV)
            out, aux = model(b["inputs"], b["candidates"], return_aux=True)
            out = out.float().cpu()
            B = out.shape[0]
            preds["as is"].append(combined_score(torch.sigmoid(out)))
            preds["computed raw"].append(renormalized(out, comp[n:n + B]))
            preds["PDM raw (ceiling)"].append(renormalized(out, pdm_raw[n:n + B]))
            learned.append(aux["progress_raw"].float().cpu())
            n += B
    learned = torch.cat(learned)
    print(f"--- {ck}: S1's learned raw progress vs PDM's: MAE {(learned - pdm_raw).abs().mean():.2f} m, "
          f"median {(learned - pdm_raw).abs().median():.2f} m")
    for k in res:
        m = selection_metrics(true, first_max(torch.cat(preds[k])))
        res[k].append(m)
        print(f"    {k:18s} regret {m['regret']:.4f}  top-1 {m['top1']:.3f}")

print("\n3-seed summary (mean ± sd):")
for k, v in res.items():
    r = np.array([m["regret"] for m in v]); t = np.array([m["top1"] for m in v])
    print(f"  {k:18s} regret {r.mean():.4f} ± {r.std(ddof=1) if len(r) > 1 else 0:.4f}   top-1 {t.mean():.3f}")
