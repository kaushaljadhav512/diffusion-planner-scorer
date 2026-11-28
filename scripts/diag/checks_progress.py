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



EGO = Path(DATA).expanduser().parent / (Path(DATA).name + "-ego")


def ego_speed(path: Path) -> float:
    """Real ego speed [m/s] from the patched copy (train-labels-2k-ego), nan if missing."""
    p = EGO / path.parent.name / path.name
    if not p.exists():
        return float("nan")
    z = np.load(p)
    return float(z["input_ego_current_state"][4] * 20.0)


files = find_samples(DATA)
_, val = split_by_scenario(files, 0.1, 0)
loader = DataLoader(LabelDataset(val), batch_size=64, shuffle=False, num_workers=4)
speed = np.array([ego_speed(f) for f in val])
print(f"validation samples {len(val)}; real ego speed found for {np.isfinite(speed).sum()}")

comp, pdm_raw, labels = [], [], []
for batch in loader:
    comp.append(computed_progress(batch["inputs"]["route_lanes"], batch["candidates"]))
    pdm_raw.append(batch["progress_cv"].float())
    labels.append(batch["metrics_cv"].float())
comp, pdm_raw, labels = torch.cat(comp), torch.cat(pdm_raw), torch.cat(labels)
true = combined_score(labels)
unsafe = {"collision": labels[..., 0] < 1, "drivable": labels[..., 1] < 1, "ttc": labels[..., 4] < 1,
          "any multiplier": labels[..., :3].prod(-1) < 1}

bands = [(0, 0.5), (0.5, 3), (3, 8), (8, 100)]
acc_err = {k: [] for k in unsafe}
acc_band = []
for ck in CKPTS:
    model = Verifier.load(ck, ARGS, device=DEV); model.eval()
    learned, outs = [], []
    with torch.no_grad():
        for batch in loader:
            b = to_device(batch, DEV)
            out, aux = model(b["inputs"], b["candidates"], return_aux=True)
            outs.append(out.float().cpu()); learned.append(aux["progress_raw"].float().cpu())
    out, learned = torch.cat(outs), torch.cat(learned)
    err = learned - pdm_raw
    print(f"\n--- {ck}")
    print("  check 1: S1's learned raw progress minus PDM's [m]")
    print(f"    {'label':15s} {'unsafe: n':>10s} {'mean err':>9s}   {'safe: n':>9s} {'mean err':>9s}   "
          f"{'within mixed scenes: unsafe - safe':>36s} {'(scenes)':>9s}")
    for k, u in unsafe.items():
        mixed = u.any(1) & (~u).any(1)
        diffs = [(err[i][u[i]].mean() - err[i][~u[i]].mean()).item() for i in torch.nonzero(mixed)[:, 0]]
        # the same for PDM's own raw progress: are unsafe candidates genuinely slower or faster?
        rel = [(pdm_raw[i][u[i]].mean() - pdm_raw[i][~u[i]].mean()).item() for i in torch.nonzero(mixed)[:, 0]]
        acc_err[k].append(np.mean(diffs))
        print(f"    {k:15s} {int(u.sum()):10d} {err[u].mean():+9.2f}   {int((~u).sum()):9d} {err[~u].mean():+9.2f}   "
              f"{np.mean(diffs):+36.2f} {len(diffs):9d}   (PDM raw progress, unsafe - safe: {np.mean(rel):+.2f} m)")

    reg = {}
    for name, score in [("as is", combined_score(torch.sigmoid(out))), ("computed", renormalized(out, comp)),
                        ("PDM raw", renormalized(out, pdm_raw))]:
        pick = first_max(score)
        reg[name] = (true.max(1).values - true.gather(1, pick[:, None])[:, 0]).numpy()
    print("  check 2: regret by real ego speed")
    print(f"    {'speed [m/s]':12s} {'samples':>8s} {'as is':>8s} {'computed':>9s} {'PDM raw':>8s}   "
          f"{'computed - as is, share of total change':>40s}")
    total = (reg["computed"] - reg["as is"]).sum()
    row = []
    for lo, hi in bands:
        m = (speed >= lo) & (speed < hi)
        ch = (reg["computed"][m] - reg["as is"][m]).sum()
        row.append((m.sum(), reg["as is"][m].mean(), reg["computed"][m].mean(), reg["PDM raw"][m].mean(), ch / total))
        print(f"    [{lo:4.1f}, {hi:5.1f}) {m.sum():8d} {reg['as is'][m].mean():8.4f} {reg['computed'][m].mean():9.4f} "
              f"{reg['PDM raw'][m].mean():8.4f}   {100 * ch / total:39.0f}%")
    acc_band.append(row)

print("\n=== 3-seed means")
print("  check 1, within mixed scenes, learned-progress error of unsafe minus safe candidates [m]:")
for k, v in acc_err.items():
    print(f"    {k:15s} {np.mean(v):+.2f}  (seeds: {', '.join(f'{x:+.2f}' for x in v)})")
print("  check 2, regret by real ego speed (mean over seeds):")
A = np.array(acc_band)  # [seeds, bands, 5]
for j, (lo, hi) in enumerate(bands):
    print(f"    [{lo:4.1f}, {hi:5.1f})  n {int(A[0, j, 0]):5d}   as is {A[:, j, 1].mean():.4f}   computed {A[:, j, 2].mean():.4f}"
          f"   PDM raw {A[:, j, 3].mean():.4f}   share of the regret increase {100 * A[:, j, 4].mean():.0f}%")
