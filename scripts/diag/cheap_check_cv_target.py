"""Cheap offline check: how well do S1's picks score under a CLS-style target, as is and with its progress term scored
against constant-velocity E? No retraining, validation split only.

Target per candidate: PDM+SL's constant-velocity labels (metrics_cv) for every metric except progress; progress =
min(1, max(raw, 2 m) / max(E_cv, 2 m)) on PDM's raw progress (progress_cv), 0 if raw < -2 m; making-progress
multiplier (score x 0 if that ratio < 0.2). E_cv = the simulated ego's speed x 4 s, read from the label run's
simulation logs (the label files hold zeroed ego state; train-labels-2k-ego holds the human's speed, not the ego's).

    python cheap_check_cv_target.py <labels dir> <label-run simulation_log dir> <args.json> <ckpt> [ckpt ...]
"""
import glob
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from diffusion_planner.scorer.dataset import LabelDataset, find_samples, split_by_scenario
from diffusion_planner.scorer.metrics import EXPERT_PROGRESS_THRESHOLD as THR, MIN_PROGRESS_RATIO, combined_score
from diffusion_planner.scorer.model import Verifier
from diffusion_planner.scorer.train import first_max, to_device

DATA, SIMLOG, ARGS, CKPTS = sys.argv[1], Path(sys.argv[2]).expanduser(), sys.argv[3], sys.argv[4:]
DEV = "cuda" if torch.cuda.is_available() else "cpu"
H = 4.0

files = find_samples(DATA)
_, val = split_by_scenario(files, 0.1, 0)
print(f"validation samples {len(val)}, scenarios {len({f.parent for f in val})}")

# ---- step 1: simulated ego speed at each validation sample's iteration, from the label run's logs
cache = Path(__file__).with_name("val_ego_speed_split0.npy")
if cache.exists():
    speed = np.load(cache)
else:
    from nuplan.planning.simulation.simulation_log import SimulationLog
    by_dir = {}
    for i, f in enumerate(val):
        by_dir.setdefault(f.parent.name, []).append((i, int(f.stem)))
    speed = np.full(len(val), np.nan)
    for n, (d, items) in enumerate(by_dir.items()):
        log, token, _ = d.split("__", 2)
        hits = glob.glob(str(SIMLOG / "*" / "*" / log / token / f"{token}.msgpack.xz"))
        if not hits:
            continue
        hist = SimulationLog.load_data(Path(hits[0])).simulation_history.data
        at = {s.iteration.index: s for s in hist}
        for i, it in items:
            if it in at:
                speed[i] = float(at[it].ego_state.dynamic_car_state.rear_axle_velocity_2d.x)
        if n % 40 == 0:
            print(f"  read {n + 1}/{len(by_dir)} scenario logs")
    np.save(cache, speed)
ok = np.isfinite(speed)
print(f"simulated ego speed found for {ok.sum()} of {len(val)} samples; mean {np.nanmean(speed):.2f} m/s")
E = torch.from_numpy(np.clip(np.nan_to_num(speed), 0, None) * H).float()      # [N] E_cv in metres


def cls_style(metrics: torch.Tensor, raw: torch.Tensor, e: torch.Tensor):
    """[N, K] score with progress against E (as combined_score_expert), and the progress ratio itself."""
    ratio = torch.clamp(torch.clamp(raw, min=THR) / torch.clamp(e, min=THR)[:, None], max=1.0)
    ratio = torch.where(raw < -THR, torch.zeros_like(ratio), ratio)
    m = metrics.clone()
    m[..., 3] = ratio
    return combined_score(m) * (ratio >= MIN_PROGRESS_RATIO).float(), ratio


# ---- step 2: labels and target
loader = DataLoader(LabelDataset(val), batch_size=64, shuffle=False, num_workers=4)
metrics, raw_lab = [], []
for b in loader:
    metrics.append(b["metrics_cv"].float()); raw_lab.append(b["progress_cv"].float())
metrics, raw_lab = torch.cat(metrics), torch.cat(raw_lab)
target, target_ratio = cls_style(metrics, raw_lab, E)
best = target.max(1).values
mixed_col = ((metrics[..., 0] < 1).any(1) & (metrics[..., 0] == 1).any(1)).numpy()
keep = torch.from_numpy(ok)


def report(name, picks):
    p = picks[:, None].long()
    got = target.gather(1, p)[:, 0]
    reg = (best - got).numpy()
    top1 = (got >= best - 1e-6).float().numpy()
    reach = (target_ratio.gather(1, p)[:, 0] >= 1 - 1e-6).float().numpy()  # pick matches or beats E (label raw)
    over = (raw_lab.gather(1, p)[:, 0] - torch.clamp(E, min=THR)).numpy()  # metres beyond E
    k = ok
    return {"name": name, "regret": reg[k].mean(), "top1": top1[k].mean(), "reach_E": reach[k].mean(),
            "beyond_E_m": np.median(over[k]), "regret_mixed_col": reg[k & mixed_col].mean(), "reg": reg}


rows = [report("pdm_sl labels (current progress)", first_max(combined_score(metrics)))]
for ck in CKPTS:
    model = Verifier.load(ck, ARGS, device=DEV); model.eval()
    outs, raws = [], []
    with torch.no_grad():
        for b in loader:
            bb = to_device(b, DEV)
            out, aux = model(bb["inputs"], bb["candidates"], return_aux=True)
            outs.append(out.float().cpu()); raws.append(aux["progress_raw"].float().cpu())
    out, raw_pred = torch.cat(outs), torch.cat(raws)
    probs = torch.sigmoid(out)
    rows.append(report(f"S1 as is  [{Path(ck).parent.name}]", first_max(combined_score(probs))))
    s_cv, _ = cls_style(probs, raw_pred, E)
    rows.append(report(f"S1 + cv E [{Path(ck).parent.name}]", first_max(s_cv)))

print(f"\ntarget = PDM+SL labels x CLS-style progress against E_cv; {ok.sum()} samples, "
      f"{(ok & mixed_col).sum()} with mixed collision labels")
print(f"{'variant':38s} {'regret':>7s} {'top-1':>6s} {'pick reaches E':>14s} {'median m beyond E':>17s} "
      f"{'regret, mixed collision':>23s}")
for r in rows:
    print(f"{r['name']:38s} {r['regret']:7.4f} {r['top1']:6.3f} {r['reach_E']:14.3f} {r['beyond_E_m']:17.2f} "
          f"{r['regret_mixed_col']:23.4f}")

print("\n3-seed means:")
for tag in ("S1 as is", "S1 + cv E"):
    rs = [r for r in rows if r["name"].startswith(tag)]
    print(f"  {tag:10s} regret {np.mean([r['regret'] for r in rs]):.4f} +- {np.std([r['regret'] for r in rs], ddof=1):.4f}"
          f"  top-1 {np.mean([r['top1'] for r in rs]):.3f}  reaches E {np.mean([r['reach_E'] for r in rs]):.3f}"
          f"  regret on mixed collision {np.mean([r['regret_mixed_col'] for r in rs]):.4f}")
print("\nregret by simulated ego speed (3-seed means for S1):")
for lo, hi in [(-1, 0.5), (0.5, 3), (3, 8), (8, 100)]:
    m = ok & (speed >= lo) & (speed < hi)
    line = f"  [{lo:4.1f},{hi:5.1f}) n {m.sum():5d}   pdm_sl {rows[0]['reg'][m].mean():.4f}"
    for tag in ("S1 as is", "S1 + cv E"):
        rs = [r for r in rows if r["name"].startswith(tag)]
        line += f"   {tag} {np.mean([r['reg'][m].mean() for r in rs]):.4f}"
    print(line)
