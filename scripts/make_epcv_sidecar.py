"""
Sidecar for --progress-mode ep_cv: per label sample, the simulated ego's speed at that step (read from the label run's
simulation logs; the label files hold DP's zeroed ego state, and train-labels-2k-ego holds the human's logged speed,
not the ego's) and each candidate's computed progress along the route (diffusion_planner.scorer.route_progress).

Writes <out>/<scenario dir>/<iteration>.npz with ego_speed (float32 scalar, m/s) and progress_route ([K] float32, m),
mirroring the label dir.

    python scripts/make_epcv_sidecar.py --labels ~/nuplan/labels/train-labels-2k \
        --simlog $NUPLAN_EXP_ROOT/exp/simulation/closed_loop_nonreactive_agents/dp_k16_label_noise1.5/train-labels-2k/<timestamp>/simulation_log \
        --out ~/nuplan/labels/train-labels-2k-epcv --workers 8
"""
import argparse
import glob
from multiprocessing import Pool
from pathlib import Path

import numpy as np


def one_scenario(job):
    scen_dir, simlog, out = job
    from nuplan.planning.simulation.simulation_log import SimulationLog

    from diffusion_planner.scorer.labels import load_sample
    from diffusion_planner.scorer.route_progress import route_progress

    log, token, _ = scen_dir.name.split("__", 2)
    hits = glob.glob(str(Path(simlog) / "*" / "*" / log / token / f"{token}.msgpack.xz"))
    if not hits:
        return scen_dir.name, 0, len(list(scen_dir.glob("*.npz")))
    hist = SimulationLog.load_data(Path(hits[0])).simulation_history.data
    speed_at = {s.iteration.index: float(s.ego_state.dynamic_car_state.rear_axle_velocity_2d.x) for s in hist}
    done, missing = 0, 0
    (Path(out) / scen_dir.name).mkdir(parents=True, exist_ok=True)
    for f in sorted(scen_dir.glob("*.npz")):
        it = int(f.stem)
        if it not in speed_at:
            missing += 1
            continue
        s = load_sample(f)
        prog = route_progress(s["inputs"]["route_lanes"], s["candidates"])
        np.savez(Path(out) / scen_dir.name / f.name, ego_speed=np.float32(speed_at[it]),
                 progress_route=prog.astype(np.float32))
        done += 1
    return scen_dir.name, done, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--simlog", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    dirs = sorted(d for d in Path(args.labels).expanduser().iterdir() if d.is_dir())
    jobs = [(d, args.simlog, str(Path(args.out).expanduser())) for d in dirs]
    total, miss, nolog = 0, 0, []
    with Pool(args.workers) as pool:
        for n, (name, done, missing) in enumerate(pool.imap_unordered(one_scenario, jobs)):
            total += done
            miss += missing
            if done == 0:
                nolog.append(name)
            if n % 100 == 0:
                print(f"[sidecar] {n + 1}/{len(jobs)} scenarios, {total} samples written", flush=True)
    print(f"[sidecar] done: {total} samples written, {miss} without a matching log step, "
          f"{len(nolog)} scenarios without a log")


if __name__ == "__main__":
    main()
