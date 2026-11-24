"""
Do car parks explain drivable-area labels? Replays PDM's drivable-area check for every recorded label sample, with
car parks kept apart from lanes.

PDM's drivable map (get_drivable_area_map) is the lanes of every roadblock and roadblock connector within 50 m, plus
every car-park area. A candidate fails if any corner of ego's box, at any of the 41 LQR-tracked states, is outside all
of them. Car parks therefore never cause a failure: they can only rescue a plan that leaves the lanes. The verifier sees
neither car parks nor exact lane polygons, so rescued plans are where it should struggle.

For each sample this script takes ego's closed-loop state from the label run's simulation log, tracks the 16
candidates with PDM's LQR simulator, and tests every corner against
  - PDM's lanes only                (lanes_off: some corner outside every lane at some step)
  - PDM's lanes + car parks         (pdm_off: should equal the cv label, checked as label_agree)
  - DP's 70 input lane polygons     (dp_off, same tracked corners in ego frame)
and, for candidates that fail, whether a failing corner lies within --edge-dist of a car park (off_near_carpark).

    python scripts/check_carpark_drivable.py --labels $NUPLAN_HOME/labels/train-labels-2k \
        --simlogs $NUPLAN_EXP_ROOT/exp/simulation/closed_loop_nonreactive_agents/dp_k16_label_noise1.5/train-labels-2k \
        --out runs/carpark_check --workers 6

Writes <out>/candidates.csv (one row per candidate) and <out>/summary.json, and prints the summary.
"""
import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)) # this worktree's diffusion_planner, also in pool workers (they import this module)

FIELDS = ["scenario", "scenario_type", "iteration", "candidate", "label_drivable", "pdm_off", "lanes_off",
          "carpark_rescued", "dp_off", "off_near_carpark", "n_carparks", "label_agree"]


def find_simlogs(root: Path) -> Dict[str, Path]:
    """<scenario_type>/<log_name>/<token> -> msgpack path, newest run first wins."""
    logs = {}
    for path in sorted(root.glob("*/simulation_log/*/*/*/*/*.msgpack.xz"), reverse=True):
        token, log_name, scenario_type = path.parents[0].name, path.parents[1].name, path.parents[2].name
        logs.setdefault(f"{log_name}__{token}__{scenario_type}", path)
    return logs


def check_scenario(job) -> List[dict]:
    scenario_dir, simlog_path, edge_dist, horizon = job
    # Imports here so each worker loads nuPlan once
    from shapely.geometry import Point
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer
    from nuplan.planning.simulation.simulation_log import SimulationLog
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from tuplan_garage.planning.simulation.planner.pdm_planner.observation.pdm_observation_utils import (
        DRIVABLE_MAP_LAYERS,
    )
    from tuplan_garage.planning.simulation.planner.pdm_planner.observation.pdm_occupancy_map import PDMOccupancyMap
    from tuplan_garage.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
    from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import (
        ego_state_to_state_array,
        state_array_to_coords_array,
    )
    from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_enums import StateIndex

    from diffusion_planner.scorer.analytic_metrics import lane_polygons, points_in_polygons
    from diffusion_planner.scorer.labels import load_sample
    from diffusion_planner.scorer.metrics import METRICS
    from diffusion_planner.scorer.pdm_scoring import relative_to_global_poses

    drivable_idx = METRICS.index("drivable_area")
    log = SimulationLog.load_data(simlog_path)
    map_api = log.scenario.map_api
    ego_states = {s.iteration.index: s.ego_state for s in log.simulation_history.data}
    simulator = PDMSimulator(TrajectorySampling(num_poses=horizon, interval_length=0.1))

    rows = []
    for npz in sorted(Path(scenario_dir).glob("*.npz")):
        sample = load_sample(npz)
        ego_state = ego_states.get(int(sample["iteration"]))
        if ego_state is None:
            continue
        candidates = np.asarray(sample["candidates"], dtype=np.float64)[:, :horizon]
        K = len(candidates)

        # Candidates tracked as PDMCandidateScorer.score does: current state, then the plan's global poses
        states = np.zeros((K, horizon + 1, StateIndex.size()), dtype=np.float64)
        states[:, 0] = ego_state_to_state_array(ego_state)
        states[:, 1:, StateIndex.STATE_SE2] = relative_to_global_poses(ego_state, candidates)
        tracked = simulator.simulate_proposals(states, ego_state)
        coords = state_array_to_coords_array(tracked, ego_state.car_footprint.vehicle_parameters)  # [K, T, 5, 2]
        corners = coords[:, :, :-1]                                                                # [K, T, 4, 2]
        points = corners.reshape(-1, 2)

        # PDM's drivable map, lanes and car parks kept apart (get_drivable_area_map)
        objects = map_api.get_proximal_map_objects(ego_state.center.point, 50.0, DRIVABLE_MAP_LAYERS)
        lanes = [lane.polygon for layer in (SemanticMapLayer.ROADBLOCK, SemanticMapLayer.ROADBLOCK_CONNECTOR)
                 for block in objects[layer] for lane in block.interior_edges]
        carparks = [c.polygon for c in objects[SemanticMapLayer.CARPARK_AREA]]

        def inside(polygons):
            if not polygons:
                return np.zeros(corners.shape[:3], dtype=bool)
            occupancy = PDMOccupancyMap([str(i) for i in range(len(polygons))], polygons)
            return occupancy.points_in_polygons(points).any(axis=0).reshape(corners.shape[:3])

        in_lane, in_carpark = inside(lanes), inside(carparks)
        off_corner = ~(in_lane | in_carpark)                                                      # [K, T, 4]
        pdm_off = off_corner.any(axis=(1, 2))
        lanes_off = (~in_lane).any(axis=(1, 2))

        # DP's input lanes (ego rear-axle frame, as the verifier sees them), same tracked corners
        x0, y0, h0 = ego_state.rear_axle.serialize()
        c, s = np.cos(h0), np.sin(h0)
        dx, dy = points[:, 0] - x0, points[:, 1] - y0
        local = np.stack([c * dx + s * dy, -s * dx + c * dy], axis=-1)
        dp_polygons = lane_polygons(sample["inputs"]["lanes"])
        if len(dp_polygons):
            in_dp = points_in_polygons(local, dp_polygons).any(axis=0).reshape(corners.shape[:3])
        else:
            in_dp = np.zeros(corners.shape[:3], dtype=bool)
        dp_off = (~in_dp).any(axis=(1, 2))

        label = np.asarray(sample["metrics_cv"])[:, drivable_idx]
        for k in range(K):
            near = False
            if pdm_off[k] and carparks:
                failing = corners[k][off_corner[k]]
                near = any(min(p.distance(Point(*xy)) for p in carparks) <= edge_dist for xy in failing)
            rows.append({
                "scenario": Path(scenario_dir).name,
                "scenario_type": sample["scenario_type"],
                "iteration": int(sample["iteration"]),
                "candidate": k,
                "label_drivable": float(label[k]),
                "pdm_off": int(pdm_off[k]),
                "lanes_off": int(lanes_off[k]),
                "carpark_rescued": int(lanes_off[k] and not pdm_off[k]),
                "dp_off": int(dp_off[k]),
                "off_near_carpark": int(near),
                "n_carparks": len(carparks),
                "label_agree": int(pdm_off[k] == (label[k] == 0.0)),
            })
    return rows


def summarize(rows: List[dict]) -> dict:
    def stats(rs):
        n = len(rs)
        fails = [r for r in rs if r["label_drivable"] == 0.0]
        passes = [r for r in rs if r["label_drivable"] == 1.0]
        left_lanes = [r for r in passes if r["lanes_off"]]
        return {
            "candidates": n,
            "label_agree": sum(r["label_agree"] for r in rs) / max(n, 1),
            "label_fail": len(fails),
            "fail_near_carpark": sum(r["off_near_carpark"] for r in fails),
            "pass_but_leaves_lanes": len(left_lanes),
            "carpark_rescued": sum(r["carpark_rescued"] for r in rs),
            "carpark_in_range": sum(r["n_carparks"] > 0 for r in rs),
            "dp_off_label_pass": sum(r["dp_off"] for r in passes),
            "dp_off_pdm_lanes_on": sum(r["dp_off"] and not r["lanes_off"] for r in rs),
        }

    by_type = defaultdict(list)
    for r in rows:
        by_type[r["scenario_type"]].append(r)
    scenes = defaultdict(list)
    for r in rows:
        scenes[(r["scenario"], r["iteration"])].append(r)
    mixed = [g for g in scenes.values() if len({r["label_drivable"] for r in g}) > 1]
    return {
        "all": stats(rows),
        "samples": len(scenes),
        "samples_mixed_drivable": len(mixed),
        "samples_mixed_with_rescue": sum(any(r["carpark_rescued"] for r in g) for g in mixed),
        "by_scenario_type": {t: stats(rs) for t, rs in sorted(by_type.items())},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True, help="label directory (one subdirectory per scenario)")
    ap.add_argument("--simlogs", required=True, help="simulation output of the label run (timestamped runs inside)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--edge-dist", type=float, default=1.0, help="[m] a failing corner this close to a car park counts")
    ap.add_argument("--horizon", type=int, default=40)
    ap.add_argument("--limit", type=int, default=0, help="only the first N scenarios (smoke test)")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    simlogs = find_simlogs(Path(args.simlogs))
    scenario_dirs = sorted(p for p in Path(args.labels).iterdir() if p.is_dir())
    jobs = [(str(d), simlogs[d.name], args.edge_dist, args.horizon) for d in scenario_dirs if d.name in simlogs]
    missing = len(scenario_dirs) - len(jobs)
    if args.limit:
        jobs = jobs[:args.limit]
    print(f"{len(jobs)} scenarios, {missing} without a simulation log", flush=True)

    rows = []
    with Pool(args.workers) as pool:
        for i, scenario_rows in enumerate(pool.imap_unordered(check_scenario, jobs), 1):
            rows += scenario_rows
            if i % 100 == 0:
                print(f"{i}/{len(jobs)} scenarios, {len(rows)} candidates", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "candidates.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows)
    summary["scenarios_without_simlog"] = missing
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "by_scenario_type"}, indent=2))


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
