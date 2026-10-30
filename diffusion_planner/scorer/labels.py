"""
Verifier training labels, recorded during closed-loop non-reactive simulation (task 2.7).

LabelRecorder is a selector: it drives with a PDM selector (constant-velocity forecasts, as at test time), so the
recorded states are on-policy. Every record_every steps it also scores the same K candidates with the oracle (logged
futures, exact in non-reactive simulation) and saves one sample:

    out_dir/<log_name>__<scenario_token>__<scenario_type>/<iteration:04d>.npz

        input_<key>       normalized model inputs of the step (what the verifier's frozen encoder reads), no batch dim
        candidates        [K, T, 4] float32 ego plans (x, y, cos, sin), ego frame, unnormalized
        neighbors         [K, P - 1, T, 4] float16 predicted neighbor futures (only with save_neighbors)
        metrics_oracle    [K, 7] per-metric scores with logged futures, order of diffusion_planner.scorer.metrics
        metrics_cv        [K, 7] the same with constant-velocity forecasts (what PDMSelector sees)
        progress_oracle   [K] raw progress along the centerline [m] (metrics_*[:, 3] is it normalized by the best)
        progress_cv       [K]
        selected          index of the executed candidate
        noise_scale       [K] initial-noise std of each candidate
        scenario_token, scenario_type, log_name, iteration

    bash scripts/run_sim.sh dp_k train-labels-2k nr  with SELECTOR=label (see run_sim.sh)
"""
import os
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional

import numpy as np
import torch

from diffusion_planner.scorer.selectors import AbstractSelector, SelectionContext

if TYPE_CHECKING:
    from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization

INPUT_PREFIX = "input_"


def save_sample(
        path: Path,
        inputs: Dict[str, np.ndarray],
        candidates: np.ndarray,
        metrics_oracle: np.ndarray,
        metrics_cv: np.ndarray,
        progress_oracle: np.ndarray,
        progress_cv: np.ndarray,
        selected: int,
        noise_scale: np.ndarray,
        scenario_token: str,
        scenario_type: str,
        log_name: str,
        iteration: int,
        neighbors: Optional[np.ndarray] = None,
    ) -> None:
    """Writes one label sample (format in the module docstring); atomic, so an interrupted run leaves no partial file."""
    arrays = {INPUT_PREFIX + k: v for k, v in inputs.items()}
    arrays.update(
        candidates=candidates.astype(np.float32),
        metrics_oracle=metrics_oracle.astype(np.float32),
        metrics_cv=metrics_cv.astype(np.float32),
        progress_oracle=progress_oracle.astype(np.float32),
        progress_cv=progress_cv.astype(np.float32),
        selected=np.int64(selected),
        noise_scale=np.asarray(noise_scale, dtype=np.float32),
        scenario_token=np.str_(scenario_token),
        scenario_type=np.str_(scenario_type),
        log_name=np.str_(log_name),
        iteration=np.int64(iteration),
    )
    if neighbors is not None:
        arrays["neighbors"] = neighbors.astype(np.float16)

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def load_sample(path: Path) -> Dict[str, object]:
    """
    Returns:
        dict with "inputs" (dict of arrays) and every other key of the file (0-d arrays as Python scalars / str)
    """
    with np.load(path) as data:
        sample = {"inputs": {k[len(INPUT_PREFIX):]: data[k] for k in data.files if k.startswith(INPUT_PREFIX)}}
        for k in data.files:
            if not k.startswith(INPUT_PREFIX):
                value = data[k]
                sample[k] = value.item() if value.ndim == 0 else value
    return sample


def sample_dir(out_dir: Path, log_name: str, scenario_token: str, scenario_type: str) -> Path:
    """One directory per scenario (a token tagged with two types runs, and is recorded, twice)."""
    return Path(out_dir) / f"{log_name}__{scenario_token}__{scenario_type}"


class LabelRecorder(AbstractSelector):
    """
    Drives with PDMSelector / PDMSpeedLimitSelector and records verifier labels every record_every steps (see the
    module docstring). Non-reactive simulation only: the oracle labels replay the logged future.
    """

    def __init__(
            self,
            out_dir: str,
            drive: str = "pdm_sl",
            record_every: int = 5,
            map_radius: float = 50.0,
            proposal_poses: int = 40,
            save_neighbors: bool = False,
        ):
        # Lazy imports: nuplan / tuplan_garage are only needed when recording
        from diffusion_planner.scorer.pdm_scoring import PDMOracleCandidateScorer
        from diffusion_planner.scorer.selectors import PDMSelector, PDMSpeedLimitSelector

        drivers = {"pdm": PDMSelector, "pdm_sl": PDMSpeedLimitSelector}
        assert drive in drivers, f"drive must be one of {list(drivers)}, got {drive}"
        self._driver = drivers[drive](map_radius=map_radius, proposal_poses=proposal_poses)
        self._oracle = PDMOracleCandidateScorer(map_radius=map_radius, proposal_poses=proposal_poses)
        self._out_dir = Path(out_dir)
        self._record_every = record_every
        self._save_neighbors = save_neighbors
        self._step = 0

    def initialize(self, initialization: "PlannerInitialization") -> None:
        self._driver.initialize(initialization)
        self._oracle.initialize(initialization)
        self._step = 0

    def select(self, candidates: torch.Tensor, context: SelectionContext) -> torch.Tensor:
        indices = self._driver.select(candidates, context)
        if self._step % self._record_every == 0:
            self._record(candidates, context, int(indices[0]))
        self._step += 1
        return indices

    def _record(self, candidates: torch.Tensor, context: SelectionContext, selected: int) -> None:
        from diffusion_planner.scorer.metrics import from_pdm

        assert context.scenario is not None and context.inputs is not None, "LabelRecorder needs the scenario and inputs"
        scenario = context.scenario
        K = candidates.shape[1]

        # Constant-velocity metrics: the driver just scored these candidates
        cv_scorer = self._driver._scorer
        metrics_cv = from_pdm(cv_scorer.last_metrics(), cv_scorer.speed_limit_compliance())
        progress_cv = cv_scorer.last_metrics()["progress_raw"]

        # Oracle metrics of the same candidates
        ego_plans = candidates[0, :, 0].detach().cpu().numpy().astype(np.float64) # [K, T, 4]
        self._oracle.scenario = scenario
        self._oracle.score(context.planner_input, ego_plans)
        metrics_oracle = from_pdm(self._oracle.last_metrics(), self._oracle.speed_limit_compliance())
        progress_oracle = self._oracle.last_metrics()["progress_raw"]

        iteration = context.planner_input.iteration.index
        save_sample(
            sample_dir(self._out_dir, scenario.log_name, scenario.token, scenario.scenario_type) / f"{iteration:04d}.npz",
            inputs={k: v[0].detach().cpu().numpy() for k, v in context.inputs.items()},
            candidates=ego_plans,
            neighbors=candidates[0, :, 1:].detach().cpu().numpy() if self._save_neighbors else None,
            metrics_oracle=metrics_oracle,
            metrics_cv=metrics_cv,
            progress_oracle=progress_oracle,
            progress_cv=progress_cv,
            selected=selected,
            noise_scale=np.asarray(context.noise_scale if context.noise_scale is not None else [np.nan] * K),
            scenario_token=scenario.token,
            scenario_type=scenario.scenario_type,
            log_name=scenario.log_name,
            iteration=iteration,
        )
