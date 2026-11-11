"""
Agreement check: in reactive simulation, does PDM+SL pick the candidate that reactive traffic rewards?

ReactiveAgreementRecorder is a selector for reactive simulation. It drives with PDMSpeedLimitSelector (so the states are
the ones PDM+SL meets) and every record_every steps scores the same K candidates three ways, with identical metric
definitions and only the other agents' model changed:

    cv        constant-velocity forecasts (what PDM+SL sees and picks with)
    log       the logged futures (the non-reactive oracle; agents ignore ego)
    reactive  the simulation's own IDM agents, forked and rolled forward while ego follows each candidate
              (diffusion_planner.scorer.reactive)

    out_dir/<log_name>__<scenario_token>__<scenario_type>/<iteration:04d>.npz
        input_<key>, candidates, selected, noise_scale, scenario_token, scenario_type, log_name, iteration
                            as in the label files (diffusion_planner.scorer.labels), so they can train a verifier
        metrics_cv, metrics_log, metrics_reactive    [K, 7], order of diffusion_planner.scorer.metrics
        progress_cv, progress_log, progress_reactive [K] raw progress [m]
        reactive_seconds    time spent scoring the K candidates reactively
    out_dir/fork_check.jsonl
        one line per recorded step: the forked agents one step ahead against the simulation's real next step, which
        both compute from the same ego state, so they should match exactly

Summarize with scripts/agreement_report.py.
"""
import json
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch

from diffusion_planner.scorer.labels import INPUT_PREFIX, sample_dir
from diffusion_planner.scorer.selectors import AbstractSelector, SelectionContext

if TYPE_CHECKING:
    from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization


class ReactiveAgreementRecorder(AbstractSelector):

    def __init__(self, out_dir: str, record_every: int = 20, map_radius: float = 50.0, proposal_poses: int = 40):
        from diffusion_planner.scorer.pdm_scoring import PDMOracleCandidateScorer
        from diffusion_planner.scorer.reactive import PDMReactiveCandidateScorer
        from diffusion_planner.scorer.selectors import PDMSpeedLimitSelector

        self._out_dir = Path(out_dir)
        self._record_every = record_every
        self._driver = PDMSpeedLimitSelector(map_radius=map_radius, proposal_poses=proposal_poses)
        self._log = PDMOracleCandidateScorer(map_radius=map_radius, proposal_poses=proposal_poses)
        self._reactive = PDMReactiveCandidateScorer(map_radius=map_radius, proposal_poses=proposal_poses)
        self._step = 0
        self._pending_check: Optional[dict] = None

    def initialize(self, initialization: "PlannerInitialization") -> None:
        self._driver.initialize(initialization)
        self._log.initialize(initialization)
        self._reactive.initialize(initialization)
        self._step = 0
        self._pending_check = None

    def select(self, candidates: torch.Tensor, context: SelectionContext) -> torch.Tensor:
        from diffusion_planner.scorer.reactive import find_simulation

        assert context.scenario is not None and context.inputs is not None, "needs the scenario and inputs"
        simulation = find_simulation()
        assert type(simulation._observations).__name__ == "IDMAgents", \
            "ReactiveAgreementRecorder needs reactive simulation (IDM agents): run the r challenge"

        if self._pending_check is not None:
            self._finish_fork_check(context)

        indices = self._driver.select(candidates, context)
        iteration = context.planner_input.iteration.index
        # The rollout needs rollout_steps logged iterations ahead (traffic lights, open-loop objects)
        fits = iteration + self._reactive.rollout_steps < context.scenario.get_number_of_iterations()
        if self._step % self._record_every == 0 and fits:
            self._record(candidates, context, int(indices[0]), simulation)
        self._step += 1
        return indices

    def _record(self, candidates: torch.Tensor, context: SelectionContext, selected: int, simulation) -> None:
        from diffusion_planner.scorer.metrics import from_pdm
        from diffusion_planner.scorer.reactive import detections_centers, idm_state_snapshot

        scenario = context.scenario
        K = candidates.shape[1]
        ego_plans = candidates[0, :, 0].detach().cpu().numpy().astype(np.float64) # [K, T, 4]

        # cv: the driver just scored these candidates
        cv = self._driver._scorer
        metrics_cv = from_pdm(cv.last_metrics(), cv.speed_limit_compliance())
        progress_cv = cv.last_metrics()["progress_raw"]

        self._log.scenario = scenario
        self._log.score(context.planner_input, ego_plans)
        metrics_log = from_pdm(self._log.last_metrics(), self._log.speed_limit_compliance())
        progress_log = self._log.last_metrics()["progress_raw"]

        before = idm_state_snapshot(simulation._observations)
        start = time.perf_counter()
        metrics_reactive, progress_reactive = self._reactive.score_reactive(
            context.planner_input, ego_plans, simulation._observations, scenario
        )
        seconds = time.perf_counter() - start
        if idm_state_snapshot(simulation._observations) != before:
            raise RuntimeError("reactive scoring changed the simulation's IDM agents")

        iteration = context.planner_input.iteration.index
        self._pending_check = {
            "scenario": f"{scenario.log_name}__{scenario.token}__{scenario.scenario_type}",
            "iteration": iteration + 1,
            "predicted": detections_centers(self._reactive.first_frames[0]),
        }

        arrays = {INPUT_PREFIX + k: v[0].detach().cpu().numpy() for k, v in context.inputs.items()}
        arrays.update(
            candidates=ego_plans.astype(np.float32),
            metrics_cv=metrics_cv.astype(np.float32),
            metrics_log=metrics_log.astype(np.float32),
            metrics_reactive=metrics_reactive.astype(np.float32),
            progress_cv=np.asarray(progress_cv, dtype=np.float32),
            progress_log=np.asarray(progress_log, dtype=np.float32),
            progress_reactive=np.asarray(progress_reactive, dtype=np.float32),
            selected=np.int64(selected),
            noise_scale=np.asarray(context.noise_scale if context.noise_scale is not None else [np.nan] * K,
                                   dtype=np.float32),
            reactive_seconds=np.float32(seconds),
            scenario_token=np.str_(scenario.token),
            scenario_type=np.str_(scenario.scenario_type),
            log_name=np.str_(scenario.log_name),
            iteration=np.int64(iteration),
        )
        path = sample_dir(self._out_dir, scenario.log_name, scenario.token, scenario.scenario_type) / \
            f"{iteration:04d}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.stem + ".tmp.npz")
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, path)

    def _finish_fork_check(self, context: SelectionContext) -> None:
        """Compares the forked agents' first step with the simulation's actual next step."""
        from diffusion_planner.scorer.reactive import detections_centers

        check, self._pending_check = self._pending_check, None
        if context.planner_input.iteration.index != check["iteration"]:
            return
        _, observation = context.planner_input.history.current_state
        actual = detections_centers(observation)
        predicted = check["predicted"]
        common = predicted.keys() & actual.keys()
        deviation = max((float(np.hypot(predicted[t][0] - actual[t][0], predicted[t][1] - actual[t][1]))
                         for t in common), default=0.0)
        line = {
            "scenario": check["scenario"], "iteration": check["iteration"], "n_common": len(common),
            "only_predicted": len(predicted.keys() - actual.keys()), "only_actual": len(actual.keys() - predicted.keys()),
            "max_deviation_m": deviation,
        }
        self._out_dir.mkdir(parents=True, exist_ok=True)
        with open(self._out_dir / "fork_check.jsonl", "a") as f: # short lines: atomic appends across workers
            f.write(json.dumps(line) + "\n")
