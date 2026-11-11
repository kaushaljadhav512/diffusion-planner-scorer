"""
Tests for diffusion_planner.scorer.reactive and scripts/agreement_report.py. Need nuplan-devkit and tuplan_garage
(skipped otherwise), no map or nuPlan data. Fork fidelity against a real simulation is checked at run time
(fork_check.jsonl, see diffusion_planner/scorer/agreement.py).

    pytest tests/test_reactive.py
"""
import subprocess
import sys
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("nuplan")
pytest.importorskip("tuplan_garage")

from nuplan.planning.simulation.occupancy_map.strtree_occupancy_map import STRTreeOccupancyMap
from shapely.geometry import box
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_enums import StateIndex

from diffusion_planner.scorer.metrics import METRICS
from diffusion_planner.scorer.reactive import (
    CachedScenario,
    extend_constant_velocity,
    fork_idm_agents,
    idm_state_snapshot,
    normalized_progress,
)

ROOT = Path(__file__).resolve().parents[1]


class _FakeIDMAgents:
    """What fork_idm_agents and idm_state_snapshot touch of IDMAgents / IDMAgentManager / IDMAgent."""

    def __init__(self):
        agent = SimpleNamespace(
            _state=SimpleNamespace(progress=1.0, velocity=5.0),
            _route=deque(["lane_a"], maxlen=5),
            _policy=SimpleNamespace(target_velocity=10.0),
        )
        self._idm_agent_manager = SimpleNamespace(
            agents={"car": agent},
            agent_occupancy=STRTreeOccupancyMap({"car": box(0, 0, 1, 1)}),
        )
        self.current_iteration = 3

    def _get_idm_agent_manager(self):
        return self._idm_agent_manager


def _step(observation):
    """What one IDM step mutates."""
    manager = observation._get_idm_agent_manager()
    agent = manager.agents["car"]
    agent._state.progress += 2.0
    agent._state.velocity = 6.0
    agent._route.append("lane_b")
    agent._policy.target_velocity = 15.0
    manager.agent_occupancy.set("ego", box(5, 5, 6, 6))
    manager.agents.pop("car")
    observation.current_iteration += 1


def test_fork_leaves_original_untouched():
    live = _FakeIDMAgents()
    before = idm_state_snapshot(live)
    fork = fork_idm_agents(live)
    forked_agent = fork._get_idm_agent_manager().agents["car"]
    _step(fork)

    assert idm_state_snapshot(live) == before
    assert live._get_idm_agent_manager().agents["car"]._state.progress == 1.0
    assert list(live._get_idm_agent_manager().agents["car"]._route) == ["lane_a"]
    assert forked_agent._state.progress == 3.0 and list(forked_agent._route) == ["lane_a", "lane_b"]
    assert live._get_idm_agent_manager().agent_occupancy.get_all_ids() == ["car"]
    assert idm_state_snapshot(fork) != before


def test_extend_constant_velocity():
    states = np.zeros((3, StateIndex.size()))
    states[:, StateIndex.X] = [0.0, 1.0, 2.0]
    states[:, StateIndex.HEADING] = np.pi / 2
    states[:, StateIndex.VELOCITY_X] = 10.0       # forward, i.e. along +y at heading pi/2
    states[:, StateIndex.ACCELERATION_X] = 3.0

    out = extend_constant_velocity(states, 5, dt=0.1)
    assert out.shape == (5, StateIndex.size())
    np.testing.assert_array_equal(out[:3], states)
    np.testing.assert_allclose(out[3:, StateIndex.X], [2.0, 2.0], atol=1e-9)
    np.testing.assert_allclose(out[3:, StateIndex.Y], [1.0, 2.0])
    np.testing.assert_allclose(out[3:, StateIndex.ACCELERATION_X], 0.0)
    np.testing.assert_array_equal(extend_constant_velocity(states, 2, dt=0.1), states[:2])


def test_normalized_progress_matches_pdm():
    np.testing.assert_allclose(normalized_progress(np.array([10.0, 5.0, 20.0]), np.array([1.0, 1.0, 0.0])),
                               [1.0, 0.5, 0.0])  # the failed candidate does not set the best
    np.testing.assert_allclose(normalized_progress(np.array([0.05, 0.0]), np.array([1.0, 0.0])), [1.0, 0.0])


def test_cached_scenario_memoizes_per_iteration():
    calls = []

    class Scenario:
        token = "tok"

        def get_traffic_light_status_at_iteration(self, iteration):
            calls.append(("tl", iteration))
            yield from [iteration]

        def get_tracked_objects_at_iteration(self, iteration):
            calls.append(("obj", iteration))
            return iteration

        def get_time_point(self, iteration):
            calls.append(("t", iteration))
            return iteration

    cached = CachedScenario(Scenario())
    for _ in range(3):
        assert cached.get_traffic_light_status_at_iteration(4) == [4]  # a list: reusable, unlike the generator
        assert cached.get_tracked_objects_at_iteration(4) == 4
        assert cached.get_time_point(4) == 4
    assert sorted(calls) == [("obj", 4), ("t", 4), ("tl", 4)]
    assert cached.token == "tok"


def _write_sample(path, cv_best, reactive_best, scenario_type="following_lane_with_lead", K=4):
    def metrics(best):
        m = np.ones((K, len(METRICS)), dtype=np.float32)
        m[:, 3] = 0.5
        m[best, 3] = 1.0
        return m

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, metrics_cv=metrics(cv_best), metrics_log=metrics(cv_best), metrics_reactive=metrics(reactive_best),
             selected=np.int64(cv_best), reactive_seconds=np.float32(2.0), scenario_type=np.str_(scenario_type),
             iteration=np.int64(int(path.stem)))


def test_agreement_report(tmp_path):
    _write_sample(tmp_path / "s1" / "0000.npz", cv_best=1, reactive_best=1)
    _write_sample(tmp_path / "s1" / "0020.npz", cv_best=1, reactive_best=2)
    _write_sample(tmp_path / "s2" / "0000.npz", cv_best=0, reactive_best=0, scenario_type="changing_lane")
    _write_sample(tmp_path / "s2" / "0020.npz", cv_best=3, reactive_best=3, scenario_type="changing_lane")
    (tmp_path / "fork_check.jsonl").write_text(
        '{"scenario": "s1", "iteration": 1, "n_common": 3, "only_predicted": 0, "only_actual": 0, '
        '"max_deviation_m": 0.0}\n')

    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "agreement_report.py"), str(tmp_path),
                          "--csv", str(tmp_path / "out.csv")], capture_output=True, text=True, check=True).stdout
    assert "4 samples from 2 scenarios, K = 4" in out
    # PDM+SL agrees on 3 of 4; the miss costs (5 * 0.5 vs 5 * 1) / 16 = 0.15625, mean 0.0391
    assert "PDM+SL (cv)        75.0%" in out
    assert "0.0391" in out
    assert "Fork fidelity (1 checks)" in out
    assert len((tmp_path / "out.csv").read_text().splitlines()) == 5
