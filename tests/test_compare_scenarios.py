"""
Tests for scripts/compare_scenarios.py on a fake experiment tree (no nuPlan).

    pytest tests/test_compare_scenarios.py
"""
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "compare_scenarios.py"
_spec = importlib.util.spec_from_file_location("compare_scenarios", SCRIPT)
cs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cs)

NR = "closed_loop_nonreactive_agents"


def _run(root, planner, scores, split="sub", challenge=NR, timestamp="2026-10-30-00-00-00"):
    """scores: {scenario: score in [0, 1]}; scenario types are the scenario name's first letter."""
    run_dir = root / "simulation" / challenge / planner / split / timestamp
    (run_dir / "aggregator_metric").mkdir(parents=True)
    names = list(scores)
    pd.DataFrame({
        "scenario": names + ["final_score"], "log_name": ["log"] * len(names) + [None],
        "scenario_type": [n[0] for n in names] + [None], "score": list(scores.values()) + [0.0],
    }).to_parquet(run_dir / "aggregator_metric" / "x_weighted_average_metrics_y.parquet")


def _tree(root):
    # s1: PDM gains in both seeds; s2: gain in one seed, loss in the other; s3: nothing ever changes
    _run(root, "dp_k16_first_noise1.5", {"a1": 0.5, "a2": 0.8, "b3": 1.0})
    _run(root, "dp_k16_pdm_noise1.5", {"a1": 0.9, "a2": 0.9, "b3": 1.0})
    _run(root, "dp_k16_first_noise1.5_seed1", {"a1": 0.4, "a2": 0.8, "b3": 1.0})
    _run(root, "dp_k16_pdm_noise1.5_seed1", {"a1": 0.6, "a2": 0.7, "b3": 1.0})
    _run(root, "dp_k16_pdm_sl_noise0.5-1.5", {"a1": 0.6, "a2": 0.8, "b3": 1.0})
    _run(root, "dp_k16_first_noise0.5-1.5", {"a1": 0.5, "a2": 0.8, "b3": 1.0})
    _run(root, "dp_k16_pdm_noise1.5", {"a1": 0.0, "a2": 0.0, "b3": 0.0}, split="other")


def test_deltas_match_first_of_same_seed_and_noise(tmp_path):
    _tree(tmp_path)
    deltas = cs.deltas_vs_first(cs.scenario_scores(tmp_path, "sub")).set_index(["config", "seed", "scenario"])

    assert deltas.loc[("dp_k16_pdm_noise1.5", 0, "a1"), "delta"] == pytest.approx(40)
    assert deltas.loc[("dp_k16_pdm_noise1.5", 1, "a1"), "delta"] == pytest.approx(20)
    assert deltas.loc[("dp_k16_pdm_sl_noise0.5-1.5", 0, "a1"), "delta"] == pytest.approx(10)
    assert len(deltas) == 9 # 3 non-first runs x 3 scenarios; the other split is ignored


def test_per_scenario_consistency_and_insensitive(tmp_path):
    _tree(tmp_path)
    deltas = cs.deltas_vs_first(cs.scenario_scores(tmp_path, "sub"))
    scenarios = cs.per_scenario(deltas).set_index(["config", "scenario"])

    pdm = "dp_k16_pdm_noise1.5"
    assert scenarios.loc[(pdm, "a1"), "consistent"] and scenarios.loc[(pdm, "a1"), "gains"] == 2
    assert not scenarios.loc[(pdm, "a2"), "consistent"]
    assert scenarios.loc[(pdm, "b3"), "insensitive"] and not scenarios.loc[(pdm, "a2"), "insensitive"]

    summary = cs.per_config(deltas, cs.per_scenario(deltas)).set_index("config")
    assert summary.loc[pdm, "seeds"] == 2 and summary.loc[pdm, "improved"] == pytest.approx(1.5)
    assert summary.loc[pdm, "consistent_gains"] == 1 and summary.loc[pdm, "insensitive"] == 1


def test_cli(tmp_path):
    _tree(tmp_path)
    out = tmp_path / "per_scenario.csv"
    result = subprocess.run([sys.executable, str(SCRIPT), "--exp-root", str(tmp_path), "--split", "sub",
                             "--config", "dp_k16_pdm_noise1.5", "--out", str(out)],
                            capture_output=True, text=True, check=True)
    assert "top 10 gains" in result.stdout
    assert len(pd.read_csv(out)) == 6
