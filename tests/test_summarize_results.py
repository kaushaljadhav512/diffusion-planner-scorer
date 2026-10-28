"""
Tests for scripts/summarize_results.py on a fake experiment tree (no nuPlan).

    pytest tests/test_summarize_results.py
"""
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "summarize_results.py"
_spec = importlib.util.spec_from_file_location("summarize_results", SCRIPT)
sr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sr)


def _run(root, planner, split, challenge, cls, runtimes=None, failed=0):
    """One run dir laid out like run_sim.sh's, with nuPlan's aggregator and (optionally) runner report."""
    run_dir = root / "simulation" / challenge / planner / split / "2026-10-27-12-00-00"
    (run_dir / "aggregator_metric").mkdir(parents=True)
    pd.DataFrame({
        "scenario": ["s1", "s2", "final_score"], "log_name": ["log", "log", None],
        "scenario_type": ["a", "b", None], "num_scenarios": [1, 1, 2], "score": [cls / 100] * 3,
    }).to_parquet(run_dir / "aggregator_metric" / "x_weighted_average_metrics_y.parquet")
    if runtimes is not None:
        report = pd.DataFrame({"succeeded": [True, not failed], **{col: [v, v] for col, v in runtimes.items()}})
        report.to_parquet(run_dir / "runner_report.parquet")


def test_runner_report_runtimes_and_failures(tmp_path):
    _run(tmp_path, "dp_k16_pdm", "sub", "closed_loop_nonreactive_agents", 70.0, failed=1, runtimes={
        "compute_trajectory_runtimes_mean": 0.25, "sampling_runtimes_mean": 0.1, "selection_runtimes_mean": 0.12})
    out = sr.parse_runner_report(next(tmp_path.rglob("runner_report.parquet")))

    assert out["failed"] == 1
    assert out["step_ms"] == pytest.approx(250) and out["sample_ms"] == pytest.approx(100)
    assert out["select_ms"] == pytest.approx(120)
    assert sr.parse_runner_report(tmp_path / "missing.parquet") == {}


def test_gain_over_first_matches_k_noise_split_challenge():
    df = pd.DataFrame([
        ("dp_k16_first", "sub", "nr", 70.0), ("dp_k16_oracle", "sub", "nr", 75.0), ("dp_k16_pdm", "sub", "r", 68.0),
        ("dp_k16_first", "sub", "r", 66.0), ("dp_k8_pdm", "sub", "nr", 71.0),  # no K=8 FirstSample run
        ("dp_k16_first_noise1.0", "sub", "nr", 60.0), ("dp_k16_pdm_noise1.0", "sub", "nr", 64.0),
        ("dp", "sub", "nr", 70.0),
    ], columns=["planner", "split", "challenge", "CLS"])

    gains = sr.gain_over_first(df).tolist()

    assert gains[:2] == [0.0, 5.0] and gains[2:4] == [2.0, 0.0]
    assert pd.isna(gains[4]) and gains[5:7] == [0.0, 4.0] and pd.isna(gains[7])


def test_cli_split_filter_and_csv(tmp_path):
    rt = {"compute_trajectory_runtimes_mean": 0.2, "sampling_runtimes_mean": 0.1, "selection_runtimes_mean": 0.05}
    _run(tmp_path, "dp_k16_first", "sub_smoke5", "closed_loop_nonreactive_agents", 70.0, rt)
    _run(tmp_path, "dp_k16_oracle", "sub_smoke5", "closed_loop_nonreactive_agents", 73.5, rt)
    _run(tmp_path, "dp", "test14-hard", "closed_loop_nonreactive_agents", 76.0, {"compute_trajectory_runtimes_mean": 0.1})
    csv = tmp_path / "out.csv"

    result = subprocess.run([sys.executable, str(SCRIPT), "--exp-root", str(tmp_path), "--split", "sub_smoke5",
                             "--csv", str(csv)], capture_output=True, text=True, check=True)

    assert "gain over FirstSample" in result.stdout
    table = pd.read_csv(csv).set_index("planner")
    assert set(table.index) == {"dp_k16_first", "dp_k16_oracle"}
    assert table.loc["dp_k16_oracle", "vs_first"] == pytest.approx(3.5)
    assert table.loc["dp_k16_oracle", "select_ms"] == pytest.approx(50)
