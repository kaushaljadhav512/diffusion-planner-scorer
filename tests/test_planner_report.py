"""
Tests for the per-step runtimes MultiSampleDiffusionPlanner reports to nuPlan. Need nuplan-devkit (skipped
otherwise), no checkpoint or data.

    pytest tests/test_planner_report.py
"""
import pytest

pytest.importorskip("nuplan")

from unittest.mock import MagicMock

from diffusion_planner.planner import planner as planner_module
from diffusion_planner.scorer.planner import MultiSampleDiffusionPlanner, MultiSamplePlannerReport


def test_report_summary_has_sampling_and_selection_columns():
    report = MultiSamplePlannerReport(
        compute_trajectory_runtimes=[0.3, 0.2], sampling_runtimes=[0.1, 0.1], selection_runtimes=[0.15, 0.05])
    summary = report.compute_summary_statistics()  # what nuPlan writes to runner_report.parquet

    assert summary["selection_runtimes_mean"] == pytest.approx(0.1)
    assert summary["sampling_runtimes_median"] == pytest.approx(0.1)
    assert {"compute_trajectory_runtimes_mean", "sampling_runtimes_mean", "selection_runtimes_mean"} <= set(summary)


def test_generate_planner_report_clears_per_scenario():
    planner = MultiSampleDiffusionPlanner.__new__(MultiSampleDiffusionPlanner) # skip loading the checkpoint
    planner._compute_trajectory_runtimes = [0.3]
    planner._sampling_runtimes = [0.1]
    planner._selection_runtimes = [0.2]

    report = planner.generate_planner_report()
    assert report.sampling_runtimes == [0.1] and report.selection_runtimes == [0.2]

    empty = planner.generate_planner_report()
    assert empty.compute_trajectory_runtimes == empty.sampling_runtimes == empty.selection_runtimes == []


def test_model_is_built_in_initialize_not_in_the_constructor(monkeypatch):
    # nuPlan constructs one planner per scenario in the main process; a model in each runs it out of RAM
    build = MagicMock()
    monkeypatch.setattr(planner_module, "Diffusion_Planner", build)
    monkeypatch.setattr(planner_module, "DataProcessor", MagicMock())
    planner = planner_module.DiffusionPlanner(MagicMock(), None, MagicMock(), MagicMock())
    build.assert_not_called()

    planner.initialize(MagicMock())
    build.assert_called_once()
