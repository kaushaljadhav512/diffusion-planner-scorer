"""
Tests for scripts/make_subset_filter.py on tiny synthetic DBs (only the scenario_tag table), no nuPlan data.

    pytest tests/test_make_subset_filter.py
"""
import importlib.util
import sqlite3
from collections import Counter
from pathlib import Path

import pytest
import yaml

_spec = importlib.util.spec_from_file_location(
    "make_subset_filter", Path(__file__).resolve().parent.parent / "scripts" / "make_subset_filter.py")
msf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(msf)


def _db(path, tags):
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE scenario_tag (token BLOB, lidar_pc_token BLOB, type TEXT)")
    con.executemany("INSERT INTO scenario_tag VALUES (?, ?, ?)",
                    [(bytes(8), bytes.fromhex(token), type_) for token, type_ in tags])
    con.commit()
    con.close()


def test_tagged_scenarios_across_dbs(tmp_path):
    _db(tmp_path / "a.db", [("00000000000000a1", "turn"), ("00000000000000a1", "unlisted"),
                            ("00000000000000a2", "stop"), ("00000000000000ff", "turn")]) # ff: token not asked for
    _db(tmp_path / "b.db", [("00000000000000b1", "turn"), ("00000000000000b1", "stop")])

    pairs = msf.tagged_scenarios(sorted(tmp_path.glob("*.db")),
                                 ["00000000000000a1", "00000000000000a2", "00000000000000b1"], {"turn", "stop"})

    assert pairs == [("00000000000000a1", "turn"), ("00000000000000a2", "stop"),
                     ("00000000000000b1", "stop"), ("00000000000000b1", "turn")]


@pytest.mark.parametrize("available, n, expected", [
    ({"a": 20, "b": 20, "c": 20}, 9, {"a": 3, "b": 3, "c": 3}),
    ({"a": 20, "b": 30, "c": 20}, 7, {"a": 2, "b": 3, "c": 2}),   # leftover to the largest type
    ({"a": 20, "b": 20, "c": 20}, 8, {"a": 3, "b": 3, "c": 2}),   # tie: by name
    ({"a": 1, "b": 20, "c": 20}, 9, {"a": 1, "b": 4, "c": 4}),    # small type capped, rest redistributed
])
def test_allocate(available, n, expected):
    assert msf.allocate(available, n) == expected


def test_allocate_too_many():
    with pytest.raises(ValueError):
        msf.allocate({"a": 2, "b": 3}, 6)


def _pairs():
    # 14 types x 20 single-tag tokens, plus one token tagged with two types
    pairs = [(f"{i:04x}{j:012x}", f"type_{i:02d}") for i in range(14) for j in range(20)]
    return sorted(pairs + [("ffff000000000000", "type_00"), ("ffff000000000000", "type_01")])


def test_pick_balanced_deterministic_single_tag():
    pairs = _pairs()
    chosen = msf.pick(pairs, 100, seed=0)

    assert len(chosen) == 100 and len({token for token, _ in chosen}) == 100
    assert sorted(Counter(type_ for _, type_ in chosen).values()) == [7] * 12 + [8] * 2
    assert "ffff000000000000" not in {token for token, _ in chosen}
    assert set(chosen) <= set(pairs)

    assert msf.pick(pairs, 100, seed=0) == chosen
    assert msf.pick(pairs, 100, seed=1) != chosen


def test_write_filter_keeps_settings(tmp_path):
    source = {"_target_": "x.ScenarioFilter", "scenario_types": ["a", "b", "c"], "scenario_tokens": ["1", "2"],
              "timestamp_threshold_s": 15, "remove_invalid_goals": True, "shuffle": False}
    # all-digit / exponent-looking tokens must stay strings
    chosen = [("1234567890123456", "b"), ("12345e6789012345", "a")]
    path = tmp_path / "out.yaml"
    msf.write_filter(source, chosen, path, "# header\n")

    written = yaml.safe_load(path.read_text())
    assert path.read_text().startswith("# header\n")
    assert written["scenario_types"] == ["a", "b"]
    assert written["scenario_tokens"] == ["1234567890123456", "12345e6789012345"]
    assert {k: v for k, v in written.items() if not k.startswith("scenario_")} == \
        {"_target_": "x.ScenarioFilter", "timestamp_threshold_s": 15, "remove_invalid_goals": True, "shuffle": False}
