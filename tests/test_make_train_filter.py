"""
Tests for scripts/make_train_filter.py on tiny synthetic DBs (scenario_tag + lidar_pc), no nuPlan data.

    pytest tests/test_make_train_filter.py
"""
import importlib.util
import random
import sqlite3
import subprocess
import sys
from collections import Counter
from pathlib import Path

import yaml

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "make_train_filter.py"
_spec = importlib.util.spec_from_file_location("make_train_filter", SCRIPT)
mtf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mtf)

S = 1_000_000 # us


def _db(path, tags, duration_s=100, scenes=None):
    """
    tags: (token, type, time [s]) in a log covering [0, duration_s] s. They go in scene 2, a valid scene with a goal,
    unless scenes maps their token to another scene (0-4, or 9: valid but without a goal).
    """
    scenes = scenes or {}
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE lidar_pc (token BLOB, timestamp INTEGER, scene_token BLOB)")
    con.execute("CREATE TABLE scenario_tag (token BLOB, lidar_pc_token BLOB, type TEXT)")
    con.execute("CREATE TABLE scene (token BLOB, name TEXT, goal_ego_pose_token BLOB)")
    con.execute("CREATE TABLE ego_pose (token BLOB)")
    con.execute("INSERT INTO ego_pose VALUES (?)", (b"goal",))
    con.executemany("INSERT INTO scene VALUES (?, ?, ?)",
                    [(_scene(i), f"scene-{i}", b"goal") for i in range(5)] + [(_scene(9), "scene-2b", None)])
    rows = [(b"start", 0, _scene(0)), (b"end", duration_s * S, _scene(4))]
    rows += [(bytes.fromhex(token), t * S, _scene(scenes.get(token, 2))) for token, _, t in tags]
    con.executemany("INSERT INTO lidar_pc VALUES (?, ?, ?)", rows)
    con.executemany("INSERT INTO scenario_tag VALUES (?, ?, ?)",
                    [(bytes(8), bytes.fromhex(token), type_) for token, type_, _ in tags])
    con.commit()
    con.close()


def _scene(i):
    return bytes([i])


def _token(t):
    return f"{t:016x}"


def test_read_tags_skips_log_edges_and_unlisted_types(tmp_path):
    _db(tmp_path / "log1.db", [(_token(1), "turn", 1), (_token(10), "turn", 10), (_token(90), "turn", 90),
                               (_token(20), "other", 20)])
    tags = mtf.read_tags([tmp_path / "log1.db"], {"turn"})
    # t=1 s has < 2 s of history, t=90 s < 15 s of future
    assert tags == [mtf.Tag(_token(10), "turn", "log1", 10 * S)]


def test_read_tags_skips_scenes_the_builder_drops(tmp_path):
    tags = [(_token(t), "turn", t) for t in (10, 20, 30, 40)]
    # Sorted by name the 6 scenes are 0, 1, 2, 9, 3, 4; the first and last two are outside the builder's valid scenes
    _db(tmp_path / "log1.db", tags, scenes={_token(10): 1, _token(20): 3, _token(30): 9})
    kept = mtf.read_tags([tmp_path / "log1.db"], {"turn"})
    # t=10, 20: outside the valid scenes; t=30: valid scene without a goal
    assert [tag.token for tag in kept] == [_token(40)]


def test_candidates_cap_gap_and_single_tag():
    tags = [mtf.Tag(_token(t), "turn", "log1", t * S) for t in (10, 12, 30, 50, 70)]
    tags += [mtf.Tag(_token(40), "turn", "log2", 40 * S), mtf.Tag(_token(40), "stop", "log2", 40 * S)] # 2 tags: skipped
    pool = mtf.candidates_per_type(tags, max_per_log_type=3, min_gap_s=15, rng=random.Random(0))

    picked = sorted(tag.timestamp_us // S for tag in pool["turn"])
    assert len(picked) == 3 and all(b - a >= 15 for a, b in zip(picked, picked[1:]))
    assert "stop" not in pool and all(tag.log == "log1" for tag in pool["turn"])


def test_main_writes_balanced_train_only_filter(tmp_path, monkeypatch):
    rng = random.Random(0)
    for log in range(6):
        _db(tmp_path / f"train{log}.db", [(_token(1000 * log + t), rng.choice(["turn", "stop"]), t)
                                          for t in range(5, 80, 16)])
    _db(tmp_path / "val0.db", [(_token(9999), "turn", 30)])
    splitter = tmp_path / "splitter.yaml"
    splitter.write_text(yaml.safe_dump({"log_splits": {"train": [f"train{i}" for i in range(6)], "val": ["val0"]}}))

    filter_dir = tmp_path / "filters"
    filter_dir.mkdir()
    (filter_dir / "src.yaml").write_text(yaml.safe_dump(
        {"scenario_types": ["turn", "stop"], "scenario_tokens": [], "timestamp_threshold_s": 15}))
    monkeypatch.setattr(mtf, "FILTER_DIR", filter_dir)
    monkeypatch.setattr(sys, "argv", ["x", "--n", "10", "--types-from", "src", "--out", "out",
                                      "--db-dir", str(tmp_path), "--splitter", str(splitter)])
    mtf.main()

    written = yaml.safe_load((filter_dir / "out.yaml").read_text())
    assert len(written["scenario_tokens"]) == 10
    assert _token(9999) not in written["scenario_tokens"]   # val log never used
    assert written["timestamp_threshold_s"] == 15            # source settings kept
    tokens = set(written["scenario_tokens"])
    types = Counter(t for log in range(6) for tok, t in _tags_of(tmp_path / f"train{log}.db") if tok in tokens)
    assert types == {"turn": 5, "stop": 5}


def _tags_of(db):
    con = sqlite3.connect(db)
    try:
        return [(tok.hex(), t) for tok, t in con.execute("SELECT lidar_pc_token, type FROM scenario_tag")]
    finally:
        con.close()
