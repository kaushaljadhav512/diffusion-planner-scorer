"""Build the train scenario filter for verifier label generation (task 2.6).

Picks --n scenarios from the train DBs, balanced over Test14-hard's 14 scenario types, and writes a scenario filter
with Test14-hard's other settings. Only DBs whose log is in nuPlan's official train split are used (nuplan-devkit's
splitter config), so Val14 / Test14 logs can never leak in even though val DBs share the trainval/ directory.
Needs only the DBs (no nuplan-devkit install, no GPU).

    python scripts/make_train_filter.py                       # 2000 scenarios -> train-labels-2k
    python scripts/make_train_filter.py --n 200 --out train-labels-200
    SMOKE=5 SELECTOR=label bash scripts/run_sim.sh dp_k train-labels-2k nr

Within a log, scenarios of the same type are kept at least --min-gap-s apart (nuPlan's filter drops same-type
scenarios closer than Test14-hard's timestamp_threshold_s=15 anyway) and at most --max-per-log-type of them are
picked, so the labels cover many logs. Tags too close to the start or end of their log are skipped: a scenario needs
2 s of history and 15 s of future. As in make_subset_filter.py, only tokens with a single listed tag are picked, so
the written filter yields exactly --n scenarios.
"""
import argparse
import os
import random
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, NamedTuple, Set

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_subset_filter import FILTER_DIR, allocate, write_filter  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SPLITTER = ROOT / "third_party" / "nuplan-devkit" / "nuplan" / "planning" / "script" / "config" / "common" / "splitter" / "nuplan.yaml"
HISTORY_S, FUTURE_S = 2.0, 15.0


class Tag(NamedTuple):
    token: str
    type: str
    log: str
    timestamp_us: int


def train_logs(splitter: Path) -> Set[str]:
    return set(yaml.safe_load(splitter.read_text())["log_splits"]["train"])


def read_tags(db_files: List[Path], types: Set[str]) -> List[Tag]:
    """Tags of the listed types that leave HISTORY_S before and FUTURE_S after them in their log (one DB = one log)."""
    query = ("SELECT st.lidar_pc_token, st.type, lp.timestamp FROM scenario_tag st "
             "JOIN lidar_pc lp ON lp.token = st.lidar_pc_token")
    tags = []
    for db in db_files:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            start, end = con.execute("SELECT MIN(timestamp), MAX(timestamp) FROM lidar_pc").fetchone()
            if start is None:
                continue
            for token, type_, timestamp in con.execute(query):
                if type_ in types and start + HISTORY_S * 1e6 <= timestamp <= end - FUTURE_S * 1e6:
                    tags.append(Tag(token.hex(), type_, db.stem, int(timestamp)))
        finally:
            con.close()
    return sorted(tags)


def candidates_per_type(tags: List[Tag], max_per_log_type: int, min_gap_s: float, rng: random.Random
                        ) -> Dict[str, List[Tag]]:
    """
    Per type, the tags that may be picked: single-tag tokens only, at most max_per_log_type per (log, type), and
    same-type tags of a log at least min_gap_s apart.
    """
    tags_per_token = defaultdict(int)
    for tag in tags:
        tags_per_token[tag.token] += 1

    by_log_type = defaultdict(list)
    for tag in tags:
        if tags_per_token[tag.token] == 1:
            by_log_type[(tag.log, tag.type)].append(tag)

    pool = defaultdict(list)
    for (_, type_), group in sorted(by_log_type.items()):
        kept = []
        for tag in rng.sample(group, len(group)):
            if len(kept) == max_per_log_type:
                break
            if all(abs(tag.timestamp_us - other.timestamp_us) >= min_gap_s * 1e6 for other in kept):
                kept.append(tag)
        pool[type_] += kept
    return pool


def pick(pool: Dict[str, List[Tag]], n: int, rng: random.Random) -> List[Tag]:
    """Type-balanced random subset of the pool (allocate: equal per type, capped by availability)."""
    quota = allocate({t: len(tags) for t, tags in pool.items()}, n)
    chosen = []
    for type_ in sorted(pool):
        chosen += rng.sample(sorted(pool[type_]), quota[type_])
    return sorted(chosen)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="train-labels-2k", help="output filter name")
    parser.add_argument("--types-from", default="test14-hard", help="filter whose scenario types and settings to use")
    parser.add_argument("--max-per-log-type", type=int, default=2)
    parser.add_argument("--min-gap-s", type=float, default=15.0)
    parser.add_argument("--db-dir", default=None, help="train DBs (default: $NUPLAN_DATA_ROOT/nuplan-v1.1/trainval)")
    parser.add_argument("--splitter", default=str(SPLITTER), help="nuplan-devkit splitter config with the train log list")
    args = parser.parse_args()

    db_dir = Path(args.db_dir or Path(os.environ["NUPLAN_DATA_ROOT"]) / "nuplan-v1.1" / "trainval")
    all_dbs = sorted(db_dir.glob("*.db"))
    logs = train_logs(Path(args.splitter))
    db_files = [db for db in all_dbs if db.stem in logs]
    assert db_files, f"none of the {len(all_dbs)} DBs in {db_dir} is a train-split log"

    source = yaml.safe_load((FILTER_DIR / f"{args.types_from}.yaml").read_text())
    types = set(source["scenario_types"])

    rng = random.Random(args.seed)
    tags = read_tags(db_files, types)
    pool = candidates_per_type(tags, args.max_per_log_type, args.min_gap_s, rng)
    chosen = pick(pool, args.n, rng)

    header = (f"# {args.n} train-split scenarios, balanced over the {args.types_from} types, seed {args.seed}, "
              f"<= {args.max_per_log_type} per log and type.\n"
              f"# Generated by scripts/make_train_filter.py; do not edit by hand. Never contains val / test logs.\n")
    write_filter(source, [(tag.token, tag.type) for tag in chosen], FILTER_DIR / f"{args.out}.yaml", header)

    print(f"{len(db_files)} train DBs ({len(all_dbs) - len(db_files)} non-train skipped) -> {len(tags)} tags; "
          f"picked {len(chosen)} from {len({tag.log for tag in chosen})} logs")
    print(f"{'scenario type':<58}{'pool':>6}{'picked':>8}")
    for t in sorted(types):
        print(f"{t:<58}{len(pool.get(t, [])):>6}{sum(1 for tag in chosen if tag.type == t):>8}")
    print(f"wrote {FILTER_DIR / args.out}.yaml")


if __name__ == "__main__":
    main()
