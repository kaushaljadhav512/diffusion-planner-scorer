"""Build a fixed, type-balanced subset of a scenario filter (task 1.9).

Looks up the scenario type of every token of the source filter in the nuPlan DBs (scenario_tag table), picks
the same number of scenarios per type with a fixed seed, and writes a filter with the source's other settings.
Needs only the DBs (no nuplan-devkit, no GPU).

    python scripts/make_subset_filter.py                                  # test14-hard -> test14-hard-100
    python scripts/make_subset_filter.py --n 50 --seed 1 --out test14-hard-50
    bash scripts/run_sim.sh dp_k test14-hard-100 nr                       # then use it as a split

A scenario is a (token, type) pair: nuPlan's builder keeps every tag of a listed token whose type is listed, so a
token tagged with two listed types runs twice. The subset is chosen so the written filter yields exactly --n.
"""
import argparse
import os
import random
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import yaml

FILTER_DIR = Path(__file__).resolve().parent.parent / "diffusion_planner" / "config" / "scenario_filter"


def tagged_scenarios(db_files: List[Path], tokens: List[str], types: Set[str]) -> List[Tuple[str, str]]:
    """
    Returns:
        sorted (token, type) pairs of every tag of the given tokens whose type is in types
    """
    blobs = [bytes.fromhex(t) for t in tokens]
    query = f"SELECT lidar_pc_token, type FROM scenario_tag WHERE lidar_pc_token IN ({','.join('?' * len(blobs))})"
    pairs = set()
    for db in db_files:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            pairs.update((token.hex(), type_) for token, type_ in con.execute(query, blobs) if type_ in types)
        finally:
            con.close()
    return sorted(pairs)


def allocate(available: Dict[str, int], n: int) -> Dict[str, int]:
    """Equal count per type, capped by what each type has; leftovers go one each to the largest types (then by name)."""
    if n > sum(available.values()):
        raise ValueError(f"asked for {n} scenarios, only {sum(available.values())} exist")
    quota = {t: 0 for t in available}
    order = sorted(available, key=lambda t: (-available[t], t))
    for _ in range(n):
        open_types = [t for t in order if quota[t] < available[t]]
        lowest = min(quota[t] for t in open_types)
        quota[next(t for t in open_types if quota[t] == lowest)] += 1
    return quota


def pick(pairs: List[Tuple[str, str]], n: int, seed: int) -> List[Tuple[str, str]]:
    """
    Type-balanced random subset of the (token, type) pairs. Only tokens with a single listed tag are picked, so
    selecting a token selects exactly one scenario; tokens with several listed tags are left out.

    Returns:
        sorted (token, type) pairs, len n
    """
    tags_per_token = defaultdict(int)
    for token, _ in pairs:
        tags_per_token[token] += 1

    by_type = defaultdict(list)
    for token, type_ in pairs:
        if tags_per_token[token] == 1:
            by_type[type_].append(token)

    quota = allocate({t: len(tokens) for t, tokens in by_type.items()}, n)
    rng = random.Random(seed)
    chosen = []
    for type_ in sorted(by_type): # fixed order so the seed fully determines the subset
        chosen += [(token, type_) for token in rng.sample(sorted(by_type[type_]), quota[type_])]
    return sorted(chosen)


def write_filter(source: dict, chosen: List[Tuple[str, str]], path: Path, header: str) -> None:
    config = dict(source)
    config["scenario_types"] = sorted({type_ for _, type_ in chosen})
    config["scenario_tokens"] = sorted(token for token, _ in chosen)
    with open(path, "w") as f:
        f.write(header)
        yaml.safe_dump(config, f, sort_keys=False, default_flow_style=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="test14-hard", help="scenario filter to subsample, by name (default: %(default)s)")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None, help="output filter name (default: <source>-<n>)")
    parser.add_argument("--db-dir", default=None,
                        help="dir of the DBs holding the source's tokens (default: $NUPLAN_DATA_ROOT/nuplan-v1.1/test)")
    args = parser.parse_args()

    db_dir = Path(args.db_dir or Path(os.environ["NUPLAN_DATA_ROOT"]) / "nuplan-v1.1" / "test")
    db_files = sorted(db_dir.glob("*.db"))
    assert db_files, f"no .db files in {db_dir} (source setup/config.env, or pass --db-dir)"

    source_path = FILTER_DIR / f"{args.source}.yaml"
    source = yaml.safe_load(source_path.read_text())
    tokens, types = [str(t) for t in source["scenario_tokens"]], set(source["scenario_types"])
    bad = [t for t in tokens if len(t) != 16 or any(c not in "0123456789abcdef" for c in t)]
    assert not bad, f"tokens that are not 16 hex chars (unquoted in {source_path.name}?): {bad[:3]}"

    pairs = tagged_scenarios(db_files, tokens, types)
    missing = set(tokens) - {token for token, _ in pairs}
    assert not missing, f"{len(missing)} of {len(tokens)} tokens not found in {db_dir}, e.g. {sorted(missing)[:3]}"

    chosen = pick(pairs, args.n, args.seed)

    # Sanity check: what nuPlan's builder will select from the written filter (every listed tag of every token)
    chosen_tokens = {token for token, _ in chosen}
    selected = [(token, type_) for token, type_ in pairs if token in chosen_tokens and type_ in {t for _, t in chosen}]
    assert sorted(selected) == chosen, "written filter would select different scenarios than chosen"

    out = args.out or f"{args.source}-{args.n}"
    header = (f"# {args.n} scenarios of {args.source}, balanced per type, seed {args.seed}.\n"
              f"# Generated by scripts/make_subset_filter.py; do not edit by hand.\n")
    write_filter(source, chosen, FILTER_DIR / f"{out}.yaml", header)

    count = lambda items: {t: sum(1 for _, x in items if x == t) for t in sorted(types)}
    total, picked = count(pairs), count(chosen)
    print(f"{len(tokens)} tokens -> {len(pairs)} scenarios in {len(db_files)} DBs; picked {len(chosen)}")
    print(f"{'scenario type':<58}{'all':>5}{'picked':>8}")
    for t in sorted(types):
        print(f"{t:<58}{total[t]:>5}{picked[t]:>8}")
    print(f"wrote {FILTER_DIR / out}.yaml")


if __name__ == "__main__":
    main()
