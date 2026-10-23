#!/usr/bin/env python3
"""Plot per-scenario-type CLS (NR vs R) from a summarize_results.py --by-type text dump.

    python scripts/plot_baseline.py results/dp_test14hard.txt report/figures/dp_by_type.pdf
"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

NR_COLOR, R_COLOR = "#2a78d6", "#eb6834"  # categorical slots 1-2


def parse_by_type(path):
    rows, in_table = {}, False
    for line in open(path):
        if line.startswith("=== CLS by scenario type"):
            in_table = True
            continue
        if not in_table:
            continue
        parts = line.split()
        if len(parts) == 3 and parts[0] not in ("challenge",):
            try:
                rows[parts[0]] = (float(parts[1]), float(parts[2]))
            except ValueError:
                pass
        elif line.startswith("wrote") or line.startswith("==="):
            break
    return rows


def main(src, dst):
    rows = parse_by_type(src)
    names = sorted(rows, key=lambda k: rows[k][0])
    nr = [rows[k][0] for k in names]
    r = [rows[k][1] for k in names]
    labels = [k.replace("starting_straight_traffic_light_intersection_traversal", "straight_tl_intersection")
              .replace("_", " ") for k in names]

    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    y = range(len(names))
    h = 0.38
    ax.barh([i + h / 2 + 0.01 for i in y], nr, height=h, color=NR_COLOR, label="Non-reactive (NR)")
    ax.barh([i - h / 2 - 0.01 for i in y], r, height=h, color=R_COLOR, label="Reactive (R)")
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Closed-loop score (CLS)")
    ax.grid(axis="x", color="#dddddd", linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(dst)
    print(f"wrote {dst}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
