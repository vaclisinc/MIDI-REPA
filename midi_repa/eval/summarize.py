"""Aggregate per-seed scores into the design-study tables: mean +/- 95% CI over seeds.

The CI is Student-t over generation seeds (``t_{0.975, n-1} * sd / sqrt(n)``); it describes
sampling variability for a fixed trained model, not independent training runs.

    python -m midi_repa.eval.summarize scores.csv

``scores.csv`` has columns ``arm, seed, fad, clap`` (one row per arm and seed).
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

from scipy import stats

SEEDS = (42, 1234, 7777, 9999, 31415, 6798, 2024, 8888)  # the paper's eight seed sets


def mean_ci(values: list[float], level: float = 0.95) -> tuple[float, float]:
    n = len(values)
    if n < 2:
        return values[0], float("nan")
    half = stats.t.ppf(0.5 + level / 2, n - 1) * statistics.stdev(values) / math.sqrt(n)
    return statistics.fmean(values), half


def summarize(path: Path) -> dict[str, dict[str, tuple[float, float, int]]]:
    per_arm: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            for metric in ("fad", "clap"):
                if row.get(metric):
                    per_arm[row["arm"]][metric].append(float(row[metric]))
    return {arm: {m: (*mean_ci(v), len(v)) for m, v in metrics.items()}
            for arm, metrics in per_arm.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scores", type=Path)
    args = parser.parse_args()
    print(f"{'arm':28s} {'FAD (lower)':>20s} {'CLAP (higher)':>20s}  seeds")
    for arm, metrics in summarize(args.scores).items():
        cells = [f"{metrics[m][0]:.4f} +/- {metrics[m][1]:.4f}" if m in metrics else "-"
                 for m in ("fad", "clap")]
        n = max(v[2] for v in metrics.values())
        print(f"{arm:28s} {cells[0]:>20s} {cells[1]:>20s}  {n}")


if __name__ == "__main__":
    main()
