"""Build the comparison table from the eval rows. No hand-typed numbers.

Every figure in FINDINGS.md comes out of this script, so the report cannot drift
away from the measurements the way a hand-maintained table always eventually does.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

RESULTS = Path("results/eval.jsonl")


@dataclass
class Cell:
    n: int = 0
    correct: int = 0
    searches: int = 0
    cost: float = 0.0
    latency: float = 0.0

    def add(self, row: dict) -> None:
        self.n += 1
        self.correct += int(row["correct"])
        self.searches += row["searches"]
        self.cost += row["cost_usd"]
        self.latency += row["latency_s"]

    @property
    def accuracy(self) -> float:
        return self.correct / self.n if self.n else 0.0

    @property
    def cost_per_question(self) -> float:
        return self.cost / self.n if self.n else 0.0

    @property
    def cost_per_correct(self) -> float:
        return self.cost / self.correct if self.correct else float("inf")


def main() -> None:
    cells: dict[tuple[str, str], Cell] = defaultdict(Cell)
    for line in RESULTS.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        cells[(row["kind"], row["strategy"])].add(row)

    header = f"{'set':10s} {'strategy':12s} {'n':>4s} {'acc':>7s} {'hops':>6s} {'$/q':>9s} {'$/correct':>11s} {'p50 s':>7s}"
    print(header)
    print("-" * len(header))
    for (kind, strategy), cell in sorted(cells.items()):
        print(
            f"{kind:10s} {strategy:12s} {cell.n:4d} {cell.accuracy:6.1%} "
            f"{cell.searches / cell.n:6.1f} {cell.cost_per_question:9.4f} "
            f"{cell.cost_per_correct:11.4f} {cell.latency / cell.n:7.1f}"
        )

    agentic = cells.get(("singlehop", "agentic"))
    single = cells.get(("singlehop", "single_shot"))
    if agentic and single and single.cost_per_question:
        multiple = agentic.cost_per_question / single.cost_per_question
        delta = agentic.accuracy - single.accuracy
        print(
            f"\nON SINGLE-HOP QUESTIONS agentic costs {multiple:.1f}x "
            f"for an accuracy change of {delta:+.1%}."
        )


if __name__ == "__main__":
    main()
