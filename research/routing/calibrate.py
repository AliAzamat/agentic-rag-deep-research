"""Check the router against the eval set's ground-truth kinds.

This prints a confusion matrix, not an accuracy number, because the two error types
have completely different costs and averaging them hides the tradeoff.
"""
from __future__ import annotations

from research.eval.dataset import load
from research.routing.router import route

"""Measured on the bounded eval run. Used to price each routing mistake."""
COST_SINGLE_SHOT = 0.0021
COST_AGENTIC = 0.0169


def main() -> None:
    matrix = {("multihop", "agentic"): 0, ("multihop", "single_shot"): 0,
              ("singlehop", "agentic"): 0, ("singlehop", "single_shot"): 0}

    for q in load():
        matrix[(q.kind, route(q.question).strategy)] += 1

    print("                routed->agentic  routed->single_shot")
    print(f"multihop  (60)       {matrix[('multihop', 'agentic')]:3d}                  {matrix[('multihop', 'single_shot')]:3d}")
    print(f"singlehop (60)       {matrix[('singlehop', 'agentic')]:3d}                  {matrix[('singlehop', 'single_shot')]:3d}")

    agentic_count = matrix[("multihop", "agentic")] + matrix[("singlehop", "agentic")]
    single_count = 120 - agentic_count
    blended = (agentic_count * COST_AGENTIC + single_count * COST_SINGLE_SHOT) / 120
    print(f"\nblended cost/question: ${blended:.4f}")
    print(f"vs all-agentic:        ${COST_AGENTIC:.4f}")
    print(f"vs all-single-shot:    ${COST_SINGLE_SHOT:.4f}")


if __name__ == "__main__":
    main()
