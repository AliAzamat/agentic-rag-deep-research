"""Decide whether a question needs the agentic loop.

The signal we want is decomposability: does answering this require a fact you must
first look up? The planner already answers that question as a side effect, and it is
by far the cheapest classifier available because we would run it anyway.
"""
from __future__ import annotations

from dataclasses import dataclass

from research.planning.decompose import InvalidPlan, decompose
from research.planning.schema import Plan

"""Calibrated in research/routing/calibrate.py against the 120-question set.
Errors are asymmetric: routing a multi-hop question to single-shot costs ~40
accuracy points, while routing a single-hop question to agentic costs ~8x money.
At our volumes the accuracy loss hurts more, so the threshold leans agentic."""
MIN_HOPS_FOR_AGENTIC = 2


@dataclass
class Route:
    strategy: str          # "single_shot" | "agentic"
    reason: str
    plan: Plan | None      # reused by the agentic path so planning is never paid twice
    prompt_tokens: int
    completion_tokens: int


def route(question: str) -> Route:
    try:
        plan, usage = decompose(question)
    except InvalidPlan as exc:
        # A plan we can't execute is not a reason to fail the request. Fall back to
        # the strategy that needs no plan at all.
        return Route("single_shot", f"invalid plan: {exc}", None, 0, 0)

    has_dependency = any(sq.depends_on for sq in plan.sub_questions)
    hop_count = len(plan.sub_questions)

    if has_dependency:
        strategy, reason = "agentic", "plan contains a dependent hop"
    elif hop_count >= MIN_HOPS_FOR_AGENTIC:
        strategy, reason = "agentic", f"plan has {hop_count} independent hops"
    else:
        strategy, reason = "single_shot", "plan is a single atomic hop"

    return Route(
        strategy=strategy,
        reason=reason,
        plan=plan,
        prompt_tokens=usage["prompt_tokens"],
        completion_tokens=usage["completion_tokens"],
    )
