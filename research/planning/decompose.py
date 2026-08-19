"""Ask the model to write the plan, then validate it hard before trusting it.

A planner that emits a cycle or a dangling dependency will hang the executor
forever, so validation is not optional politeness — it is a liveness guarantee.
"""
from __future__ import annotations

from research.llm.client import chat_json
from research.planning.schema import Plan, SubQuestion

SYSTEM = """You decompose a research question into the minimum set of sub-questions
needed to answer it.

Rules:
- Emit between 1 and 4 sub-questions. One is correct when the question is already atomic.
- If a sub-question needs the ANSWER to an earlier one, write that earlier id as a
  placeholder inside the text, like: "Which EU region did {{sq-1}} choose?"
- Any sub-question containing {{sq-N}} MUST list "sq-N" in its depends_on.
- Never create a cycle. Ids are sq-1, sq-2, ... in order.

Return ONLY JSON:
{"sub_questions": [{"id": "sq-1", "text": str, "depends_on": [str]}]}"""


class InvalidPlan(Exception):
    """Raised when the model returns a plan the executor cannot safely run."""


def _validate(plan: Plan) -> None:
    ids = {sq.sq_id for sq in plan.sub_questions}
    for sq in plan.sub_questions:
        for dep in sq.depends_on:
            if dep not in ids:
                raise InvalidPlan(f"{sq.sq_id} depends on unknown {dep}")
            if dep == sq.sq_id:
                raise InvalidPlan(f"{sq.sq_id} depends on itself")
        # Every placeholder must be declared. Otherwise the executor would happily
        # run a hop whose text still says "{{sq-1}}" and search for literal braces.
        for ref in sq.placeholders():
            if ref not in sq.depends_on:
                raise InvalidPlan(f"{sq.sq_id} references {ref} without declaring the dependency")

    # Cycle check by repeated peeling: anything left over sits in a cycle.
    resolved: set[str] = set()
    remaining = list(plan.sub_questions)
    while remaining:
        peelable = [sq for sq in remaining if all(d in resolved for d in sq.depends_on)]
        if not peelable:
            raise InvalidPlan(f"cycle among {[sq.sq_id for sq in remaining]}")
        resolved.update(sq.sq_id for sq in peelable)
        remaining = [sq for sq in remaining if sq.sq_id not in resolved]


def decompose(question: str) -> tuple[Plan, dict[str, int]]:
    payload, usage = chat_json(system=SYSTEM, user=f"Question: {question}")
    plan = Plan(
        question=question,
        sub_questions=[
            SubQuestion(
                sq_id=item["id"],
                text=item["text"],
                depends_on=list(item.get("depends_on", [])),
            )
            for item in payload.get("sub_questions", [])
        ],
    )
    if not plan.sub_questions:
        # Degenerate plans are common on short questions. Treat the whole question
        # as one hop rather than erroring — an atomic question is a valid plan.
        plan.sub_questions = [SubQuestion(sq_id="sq-1", text=question)]
    _validate(plan)
    return plan, usage
