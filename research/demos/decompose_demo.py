"""Show the plan for the bridge question, including the dependency edge."""
from __future__ import annotations

from research.demos.single_shot_fails import QUESTION
from research.planning.decompose import decompose

if __name__ == "__main__":
    plan, usage = decompose(QUESTION)
    for sq in plan.sub_questions:
        deps = f"  <- {', '.join(sq.depends_on)}" if sq.depends_on else ""
        print(f"{sq.sq_id}: {sq.text}{deps}")
    print(f"\nfirst parallel batch: {[sq.sq_id for sq in plan.ready()]}")
    print(f"planner tokens: {usage['prompt_tokens']} + {usage['completion_tokens']}")
