"""Grading. Exact match is useless for free-text answers, so we use an LLM judge
pinned to the gold answer, plus a mechanical evidence check.

An answer only counts as correct if the judge says it matches AND every gold passage
was actually retrieved. That second condition kills lucky guesses, which otherwise
inflate the agentic numbers most.
"""
from __future__ import annotations

from research.eval.dataset import EvalQuestion
from research.llm.client import chat_json

JUDGE_SYSTEM = """Compare a candidate answer to a gold answer. They match if the
candidate states the same facts, even with different wording. Extra correct detail is
fine; a missing or contradicted fact is not a match.
Return ONLY JSON: {"match": bool, "reason": str}"""


def grade(q: EvalQuestion, candidate: str, retrieved: set[str]) -> tuple[bool, str]:
    payload, _usage = chat_json(
        system=JUDGE_SYSTEM,
        user=f"Gold: {q.gold_answer}\n\nCandidate: {candidate}",
    )
    matched = bool(payload.get("match", False))
    grounded = all(pid in retrieved for pid in q.gold_passages)
    if matched and not grounded:
        return False, "answer matched but gold evidence was never retrieved"
    return matched, str(payload.get("reason", ""))
