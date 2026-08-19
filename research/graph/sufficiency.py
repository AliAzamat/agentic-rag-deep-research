"""The reflection step: given the evidence so far, can we answer the question yet?

The critical design choice is the SHAPE of the output. A confidence score tells you
nothing actionable. A named missing fact is directly executable as the next hop.
"""
from __future__ import annotations

from dataclasses import dataclass

from research.graph.state import ResearchState
from research.llm.client import chat_json
from research.planning.schema import SubQuestion

SYSTEM = """You judge whether the gathered evidence is sufficient to answer the
original question.

You are given the question, the sub-questions answered so far with their answers,
and the evidence passages.

If every part of the question is supported by evidence, return sufficient: true.
If not, return sufficient: false AND write ONE concrete follow-up sub-question that
would close the largest remaining gap. The follow-up must be answerable by a single
document search — not "find more context", but a specific factual question.

Return ONLY JSON:
{"sufficient": bool, "missing": str, "follow_up": str}
Use "" for missing and follow_up when sufficient is true."""


@dataclass
class Verdict:
    sufficient: bool
    missing: str
    follow_up: str
    prompt_tokens: int
    completion_tokens: int


def _evidence_block(state: ResearchState, limit: int = 12) -> str:
    """Deduplicate evidence by passage_id, keep the highest-scoring copy, and cap it.

    Without the cap this prompt grows with every hop, so a runaway loop would also
    be a runaway token bill — the two failure modes compound.
    """
    best: dict[str, object] = {}
    for hit in state.get("evidence", []):
        prior = best.get(hit.passage_id)
        if prior is None or hit.score > prior.score:  # type: ignore[union-attr]
            best[hit.passage_id] = hit
    ranked = sorted(best.values(), key=lambda h: h.score, reverse=True)[:limit]  # type: ignore[attr-defined]
    return "\n\n".join(f"({h.doc_title}) {h.text}" for h in ranked)  # type: ignore[attr-defined]


def check(state: ResearchState) -> Verdict:
    answered = "\n".join(
        f"- {sq.sq_id} {sq.text} -> {sq.answer}"
        for sq in state["plan"].sub_questions
        if sq.is_resolved
    )
    payload, usage = chat_json(
        system=SYSTEM,
        user=(
            f"Original question: {state['question']}\n\n"
            f"Answered so far:\n{answered}\n\n"
            f"Evidence:\n{_evidence_block(state)}"
        ),
    )
    return Verdict(
        sufficient=bool(payload.get("sufficient", False)),
        missing=str(payload.get("missing", "")),
        follow_up=str(payload.get("follow_up", "")),
        prompt_tokens=usage["prompt_tokens"],
        completion_tokens=usage["completion_tokens"],
    )


def reflect_node(state: ResearchState) -> dict:
    """LangGraph node. When the verdict is insufficient, append the follow-up to the
    plan as a new, dependency-free sub-question so the executor picks it up."""
    verdict = check(state)
    if verdict.sufficient or not verdict.follow_up:
        return {"done": True}

    plan = state["plan"]
    plan.sub_questions.append(
        SubQuestion(sq_id=f"sq-{len(plan.sub_questions) + 1}", text=verdict.follow_up)
    )
    return {"done": False}
