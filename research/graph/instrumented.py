"""The graph nodes, wrapped in tracing.

Kept in a separate module from build.py on purpose: the untraced graph stays readable
as the reference implementation, and the traced one is what runs in the eval and in
production. Neither file has to pretend to be the other.
"""
from __future__ import annotations

from research.graph.budget import stop_reason
from research.graph.hops import run_hop
from research.graph.state import HopRecord, ResearchState
from research.graph.sufficiency import check
from research.llm.client import MODEL
from research.observability.tracing import record_generation, span
from research.planning.decompose import decompose
from research.planning.schema import SubQuestion
from research.retrieval.hybrid import search
from research.synthesis.synthesize import _dedup_evidence, synthesize
from research.synthesis.verify import verify

HOP_K = 6


def traced_plan(state: ResearchState, trace) -> dict:
    with span(trace, "plan") as sp:
        plan, usage = decompose(state["question"])
        record_generation(sp, "decompose", MODEL, usage["prompt_tokens"], usage["completion_tokens"],
                          hop_count=len(plan.sub_questions))
    return {"plan": plan}


def traced_execute(state: ResearchState, trace) -> dict:
    plan = state["plan"]
    seen = set(state.get("seen_passages", []))
    batch = plan.ready()

    records: list[HopRecord] = []
    fresh_hits = []
    for sq in batch:
        with span(trace, f"hop:{sq.sq_id}", depends_on=sq.depends_on) as sp:
            record = run_hop(plan, sq, seen)
            record_generation(
                sp, "hop_answer", MODEL, record["prompt_tokens"], record["completion_tokens"],
                query=record["query"],
                retrieved=len(record["passage_ids"]),
                new_passages=len(record["new_passage_ids"]),
            )
        sq.answer = record["answer"] or None
        sq.evidence = record["passage_ids"]
        records.append(record)
        seen.update(record["passage_ids"])
        fresh_hits.extend(search(record["query"], k=HOP_K))

    return {
        "hops": records,
        "seen_passages": [pid for r in records for pid in r["new_passage_ids"]],
        "evidence": fresh_hits,
        "hop_count": state.get("hop_count", 0) + len(records),
    }


def traced_reflect(state: ResearchState, trace) -> dict:
    with span(trace, "reflect", hop_count=state.get("hop_count", 0)) as sp:
        verdict = check(state)
        record_generation(sp, "sufficiency", MODEL, verdict.prompt_tokens, verdict.completion_tokens,
                          sufficient=verdict.sufficient, missing=verdict.missing)
    if verdict.sufficient or not verdict.follow_up:
        return {"done": True}
    plan = state["plan"]
    plan.sub_questions.append(
        SubQuestion(sq_id=f"sq-{len(plan.sub_questions) + 1}", text=verdict.follow_up)
    )
    return {"done": False}


def traced_synthesize(state: ResearchState, trace) -> dict:
    with span(trace, "synthesize", stop_reason=stop_reason(state) or "sufficient") as sp:
        final = verify(synthesize(state), _dedup_evidence(state))
        record_generation(sp, "synthesis", MODEL, final.prompt_tokens, final.completion_tokens,
                          claims=len(final.claims), unverified=len(final.unverified))
    trace.update(
        output={"answer": final.render()},
        metadata={
            "hops": state.get("hop_count", 0),
            "stop_reason": stop_reason(state) or "sufficient",
            "unverified_claims": len(final.unverified),
        },
    )
    return {"final": {"answer": final.render(), "unverified_count": len(final.unverified)}, "done": True}
