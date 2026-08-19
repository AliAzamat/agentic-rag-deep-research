"""Executing one hop: resolve placeholders, search, extract an answer from the hits."""
from __future__ import annotations

from research.graph.state import HopRecord, ResearchState
from research.llm.client import chat_json
from research.planning.resolve import resolve_text
from research.planning.schema import Plan, SubQuestion
from research.retrieval.hybrid import search

HOP_SYSTEM = """Answer the sub-question using ONLY the numbered passages.
Be terse — a name, a number, a short phrase. If the passages do not answer it,
return "" for answer.
Return ONLY JSON: {"answer": str, "citations": [int]}"""

HOP_K = 6


def run_hop(plan: Plan, sq: SubQuestion, seen: set[str]) -> HopRecord:
    """Run a single sub-question end to end and return its record.

    `seen` is every passage_id retrieved by earlier hops. The difference between
    what this hop retrieved and what was already seen is the hop's marginal value,
    and step 5 turns that number into a stopping rule.
    """
    query = resolve_text(plan, sq)
    hits = search(query, k=HOP_K)

    passages = "\n\n".join(
        f"[{i}] ({h.doc_title}, {h.published})\n{h.text}" for i, h in enumerate(hits, start=1)
    )
    payload, usage = chat_json(
        system=HOP_SYSTEM,
        user=f"Passages:\n{passages}\n\nSub-question: {query}",
    )

    cited = [hits[t - 1].passage_id for t in payload.get("citations", []) if 1 <= t <= len(hits)]
    retrieved = [h.passage_id for h in hits]

    return HopRecord(
        sq_id=sq.sq_id,
        query=query,
        answer=payload.get("answer", ""),
        passage_ids=retrieved,
        new_passage_ids=[pid for pid in retrieved if pid not in seen],
        prompt_tokens=usage["prompt_tokens"],
        completion_tokens=usage["completion_tokens"],
    )


def execute_batch(state: ResearchState) -> dict:
    """LangGraph node: run every currently-ready hop, then write the answers back
    onto the plan so the next batch's placeholders can resolve."""
    plan: Plan = state["plan"]
    seen = set(state.get("seen_passages", []))
    batch = plan.ready()

    records: list[HopRecord] = []
    fresh_hits = []
    for sq in batch:
        record = run_hop(plan, sq, seen)
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
