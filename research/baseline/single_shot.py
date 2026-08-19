"""The baseline every later measurement is compared against: one retrieval pass,
one generation call. No planning, no reflection, no second look."""
from __future__ import annotations

from dataclasses import dataclass

from research.llm.client import chat_json
from research.retrieval.hybrid import Hit, search

SYSTEM = """You are a research assistant. Answer the question using ONLY the numbered
passages below. If the passages do not contain enough information, set "answer" to
"I could not verify this from the provided sources" and leave "citations" empty.
Return ONLY JSON: {"answer": str, "citations": [int]}"""


@dataclass
class Answer:
    text: str
    citations: list[str]      # passage_ids actually used
    hits: list[Hit]           # everything that was retrieved, for auditing
    prompt_tokens: int
    completion_tokens: int


def _format_passages(hits: list[Hit]) -> str:
    return "\n\n".join(
        f"[{i}] ({hit.doc_title}, {hit.published})\n{hit.text}"
        for i, hit in enumerate(hits, start=1)
    )


def answer(question: str, k: int = 8) -> Answer:
    hits = search(question, k=k)
    payload, usage = chat_json(
        system=SYSTEM,
        user=f"Passages:\n{_format_passages(hits)}\n\nQuestion: {question}",
    )
    used = [hits[tag - 1].passage_id for tag in payload.get("citations", []) if 1 <= tag <= len(hits)]
    return Answer(
        text=payload.get("answer", ""),
        citations=used,
        hits=hits,
        prompt_tokens=usage["prompt_tokens"],
        completion_tokens=usage["completion_tokens"],
    )
