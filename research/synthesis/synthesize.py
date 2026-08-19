"""Final synthesis. The output is a list of CLAIMS, not a paragraph.

Citing a whole paragraph is theater: a reader cannot tell which of five sentences a
citation supports. Forcing the model to emit discrete claims, each with its own
passage ids, makes every sentence independently checkable — including the ones the
evidence doesn't support.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from research.graph.state import ResearchState
from research.llm.client import chat_json
from research.retrieval.hybrid import Hit

SYSTEM = """You write a final research answer as a list of discrete claims.

Rules:
- Each claim is ONE checkable statement.
- Each claim cites the tag numbers of the passages that support it.
- If the question asks for something the passages do not support, still emit a claim
  for it with an empty citation list and supported: false. Do NOT silently drop it,
  and do NOT assert it as fact.
- Never state anything not present in the passages.

Return ONLY JSON:
{"claims": [{"text": str, "citations": [int], "supported": bool}]}"""


@dataclass
class Claim:
    text: str
    passage_ids: list[str] = field(default_factory=list)
    supported: bool = True


@dataclass
class FinalAnswer:
    claims: list[Claim]
    prompt_tokens: int
    completion_tokens: int

    @property
    def unverified(self) -> list[Claim]:
        return [c for c in self.claims if not c.supported]

    def render(self) -> str:
        lines: list[str] = []
        for claim in self.claims:
            if claim.supported:
                tags = ", ".join(claim.passage_ids)
                lines.append(f"{claim.text} [{tags}]")
            else:
                lines.append(f"{claim.text} — I could not verify this from the sources.")
        return "\n".join(lines)


def _dedup_evidence(state: ResearchState, limit: int = 16) -> list[Hit]:
    best: dict[str, Hit] = {}
    for hit in state.get("evidence", []):
        prior = best.get(hit.passage_id)
        if prior is None or hit.score > prior.score:
            best[hit.passage_id] = hit
    return sorted(best.values(), key=lambda h: h.score, reverse=True)[:limit]


def synthesize(state: ResearchState) -> FinalAnswer:
    evidence = _dedup_evidence(state)
    passages = "\n\n".join(
        f"[{i}] ({h.doc_title}, {h.published})\n{h.text}" for i, h in enumerate(evidence, start=1)
    )
    hop_summary = "\n".join(
        f"- {sq.text} -> {sq.answer}" for sq in state["plan"].sub_questions if sq.is_resolved
    )
    payload, usage = chat_json(
        system=SYSTEM,
        user=(
            f"Question: {state['question']}\n\n"
            f"Sub-answers:\n{hop_summary}\n\n"
            f"Passages:\n{passages}"
        ),
    )

    claims: list[Claim] = []
    for item in payload.get("claims", []):
        tags = [t for t in item.get("citations", []) if 1 <= t <= len(evidence)]
        claims.append(
            Claim(
                text=str(item.get("text", "")),
                passage_ids=[evidence[t - 1].passage_id for t in tags],
                supported=bool(item.get("supported", True)) and bool(tags),
            )
        )
    return FinalAnswer(claims=claims, prompt_tokens=usage["prompt_tokens"],
                       completion_tokens=usage["completion_tokens"])
