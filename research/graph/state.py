"""The state object every LangGraph node reads and writes.

Reducers matter more than they look. When several hop nodes run in the same
superstep, LangGraph merges their partial updates using the annotated reducer, so
`operator.add` on a list means "append what each branch produced" rather than
"last writer wins".
"""
from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from research.planning.schema import Plan
from research.retrieval.hybrid import Hit


class HopRecord(TypedDict):
    sq_id: str
    query: str            # the resolved query text actually sent to the retriever
    answer: str
    passage_ids: list[str]
    new_passage_ids: list[str]   # passages not seen in any earlier hop
    prompt_tokens: int
    completion_tokens: int


class ResearchState(TypedDict, total=False):
    question: str
    plan: Plan
    hops: Annotated[list[HopRecord], operator.add]
    seen_passages: Annotated[list[str], operator.add]
    evidence: Annotated[list[Hit], operator.add]
    hop_count: int
    done: bool
    final: dict


def new_state(question: str) -> ResearchState:
    return ResearchState(
        question=question,
        hops=[],
        seen_passages=[],
        evidence=[],
        hop_count=0,
        done=False,
    )
