"""Langfuse instrumentation.

One question is one trace. Every hop, reflection, and synthesis call is a span inside
it. The reason this matters more for an agent than for a pipeline: with a variable
number of LLM calls per request, an average cost per CALL tells you nothing. You need
cost per QUESTION, which means the spans have to roll up to a trace.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Iterator

from langfuse import Langfuse

PRICE_IN_PER_MTOK = 0.09
PRICE_OUT_PER_MTOK = 0.36

_client = Langfuse(
    public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
    secret_key=os.environ["LANGFUSE_SECRET_KEY"],
    host=os.environ.get("LANGFUSE_HOST", "http://localhost:3000"),
)


def cost_usd(prompt_tokens: int, completion_tokens: int) -> float:
    return (
        prompt_tokens / 1_000_000 * PRICE_IN_PER_MTOK
        + completion_tokens / 1_000_000 * PRICE_OUT_PER_MTOK
    )


@contextmanager
def question_trace(question: str, qid: str | None = None) -> Iterator[Any]:
    """One trace per question. Everything downstream attaches to it."""
    trace = _client.trace(
        name="research_question",
        input={"question": question},
        metadata={"qid": qid} if qid else None,
        tags=["agentic-rag"],
    )
    try:
        yield trace
    finally:
        _client.flush()


@contextmanager
def span(trace: Any, name: str, **metadata: Any) -> Iterator[Any]:
    """A named child span. Node code calls span.update(...) to record usage."""
    child = trace.span(name=name, metadata=metadata or None)
    try:
        yield child
    finally:
        child.end()


def record_generation(
    parent: Any, name: str, model: str, prompt_tokens: int, completion_tokens: int, **meta: Any
) -> None:
    """Attach an LLM call to a span with its usage AND its dollar cost.

    Recording the cost explicitly rather than deriving it in the dashboard means the
    number in Langfuse always matches the number in the eval report — one formula,
    one source, no drift between two places that both claim to know the price.
    """
    parent.generation(
        name=name,
        model=model,
        usage={"input": prompt_tokens, "output": completion_tokens, "unit": "TOKENS"},
        metadata={**meta, "cost_usd": cost_usd(prompt_tokens, completion_tokens)},
    ).end()
