"""Pull the traces back out and aggregate cost per question.

The dashboard shows you averages. This script answers the question that actually
drives decisions: which questions are expensive, and WHY did they stop?
"""
from __future__ import annotations

import os
from collections import Counter

from langfuse import Langfuse

_client = Langfuse(
    public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
    secret_key=os.environ["LANGFUSE_SECRET_KEY"],
    host=os.environ.get("LANGFUSE_HOST", "http://localhost:3000"),
)


def main(limit: int = 200) -> None:
    traces = _client.fetch_traces(tags=["agentic-rag"], limit=limit).data

    stop_reasons: Counter[str] = Counter()
    rows: list[tuple[str, float, int, str]] = []

    for trace in traces:
        meta = trace.metadata or {}
        observations = _client.fetch_observations(trace_id=trace.id).data
        total = sum((obs.metadata or {}).get("cost_usd", 0.0) for obs in observations)
        reason = str(meta.get("stop_reason", "unknown"))
        stop_reasons[reason] += 1
        rows.append((meta.get("qid") or trace.id, total, int(meta.get("hops", 0)), reason))

    rows.sort(key=lambda r: r[1], reverse=True)
    print("most expensive questions:")
    for qid, cost, hops, reason in rows[:10]:
        print(f"  {qid:8s} ${cost:.4f}  hops={hops}  stopped={reason}")

    print("\nstop reasons:")
    for reason, count in stop_reasons.most_common():
        print(f"  {reason:22s} {count:4d}  ({count / max(len(rows), 1):.0%})")


if __name__ == "__main__":
    main()
