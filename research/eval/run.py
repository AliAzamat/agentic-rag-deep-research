"""The eval harness. Runs a question set through a strategy and reports accuracy,
cost, latency, and search count.

Now supports a third strategy, `routed`, which is the blend the router produces and
the only number that describes the system we would actually deploy.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass

from research.baseline.single_shot import answer as single_shot_answer
from research.eval.dataset import load
from research.eval.grade import grade
from research.graph.build import research
from research.routing.router import route

PRICE_IN_PER_MTOK = 0.09
PRICE_OUT_PER_MTOK = 0.36


@dataclass
class Row:
    qid: str
    kind: str
    strategy: str
    correct: bool
    searches: int
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    latency_s: float
    note: str


def cost_of(prompt_tokens: int, completion_tokens: int) -> float:
    return (
        prompt_tokens / 1_000_000 * PRICE_IN_PER_MTOK
        + completion_tokens / 1_000_000 * PRICE_OUT_PER_MTOK
    )


def run_single_shot(q) -> Row:
    started = time.perf_counter()
    result = single_shot_answer(q.question)
    correct, note = grade(q, result.text, {h.passage_id for h in result.hits})
    return Row(
        qid=q.qid, kind=q.kind, strategy="single_shot", correct=correct, searches=1,
        prompt_tokens=result.prompt_tokens, completion_tokens=result.completion_tokens,
        cost_usd=cost_of(result.prompt_tokens, result.completion_tokens),
        latency_s=time.perf_counter() - started, note=note,
    )


def run_agentic(q) -> Row:
    started = time.perf_counter()
    state = research(q.question)
    hops = state.get("hops", [])
    prompt_tokens = sum(h["prompt_tokens"] for h in hops)
    completion_tokens = sum(h["completion_tokens"] for h in hops)
    retrieved = {pid for h in hops for pid in h["passage_ids"]}
    correct, note = grade(q, state.get("final", {}).get("answer", ""), retrieved)
    return Row(
        qid=q.qid, kind=q.kind, strategy="agentic", correct=correct, searches=len(hops),
        prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
        cost_usd=cost_of(prompt_tokens, completion_tokens),
        latency_s=time.perf_counter() - started, note=note,
    )


def run_routed(q) -> Row:
    """Route, then run the chosen strategy. The router's own planner tokens are
    charged to the row, because in production you pay for the decision too."""
    decision = route(q.question)
    row = run_single_shot(q) if decision.strategy == "single_shot" else run_agentic(q)
    row.strategy = "routed"
    row.prompt_tokens += decision.prompt_tokens
    row.completion_tokens += decision.completion_tokens
    row.cost_usd = cost_of(row.prompt_tokens, row.completion_tokens)
    row.note = f"{decision.strategy}: {decision.reason}"
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", choices=["multihop", "singlehop", "all"], default="all")
    parser.add_argument("--strategy", choices=["single_shot", "agentic", "routed", "both"], default="both")
    parser.add_argument("--budget", type=int, default=900, help="hard cap on total searches")
    parser.add_argument("--out", default="results/eval.jsonl")
    args = parser.parse_args()

    questions = load(None if args.set == "all" else args.set)
    strategies = ["single_shot", "agentic"] if args.strategy == "both" else [args.strategy]
    runners = {"single_shot": run_single_shot, "agentic": run_agentic, "routed": run_routed}

    rows: list[Row] = []
    spent = 0
    for q in questions:
        for strategy in strategies:
            if spent >= args.budget:
                print(f"BUDGET EXHAUSTED after {len(rows)} rows ({spent} searches)")
                _write(rows, args.out)
                return
            row = runners[strategy](q)
            spent += row.searches
            rows.append(row)
            print(f"{row.qid} {row.strategy:11s} correct={row.correct} searches={row.searches:2d} spent={spent}")
    _write(rows, args.out)


def _write(rows: list[Row], path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(asdict(row)) + "\n")
    print(f"wrote {len(rows)} rows to {path}")


if __name__ == "__main__":
    main()
