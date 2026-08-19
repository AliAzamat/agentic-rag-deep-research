"""The eval set: 120 questions, 60 multi-hop and 60 single-hop.

The single-hop half exists purely to keep us honest. It is very easy to build an
agentic system, measure it only on questions that need agency, and conclude it is
better than everything. The single-hop half is the control group.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

DATASET_PATH = Path("data/eval_questions.jsonl")


@dataclass(frozen=True)
class EvalQuestion:
    qid: str
    question: str
    kind: str                 # "multihop" | "singlehop"
    gold_answer: str
    gold_passages: list[str]  # passage_ids that must appear for the answer to count


def load(kind: str | None = None) -> list[EvalQuestion]:
    rows = [json.loads(line) for line in DATASET_PATH.read_text().splitlines() if line.strip()]
    questions = [
        EvalQuestion(
            qid=r["qid"],
            question=r["question"],
            kind=r["kind"],
            gold_answer=r["gold_answer"],
            gold_passages=r["gold_passages"],
        )
        for r in rows
    ]
    return [q for q in questions if kind is None or q.kind == kind]
