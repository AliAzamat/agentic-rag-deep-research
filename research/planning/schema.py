"""The plan is the contract between the planner and the executor.

A SubQuestion can name other sub-questions it depends on. That single field is what
turns a flat list of searches into a graph, and it is the difference between an
agent that can answer a bridge question and one that cannot.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

PLACEHOLDER = re.compile(r"\{\{(sq-\d+)\}\}")


@dataclass
class SubQuestion:
    sq_id: str                      # "sq-1", "sq-2", ...
    text: str                       # may contain {{sq-1}} placeholders
    depends_on: list[str] = field(default_factory=list)
    answer: str | None = None       # filled in once this hop resolves
    evidence: list[str] = field(default_factory=list)  # passage_ids

    @property
    def is_resolved(self) -> bool:
        return self.answer is not None

    def placeholders(self) -> list[str]:
        return PLACEHOLDER.findall(self.text)


@dataclass
class Plan:
    question: str
    sub_questions: list[SubQuestion]

    def by_id(self, sq_id: str) -> SubQuestion:
        for sq in self.sub_questions:
            if sq.sq_id == sq_id:
                return sq
        raise KeyError(sq_id)

    def ready(self) -> list[SubQuestion]:
        """Sub-questions that are unresolved and whose dependencies are all resolved.
        This list IS the next parallel batch — everything in it can run at once."""
        return [
            sq
            for sq in self.sub_questions
            if not sq.is_resolved and all(self.by_id(dep).is_resolved for dep in sq.depends_on)
        ]

    def unresolved(self) -> list[SubQuestion]:
        return [sq for sq in self.sub_questions if not sq.is_resolved]
