"""Substitute resolved answers into a dependent sub-question's placeholders.

This is the actual mechanism that builds the bridge: {{sq-1}} becomes "Northwind
Data", and the second hop finally has a query that can reach the right passages.
"""
from __future__ import annotations

from research.planning.schema import PLACEHOLDER, Plan, SubQuestion


def resolve_text(plan: Plan, sq: SubQuestion) -> str:
    """Return sq.text with every {{sq-N}} replaced by that hop's answer."""

    def swap(match) -> str:
        dep = plan.by_id(match.group(1))
        if dep.answer is None:
            # Should be unreachable: Plan.ready() only returns hops whose deps are
            # resolved. Fail loudly rather than searching for literal braces.
            raise RuntimeError(f"{sq.sq_id} ran before {dep.sq_id} resolved")
        return dep.answer.strip()

    return PLACEHOLDER.sub(swap, sq.text)
