"""Two bounds on the loop, added AFTER watching it run away.

1. A hard hop cap. Non-negotiable, applies regardless of what reflection thinks.
2. A diminishing-returns gate. If the last two hops brought in almost no passages
   the corpus has not seen, more hops will not help either — the agent is
   re-retrieving the same evidence and re-asking whether it is enough.

The second one is the interesting one. The cap stops a disaster; the gate stops
the waste that happens well before the cap.
"""
from __future__ import annotations

from research.graph.state import ResearchState

MAX_HOPS = 6                 # hard ceiling per question
STALL_WINDOW = 2             # look at the last N hops
STALL_THRESHOLD = 1          # fewer than this many NEW passages across the window = stalled


def hop_cap_reached(state: ResearchState) -> bool:
    return state.get("hop_count", 0) >= MAX_HOPS


def diminishing_returns(state: ResearchState) -> bool:
    """True when the recent hops stopped bringing in unseen evidence."""
    hops = state.get("hops", [])
    if len(hops) < STALL_WINDOW:
        return False
    recent = hops[-STALL_WINDOW:]
    new_evidence = sum(len(h["new_passage_ids"]) for h in recent)
    return new_evidence < STALL_THRESHOLD


def stop_reason(state: ResearchState) -> str | None:
    """The single place that decides the loop must end regardless of reflection.
    Returning a REASON rather than a bool is what makes the traces readable later."""
    if hop_cap_reached(state):
        return "hop_cap"
    if diminishing_returns(state):
        return "diminishing_returns"
    return None
