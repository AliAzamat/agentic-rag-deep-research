"""Mechanical post-check on the model's own citation claims.

The model asserts which passages support each claim. We do not take that on faith:
a claim whose cited passages share almost no content words with it gets demoted to
unsupported. This is a cheap lexical guard, not a proof — but it catches the common
case where the model cites a plausible-looking neighbouring passage.
"""
from __future__ import annotations

import re

from research.retrieval.hybrid import Hit
from research.synthesis.synthesize import Claim, FinalAnswer

STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "for", "and", "or", "is", "was", "were",
    "that", "with", "on", "at", "by", "it", "its", "as", "from", "this", "their",
}
MIN_OVERLAP = 0.30


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9\-]+", text.lower()) if w not in STOPWORDS and len(w) > 2}


def verify(final: FinalAnswer, evidence: list[Hit]) -> FinalAnswer:
    by_id = {h.passage_id: h for h in evidence}
    for claim in final.claims:
        if not claim.supported:
            continue
        claim_words = _content_words(claim.text)
        if not claim_words:
            continue
        cited_words: set[str] = set()
        for pid in claim.passage_ids:
            hit = by_id.get(pid)
            if hit is not None:
                cited_words |= _content_words(hit.text)
        overlap = len(claim_words & cited_words) / len(claim_words)
        if overlap < MIN_OVERLAP:
            # Demote rather than delete: the reader should see the claim was made
            # and that we could not stand behind it.
            claim.supported = False
            claim.passage_ids = []
    return final


def demoted(final: FinalAnswer) -> list[Claim]:
    return [c for c in final.claims if not c.supported]
