"""Run one multi-hop question through the single-shot baseline and show exactly
where it breaks. Run this before writing any agent code."""
from __future__ import annotations

from research.baseline.single_shot import answer

QUESTION = (
    "Which cloud region did the company that acquired Sparkplug Analytics in 2025 "
    "choose for its first EU data residency zone, and what latency target did they "
    "publish for it?"
)

if __name__ == "__main__":
    result = answer(QUESTION)

    print("QUESTION:", QUESTION, "\n")
    print("RETRIEVED PASSAGES (top 8, hybrid + rerank):")
    for i, hit in enumerate(result.hits, start=1):
        print(f"  [{i}] {hit.score:6.3f}  {hit.doc_title}")
    print("\nANSWER:", result.text)
    print("CITATIONS:", result.citations)
    print(f"TOKENS: prompt={result.prompt_tokens} completion={result.completion_tokens}")
