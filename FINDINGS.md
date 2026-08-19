# Agentic retrieval vs single-shot RAG: what we measured

Corpus: 4,200 passages, hybrid Qdrant + BM25 retrieval with a `bge-reranker-v2-m3`
rerank. Model: `Qwen3-8B-Instruct` self-hosted on vLLM 0.9.2, one L40S, temperature 0.
Eval set: 120 questions, 60 multi-hop and 60 single-hop, graded by an LLM judge pinned
to a gold answer AND required to have retrieved every gold passage.

## The table

| set        | strategy    |  n | accuracy | avg hops |    $/question |  $/correct | avg latency |
|------------|-------------|---:|---------:|---------:|--------------:|-----------:|------------:|
| multihop   | single_shot | 60 |    31.7% |      1.0 |     $0.0021   |   $0.0066  |      1.4 s  |
| multihop   | agentic     | 60 |    71.7% |      3.4 |     $0.0169   |   $0.0236  |      6.1 s  |
| singlehop  | single_shot | 60 |    81.7% |      1.0 |     $0.0021   |   $0.0026  |      1.3 s  |
| singlehop  | agentic     | 60 |    80.0% |      3.1 |     $0.0168   |   $0.0210  |      5.8 s  |
| all        | routed      |120 |    75.0% |      2.3 |     $0.0104   |   $0.0139  |      4.1 s  |

## What it says

**Agentic retrieval wins decisively on multi-hop questions.** 31.7% -> 71.7%, a
40-point gain. The mechanism is not "more search". It is that a dependent hop can use
the answer to a previous hop as a query term, reaching passages the original question
had no lexical or semantic path to. No amount of reranking or higher `k` substitutes
for that, because the connection does not exist in the corpus until you know the
bridge entity.

**Agentic retrieval is a loss on single-hop questions.** 81.7% -> 80.0%, which is
inside the noise of a 60-question set, at 8.0x the cost and 4.5x the latency. Two
mechanisms showed up in the traces: the planner sometimes splits an atomic question
into two narrower hops, and extra hops pull in loosely related passages that dilute
the synthesis prompt.

**Do not deploy agentic retrieval globally.** If your traffic is mostly single-hop —
and most product search traffic is — turning this on everywhere multiplies your
inference bill by roughly eight and your p50 latency by four and a half, for nothing.

**Route per query.** Using the planner's own output as the decomposability signal:
56/60 multi-hop questions routed to agentic, 49/60 single-hop routed to single-shot.
Blended cost $0.0104/question against $0.0169 for all-agentic — a 38% cost reduction
— while keeping nearly all of the multi-hop accuracy gain.

## Bounding the loop is not optional

With unbounded reflection, the multi-hop run exhausted a 900-search budget after 58
of 60 questions. One question (mh-029) consumed 31 searches; from hop 5 onward it
retrieved zero new passages and re-asked the identical follow-up. The sufficiency
prompt is biased in one direction: asked to name the largest remaining gap, it always
finds one, so it never concludes a question is unanswerable.

Two bounds fixed it, and they do different jobs:
- `MAX_HOPS = 6` — a hard ceiling. Chosen from the data: every question that ever
  produced a correct answer did so within 5 hops.
- Diminishing returns — stop when the last 2 hops brought in fewer than 1 previously
  unseen passage. This fires well before the cap and is what removes the ordinary
  waste rather than just the catastrophe.

Stop-reason distribution on the multi-hop set after bounding: 68% `sufficient`,
25% `diminishing_returns`, 7% `hop_cap`.

## When NOT to use this

- Your questions are answerable from a single retrieval pass. Check this first; it is
  usually true and it is cheap to verify.
- You cannot bound the loop, whether by hop cap or budget. An unbounded reflection
  loop is an availability risk, not just a cost risk.
- Latency matters more than recall. 6.1 s p50 is not an interactive experience.
- Your corpus doesn't contain the answer. Agentic retrieval searches harder; it does
  not create evidence. On our 15 stalled questions the correct outcome was an
  "I could not verify this" claim, not more hops.

## Reproducing

```bash
docker compose up -d qdrant vllm langfuse
python -m research.corpus.loader
python -m research.eval.run --set all --strategy both --budget 900
python -m research.eval.run --set all --strategy routed --budget 900 --out results/routed.jsonl
python -m research.report.build
```
