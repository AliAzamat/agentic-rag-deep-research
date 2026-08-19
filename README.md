# Agentic Retrieval: The Research Agent That Knows When It Has Enough

An advanced LangGraph project that builds an iterative research agent over a hybrid Qdrant + BM25 corpus. You start from a multi-hop question that single-shot RAG cannot answer even with perfect retrieval, then build query decomposition with a real dependency graph, parallel and sequential hop execution as a LangGraph state machine, and an LLM sufficiency check that names what is still missing. You then let reflection run unbounded, watch it burn 31 searches on one question and blow a 900-search budget 29 questions into a 60-question eval, and fix it with a hop cap plus a diminishing-returns gate. Then comes the honest part: you run the single-hop half of the eval and find agentic retrieval costs 8x for zero accuracy gain, so you build the router that picks single-shot or agentic per query. You finish with per-claim citation synthesis that can say "I could not verify this", self-host the model on vLLM with continuous batching tuned for many short agent calls, trace every hop in Langfuse with per-question cost attribution, and ship a comparison report that tells a reader when NOT to use what you just built.

## Stack
- Python
- LangGraph
- Qdrant
- rank_bm25
- bge-reranker
- vLLM
- Langfuse
