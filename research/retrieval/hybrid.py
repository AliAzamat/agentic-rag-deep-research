"""Hybrid dense + sparse retrieval with a cross-encoder rerank.

This is deliberately a GOOD retriever. The point of step 1 is that a multi-hop
question fails even when retrieval is not the weak link, so the baseline has to be
strong enough that nobody can blame recall for the wrong answer.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from research.corpus.loader import COLLECTION, EMBED_MODEL, Passage, read_corpus

RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
DENSE_K = 30
SPARSE_K = 30
FINAL_K = 8


@dataclass
class Hit:
    passage_id: str
    doc_title: str
    text: str
    published: str
    score: float


@lru_cache(maxsize=1)
def _resources() -> tuple[QdrantClient, SentenceTransformer, CrossEncoder, BM25Okapi, list[Passage]]:
    """Load clients and models once per process. Cross-encoders are expensive to
    construct, so a module-level cache matters more here than it looks."""
    client = QdrantClient(url=os.environ.get("QDRANT_URL", "http://localhost:6333"))
    encoder = SentenceTransformer(EMBED_MODEL)
    reranker = CrossEncoder(RERANK_MODEL, max_length=512)
    passages = read_corpus()
    bm25 = BM25Okapi([p.text.lower().split() for p in passages])
    return client, encoder, reranker, bm25, passages


def search(query: str, k: int = FINAL_K) -> list[Hit]:
    """Dense recall + sparse recall, union, then cross-encoder rerank to top k."""
    client, encoder, reranker, bm25, passages = _resources()

    qvec = encoder.encode([query], normalize_embeddings=True)[0].tolist()
    dense = client.search(collection_name=COLLECTION, query_vector=qvec, limit=DENSE_K)

    sparse_scores = bm25.get_scores(query.lower().split())
    top_sparse = sorted(range(len(passages)), key=lambda i: sparse_scores[i], reverse=True)[:SPARSE_K]

    # Union by passage_id. Dense and sparse find different things: dense catches
    # paraphrase, sparse catches exact identifiers like "FY2027" or a product SKU.
    pool: dict[str, tuple[str, str, str]] = {}
    for point in dense:
        payload = point.payload or {}
        pool[payload["passage_id"]] = (payload["doc_title"], payload["text"], payload["published"])
    for idx in top_sparse:
        p = passages[idx]
        pool[p.passage_id] = (p.doc_title, p.text, p.published)

    if not pool:
        return []

    items = list(pool.items())
    scores = reranker.predict([(query, text) for _, (_, text, _) in items])
    ranked = sorted(zip(items, scores), key=lambda pair: pair[1], reverse=True)[:k]

    return [
        Hit(passage_id=pid, doc_title=title, text=text, published=published, score=float(score))
        for (pid, (title, text, published)), score in ranked
    ]
