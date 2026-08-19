"""Loads the research corpus into Qdrant (dense) and an in-memory BM25 index (sparse).

The corpus is 4,200 passages sliced from company filings, product changelogs, and
engineering blog posts. It is deliberately built so that some facts only connect
through a bridge entity that the question never names.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

COLLECTION = "research_corpus"
EMBED_MODEL = "BAAI/bge-base-en-v1.5"
VECTOR_SIZE = 768
CORPUS_PATH = Path(os.environ.get("CORPUS_PATH", "data/corpus.jsonl"))


@dataclass(frozen=True)
class Passage:
    passage_id: str
    doc_id: str
    doc_title: str
    text: str
    published: str  # ISO date; used for recency filters and for citation display


def read_corpus() -> list[Passage]:
    """Read the JSONL corpus off disk. One JSON object per line."""
    passages: list[Passage] = []
    with CORPUS_PATH.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            passages.append(
                Passage(
                    passage_id=row["passage_id"],
                    doc_id=row["doc_id"],
                    doc_title=row["doc_title"],
                    text=row["text"],
                    published=row.get("published", ""),
                )
            )
    return passages


def build_index(client: QdrantClient, passages: list[Passage], batch: int = 128) -> None:
    """Embed every passage and upsert into Qdrant. Point ids are the passage index,
    so re-running replaces vectors in place instead of duplicating them."""
    encoder = SentenceTransformer(EMBED_MODEL)

    client.recreate_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )

    for start in range(0, len(passages), batch):
        window = passages[start : start + batch]
        vectors = encoder.encode(
            [p.text for p in window],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        client.upsert(
            collection_name=COLLECTION,
            points=[
                PointStruct(
                    id=start + offset,
                    vector=vector.tolist(),
                    payload={
                        "passage_id": p.passage_id,
                        "doc_id": p.doc_id,
                        "doc_title": p.doc_title,
                        "text": p.text,
                        "published": p.published,
                    },
                )
                for offset, (p, vector) in enumerate(zip(window, vectors))
            ],
        )


if __name__ == "__main__":
    corpus = read_corpus()
    build_index(QdrantClient(url=os.environ.get("QDRANT_URL", "http://localhost:6333")), corpus)
    print(f"indexed {len(corpus)} passages into '{COLLECTION}'")
