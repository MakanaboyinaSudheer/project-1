"""RAG retrieval used by the agents. Owner: Member 1 (Week 2)."""

from datetime import UTC, datetime

import numpy as np

from discovery_agent.retrieval.embeddings import embed_texts
from discovery_agent.retrieval.vector_store import VectorStore
from discovery_agent.schemas import RetrievedChunk


def retrieve(query: str, k: int = 8, peer_reviewed_only: bool = False) -> list[RetrievedChunk]:
    """Semantic search; the Hypothesis Generator calls this with peer_reviewed_only=True."""
    if k < 1:
        raise ValueError("k must be positive")
    store = VectorStore()
    try:
        candidates = store.search(
            query,
            k=max(k * 4, k),
            filter_expr="peer_reviewed == true" if peer_reviewed_only else "",
        )
    finally:
        store.close()
    if peer_reviewed_only:
        candidates = [
            item
            for item in candidates
            if item.metadata.get("peer_reviewed") is True and item.metadata.get("source") != "arxiv"
        ]
    if not candidates:
        return []

    vectors = np.asarray(embed_texts([query, *(item.text for item in candidates)]), dtype=float)
    query_vector, document_vectors = vectors[0], vectors[1:]
    similarities = document_vectors @ query_vector
    now = datetime.now(UTC).timestamp()
    recency_scores = []
    for item in candidates:
        published = item.metadata.get("published")
        timestamp = published.timestamp() if isinstance(published, datetime) else 0
        age = max(0.0, now - timestamp)
        recency_scores.append(max(0.0, 1.0 - age / (10 * 365.25 * 24 * 60 * 60)))
    recency = np.asarray(recency_scores)
    relevance = 0.85 * similarities + 0.15 * recency
    selected: list[int] = []
    remaining = set(range(len(candidates)))
    while remaining and len(selected) < k:
        best = max(
            remaining,
            key=lambda index: (
                relevance[index]
                if not selected
                else 0.7 * relevance[index]
                - 0.3 * max(float(document_vectors[index] @ document_vectors[chosen]) for chosen in selected)
            ),
        )
        selected.append(best)
        remaining.remove(best)
    return [candidates[index] for index in selected]
