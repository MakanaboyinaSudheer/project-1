"""RAG retrieval used by the agents. Owner: Member 1 (Week 2)."""

from __future__ import annotations

from datetime import datetime

from discovery_agent.retrieval.embeddings import embed_texts
from discovery_agent.retrieval.vector_store import VectorStore
from discovery_agent.schemas import RetrievedChunk


def retrieve(query: str, k: int = 8, peer_reviewed_only: bool = False) -> list[RetrievedChunk]:
    """Semantic search; the Hypothesis Generator calls this with peer_reviewed_only=True."""
    if not query or not str(query).strip():
        return []

    store = VectorStore()
    try:
        store.ensure_collection()
        filter_expr = "peer_reviewed == true" if peer_reviewed_only else ""
        result_count = max(1, min(100, k * 4))
        results = store.search(query, k=result_count, filter_expr=filter_expr)
    finally:
        store.close()
    if not results:
        return []

    def _recency_boost(chunk: RetrievedChunk) -> float:
        metadata = chunk.metadata or {}
        published = metadata.get("published_ts")
        if not published:
            return 0.0
        try:
            dt = datetime.fromisoformat(str(published).replace("Z", "+00:00"))
        except ValueError:
            return 0.0
        now = datetime.now(dt.tzinfo or None)
        age_seconds = (now - dt).total_seconds()
        age_days = max(0.0, age_seconds / 86400.0)
        return max(0.0, 0.2 - (age_days / 3650.0))

    relevance = [max(0.0, float(chunk.score)) + _recency_boost(chunk) for chunk in results]
    vectors = embed_texts([chunk.text for chunk in results])
    selected: list[int] = []
    remaining = list(range(len(results)))
    limit = min(max(1, k), len(results))
    diversity_weight = 0.3

    while remaining and len(selected) < limit:
        best_index = max(
            remaining,
            key=lambda index: (
                (1.0 - diversity_weight) * relevance[index]
                - diversity_weight
                * max(
                    (
                        sum(left * right for left, right in zip(vectors[index], vectors[chosen], strict=True))
                        for chosen in selected
                    ),
                    default=0.0,
                ),
                -index,
            ),
        )
        selected.append(best_index)
        remaining.remove(best_index)

    return [results[index] for index in selected]
