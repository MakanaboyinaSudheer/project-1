"""RAG retrieval used by the agents. Owner: Member 1 (Week 2)."""

from __future__ import annotations

from datetime import datetime

from discovery_agent.retrieval.vector_store import VectorStore
from discovery_agent.schemas import RetrievedChunk


def retrieve(query: str, k: int = 8, peer_reviewed_only: bool = False) -> list[RetrievedChunk]:
    """Semantic search; the Hypothesis Generator calls this with peer_reviewed_only=True."""
    if not query or not str(query).strip():
        return []

    store = VectorStore()
    store.ensure_collection()
    filter_expr = "peer_reviewed == true" if peer_reviewed_only else ""
    results = store.search(query, k=max(1, k), filter_expr=filter_expr)
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

    ranked = []
    for chunk in results:
        score = float(chunk.score)
        if score < 0:
            score = 0.0
        score += _recency_boost(chunk)
        ranked.append((score, chunk))

    ranked.sort(key=lambda item: item[0], reverse=True)
    final = [chunk for _, chunk in ranked[: max(1, k)]]
    return final
