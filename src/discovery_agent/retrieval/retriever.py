"""RAG retrieval used by the agents. Owner: Member 1 (Week 2)."""

from discovery_agent.schemas import RetrievedChunk


def retrieve(query: str, k: int = 8, peer_reviewed_only: bool = False) -> list[RetrievedChunk]:
    """Semantic search; the Hypothesis Generator calls this with peer_reviewed_only=True."""
    raise NotImplementedError
