"""Milvus access layer. Owner: Member 1 (Week 1)."""

from discovery_agent.schemas import Chunk, RetrievedChunk


class VectorStore:
    def __init__(self, uri: str | None = None, collection: str | None = None) -> None:
        raise NotImplementedError

    def ensure_collection(self) -> None:
        """Create collection + index if missing. Fields: id, paper_id, kind, source,
        peer_reviewed, published_ts, text, vector."""
        raise NotImplementedError

    def upsert(self, chunks: list[Chunk]) -> int:
        raise NotImplementedError

    def search(self, query: str, k: int = 8, filter_expr: str = "") -> list[RetrievedChunk]:
        raise NotImplementedError
