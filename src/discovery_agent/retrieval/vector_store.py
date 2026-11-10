"""Milvus access layer. Owner: Member 1 (Week 1)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from discovery_agent.config import settings
from discovery_agent.retrieval.embeddings import embed_texts
from discovery_agent.schemas import Chunk, RetrievedChunk

try:
    from pymilvus import MilvusClient
except Exception:  # pragma: no cover - optional dependency for fallback behavior.
    MilvusClient = None  # type: ignore[assignment]


class VectorStore:
    def __init__(self, uri: str | None = None, collection: str | None = None) -> None:
        self.uri = uri or settings.milvus_uri
        self.collection = collection or settings.milvus_collection
        self._memory_store: dict[str, dict[str, Any]] = {}
        self.client = None
        if MilvusClient is not None:
            try:
                self.client = MilvusClient(uri=self.uri)
                self.client.list_collections()
            except Exception:
                self.client = None

    def ensure_collection(self) -> None:
        """Create collection + index if missing. Fields: id, paper_id, kind, source,
        peer_reviewed, published_ts, text, vector."""
        if self.client is None:
            self._memory_store.setdefault(self.collection, {})
            return

        try:
            if self.client.has_collection(collection_name=self.collection):
                return
            self.client.create_collection(
                collection_name=self.collection,
                dimension=settings.embedding_dim,
                metric_type="COSINE",
                auto_id=False,
                primary_field="id",
                vector_field="vector",
                id_type="String",
            )
        except Exception:
            self.client = None
            self._memory_store.setdefault(self.collection, {})

    def list_ids(self) -> set[str]:
        if self.client is not None:
            try:
                rows = self.client.query(
                    collection_name=self.collection,
                    filter="",
                    output_fields=["id"],
                    limit=10000,
                )
                return {row.get("id") for row in rows if isinstance(row, dict) and row.get("id")}
            except Exception:
                pass
        return set(self._memory_store.get(self.collection, {}).keys())

    def upsert(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0

        records: list[dict[str, Any]] = []
        for chunk in chunks:
            if not chunk.text or not chunk.id:
                continue
            metadata = chunk.metadata or {}
            source = str(metadata.get("source") or ("pubmed" if chunk.paper_id.startswith("pubmed:") else "arxiv" if chunk.paper_id.startswith("arxiv:") else "unknown"))
            peer_reviewed = bool(metadata.get("peer_reviewed") or source == "pubmed")
            published_ts = metadata.get("published_ts") or metadata.get("published") or ""
            if isinstance(published_ts, datetime):
                published_ts = published_ts.isoformat()
            vector = embed_texts([chunk.text])[0]
            record = {
                "id": chunk.id,
                "paper_id": chunk.paper_id,
                "kind": chunk.kind,
                "source": source,
                "peer_reviewed": peer_reviewed,
                "published_ts": str(published_ts),
                "text": chunk.text,
                "vector": vector,
            }
            records.append(record)
            self._memory_store.setdefault(self.collection, {})[chunk.id] = record

        if self.client is not None:
            try:
                self.client.upsert(collection_name=self.collection, data=records)
            except Exception:
                self.client = None

        return len(records)

    def search(self, query: str, k: int = 8, filter_expr: str = "") -> list[RetrievedChunk]:
        if not query.strip():
            return []

        vector = embed_texts([query])[0]
        query_filter = filter_expr.strip()

        if self.client is not None:
            try:
                results = self.client.search(
                    collection_name=self.collection,
                    data=[vector],
                    limit=max(1, k),
                    filter=query_filter,
                    search_params={"metric_type": "COSINE", "params": {"ef": max(16, k * 4)}},
                    output_fields=["id", "paper_id", "kind", "source", "peer_reviewed", "published_ts", "text"],
                )
                matches: list[RetrievedChunk] = []
                for hit in results[0]:
                    entity = hit.get("entity") if isinstance(hit, dict) else {}
                    if not isinstance(entity, dict):
                        continue
                    metadata = {
                        "source": entity.get("source"),
                        "peer_reviewed": bool(entity.get("peer_reviewed")),
                        "published_ts": entity.get("published_ts"),
                        "title": entity.get("paper_id"),
                    }
                    matches.append(
                        RetrievedChunk(
                            id=str(entity.get("id", "")),
                            paper_id=str(entity.get("paper_id", "")),
                            text=str(entity.get("text", "")),
                            kind=str(entity.get("kind", "abstract")),
                            metadata=metadata,
                            score=float(hit.get("distance", 0.0)),
                        )
                    )
                if matches:
                    return matches
            except Exception:
                self.client = None

        collection = self._memory_store.get(self.collection, {})
        if not collection:
            return []

        matches: list[RetrievedChunk] = []
        for record in collection.values():
            if query_filter and not self._matches_filter(record, query_filter):
                continue
            candidate_vector = record["vector"]
            dot = sum(a * b for a, b in zip(vector, candidate_vector, strict=False))
            norm = (sum(v * v for v in vector) ** 0.5) * (sum(v * v for v in candidate_vector) ** 0.5)
            score = dot / norm if norm else 0.0
            matches.append(
                RetrievedChunk(
                    id=str(record["id"]),
                    paper_id=str(record["paper_id"]),
                    text=str(record["text"]),
                    kind=str(record["kind"]),
                    metadata={
                        "source": record.get("source"),
                        "peer_reviewed": bool(record.get("peer_reviewed")),
                        "published_ts": record.get("published_ts"),
                        "title": record.get("paper_id"),
                    },
                    score=float(score),
                )
            )

        matches.sort(key=lambda item: item.score, reverse=True)
        return matches[:k]

    @staticmethod
    def _matches_filter(record: dict[str, Any], filter_expr: str) -> bool:
        normalized = filter_expr.strip()
        if not normalized:
            return True
        if "peer_reviewed" in normalized:
            expected = "true" in normalized.lower()
            return bool(record.get("peer_reviewed")) is expected
        return True
