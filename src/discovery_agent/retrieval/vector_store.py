"""Milvus access layer. Owner: Member 1 (Week 1)."""

from __future__ import annotations

import logging
from contextlib import suppress
from datetime import datetime
from typing import Any

from discovery_agent.config import settings
from discovery_agent.retrieval.embeddings import embed_texts
from discovery_agent.schemas import Chunk, RetrievedChunk

try:
    from pymilvus import MilvusClient
except ImportError:  # pragma: no cover - optional dependency for explicit fallback mode.
    MilvusClient = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)
_EMBEDDING_BATCH_SIZE = 128


class VectorStore:
    def __init__(
        self,
        uri: str | None = None,
        collection: str | None = None,
        *,
        allow_memory_fallback: bool | None = None,
    ) -> None:
        self.uri = uri or settings.milvus_uri
        self.collection = collection or settings.milvus_collection
        self.allow_memory_fallback = (
            settings.milvus_allow_memory_fallback if allow_memory_fallback is None else allow_memory_fallback
        )
        self._memory_store: dict[str, dict[str, Any]] = {}
        self.client = None

        if MilvusClient is None:
            self._use_memory_fallback(RuntimeError("pymilvus is unavailable"))
            return

        client = None
        try:
            client = MilvusClient(uri=self.uri)
            client.list_collections()
        except Exception as exc:
            if client is not None:
                with suppress(Exception):
                    client.close()
            self._use_memory_fallback(exc)
        else:
            self.client = client

    def _use_memory_fallback(self, error: Exception) -> None:
        client = self.client
        self.client = None
        if client is not None:
            with suppress(Exception):
                client.close()
        if not self.allow_memory_fallback:
            raise RuntimeError(
                f"Milvus is unavailable at {self.uri}; persistent vector storage is required"
            ) from error
        logger.warning(
            "Milvus is unavailable at %s; using non-persistent in-memory storage: %s",
            self.uri,
            error,
        )
        self._memory_store.setdefault(self.collection, {})

    def close(self) -> None:
        """Close the Milvus client when one was initialized."""
        if self.client is not None:
            client = self.client
            self.client = None
            client.close()

    def ensure_collection(self) -> None:
        """Create collection + index if missing. Fields: id, paper_id, kind, source,
        peer_reviewed, published_ts, text, vector."""
        if self.client is None:
            if not self.allow_memory_fallback:
                raise RuntimeError("Milvus is unavailable and in-memory fallback is disabled")
            self._memory_store.setdefault(self.collection, {})
            return

        from pymilvus import DataType

        try:
            if self.client.has_collection(collection_name=self.collection):
                return
            schema = self.client.create_schema(auto_id=False, enable_dynamic_field=False)
            schema.add_field(
                field_name="id",
                datatype=DataType.VARCHAR,
                is_primary=True,
                max_length=512,
            )
            schema.add_field(field_name="paper_id", datatype=DataType.VARCHAR, max_length=512)
            schema.add_field(field_name="kind", datatype=DataType.VARCHAR, max_length=32)
            schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=16)
            schema.add_field(field_name="peer_reviewed", datatype=DataType.BOOL)
            schema.add_field(
                field_name="published_ts",
                datatype=DataType.VARCHAR,
                max_length=40,
            )
            schema.add_field(
                field_name="text",
                datatype=DataType.VARCHAR,
                max_length=32768,
            )
            schema.add_field(
                field_name="vector",
                datatype=DataType.FLOAT_VECTOR,
                dim=settings.embedding_dim,
            )
            index_params = self.client.prepare_index_params()
            index_params.add_index(
                field_name="vector",
                index_type="AUTOINDEX",
                metric_type="COSINE",
            )
            index_params.add_index(field_name="peer_reviewed", index_type="INVERTED")
            self.client.create_collection(
                collection_name=self.collection,
                schema=schema,
                index_params=index_params,
            )
        except Exception as exc:
            self._use_memory_fallback(exc)

    def list_ids(self) -> set[str]:
        if self.client is not None:
            try:
                rows = self.client.query(
                    collection_name=self.collection,
                    filter="",
                    output_fields=["id"],
                    limit=10000,
                    consistency_level="Strong",
                )
                return {row.get("id") for row in rows if isinstance(row, dict) and row.get("id")}
            except Exception as exc:
                self._use_memory_fallback(exc)
        if not self.allow_memory_fallback:
            raise RuntimeError("Milvus is unavailable and in-memory fallback is disabled")
        return set(self._memory_store.get(self.collection, {}).keys())

    def upsert(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0

        valid_chunks = [chunk for chunk in chunks if chunk.text and chunk.id]
        records: list[dict[str, Any]] = []
        for start in range(0, len(valid_chunks), _EMBEDDING_BATCH_SIZE):
            batch = valid_chunks[start : start + _EMBEDDING_BATCH_SIZE]
            vectors = embed_texts([chunk.text for chunk in batch])
            for chunk, vector in zip(batch, vectors, strict=True):
                metadata = chunk.metadata or {}
                inferred_source = (
                    "pubmed"
                    if chunk.paper_id.startswith("pubmed:")
                    else "arxiv"
                    if chunk.paper_id.startswith("arxiv:")
                    else "unknown"
                )
                source = str(metadata.get("source") or inferred_source)
                peer_reviewed = bool(metadata.get("peer_reviewed") or source == "pubmed")
                published_ts = metadata.get("published_ts") or metadata.get("published") or ""
                if isinstance(published_ts, datetime):
                    published_ts = published_ts.isoformat()
                records.append(
                    {
                        "id": chunk.id,
                        "paper_id": chunk.paper_id,
                        "kind": chunk.kind,
                        "source": source,
                        "peer_reviewed": peer_reviewed,
                        "published_ts": str(published_ts),
                        "text": chunk.text,
                        "vector": vector,
                    }
                )

        if not records:
            return 0

        if self.client is not None:
            try:
                self.client.upsert(collection_name=self.collection, data=records)
            except Exception as exc:
                self._use_memory_fallback(exc)

        if self.client is None:
            if not self.allow_memory_fallback:
                raise RuntimeError("Milvus is unavailable and in-memory fallback is disabled")
            self._memory_store.setdefault(self.collection, {}).update(
                {record["id"]: record for record in records}
            )

        return len(records)

    def search(self, query: str, k: int = 8, filter_expr: str = "") -> list[RetrievedChunk]:
        if not query.strip():
            return []
        if k <= 0:
            raise ValueError("k must be greater than zero")
        if self.client is None and not self.allow_memory_fallback:
            raise RuntimeError("Milvus is unavailable and in-memory fallback is disabled")

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
                    output_fields=[
                        "id",
                        "paper_id",
                        "kind",
                        "source",
                        "peer_reviewed",
                        "published_ts",
                        "text",
                    ],
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
            except Exception as exc:
                self._use_memory_fallback(exc)

        collection = self._memory_store.get(self.collection, {})
        if not collection:
            return []

        matches = []
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
