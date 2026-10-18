"""Milvus access layer. Owner: Member 1 (Week 1)."""

from datetime import UTC, datetime

from pymilvus import DataType, MilvusClient

from discovery_agent.config import settings
from discovery_agent.retrieval.embeddings import embed_texts
from discovery_agent.schemas import Chunk, RetrievedChunk


class VectorStore:
    def __init__(self, uri: str | None = None, collection: str | None = None) -> None:
        self.uri = uri or settings.milvus_uri
        self.collection = collection or settings.milvus_collection
        self.client = MilvusClient(uri=self.uri)

    def ensure_collection(self) -> None:
        """Create collection + index if missing. Fields: id, paper_id, kind, source,
        peer_reviewed, published_ts, text, vector."""
        if self.client.has_collection(self.collection):
            return
        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=256)
        schema.add_field("paper_id", DataType.VARCHAR, max_length=256)
        schema.add_field("kind", DataType.VARCHAR, max_length=32)
        schema.add_field("source", DataType.VARCHAR, max_length=32)
        schema.add_field("peer_reviewed", DataType.BOOL)
        schema.add_field("published_ts", DataType.INT64)
        schema.add_field("title", DataType.VARCHAR, max_length=4096)
        schema.add_field("url", DataType.VARCHAR, max_length=4096)
        schema.add_field("text", DataType.VARCHAR, max_length=65535)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=settings.embedding_dim)
        indexes = self.client.prepare_index_params()
        indexes.add_index(
            field_name="vector",
            index_type="AUTOINDEX",
            metric_type="COSINE",
        )
        indexes.add_index(field_name="peer_reviewed", index_type="INVERTED")
        self.client.create_collection(
            collection_name=self.collection,
            schema=schema,
            index_params=indexes,
        )

    def upsert(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0
        self.ensure_collection()
        vectors = embed_texts([chunk.text for chunk in chunks])
        rows = []
        for chunk, vector in zip(chunks, vectors, strict=True):
            published = chunk.metadata.get("published")
            if isinstance(published, datetime):
                published_ts = int(published.timestamp())
            elif isinstance(published, (int, float)):
                published_ts = int(published)
            else:
                published_ts = 0
            rows.append(
                {
                    "id": chunk.id,
                    "paper_id": chunk.paper_id,
                    "kind": chunk.kind,
                    "source": chunk.metadata.get("source", chunk.paper_id.split(":", 1)[0]),
                    "peer_reviewed": bool(chunk.metadata.get("peer_reviewed", False)),
                    "published_ts": published_ts,
                    "title": str(chunk.metadata.get("title", ""))[:4096],
                    "url": str(chunk.metadata.get("url", ""))[:4096],
                    "text": chunk.text[:65535],
                    "vector": vector,
                }
            )
        self.client.upsert(collection_name=self.collection, data=rows)
        return len(rows)

    def get_existing_ids(self, ids: list[str]) -> set[str]:
        if not ids:
            return set()
        self.ensure_collection()
        existing: set[str] = set()
        for start in range(0, len(ids), 500):
            rows = self.client.get(
                collection_name=self.collection,
                ids=ids[start : start + 500],
                output_fields=["id"],
            )
            existing.update(str(row["id"]) for row in rows)
        return existing

    def search(self, query: str, k: int = 8, filter_expr: str = "") -> list[RetrievedChunk]:
        if k < 1:
            raise ValueError("k must be positive")
        self.ensure_collection()
        query_vector = embed_texts([query])[0]
        response = self.client.search(
            collection_name=self.collection,
            data=[query_vector],
            limit=k,
            filter=filter_expr,
            output_fields=[
                "paper_id",
                "kind",
                "source",
                "peer_reviewed",
                "published_ts",
                "title",
                "url",
                "text",
            ],
            search_params={"metric_type": "COSINE"},
        )
        results = []
        for hit in response[0]:
            entity = hit.get("entity", {})
            timestamp = entity.get("published_ts", 0)
            metadata = {
                "source": entity.get("source", ""),
                "peer_reviewed": entity.get("peer_reviewed", False),
                "title": entity.get("title", ""),
                "url": entity.get("url", ""),
            }
            if timestamp:
                from datetime import datetime

                metadata["published"] = datetime.fromtimestamp(timestamp, tz=UTC)
            results.append(
                RetrievedChunk(
                    id=str(hit["id"]),
                    paper_id=entity.get("paper_id", ""),
                    text=entity.get("text", ""),
                    kind=entity.get("kind", "abstract"),
                    metadata=metadata,
                    score=float(hit.get("distance", hit.get("score", 0.0))),
                )
            )
        return results

    def close(self) -> None:
        self.client.close()
