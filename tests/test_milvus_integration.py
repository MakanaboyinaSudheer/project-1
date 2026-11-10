from uuid import uuid4

import pytest
from pymilvus import MilvusClient

from discovery_agent.config import settings
from discovery_agent.retrieval import vector_store
from discovery_agent.schemas import Chunk


@pytest.mark.integration
def test_milvus_is_ready_and_accepts_client_requests() -> None:
    client = MilvusClient(uri=settings.milvus_uri, timeout=10)
    try:
        collections = client.list_collections()
    finally:
        client.close()

    assert isinstance(collections, list)


@pytest.mark.integration
def test_vector_store_persists_chunks_and_filters_peer_reviewed_papers(monkeypatch) -> None:
    collection_name = f"test_{uuid4().hex[:12]}"
    monkeypatch.setattr(
        vector_store,
        "embed_texts",
        lambda texts: [[1.0] + [0.0] * (settings.embedding_dim - 1) for _ in texts],
    )
    store = vector_store.VectorStore(collection=collection_name)
    try:
        store.ensure_collection()
        stored = store.upsert(
            [
                Chunk(
                    id="arxiv:test:abstract",
                    paper_id="arxiv:test",
                    text="preprint study",
                    kind="abstract",
                    metadata={"source": "arxiv", "peer_reviewed": False},
                ),
                Chunk(
                    id="pubmed:test:abstract",
                    paper_id="pubmed:test",
                    text="journal study",
                    kind="abstract",
                    metadata={"source": "pubmed", "peer_reviewed": True},
                ),
            ]
        )
        results = store.search("journal study", k=5, filter_expr="peer_reviewed == true")

        assert stored == 2
        assert {chunk.id for chunk in results} == {"pubmed:test:abstract"}
        assert store.list_ids() == {"arxiv:test:abstract", "pubmed:test:abstract"}
    finally:
        client = MilvusClient(uri=settings.milvus_uri, timeout=10)
        try:
            if client.has_collection(collection_name=collection_name):
                client.drop_collection(collection_name=collection_name)
        finally:
            client.close()
            store.close()
