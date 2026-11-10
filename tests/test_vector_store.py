import pytest

from discovery_agent.retrieval import vector_store
from discovery_agent.schemas import Chunk


def _chunk(paper_id: str) -> Chunk:
    source = "pubmed" if paper_id.startswith("pubmed:") else "arxiv"
    return Chunk(
        id=f"{paper_id}:abstract",
        paper_id=paper_id,
        text=f"Study for {paper_id}",
        kind="abstract",
        metadata={"source": source, "peer_reviewed": source == "pubmed"},
    )


def test_vector_store_fails_fast_if_milvus_is_unavailable_by_default(monkeypatch) -> None:
    monkeypatch.setattr(vector_store, "MilvusClient", None)
    monkeypatch.setattr(vector_store.settings, "milvus_allow_memory_fallback", False)

    with pytest.raises(RuntimeError, match="persistent vector storage is required"):
        vector_store.VectorStore()


def test_explicit_memory_fallback_is_non_persistent_and_filters_peer_reviewed(monkeypatch, caplog) -> None:
    monkeypatch.setattr(vector_store, "MilvusClient", None)
    monkeypatch.setattr(vector_store, "embed_texts", lambda texts: [[1.0, 0.0] for _ in texts])
    store = vector_store.VectorStore(collection="test-papers", allow_memory_fallback=True)
    store.ensure_collection()

    assert store.upsert([_chunk("arxiv:1"), _chunk("pubmed:2")]) == 2
    results = store.search("query", filter_expr="peer_reviewed == true")

    assert [chunk.paper_id for chunk in results] == ["pubmed:2"]
    assert "non-persistent in-memory storage" in caplog.text


def test_vector_store_rejects_non_positive_search_limit(monkeypatch) -> None:
    monkeypatch.setattr(vector_store, "MilvusClient", None)
    store = vector_store.VectorStore(allow_memory_fallback=True)

    with pytest.raises(ValueError, match="greater than zero"):
        store.search("query", k=0)


def test_list_ids_uses_strong_consistency(monkeypatch) -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.query_kwargs = None

        def list_collections(self):
            return []

        def query(self, **kwargs):
            self.query_kwargs = kwargs
            return [{"id": "arxiv:1:abstract"}]

    client = FakeClient()
    monkeypatch.setattr(vector_store, "MilvusClient", lambda **_: client)
    store = vector_store.VectorStore(collection="test-papers")

    assert store.list_ids() == {"arxiv:1:abstract"}
    assert client.query_kwargs["consistency_level"] == "Strong"


def test_vector_store_batches_embeddings_for_bulk_upserts(monkeypatch) -> None:
    monkeypatch.setattr(vector_store, "MilvusClient", None)
    batch_sizes = []

    def fake_embed_texts(texts):
        batch_sizes.append(len(texts))
        return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr(vector_store, "embed_texts", fake_embed_texts)
    store = vector_store.VectorStore(collection="bulk-test", allow_memory_fallback=True)
    chunks = [
        Chunk(id=f"arxiv:{index}:abstract", paper_id=f"arxiv:{index}", text="text", kind="abstract")
        for index in range(300)
    ]

    assert store.upsert(chunks) == 300
    assert batch_sizes == [128, 128, 44]
