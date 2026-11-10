from discovery_agent.retrieval import retriever
from discovery_agent.schemas import RetrievedChunk


def _chunk(identifier: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        id=identifier,
        paper_id=f"arxiv:{identifier}",
        text=identifier,
        kind="abstract",
        score=score,
    )


def test_retrieve_uses_mmr_to_balance_relevance_and_diversity(monkeypatch) -> None:
    candidates = [
        _chunk("top", 0.95),
        _chunk("near-duplicate", 0.90),
        _chunk("diverse", 0.80),
    ]

    class FakeStore:
        def ensure_collection(self):
            pass

        def search(self, query, *, k, filter_expr):
            assert query == "query"
            assert k == 8
            assert filter_expr == ""
            return candidates

        def close(self):
            pass

    monkeypatch.setattr(retriever, "VectorStore", FakeStore)
    monkeypatch.setattr(
        retriever,
        "embed_texts",
        lambda texts: [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
    )

    results = retriever.retrieve("query", k=2)

    assert [chunk.id for chunk in results] == ["top", "diverse"]


def test_retrieve_preserves_peer_reviewed_filter(monkeypatch) -> None:
    calls = []

    class FakeStore:
        def ensure_collection(self):
            pass

        def search(self, query, *, k, filter_expr):
            calls.append((query, k, filter_expr))
            return []

        def close(self):
            pass

    monkeypatch.setattr(retriever, "VectorStore", FakeStore)

    assert retriever.retrieve("query", k=5, peer_reviewed_only=True) == []
    assert calls == [("query", 20, "peer_reviewed == true")]
