from datetime import datetime
from uuid import uuid4

import pytest

from discovery_agent.ingestion import pipeline
from discovery_agent.retrieval import vector_store
from discovery_agent.schemas import Chunk, Paper


def _paper(paper_id: str, *, pdf_url: str | None = None) -> Paper:
    return Paper(
        id=paper_id,
        source="pubmed" if paper_id.startswith("pubmed:") else "arxiv",
        title=f"Title {paper_id}",
        abstract="An abstract",
        published=datetime(2024, 1, 1),
        pdf_url=pdf_url,
        peer_reviewed=paper_id.startswith("pubmed:"),
    )


def test_run_ingestion_excludes_holdout_and_indexes_abstract_body_and_figures(monkeypatch, tmp_path) -> None:
    included = _paper("arxiv:included", pdf_url="https://example.test/included.pdf")
    held_out = _paper("arxiv:held-out", pdf_url="https://example.test/holdout.pdf")
    duplicate = _paper("arxiv:included")
    ingested_chunks: list[Chunk] = []
    events = []

    class FakeStore:
        def ensure_collection(self):
            events.append("ensure")

        def list_ids(self):
            return set()

        def upsert(self, chunks):
            ingested_chunks.extend(chunks)
            return len(chunks)

        def close(self):
            events.append("close")

    monkeypatch.setattr(pipeline, "fetch_arxiv", lambda **_: [included, held_out])
    monkeypatch.setattr(pipeline, "fetch_pubmed", lambda **_: [duplicate])

    def exclude_holdouts(papers):
        events.append("exclude")
        return [paper for paper in papers if paper.id != held_out.id]

    monkeypatch.setattr(pipeline, "exclude_holdout_papers", exclude_holdouts)
    monkeypatch.setattr(pipeline, "download_pdf", lambda paper, *_args, **_kwargs: tmp_path / "paper.pdf")
    monkeypatch.setattr(
        pipeline,
        "extract_text_chunks",
        lambda paper, *_: [Chunk(id=f"{paper.id}:body:0", paper_id=paper.id, text="Body", kind="body")],
    )
    monkeypatch.setattr(pipeline, "extract_figures", lambda *_: [(tmp_path / "figure.png", "Figure 1.")])
    monkeypatch.setattr(
        pipeline,
        "describe_figure",
        lambda image, paper_id, caption: Chunk(
            id=f"{paper_id}:figure:1",
            paper_id=paper_id,
            text=f"{caption} description",
            kind="figure",
        ),
    )
    monkeypatch.setattr(pipeline, "VectorStore", FakeStore)
    monkeypatch.setattr(pipeline.settings, "ingest_figure_dir", str(tmp_path / "figures"))

    count = pipeline.run_ingestion("test query", max_results=3)

    assert count == 3
    assert events == ["exclude", "ensure", "close"]
    assert [chunk.kind for chunk in ingested_chunks] == ["abstract", "body", "figure"]
    assert {chunk.paper_id for chunk in ingested_chunks} == {included.id}


def test_run_ingestion_returns_zero_for_blank_query(monkeypatch) -> None:
    monkeypatch.setattr(pipeline, "fetch_arxiv", lambda **_: (_ for _ in ()).throw(AssertionError()))
    assert pipeline.run_ingestion("  ") == 0


def test_run_ingestion_skips_already_indexed_figures(monkeypatch, tmp_path) -> None:
    paper = _paper("arxiv:already-indexed", pdf_url="https://example.test/paper.pdf")
    existing_ids = {
        f"{paper.id}:abstract",
        f"{paper.id}:body:0000",
        f"{paper.id}:figure:figure-1",
    }

    class FakeStore:
        def ensure_collection(self):
            pass

        def list_ids(self):
            return existing_ids

        def upsert(self, chunks):
            raise AssertionError("Existing chunks should not be upserted")

        def close(self):
            pass

    monkeypatch.setattr(pipeline, "fetch_arxiv", lambda **_: [paper])
    monkeypatch.setattr(pipeline, "fetch_pubmed", lambda **_: [])
    monkeypatch.setattr(pipeline, "exclude_holdout_papers", lambda papers: list(papers))
    monkeypatch.setattr(pipeline, "VectorStore", FakeStore)
    monkeypatch.setattr(pipeline, "download_pdf", lambda *_args, **_kwargs: tmp_path / "paper.pdf")
    monkeypatch.setattr(
        pipeline,
        "extract_text_chunks",
        lambda *_: [
            Chunk(
                id=f"{paper.id}:body:0000",
                paper_id=paper.id,
                text="Body text",
                kind="body",
            )
        ],
    )
    monkeypatch.setattr(
        pipeline,
        "extract_figures",
        lambda *_: [(tmp_path / "figure-1.png", "Figure 1.")],
    )
    monkeypatch.setattr(
        pipeline,
        "describe_figure",
        lambda *_: (_ for _ in ()).throw(AssertionError("Existing figures must not be re-described")),
    )

    assert pipeline.run_ingestion("repeat query") == 0


@pytest.mark.integration
def test_run_ingestion_is_idempotent_against_milvus(monkeypatch) -> None:
    paper = _paper("arxiv:9999.99999")
    collection_name = f"test_{uuid4().hex[:12]}"
    monkeypatch.setattr(pipeline.settings, "milvus_collection", collection_name)
    monkeypatch.setattr(pipeline, "fetch_arxiv", lambda **_: [paper])
    monkeypatch.setattr(pipeline, "fetch_pubmed", lambda **_: [])
    monkeypatch.setattr(
        vector_store,
        "embed_texts",
        lambda texts: [[1.0] + [0.0] * (pipeline.settings.embedding_dim - 1) for _ in texts],
    )

    try:
        assert pipeline.run_ingestion("idempotency test", max_results=1) == 1
        assert pipeline.run_ingestion("idempotency test", max_results=1) == 0
    finally:
        from pymilvus import MilvusClient

        client = MilvusClient(uri=pipeline.settings.milvus_uri, timeout=10)
        try:
            if client.has_collection(collection_name=collection_name):
                client.drop_collection(collection_name=collection_name)
        finally:
            client.close()
