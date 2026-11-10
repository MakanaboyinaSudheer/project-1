"""Ingestion orchestration: fetch -> dedupe -> chunk -> embed -> upsert. Owner: Member 1."""

from __future__ import annotations

from discovery_agent.config import settings
from discovery_agent.ingestion.arxiv_client import fetch_arxiv
from discovery_agent.ingestion.pubmed_client import fetch_pubmed
from discovery_agent.retrieval.vector_store import VectorStore
from discovery_agent.schemas import Chunk, Paper


def _chunk_paper(paper: Paper) -> list[Chunk]:
    metadata = {
        "source": paper.source,
        "peer_reviewed": paper.peer_reviewed,
        "published_ts": paper.published.isoformat() if paper.published else None,
        "title": paper.title,
    }
    chunks: list[Chunk] = []
    abstract = (paper.abstract or paper.title).strip()
    if abstract:
        chunks.append(
            Chunk(
                id=f"{paper.id}:abstract",
                paper_id=paper.id,
                text=abstract,
                kind="abstract",
                metadata=metadata,
            )
        )
    return chunks


def run_ingestion(query: str, max_results: int | None = None) -> int:
    """Ingest papers for `query` from ArXiv + PubMed. Returns number of new chunks upserted.

    Must be idempotent (re-running does not create duplicates) so it can run on a schedule.
    """
    if not query or not str(query).strip():
        return 0

    limit = max_results if max_results is not None else settings.ingest_max_results
    limit = max(1, int(limit))

    papers: list[Paper] = []
    for fetcher in (fetch_arxiv, fetch_pubmed):
        try:
            if fetcher is fetch_arxiv:
                papers.extend(
                    fetcher(query=query, categories=settings.arxiv_category_list, max_results=limit)
                )
            else:
                papers.extend(fetcher(query=query, max_results=limit))
        except Exception:
            continue

    seen: set[str] = set()
    chunks: list[Chunk] = []
    for paper in papers:
        if paper.id in seen:
            continue
        seen.add(paper.id)
        chunks.extend(_chunk_paper(paper))

    if not chunks:
        return 0

    store = VectorStore()
    store.ensure_collection()
    existing_ids = store.list_ids()
    new_chunks = [chunk for chunk in chunks if chunk.id not in existing_ids]
    if not new_chunks:
        return 0

    store.upsert(new_chunks)
    return len(new_chunks)
