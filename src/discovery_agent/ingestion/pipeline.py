"""Ingestion orchestration: fetch -> dedupe -> chunk -> embed -> upsert. Owner: Member 1."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from discovery_agent.config import settings
from discovery_agent.evaluation.holdout import exclude_holdout_papers
from discovery_agent.ingestion.arxiv_client import fetch_arxiv
from discovery_agent.ingestion.figure_parser import describe_figure
from discovery_agent.ingestion.pdf_parser import download_pdf, extract_figures, extract_text_chunks
from discovery_agent.ingestion.pubmed_client import fetch_pubmed
from discovery_agent.retrieval.vector_store import VectorStore
from discovery_agent.schemas import Chunk, Paper

logger = logging.getLogger(__name__)


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


def _paper_figure_dir(paper: Paper) -> Path:
    paper_key = hashlib.sha256(paper.id.encode()).hexdigest()[:16]
    return Path(settings.ingest_figure_dir) / paper_key


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
            logger.exception("Literature fetch failed for query %r using %s", query, fetcher.__name__)
            continue

    seen: set[str] = set()
    unique_papers: list[Paper] = []
    for paper in papers:
        if paper.id in seen:
            continue
        seen.add(paper.id)
        unique_papers.append(paper)

    eligible_papers = exclude_holdout_papers(unique_papers)
    if not eligible_papers:
        return 0

    store = VectorStore()
    chunks: list[Chunk] = []
    try:
        store.ensure_collection()
        existing_ids = store.list_ids()
        for paper in eligible_papers:
            chunks.extend(chunk for chunk in _chunk_paper(paper) if chunk.id not in existing_ids)
            if not paper.pdf_url:
                continue

            pdf_path = download_pdf(
                paper,
                Path(settings.ingest_pdf_dir),
                max_bytes=settings.ingest_pdf_max_bytes,
            )
            if pdf_path is None:
                continue

            try:
                body_chunks = extract_text_chunks(paper, pdf_path)
                chunks.extend(chunk for chunk in body_chunks if chunk.id not in existing_ids)
                figures = extract_figures(pdf_path, _paper_figure_dir(paper))
            except Exception:
                logger.exception("PDF parsing failed for paper %s", paper.id)
                continue

            for image_path, caption in figures:
                figure_id = f"{paper.id}:figure:{image_path.stem}"
                if figure_id in existing_ids:
                    continue
                try:
                    chunks.append(describe_figure(image_path, paper.id, caption))
                except Exception:
                    logger.exception("Figure description failed for paper %s: %s", paper.id, image_path)

        if not chunks:
            return 0

        existing_ids = store.list_ids()
        new_chunks = [chunk for chunk in chunks if chunk.id not in existing_ids]
        if not new_chunks:
            return 0

        return store.upsert(new_chunks)
    finally:
        store.close()
