"""Ingestion orchestration: fetch -> dedupe -> chunk -> embed -> upsert. Owner: Member 1."""

import logging
from pathlib import Path

from discovery_agent.config import settings
from discovery_agent.evaluation.holdout import load_holdout
from discovery_agent.ingestion.arxiv_client import fetch_arxiv
from discovery_agent.ingestion.figure_parser import describe_figure
from discovery_agent.ingestion.pdf_parser import download_pdf, extract_figures, extract_text_chunks
from discovery_agent.ingestion.pubmed_client import fetch_pubmed
from discovery_agent.retrieval.vector_store import VectorStore
from discovery_agent.schemas import Chunk, Paper

logger = logging.getLogger(__name__)


def run_ingestion(query: str, max_results: int | None = None) -> int:
    """Ingest papers for `query` from ArXiv + PubMed. Returns number of new chunks upserted.

    Must be idempotent (re-running does not create duplicates) so it can run on a schedule.
    """
    limit = settings.ingest_max_results if max_results is None else max_results
    if limit < 1:
        raise ValueError("max_results must be positive")
    papers_by_id: dict[str, Paper] = {}
    papers = fetch_arxiv(query, settings.arxiv_category_list, limit)
    if settings.ncbi_email:
        papers.extend(fetch_pubmed(query, limit))
    else:
        logger.warning("Skipping PubMed ingestion because NCBI_EMAIL is not configured")
    holdout_path = Path("data/holdout.jsonl")
    if holdout_path.exists():
        holdout_ids = {paper.id for paper in load_holdout(holdout_path)}
        papers = [paper for paper in papers if paper.id not in holdout_ids]
    else:
        logger.warning("No holdout dataset at %s; no paper IDs can be excluded", holdout_path)
    for paper in papers:
        papers_by_id.setdefault(paper.id, paper)

    chunks: list[Chunk] = []
    for paper in papers_by_id.values():
        metadata = {
            "source": paper.source,
            "peer_reviewed": paper.peer_reviewed,
            "published": paper.published,
            "url": paper.url,
            "title": paper.title,
        }
        if paper.abstract:
            chunks.append(
                Chunk(
                    id=f"{paper.id}:abstract",
                    paper_id=paper.id,
                    text=f"{paper.title}\n\n{paper.abstract}",
                    kind="abstract",
                    metadata=metadata,
                )
            )
        if paper.pdf_url:
            pdf_path = download_pdf(paper, Path("data/pdfs"))
            if pdf_path is not None:
                chunks.extend(extract_text_chunks(paper, pdf_path))
                image_dir = Path("data/figures") / paper.id.replace(":", "_")
                for image_path, caption in extract_figures(pdf_path, image_dir):
                    figure_chunk = describe_figure(image_path, paper.id, caption)
                    figure_chunk.metadata.update(metadata)
                    chunks.append(figure_chunk)
    store = VectorStore()
    try:
        existing_ids = store.get_existing_ids([chunk.id for chunk in chunks])
        new_chunks = [chunk for chunk in chunks if chunk.id not in existing_ids]
        return store.upsert(new_chunks)
    finally:
        store.close()
