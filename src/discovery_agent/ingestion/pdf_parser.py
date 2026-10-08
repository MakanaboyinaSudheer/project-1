"""PDF download + text/figure extraction with PyMuPDF. Owner: Member 1 (Week 3)."""

from pathlib import Path

from discovery_agent.schemas import Chunk, Paper


def download_pdf(paper: Paper, dest_dir: Path) -> Path | None:
    raise NotImplementedError


def extract_text_chunks(paper: Paper, pdf_path: Path, chunk_size: int = 1000) -> list[Chunk]:
    raise NotImplementedError


def extract_figures(pdf_path: Path, out_dir: Path) -> list[tuple[Path, str]]:
    """Save embedded images / rendered figure regions as PNGs. Returns (path, caption) pairs."""
    raise NotImplementedError
