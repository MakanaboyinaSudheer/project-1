"""PDF download + text/figure extraction with PyMuPDF. Owner: Member 1 (Week 3)."""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

from discovery_agent.schemas import Chunk, Paper


def download_pdf(paper: Paper, dest_dir: Path) -> Path | None:
    """Download the PDF if available and save it under a stable local filename."""
    pdf_url = paper.pdf_url
    if not pdf_url:
        return None

    dest_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{paper.id.replace('/', '_')}.pdf"
    target = dest_dir / filename
    try:
        urlretrieve(pdf_url, target)
        return target if target.exists() else None
    except Exception:
        return None


def extract_text_chunks(paper: Paper, pdf_path: Path, chunk_size: int = 1000) -> list[Chunk]:
    """Return a simple fallback chunk list using the abstract if the PDF isn't available."""
    if not pdf_path or not pdf_path.exists():
        text = (paper.abstract or paper.title).strip()
        if not text:
            return []
        return [
            Chunk(
                id=f"{paper.id}:body",
                paper_id=paper.id,
                text=text[: max(1, chunk_size)],
                kind="body",
                metadata={"source": paper.source, "peer_reviewed": paper.peer_reviewed},
            )
        ]

    text = (paper.abstract or paper.title).strip()
    if not text:
        return []
    return [
        Chunk(
            id=f"{paper.id}:body",
            paper_id=paper.id,
            text=text[: max(1, chunk_size)],
            kind="body",
            metadata={"source": paper.source, "peer_reviewed": paper.peer_reviewed},
        )
    ]


def extract_figures(pdf_path: Path, out_dir: Path) -> list[tuple[Path, str]]:
    """Save embedded images / rendered figure regions as PNGs. Returns (path, caption) pairs."""
    if not pdf_path or not pdf_path.exists():
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    return []
