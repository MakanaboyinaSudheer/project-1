"""PDF download + text/figure extraction with PyMuPDF. Owner: Member 1 (Week 3)."""

import hashlib
import re
from pathlib import Path

import fitz
import httpx

from discovery_agent.schemas import Chunk, Paper


def download_pdf(paper: Paper, dest_dir: Path) -> Path | None:
    if not paper.pdf_url:
        return None
    dest_dir.mkdir(parents=True, exist_ok=True)
    safe_id = re.sub(r"[^A-Za-z0-9._-]", "_", paper.id)
    destination = dest_dir / f"{safe_id}.pdf"
    if destination.is_file():
        return destination
    response = httpx.get(paper.pdf_url, follow_redirects=True, timeout=60)
    response.raise_for_status()
    if not response.content.startswith(b"%PDF-"):
        raise ValueError(f"PDF URL for {paper.id} did not return a PDF document")
    temporary = destination.with_suffix(f".{hashlib.sha256(response.content).hexdigest()[:12]}.tmp")
    temporary.write_bytes(response.content)
    temporary.replace(destination)
    return destination


def extract_text_chunks(paper: Paper, pdf_path: Path, chunk_size: int = 1000) -> list[Chunk]:
    if chunk_size < 100:
        raise ValueError("chunk_size must be at least 100 characters")
    document = fitz.open(pdf_path)
    try:
        text = " ".join(re.sub(r"\s+", " ", page.get_text()).strip() for page in document)
        text = re.sub(r"\s+", " ", text).strip()
        overlap = min(200, chunk_size // 5)
        step = chunk_size - overlap
        metadata = {
            "source": paper.source,
            "peer_reviewed": paper.peer_reviewed,
            "title": paper.title,
            "url": paper.url,
            "published": paper.published,
        }
        return [
            Chunk(
                id=f"{paper.id}:body:{index}",
                paper_id=paper.id,
                text=text[start : start + chunk_size],
                kind="body",
                metadata=metadata,
            )
            for index, start in enumerate(range(0, len(text), step))
            if text[start : start + chunk_size].strip()
        ]
    finally:
        document.close()


def extract_figures(pdf_path: Path, out_dir: Path) -> list[tuple[Path, str]]:
    """Save embedded images / rendered figure regions as PNGs. Returns (path, caption) pairs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    document = fitz.open(pdf_path)
    saved: list[tuple[Path, str]] = []
    seen: set[str] = set()
    try:
        for page_number, page in enumerate(document, start=1):
            text_blocks = [
                block for block in page.get_text("blocks") if len(block) > 4 and isinstance(block[4], str)
            ]
            for image_index, image in enumerate(page.get_images(full=True), start=1):
                xref = image[0]
                extracted = document.extract_image(xref)
                image_bytes = extracted["image"]
                fingerprint = hashlib.sha256(image_bytes).hexdigest()
                if fingerprint in seen or len(image_bytes) < 2048:
                    continue
                seen.add(fingerprint)
                pixmap = fitz.Pixmap(document, xref)
                if pixmap.n > 4:
                    pixmap = fitz.Pixmap(fitz.csRGB, pixmap)
                if pixmap.alpha:
                    pixmap = fitz.Pixmap(pixmap, 0)
                path = out_dir / f"page-{page_number}-image-{image_index}.png"
                pixmap.save(path)
                caption = ""
                rects = page.get_image_rects(xref)
                if rects:
                    image_rect = rects[0]
                    nearby = sorted(
                        text_blocks,
                        key=lambda block: abs(block[1] - image_rect.y1) + abs(block[0] - image_rect.x0),
                    )
                    caption = next(
                        (
                            re.sub(r"\s+", " ", block[4]).strip()
                            for block in nearby
                            if re.search(r"\b(fig(?:ure)?\.?)\s*\d+", block[4], re.I)
                        ),
                        "",
                    )
                saved.append((path, caption))
    finally:
        document.close()
    return saved
