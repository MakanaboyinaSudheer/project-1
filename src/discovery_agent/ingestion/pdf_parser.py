"""Download papers and extract overlapping body-text and figure chunks."""

from __future__ import annotations

import hashlib
import logging
import re
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import httpx

from discovery_agent.schemas import Chunk, Paper

logger = logging.getLogger(__name__)
_PDF_MAGIC = b"%PDF-"
_CAPTION_PATTERN = re.compile(r"\b(?:fig(?:ure)?\.?\s*\d+|table\s+\d+)\b", re.IGNORECASE)
_MAX_IMAGE_PIXELS = 24_000_000


def _paper_filename(paper_id: str) -> str:
    safe_id = re.sub(r"[^A-Za-z0-9._-]+", "_", paper_id).strip("._-") or "paper"
    digest = hashlib.sha256(paper_id.encode()).hexdigest()[:12]
    return f"{safe_id[:80]}-{digest}.pdf"


def download_pdf(
    paper: Paper,
    dest_dir: Path,
    *,
    max_bytes: int = 50 * 1024 * 1024,
    timeout_s: float = 30.0,
) -> Path | None:
    """Download and cache a PDF, returning ``None`` for unavailable or invalid downloads."""
    if not paper.pdf_url:
        return None
    if max_bytes <= len(_PDF_MAGIC) or timeout_s <= 0:
        raise ValueError("max_bytes and timeout_s must be positive and allow a PDF header")

    parsed_url = urlparse(paper.pdf_url)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        logger.warning("Skipping non-HTTP PDF URL for %s", paper.id)
        return None

    destination = Path(dest_dir)
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / _paper_filename(paper.id)
    if target.is_file():
        try:
            with target.open("rb") as cached:
                if cached.read(len(_PDF_MAGIC)) == _PDF_MAGIC:
                    return target
        except OSError:
            logger.warning("Unable to read cached PDF for %s", paper.id, exc_info=True)

    temporary_path: Path | None = None
    try:
        with httpx.stream(
            "GET",
            paper.pdf_url,
            follow_redirects=True,
            timeout=timeout_s,
        ) as response:
            response.raise_for_status()
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=destination,
                prefix=f".{target.stem}-",
                suffix=".part",
                delete=False,
            ) as output:
                temporary_path = Path(output.name)
                header = bytearray()
                total_bytes = 0
                for block in response.iter_bytes():
                    total_bytes += len(block)
                    if total_bytes > max_bytes:
                        raise ValueError(f"PDF exceeds the configured {max_bytes}-byte limit")
                    if len(header) < len(_PDF_MAGIC):
                        header.extend(block[: len(_PDF_MAGIC) - len(header)])
                    output.write(block)
                if bytes(header) != _PDF_MAGIC:
                    raise ValueError("Downloaded content does not have a PDF signature")

        temporary_path.replace(target)
        return target
    except (httpx.HTTPError, OSError, ValueError) as exc:
        logger.warning("Could not download PDF for %s: %s", paper.id, exc)
        return None
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _chunk_ranges(text: str, chunk_size: int, overlap: int) -> list[tuple[int, int]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    ranges: list[tuple[int, int]] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start + chunk_size // 2, end)
            if boundary > start:
                end = boundary
        ranges.append((start, end))
        if end == len(text):
            break
        start = max(start + 1, end - overlap)
    return ranges


def extract_text_chunks(
    paper: Paper,
    pdf_path: Path,
    chunk_size: int = 1000,
    overlap: int = 150,
) -> list[Chunk]:
    """Extract normalized PDF text into overlapping body chunks with page metadata."""
    if not pdf_path or not Path(pdf_path).is_file():
        text = _normalize_text(paper.abstract or paper.title)
        page_ranges: list[tuple[int, int, int]] = [(0, len(text), 1)] if text else []
    else:
        import pymupdf

        with pymupdf.open(pdf_path) as document:
            pages = [_normalize_text(page.get_text()) for page in document]
        page_ranges = []
        parts: list[str] = []
        offset = 0
        for page_number, page_text in enumerate(pages, start=1):
            if not page_text:
                continue
            if parts:
                parts.append(" ")
                offset += 1
            page_ranges.append((offset, offset + len(page_text), page_number))
            parts.append(page_text)
            offset += len(page_text)
        text = "".join(parts)
        if not text:
            text = _normalize_text(paper.abstract or paper.title)
            page_ranges = [(0, len(text), 1)] if text else []

    if not text:
        return []

    metadata = {
        "source": paper.source,
        "peer_reviewed": paper.peer_reviewed,
        "published_ts": paper.published.isoformat() if paper.published else None,
        "title": paper.title,
    }
    chunks = []
    for index, (start, end) in enumerate(_chunk_ranges(text, chunk_size, overlap)):
        page_numbers = [
            page_number
            for page_start, page_end, page_number in page_ranges
            if page_start < end and page_end > start
        ]
        chunks.append(
            Chunk(
                id=f"{paper.id}:body:{index:04d}",
                paper_id=paper.id,
                text=text[start:end],
                kind="body",
                metadata={
                    **metadata,
                    "page_start": min(page_numbers) if page_numbers else None,
                    "page_end": max(page_numbers) if page_numbers else None,
                },
            )
        )
    return chunks


def _caption_blocks(page) -> list[tuple[tuple[float, float, float, float], str]]:
    captions: list[tuple[tuple[float, float, float, float], str]] = []
    for block in page.get_text("blocks"):
        if len(block) < 5 or not isinstance(block[4], str):
            continue
        caption = _normalize_text(block[4])
        if not _CAPTION_PATTERN.search(caption):
            continue
        captions.append(((block[0], block[1], block[2], block[3]), caption))
    return captions


def _caption_for_image(page, image_rect, caption_blocks) -> str:
    candidates: list[tuple[float, str]] = []
    for (x0, y0, x1, y1), caption in caption_blocks:
        vertical_distance = max(0.0, y0 - image_rect.y1, image_rect.y0 - y1)
        horizontal_gap = max(0.0, image_rect.x0 - x1, x0 - image_rect.x1)
        distance = vertical_distance + horizontal_gap
        if distance <= 160:
            candidates.append((distance, caption))
    return min(candidates, default=(0.0, ""))[1]


def extract_figures(pdf_path: Path, out_dir: Path) -> list[tuple[Path, str]]:
    """Extract embedded PDF images as PNGs and associate nearby figure captions."""
    if not pdf_path or not Path(pdf_path).is_file():
        return []

    import pymupdf

    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    extracted: list[tuple[Path, str]] = []
    with pymupdf.open(pdf_path) as document:
        for page_number, page in enumerate(document, start=1):
            figure_number = 0
            seen_xrefs: set[int] = set()
            caption_blocks = _caption_blocks(page)
            associated_captions: set[str] = set()
            for image_info in page.get_images(full=True):
                xref = int(image_info[0])
                if xref in seen_xrefs:
                    continue
                seen_xrefs.add(xref)
                width, height = int(image_info[2]), int(image_info[3])
                if width <= 0 or height <= 0 or width * height > _MAX_IMAGE_PIXELS:
                    logger.warning(
                        "Skipping image %s on page %s of %s with dimensions %sx%s",
                        xref,
                        page_number,
                        pdf_path,
                        width,
                        height,
                    )
                    continue
                try:
                    image = pymupdf.Pixmap(document, xref)
                    if image.n - image.alpha > 3:
                        image = pymupdf.Pixmap(pymupdf.csRGB, image)
                    image_rects = page.get_image_rects(xref)
                    if not image_rects:
                        image_rects = [page.rect]
                    for image_rect in image_rects:
                        figure_number += 1
                        image_path = output_dir / f"page-{page_number:04d}-figure-{figure_number:03d}.png"
                        image.save(image_path)
                        caption = _caption_for_image(page, image_rect, caption_blocks)
                        if caption:
                            associated_captions.add(caption)
                        extracted.append((image_path, caption))
                except (RuntimeError, ValueError, OSError) as exc:
                    logger.warning(
                        "Could not extract image %s on page %s of %s: %s",
                        xref,
                        page_number,
                        pdf_path,
                        exc,
                    )
            previous_caption_bottom = 0.0
            for _, (caption_rect, caption) in enumerate(caption_blocks):
                if caption in associated_captions:
                    continue
                caption_top = caption_rect[1]
                crop_top = max(
                    previous_caption_bottom,
                    caption_top - page.rect.height * 0.45,
                )
                previous_caption_bottom = caption_rect[3]
                if caption_top <= crop_top:
                    continue
                clip = pymupdf.Rect(
                    page.rect.x0,
                    crop_top,
                    page.rect.x1,
                    caption_top,
                )
                scale = min(1.5, 1600 / max(clip.width, clip.height))
                rendered = page.get_pixmap(
                    matrix=pymupdf.Matrix(scale, scale),
                    clip=clip,
                    alpha=False,
                )
                figure_number += 1
                image_path = output_dir / f"page-{page_number:04d}-figure-{figure_number:03d}.png"
                rendered.save(image_path)
                extracted.append((image_path, caption))
    return extracted
