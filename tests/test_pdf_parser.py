from contextlib import nullcontext
from datetime import datetime

import pymupdf
import pytest

from discovery_agent.ingestion import pdf_parser
from discovery_agent.schemas import Paper


def _paper(**overrides) -> Paper:
    values = {
        "id": "arxiv:2401.01234",
        "source": "arxiv",
        "title": "A study",
        "abstract": "An abstract.",
        "published": datetime(2024, 1, 1),
        "url": "https://example.test/paper",
        "pdf_url": "https://example.test/paper.pdf",
    }
    return Paper(**(values | overrides))


def test_download_pdf_streams_valid_pdf_and_uses_safe_cached_path(tmp_path, monkeypatch) -> None:
    paper = _paper(id="../../outside")

    class Response:
        def raise_for_status(self) -> None:
            pass

        def iter_bytes(self):
            yield b"%PDF-1.7\n"
            yield b"pdf body"

    calls = []

    def fake_stream(method, url, **kwargs):
        calls.append((method, url, kwargs))
        return nullcontext(Response())

    monkeypatch.setattr(pdf_parser.httpx, "stream", fake_stream)

    path = pdf_parser.download_pdf(paper, tmp_path)
    cached_path = pdf_parser.download_pdf(paper, tmp_path)

    assert path is not None
    assert path.parent == tmp_path
    assert path.read_bytes() == b"%PDF-1.7\npdf body"
    assert cached_path == path
    assert len(calls) == 1
    assert calls[0][2]["follow_redirects"] is True
    assert not list(tmp_path.glob("*.part"))


def test_download_pdf_rejects_non_pdf_and_oversized_responses(tmp_path, monkeypatch) -> None:
    class Response:
        def __init__(self, content: bytes):
            self.content = content

        def raise_for_status(self) -> None:
            pass

        def iter_bytes(self):
            yield self.content

    responses = iter([Response(b"not a pdf"), Response(b"%PDF-oversized")])
    monkeypatch.setattr(pdf_parser.httpx, "stream", lambda *_, **__: nullcontext(next(responses)))

    assert pdf_parser.download_pdf(_paper(), tmp_path) is None
    assert pdf_parser.download_pdf(_paper(id="arxiv:second"), tmp_path, max_bytes=8) is None
    assert list(tmp_path.iterdir()) == []


def test_download_pdf_skips_non_http_urls(tmp_path) -> None:
    assert pdf_parser.download_pdf(_paper(pdf_url="file:///tmp/paper.pdf"), tmp_path) is None
    assert list(tmp_path.iterdir()) == []


def test_extract_text_chunks_reads_pdf_pages_with_overlap_and_metadata(tmp_path) -> None:
    pdf_path = tmp_path / "paper.pdf"
    document = pymupdf.open()
    first_page = document.new_page()
    first_page.insert_textbox((40, 40, 550, 780), "alpha " * 150)
    second_page = document.new_page()
    second_page.insert_textbox((40, 40, 550, 780), "omega " * 100)
    document.save(pdf_path)
    document.close()

    chunks = pdf_parser.extract_text_chunks(_paper(), pdf_path, chunk_size=300, overlap=50)

    assert len(chunks) > 2
    assert chunks[0].kind == "body"
    assert chunks[0].id == "arxiv:2401.01234:body:0000"
    assert chunks[0].metadata["page_start"] == 1
    assert all(chunk.metadata["source"] == "arxiv" for chunk in chunks)
    assert any(chunk.metadata["page_start"] == 2 for chunk in chunks)
    assert chunks[0].text[-50:] in chunks[1].text


def test_extract_text_chunks_uses_abstract_when_pdf_is_missing(tmp_path) -> None:
    chunks = pdf_parser.extract_text_chunks(_paper(), tmp_path / "missing.pdf", chunk_size=20, overlap=5)

    assert chunks[0].text == "An abstract."
    assert chunks[0].metadata["page_start"] == 1


@pytest.mark.parametrize(
    ("chunk_size", "overlap"),
    [(0, 0), (10, -1), (10, 10)],
)
def test_extract_text_chunks_rejects_invalid_chunking(chunk_size, overlap, tmp_path) -> None:
    with pytest.raises(ValueError):
        pdf_parser.extract_text_chunks(
            _paper(),
            tmp_path / "missing.pdf",
            chunk_size=chunk_size,
            overlap=overlap,
        )


def test_extract_figures_saves_embedded_image_and_associates_caption(tmp_path) -> None:
    image = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 120, 80), False)
    image.clear_with(180)
    document = pymupdf.open()
    page = document.new_page()
    page.insert_image(pymupdf.Rect(80, 100, 440, 340), stream=image.tobytes("png"))
    page.insert_text((80, 370), "Figure 1. Treatment response by study group.")
    pdf_path = tmp_path / "figures.pdf"
    document.save(pdf_path)
    document.close()

    figures = pdf_parser.extract_figures(pdf_path, tmp_path / "extracted")

    assert len(figures) == 1
    image_path, caption = figures[0]
    assert image_path.is_file()
    assert image_path.suffix == ".png"
    assert caption.startswith("Figure 1.")


def test_extract_figures_renders_vector_chart_regions(tmp_path) -> None:
    document = pymupdf.open()
    page = document.new_page()
    page.draw_line((100, 100), (100, 300), color=(0, 0, 0), width=2)
    page.draw_line((100, 300), (400, 300), color=(0, 0, 0), width=2)
    page.draw_line((120, 260), (200, 190), color=(0, 0, 1), width=2)
    page.draw_line((200, 190), (300, 150), color=(0, 0, 1), width=2)
    page.insert_text((100, 350), "Figure 3. Increasing response over time.")
    pdf_path = tmp_path / "vector.pdf"
    document.save(pdf_path)
    document.close()

    figures = pdf_parser.extract_figures(pdf_path, tmp_path / "vector-extracted")

    assert len(figures) == 1
    image_path, caption = figures[0]
    assert image_path.is_file()
    assert image_path.stat().st_size > 0
    assert caption.startswith("Figure 3.")
