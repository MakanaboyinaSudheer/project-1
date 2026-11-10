import pytest

from discovery_agent.ingestion import figure_parser


class FakeVisionModel:
    def __init__(self, content):
        self.content = content
        self.messages = None

    def invoke(self, messages):
        self.messages = messages
        return type("Response", (), {"content": self.content})()


def test_describe_figure_sends_image_and_caption_to_configured_vlm(tmp_path, monkeypatch) -> None:
    image_path = tmp_path / "chart.png"
    image_path.write_bytes(b"png-bytes")
    model = FakeVisionModel("A line chart shows an increasing treatment response.")
    monkeypatch.setattr(figure_parser, "get_vlm", lambda: model)

    chunk = figure_parser.describe_figure(image_path, "pubmed:123", "Figure 2. Response over time.")

    assert chunk.kind == "figure"
    assert chunk.id == "pubmed:123:figure:chart"
    assert chunk.text.startswith("Caption: Figure 2.")
    assert "increasing treatment response" in chunk.text
    assert chunk.metadata["description"] == model.content
    content = model.messages[0].content
    assert content[0]["type"] == "text"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_describe_figure_rejects_missing_image_or_empty_vlm_response(tmp_path, monkeypatch) -> None:
    with pytest.raises(FileNotFoundError):
        figure_parser.describe_figure(tmp_path / "missing.png", "arxiv:1")

    image_path = tmp_path / "chart.png"
    image_path.write_bytes(b"png-bytes")
    monkeypatch.setattr(figure_parser, "get_vlm", lambda: FakeVisionModel(""))

    with pytest.raises(ValueError, match="no figure description"):
        figure_parser.describe_figure(image_path, "arxiv:1")


def test_describe_figure_extracts_text_from_multimodal_response(tmp_path, monkeypatch) -> None:
    image_path = tmp_path / "chart.png"
    image_path.write_bytes(b"png-bytes")
    response = [{"type": "text", "text": "Axes show time and response."}]
    monkeypatch.setattr(figure_parser, "get_vlm", lambda: FakeVisionModel(response))

    chunk = figure_parser.describe_figure(image_path, "arxiv:1")

    assert chunk.text == "Axes show time and response."
