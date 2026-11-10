"""Describe extracted research figures with the configured vision-language model."""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path

from langchain_core.messages import HumanMessage

from discovery_agent.llm import get_vlm
from discovery_agent.schemas import Chunk


def _response_text(content: object) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [
            block["text"].strip()
            for block in content
            if isinstance(block, dict) and isinstance(block.get("text"), str) and block["text"].strip()
        ]
        return "\n".join(parts)
    return ""


def describe_figure(image_path: Path, paper_id: str, caption: str = "") -> Chunk:
    """Return a VLM-grounded figure chunk describing chart type, axes, trends, and values."""
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Figure image does not exist: {path}")
    if not paper_id.strip():
        raise ValueError("paper_id must not be empty")

    mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
    image_data = base64.b64encode(path.read_bytes()).decode("ascii")
    prompt = (
        "Describe this research-paper figure for semantic search. Identify the chart or "
        "diagram type, axes and units, visible trends, and key numeric values. Use only "
        "information visible in the figure; say when something is unreadable. Do not infer "
        "causality or fabricate values."
    )
    if caption.strip():
        prompt += f"\nPaper caption: {caption.strip()}"

    response = get_vlm().invoke(
        [
            HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{image_data}"},
                    },
                ]
            )
        ]
    )
    description = _response_text(response.content)
    if not description:
        raise ValueError(f"Vision model returned no figure description for {paper_id}")

    text = (
        f"Caption: {caption.strip()}\nVisual description: {description}" if caption.strip() else description
    )
    return Chunk(
        id=f"{paper_id}:figure:{path.stem}",
        paper_id=paper_id,
        text=text,
        kind="figure",
        metadata={"image_path": str(path), "caption": caption, "description": description},
    )
