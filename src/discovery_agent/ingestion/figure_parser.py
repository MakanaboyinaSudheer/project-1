"""Chart/graph understanding with a VLM (LLaVA via get_vlm()). Owner: Member 1 (Week 3)."""

import base64
import hashlib
from pathlib import Path

from langchain_core.messages import HumanMessage

from discovery_agent.llm import get_vlm
from discovery_agent.schemas import Chunk


def describe_figure(image_path: Path, paper_id: str, caption: str = "") -> Chunk:
    """Return a kind='figure' Chunk: chart type, axes, trends, key numeric values."""
    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    image_data = base64.b64encode(image_path.read_bytes()).decode("ascii")
    mime_types = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
    mime = mime_types.get(image_path.suffix.casefold())
    if mime is None:
        raise ValueError(f"Unsupported figure image format: {image_path.suffix}")
    response = get_vlm().invoke(
        [
            HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": (
                            "Describe this scientific figure accurately. Identify the chart type, "
                            "axes and units, trends, and visible key values. Do not infer unreadable values. "
                            f"Caption: {caption or '(not provided)'}"
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_data}"}},
                ]
            )
        ]
    )
    description = response.content
    if not isinstance(description, str) or not description.strip():
        raise ValueError("Vision model returned an empty figure description")
    figure_id = hashlib.sha256(image_path.read_bytes()).hexdigest()[:16]
    return Chunk(
        id=f"{paper_id}:figure:{figure_id}",
        paper_id=paper_id,
        text=f"Figure description: {description.strip()}\nCaption: {caption}".strip(),
        kind="figure",
        metadata={"image_path": str(image_path), "caption": caption},
    )
