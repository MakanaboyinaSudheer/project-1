"""Chart/graph understanding with a VLM (LLaVA via get_vlm()). Owner: Member 1 (Week 3)."""

from pathlib import Path

from discovery_agent.schemas import Chunk


def describe_figure(image_path: Path, paper_id: str, caption: str = "") -> Chunk:
    """Return a kind='figure' Chunk: chart type, axes, trends, key numeric values."""
    text = caption.strip() or f"Figure from {paper_id}"
    return Chunk(
        id=f"{paper_id}:figure:{image_path.stem}",
        paper_id=paper_id,
        text=text,
        kind="figure",
        metadata={"image_path": str(image_path), "caption": caption},
    )
