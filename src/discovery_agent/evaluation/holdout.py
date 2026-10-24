"""Load and exclude the fixed evaluation set from ingestion."""

import json
import sys
from collections.abc import Iterable
from pathlib import Path

from discovery_agent.schemas import Paper

DEFAULT_HOLDOUT_PATH = Path("data/holdout.jsonl")


def load_holdout(path: str | Path = DEFAULT_HOLDOUT_PATH) -> list[Paper]:
    """Load and validate papers in the holdout JSONL file."""
    holdout_path = Path(path)
    if not holdout_path.is_file() and holdout_path == DEFAULT_HOLDOUT_PATH:
        installed_path = Path(sys.prefix) / "share" / "discovery-agent" / "holdout.jsonl"
        if installed_path.is_file():
            holdout_path = installed_path
    if not holdout_path.is_file():
        raise FileNotFoundError(f"Holdout dataset not found: {holdout_path}")

    papers: list[Paper] = []
    seen: set[str] = set()
    for line_number, line in enumerate(holdout_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            paper = Paper.model_validate(record)
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"Invalid holdout record on line {line_number}: {exc}") from exc
        if paper.id in seen:
            raise ValueError(f"Duplicate holdout paper id {paper.id!r} on line {line_number}")
        seen.add(paper.id)
        papers.append(paper)
    return papers


def exclude_holdout_papers(
    papers: Iterable[Paper],
    holdout_path: str | Path = DEFAULT_HOLDOUT_PATH,
) -> list[Paper]:
    """Return candidate ingestion papers excluding every holdout paper ID."""
    holdout_ids = {paper.id for paper in load_holdout(holdout_path)}
    return [paper for paper in papers if paper.id not in holdout_ids]
