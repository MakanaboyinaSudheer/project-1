"""Holdout set of recent human-authored papers, excluded from ingestion. Owner: Member 1 (Week 4)."""

import json
from pathlib import Path

from discovery_agent.schemas import Paper


def load_holdout(path: str | Path = "data/holdout.jsonl") -> list[Paper]:
    holdout_path = Path(path)
    if not holdout_path.exists():
        raise FileNotFoundError(f"Holdout dataset not found: {holdout_path}")
    papers = []
    seen: set[str] = set()
    for line_number, line in enumerate(holdout_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            paper = Paper.model_validate(json.loads(line))
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"Invalid holdout record on line {line_number}: {exc}") from exc
        if paper.id in seen:
            raise ValueError(f"Duplicate holdout paper id {paper.id!r} on line {line_number}")
        seen.add(paper.id)
        papers.append(paper)
    return papers
