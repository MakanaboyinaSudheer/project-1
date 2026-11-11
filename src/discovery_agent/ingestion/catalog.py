"""Persistent paper catalog (data/papers.json): metadata Milvus does not store, for UI and citations."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from discovery_agent.config import settings
from discovery_agent.schemas import Paper


def load_catalog(path: str | Path | None = None) -> dict[str, dict]:
    file = Path(path or settings.paper_catalog)
    try:
        data = json.loads(file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def record_papers(
    papers: list[Paper], query: str, chunk_counts: dict[str, dict[str, int]], path: str | Path | None = None
) -> None:
    """Merge ingested papers into the catalog with per-kind chunk counts and ingestion time."""
    file = Path(path or settings.paper_catalog)
    catalog = load_catalog(file)
    now = datetime.now(UTC).isoformat()
    for paper in papers:
        entry = catalog.get(paper.id, {})
        counts = entry.get("chunks", {})
        for kind, n in chunk_counts.get(paper.id, {}).items():
            counts[kind] = counts.get(kind, 0) + n
        catalog[paper.id] = {
            **paper.model_dump(mode="json"),
            "queries": sorted({*entry.get("queries", []), query}),
            "chunks": counts,
            "ingested_at": entry.get("ingested_at", now),
            "updated_at": now,
        }
    file.parent.mkdir(parents=True, exist_ok=True)
    tmp = file.with_suffix(".tmp")
    tmp.write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    tmp.replace(file)
