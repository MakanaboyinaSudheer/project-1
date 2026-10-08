"""Holdout set of recent human-authored papers, excluded from ingestion. Owner: Member 1 (Week 4)."""

from discovery_agent.schemas import Paper


def load_holdout(path: str = "data/holdout.jsonl") -> list[Paper]:
    raise NotImplementedError
