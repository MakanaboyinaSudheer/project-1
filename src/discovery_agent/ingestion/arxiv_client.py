"""ArXiv fetcher. Owner: Member 1 (Week 1)."""

from discovery_agent.schemas import Paper


def fetch_arxiv(query: str, categories: list[str], max_results: int = 50) -> list[Paper]:
    """Return newest-first preprints matching `query` within `categories` (peer_reviewed=False)."""
    raise NotImplementedError
