"""PubMed fetcher via NCBI Entrez (Bio.Entrez). Owner: Member 1 (Week 1)."""

from discovery_agent.schemas import Paper


def fetch_pubmed(query: str, max_results: int = 50) -> list[Paper]:
    """Return recent PubMed records (peer_reviewed=True for journal articles).
    Respect NCBI rate limits (3 req/s, 10 with NCBI_API_KEY)."""
    raise NotImplementedError
