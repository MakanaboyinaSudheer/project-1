import pytest

from discovery_agent.ingestion.arxiv_client import fetch_arxiv


@pytest.mark.integration
def test_fetch_at_least_ten_arxiv_papers():
    """Verify that the ArXiv client can fetch at least 10 real papers."""

    papers = fetch_arxiv(
        query="machine learning",
        categories=["cs.LG"],
        max_results=50,
    )

    assert len(papers) >= 10, f"Expected at least 10 papers, but received {len(papers)}"

    assert all(paper.source == "arxiv" for paper in papers)
    assert all(paper.peer_reviewed is False for paper in papers)

    print(f"\nSuccessfully fetched {len(papers)} ArXiv papers.")
