from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from discovery_agent.ingestion.arxiv_client import fetch_arxiv


def make_arxiv_result(
    paper_id: str,
    categories: list[str],
    published: datetime,
) -> MagicMock:
    """Create a fake ArXiv result for testing."""
    result = MagicMock()
    result.get_short_id.return_value = paper_id
    result.title = f"Test paper {paper_id}"
    result.summary = f"Abstract for {paper_id}"
    result.authors = [MagicMock(name="author")]
    result.authors[0].name = "Test Author"
    result.published = published
    result.entry_id = f"https://arxiv.org/abs/{paper_id}"
    result.pdf_url = f"https://arxiv.org/pdf/{paper_id}"
    result.categories = categories
    return result


@patch("discovery_agent.ingestion.arxiv_client.arxiv.Client")
def test_fetch_arxiv_converts_results_to_paper(mock_client_class):
    """Check that ArXiv results are converted to Paper objects."""
    published = datetime(2026, 10, 10, tzinfo=timezone.utc)
    result = make_arxiv_result("2601.00001", ["cs.LG"], published)

    mock_client = mock_client_class.return_value
    mock_client.results.return_value = [result]

    papers = fetch_arxiv("machine learning", ["cs.LG"], 10)

    assert len(papers) == 1
    assert papers[0].id == "arxiv:2601.00001"
    assert papers[0].source == "arxiv"
    assert papers[0].title == "Test paper 2601.00001"
    assert papers[0].peer_reviewed is False


@patch("discovery_agent.ingestion.arxiv_client.arxiv.Client")
def test_fetch_arxiv_filters_categories(mock_client_class):
    """Check that papers outside the requested category are excluded."""
    published = datetime(2026, 10, 10, tzinfo=timezone.utc)

    matching = make_arxiv_result("2601.00001", ["cs.LG"], published)
    non_matching = make_arxiv_result("2601.00002", ["math.AG"], published)

    mock_client = mock_client_class.return_value
    mock_client.results.return_value = [matching, non_matching]

    papers = fetch_arxiv("machine learning", ["cs.LG"], 10)

    assert len(papers) == 1
    assert papers[0].id == "arxiv:2601.00001"


@patch("discovery_agent.ingestion.arxiv_client.arxiv.Client")
def test_fetch_arxiv_returns_newest_first(mock_client_class):
    """Check that newer papers appear before older papers."""
    older = make_arxiv_result(
        "2601.00001",
        ["cs.LG"],
        datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    newer = make_arxiv_result(
        "2601.00002",
        ["cs.LG"],
        datetime(2026, 9, 1, tzinfo=timezone.utc),
    )

    mock_client = mock_client_class.return_value
    mock_client.results.return_value = [older, newer]

    papers = fetch_arxiv("machine learning", ["cs.LG"], 10)

    assert [paper.id for paper in papers] == [
        "arxiv:2601.00002",
        "arxiv:2601.00001",
    ]
    
@patch("discovery_agent.ingestion.arxiv_client.arxiv.Client")
def test_fetch_arxiv_retries_after_failure(mock_client_class, monkeypatch):
    """Check that Tenacity retries when an ArXiv request fails."""
    from tenacity import wait_none

    # Avoid waiting during this unit test.
    monkeypatch.setattr(fetch_arxiv.retry, "wait", wait_none())

    mock_client = mock_client_class.return_value

    # First call fails; second call succeeds.
    mock_client.results.side_effect = [
        RuntimeError("Temporary ArXiv error"),
        [],
    ]

    papers = fetch_arxiv("machine learning", ["cs.LG"], 10)

    assert papers == []
    assert mock_client_class.call_count == 2