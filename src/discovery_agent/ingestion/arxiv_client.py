"""ArXiv fetcher with bounded retries."""

from datetime import UTC

import arxiv
from tenacity import retry, stop_after_attempt, wait_exponential

from discovery_agent.schemas import Paper


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
def fetch_arxiv(query: str, categories: list[str], max_results: int = 50) -> list[Paper]:
    """Return newest-first preprints matching `query` within `categories` (peer_reviewed=False)."""
    if max_results < 1:
        raise ValueError("max_results must be positive")

    category_query = " OR ".join(f"cat:{category}" for category in categories)
    search_query = f"({query}) AND ({category_query})" if category_query else query
    search = arxiv.Search(
        query=search_query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )
    results = arxiv.Client().results(search)
    category_filter = set(categories)
    papers = []
    for result in results:
        result_categories = set(result.categories)
        if category_filter and not category_filter.intersection(result_categories):
            continue
        published = result.published
        if published.tzinfo is None:
            published = published.replace(tzinfo=UTC)
        papers.append(
            Paper(
                id=f"arxiv:{result.get_short_id()}",
                source="arxiv",
                title=result.title.strip(),
                abstract=result.summary.strip(),
                authors=[author.name for author in result.authors],
                published=published,
                url=result.entry_id,
                pdf_url=result.pdf_url,
                peer_reviewed=False,
            )
        )
    return sorted(
        papers,
        key=lambda paper: paper.published.timestamp() if paper.published else 0,
        reverse=True,
    )[:max_results]
