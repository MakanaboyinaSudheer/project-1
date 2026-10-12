

import arxiv

from tenacity import retry, stop_after_attempt, wait_exponential
from discovery_agent.schemas import Paper


# Retry up to 3 times if a request fails.
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
def fetch_arxiv(
    query: str,
    categories: list[str],
    max_results: int = 50,
) -> list[Paper]:
    """Return newest-first ArXiv preprints matching query and categories."""

    client = arxiv.Client(delay_seconds=3.0)

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    papers = []

    for result in client.results(search):
        # Filter papers by the requested ArXiv categories.
        if categories:
            paper_categories = set(result.categories)

            if not paper_categories.intersection(categories):
                continue

        # Convert the ArXiv result into the project's Paper model.
        paper = Paper(
            id=f"arxiv:{result.get_short_id()}",
            source="arxiv",
            title=result.title,
            abstract=result.summary,
            authors=[author.name for author in result.authors],
            published=result.published,
            url=result.entry_id,
            pdf_url=result.pdf_url,
            peer_reviewed=False,
        )

        papers.append(paper)

    # Keep the newest papers first.
    papers.sort(
        key=lambda paper: paper.published.timestamp()
        if paper.published
        else 0,
        reverse=True,
    )

    return papers