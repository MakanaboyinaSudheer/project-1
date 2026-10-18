"""PubMed fetcher via NCBI Entrez (Bio.Entrez). Owner: Member 1 (Week 1)."""

import threading
import time
from datetime import UTC, datetime

from Bio import Entrez

from discovery_agent.config import settings
from discovery_agent.schemas import Paper

_RATE_LOCK = threading.Lock()
_LAST_REQUEST = 0.0


def _respect_rate_limit() -> None:
    global _LAST_REQUEST
    interval = 0.1 if settings.ncbi_api_key else 1 / 3
    with _RATE_LOCK:
        now = time.monotonic()
        delay = interval - (now - _LAST_REQUEST)
        if delay > 0:
            time.sleep(delay)
        _LAST_REQUEST = time.monotonic()


def _published_date(article: dict) -> datetime | None:
    pub_date = article.get("MedlineDate")
    if pub_date:
        try:
            return datetime.strptime(pub_date[:4], "%Y").replace(tzinfo=UTC)
        except ValueError:
            return None

    year = article.get("Year")
    month = article.get("Month", "1")
    day = article.get("Day", "1")
    if not year:
        return None
    month = {
        "Jan": "1",
        "Feb": "2",
        "Mar": "3",
        "Apr": "4",
        "May": "5",
        "Jun": "6",
        "Jul": "7",
        "Aug": "8",
        "Sep": "9",
        "Oct": "10",
        "Nov": "11",
        "Dec": "12",
    }.get(str(month)[:3], month)
    try:
        return datetime(int(year), int(month), int(day), tzinfo=UTC)
    except (TypeError, ValueError):
        return None


def _abstract_text(article: dict) -> str:
    abstract = article.get("Abstract", {}).get("AbstractText", [])
    if isinstance(abstract, str):
        return abstract.strip()
    return " ".join(str(part).strip() for part in abstract if str(part).strip())


def fetch_pubmed(query: str, max_results: int = 50) -> list[Paper]:
    """Return recent PubMed records (peer_reviewed=True for journal articles).
    Respect NCBI rate limits (3 req/s, 10 with NCBI_API_KEY)."""
    if max_results < 1:
        raise ValueError("max_results must be positive")
    if not settings.ncbi_email:
        raise ValueError("NCBI_EMAIL must be configured before querying PubMed")

    Entrez.email = settings.ncbi_email  # type: ignore[assignment]
    Entrez.api_key = settings.ncbi_api_key or None  # type: ignore[assignment]
    _respect_rate_limit()
    search_handle = Entrez.esearch(
        db="pubmed",
        term=query,
        retmax=max_results,
        sort="pub date",
    )
    try:
        search = Entrez.read(search_handle)
    finally:
        search_handle.close()

    identifiers = search.get("IdList", [])
    if not identifiers:
        return []
    _respect_rate_limit()
    fetch_handle = Entrez.efetch(db="pubmed", id=",".join(identifiers), retmode="xml")
    try:
        records = Entrez.read(fetch_handle)
    finally:
        fetch_handle.close()

    papers = []
    for record in records.get("PubmedArticle", []):
        citation = record.get("MedlineCitation", {})
        article = citation.get("Article", {})
        identifier = str(citation.get("PMID", "")).strip()
        if not identifier:
            continue
        authors = []
        for author in article.get("AuthorList", []):
            name = " ".join(
                str(author.get(part, "")).strip() for part in ("ForeName", "LastName") if author.get(part)
            )
            if name:
                authors.append(name)
        journal_types = {
            str(item.get("PublicationType", "")).casefold() for item in article.get("PublicationTypeList", [])
        }
        pub_date = article.get("Journal", {}).get("JournalIssue", {}).get("PubDate", {})
        papers.append(
            Paper(
                id=f"pubmed:{identifier}",
                source="pubmed",
                title=str(article.get("ArticleTitle", "")).strip(),
                abstract=_abstract_text(article),
                authors=authors,
                published=_published_date(pub_date),
                url=f"https://pubmed.ncbi.nlm.nih.gov/{identifier}/",
                peer_reviewed="journal article" in journal_types,
            )
        )
    return papers
