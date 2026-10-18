"""PubMed fetcher via NCBI Entrez (Bio.Entrez)."""

from datetime import datetime
from xml.etree import ElementTree

from Bio import Entrez

from discovery_agent.config import settings
from discovery_agent.schemas import Paper


def fetch_pubmed(query: str, max_results: int = 50) -> list[Paper]:
    """Fetch recent PubMed records and convert them to Paper objects."""
    if not query.strip() or max_results <= 0:
        return []

    # Configure NCBI identification.
    Entrez.email = settings.ncbi_email

    if settings.ncbi_api_key:
        Entrez.api_key = settings.ncbi_api_key

    # Search for recent PubMed records.
    with Entrez.esearch(
        db="pubmed",
        term=query,
        sort="pub_date",
        retmax=max_results,
    ) as handle:
        search_data = Entrez.read(handle)

    paper_ids = search_data.get("IdList", [])

    if not paper_ids:
        return []

    # Fetch the selected records as XML.
    with Entrez.efetch(
        db="pubmed",
        id=",".join(paper_ids),
        retmode="xml",
    ) as handle:
        xml_data = handle.read()

    root = ElementTree.fromstring(xml_data)
    papers = []

    for article in root.findall(".//PubmedArticle"):
        citation = article.find("./MedlineCitation")

        if citation is None:
            continue

        article_data = citation.find("./Article")

        if article_data is None:
            continue

        # Extract the article title.
        title_element = article_data.find("./ArticleTitle")
        title = "".join(title_element.itertext()).strip() if title_element is not None else ""

        if not title:
            continue

        # Extract the abstract.
        abstract_parts = []

        for abstract_text in article_data.findall("./Abstract/AbstractText"):
            text = "".join(abstract_text.itertext()).strip()

            if text:
                label = abstract_text.attrib.get("Label")
                abstract_parts.append(f"{label}: {text}" if label else text)

        # Extract author names.
        authors = []

        for author in article_data.findall("./AuthorList/Author"):
            collective = author.findtext("./CollectiveName")
            last_name = author.findtext("./LastName")
            fore_name = author.findtext("./ForeName")

            name = collective or " ".join(part for part in (fore_name, last_name) if part)

            if name:
                authors.append(name)

        # Extract the publication date.
        pub_date = article_data.find("./Journal/JournalIssue/PubDate")
        published = _parse_publication_date(pub_date)

        # Identify journal articles.
        publication_types = {
            (element.text or "").strip().casefold()
            for element in article_data.findall("./PublicationTypeList/PublicationType")
        }

        peer_reviewed = "journal article" in publication_types

        pmid = citation.findtext("./PMID")

        if not pmid:
            continue

        # Convert the record into the project's Paper schema.
        papers.append(
            Paper(
                id=f"pubmed:{pmid}",
                source="pubmed",
                title=title,
                abstract=" ".join(abstract_parts),
                authors=authors,
                published=published,
                url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                pdf_url=None,
                peer_reviewed=peer_reviewed,
            )
        )

    # Return the newest records first.
    papers.sort(
        key=lambda paper: paper.published or datetime.min,
        reverse=True,
    )

    return papers


def _parse_publication_date(element) -> datetime | None:
    """Convert PubMed's partial publication date into a datetime."""
    if element is None:
        return None

    year = element.findtext("./Year")
    month = element.findtext("./Month")
    day = element.findtext("./Day")

    if not year:
        medline_date = element.findtext("./MedlineDate")

        if medline_date:
            year = medline_date[:4]

    if not year or not year.isdigit():
        return None

    try:
        month_number = int(month) if month and month.isdigit() else _month_number(month)
        day_number = int(day) if day and day.isdigit() else 1

        return datetime(
            int(year),
            month_number,
            day_number,
        )

    except ValueError:
        return None


def _month_number(month: str | None) -> int:
    """Convert a month name or abbreviation to a month number."""
    if not month:
        return 1

    try:
        return datetime.strptime(
            month[:3].title(),
            "%b",
        ).month
    except ValueError:
        return 1
