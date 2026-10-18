from datetime import datetime
from unittest.mock import MagicMock, patch

from discovery_agent.ingestion.pubmed_client import fetch_pubmed


def test_fetch_pubmed_returns_empty_for_invalid_input():
    assert fetch_pubmed("") == []
    assert fetch_pubmed("   ") == []
    assert fetch_pubmed("machine learning", 0) == []


@patch("discovery_agent.ingestion.pubmed_client.Entrez.esearch")
def test_fetch_pubmed_returns_empty_when_no_ids(mock_esearch):
    search_handle = MagicMock()
    search_handle.__enter__.return_value = search_handle
    mock_esearch.return_value = search_handle

    from unittest.mock import patch as mock_patch

    with mock_patch(
        "discovery_agent.ingestion.pubmed_client.Entrez.read",
        return_value={"IdList": []},
    ):
        assert fetch_pubmed("machine learning") == []

    mock_esearch.assert_called_once()


@patch("discovery_agent.ingestion.pubmed_client.Entrez.efetch")
@patch("discovery_agent.ingestion.pubmed_client.Entrez.esearch")
def test_fetch_pubmed_maps_article_fields(mock_esearch, mock_efetch):
    search_handle = MagicMock()
    search_handle.__enter__.return_value = search_handle
    mock_esearch.return_value = search_handle

    xml = b"""
    <PubmedArticleSet>
      <PubmedArticle>
        <MedlineCitation>
          <PMID>12345</PMID>
          <Article>
            <ArticleTitle>Machine Learning in Healthcare</ArticleTitle>
            <Abstract>
              <AbstractText Label="BACKGROUND">
                ML helps doctors.
              </AbstractText>
            </Abstract>
            <AuthorList>
              <Author>
                <ForeName>John</ForeName>
                <LastName>Smith</LastName>
              </Author>
            </AuthorList>
            <Journal>
              <JournalIssue>
                <PubDate>
                  <Year>2025</Year>
                  <Month>Jan</Month>
                  <Day>15</Day>
                </PubDate>
              </JournalIssue>
            </Journal>
            <PublicationTypeList>
              <PublicationType>Journal Article</PublicationType>
            </PublicationTypeList>
          </Article>
        </MedlineCitation>
      </PubmedArticle>
    </PubmedArticleSet>
    """

    fetch_handle = MagicMock()
    fetch_handle.__enter__.return_value = fetch_handle
    fetch_handle.read.return_value = xml
    mock_efetch.return_value = fetch_handle

    with patch(
        "discovery_agent.ingestion.pubmed_client.Entrez.read",
        return_value={"IdList": ["12345"]},
    ):
        papers = fetch_pubmed("machine learning", max_results=10)

    assert len(papers) == 1

    paper = papers[0]
    assert paper.id == "pubmed:12345"
    assert paper.source == "pubmed"
    assert paper.title == "Machine Learning in Healthcare"
    assert "ML helps doctors." in paper.abstract
    assert paper.authors == ["John Smith"]
    assert paper.published == datetime(2025, 1, 15)
    assert paper.url == "https://pubmed.ncbi.nlm.nih.gov/12345/"
    assert paper.peer_reviewed is True

    mock_esearch.assert_called_once_with(
        db="pubmed",
        term="machine learning",
        sort="pub_date",
        retmax=10,
    )
    mock_efetch.assert_called_once_with(
        db="pubmed",
        id="12345",
        retmode="xml",
    )