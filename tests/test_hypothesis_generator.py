"""Hypothesis Generator: citations must be grounded in retrieved literature."""

from unittest.mock import MagicMock, patch

import pytest

from discovery_agent.agents.hypothesis_generator import hypothesis_generator_node
from discovery_agent.schemas import Citation, CriticReport, Hypothesis, RetrievedChunk


def _chunk(paper_id: str, text: str, peer_reviewed: bool = True) -> RetrievedChunk:
    return RetrievedChunk(
        id=f"{paper_id}:abstract",
        paper_id=paper_id,
        text=text,
        score=0.9,
        metadata={"title": f"T {paper_id}", "peer_reviewed": peer_reviewed},
    )


def _hyp(*citations: Citation, statement: str = "H") -> Hypothesis:
    return Hypothesis(
        statement=statement, rationale="r", citations=list(citations), test_plan="t", expected_outcome="e"
    )


def _run(state, outputs):
    chain = MagicMock()
    chain.invoke.side_effect = outputs
    with (
        patch("discovery_agent.agents.hypothesis_generator.HYPOTHESIS_PROMPT") as prompt,
        patch("discovery_agent.agents.hypothesis_generator.get_llm"),
    ):
        prompt.__or__.return_value = chain
        return hypothesis_generator_node(state), chain


STATE = {
    "topic": "AI",
    "thread_id": "t1",
    "literature": [_chunk("pubmed:1", "Label noise hurts calibration.")],
}


def test_rejects_hallucinated_citations_and_retries():
    bad = _hyp(Citation(paper_id="pubmed:99", title="fake", quote="x"))
    good = _hyp(Citation(paper_id="pubmed:1", title="llm title", quote="Label noise hurts"), statement="ok")
    result, chain = _run(STATE, [bad, good])

    assert chain.invoke.call_count == 2
    citation = result["hypothesis"].citations[0]
    assert result["hypothesis"].statement == "ok"
    assert citation.title == "T pubmed:1"  # title comes from the source, not the LLM
    assert citation.quote == "Label noise hurts"


def test_non_verbatim_quote_is_replaced_with_source_text():
    paraphrase = _hyp(Citation(paper_id="pubmed:1", title="t", quote="noise is bad"))
    result, _ = _run(STATE, [paraphrase])
    assert result["hypothesis"].citations[0].quote == "Label noise hurts calibration."


def test_refinement_receives_previous_hypothesis_and_critique():
    cite = Citation(paper_id="pubmed:1", title="t", quote="q")
    state = {
        **STATE,
        "hypothesis": _hyp(cite, statement="old"),
        "critic_report": CriticReport(verdict="not_significant", summary="Too vague"),
    }
    result, chain = _run(state, [_hyp(cite, statement="new")])

    inputs = chain.invoke.call_args[0][0]
    assert "old" in inputs["previous_hypothesis"]
    assert "Too vague" in inputs["feedback"]
    assert result["hypotheses"][0].statement == "new"


def test_no_literature_means_no_hypothesis():
    with pytest.raises(ValueError):
        _run({"topic": "AI", "literature": []}, [])


def test_gives_up_after_repeated_hallucination():
    bad = _hyp(Citation(paper_id="nope", title="t", quote="q"))
    with pytest.raises(ValueError):
        _run(STATE, [bad, bad, bad])


def test_preprints_cannot_be_cited():
    state = {"topic": "AI", "literature": [_chunk("arxiv:1", "Preprint claim.", peer_reviewed=False)]}
    with pytest.raises(ValueError, match="peer-reviewed"):
        _run(state, [])
