"""Tests for LaTeX and Markdown research-draft synthesis."""

import json

import pytest

from discovery_agent.agents.synthesizer import synthesizer_node
from discovery_agent.schemas import (
    Citation,
    CodeAttempt,
    CriticReport,
    ExecutionResult,
    Hypothesis,
    RetrievedChunk,
)
from discovery_agent.synthesis.latex import (
    citation_key,
    render_abstract,
    render_bibliography,
    render_paper,
)


def _hypothesis() -> Hypothesis:
    return Hypothesis(
        statement="A & B",
        rationale="Prior result is 50%",
        citations=[
            Citation(paper_id="pubmed:123", title="A study_1", quote="Evidence & result"),
            Citation(paper_id="pubmed:123", title="Duplicate", quote="Same paper"),
        ],
        test_plan="Compare treatment groups",
        expected_outcome="p < 0.05",
    )


def test_latex_renderers_escape_text_and_include_citations() -> None:
    hypothesis = _hypothesis()
    state = {"hypothesis": hypothesis}

    abstract = render_abstract(state)
    paper = render_paper(state)
    bibliography = render_bibliography(hypothesis.citations)

    assert r"A \& B" in abstract
    assert r"50\%" in abstract
    assert r"\cite{paperpubmed123}" in abstract
    assert r"\bibliography{refs}" in paper
    assert "p < 0.05" in paper
    assert bibliography.count("@misc{") == 1
    assert "paperpubmed123" in bibliography
    assert r"A study\_1" in bibliography


def test_synthesis_node_writes_all_artifacts_and_returns_state(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("discovery_agent.agents.synthesizer.settings.runs_dir", str(tmp_path))
    attempt = CodeAttempt(
        iteration=1,
        code="print('done')",
        result=ExecutionResult(
            exit_code=0,
            stdout="",
            stderr="",
            duration_s=1,
            metrics={"p_value": 0.02, "result": "positive"},
        ),
        reward=0.9,
    )
    lower_reward_attempt = CodeAttempt(
        iteration=2,
        code="print('later but worse')",
        result=ExecutionResult(
            exit_code=0,
            stdout="",
            stderr="",
            duration_s=1,
            metrics={"p_value": 0.8, "result": "negative"},
        ),
        reward=0.1,
    )
    literature = RetrievedChunk(
        id="chunk-1",
        paper_id="pubmed:123",
        text="The study finds an effect of 50%.",
        score=0.9,
        metadata={"title": "Background & rationale"},
    )
    state = {
        "thread_id": "run-1",
        "topic": "A topic",
        "hypothesis": _hypothesis(),
        "literature": [literature],
        "literature_summary": "An overview.",
        "attempts": [attempt, lower_reward_attempt],
        "critic_report": CriticReport(
            verdict="significant",
            p_values={"primary": 0.02},
            summary="The result is robust.",
        ),
    }

    result = synthesizer_node(state)

    run_dir = tmp_path / "run-1"
    assert {path.name for path in run_dir.iterdir()} == {
        "abstract.tex",
        "paper.tex",
        "refs.bib",
        "draft.md",
    }
    assert result["abstract_latex"] == (run_dir / "abstract.tex").read_text(encoding="utf-8")
    assert result["draft_markdown"] == (run_dir / "draft.md").read_text(encoding="utf-8")
    assert "p_value" in result["draft_markdown"]
    assert '"p_value": 0.02' in result["draft_markdown"]
    assert '"p_value": 0.8' not in result["draft_markdown"]
    assert r"The result is robust\." in result["draft_markdown"]
    paper = (run_dir / "paper.tex").read_text(encoding="utf-8")
    assert r"Background \& rationale" in paper
    assert "0.02" in paper
    assert "0.8" not in paper
    assert "paperpubmed123" in (run_dir / "refs.bib").read_text(encoding="utf-8")
    assert result["trace"][0]["node"] == "synthesizer"


def test_synthesizer_requires_a_hypothesis() -> None:
    with pytest.raises(ValueError, match="without a hypothesis"):
        synthesizer_node({})


def test_synthesizer_rejects_unsafe_run_ids(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("discovery_agent.agents.synthesizer.settings.runs_dir", str(tmp_path))

    with pytest.raises(ValueError, match="thread_id"):
        synthesizer_node({"thread_id": "../outside", "hypothesis": _hypothesis()})

    assert not (tmp_path.parent / "outside").exists()


def test_markdown_metrics_fence_handles_backticks() -> None:
    hypothesis = _hypothesis()
    result = synthesizer_node(
        {
            "hypothesis": hypothesis,
            "attempts": [
                CodeAttempt(
                    iteration=1,
                    code="",
                    result=ExecutionResult(
                        exit_code=0,
                        stdout="",
                        stderr="",
                        duration_s=0,
                        metrics={"note": "contains ``` fence"},
                    ),
                    reward=0.5,
                )
            ],
        }
    )

    assert "````" in result["draft_markdown"]
    assert (
        json.loads(result["draft_markdown"].split("json\n", 1)[1].split("\n````", 1)[0])["note"]
        == "contains ``` fence"
    )


def test_citation_key_requires_a_sanitizable_id() -> None:
    with pytest.raises(ValueError, match="BibTeX key"):
        citation_key("///")


def test_bibliography_disambiguates_ids_that_sanitize_to_same_key() -> None:
    citations = [
        Citation(paper_id="pubmed:123", title="First", quote="Quote 1"),
        Citation(paper_id="pubmed-123", title="Second", quote="Quote 2"),
    ]

    bibliography = render_bibliography(citations)
    abstract = render_abstract(
        {
            "hypothesis": Hypothesis(
                statement="Claim",
                rationale="Reason",
                citations=citations,
                test_plan="Test",
                expected_outcome="Result",
            )
        }
    )

    assert bibliography.count("@misc{") == 2
    keys = [
        line.split("{", 1)[1].removesuffix(",")
        for line in bibliography.splitlines()
        if line.startswith("@misc")
    ]
    assert len(set(keys)) == 2
    assert all(key in abstract for key in keys)
