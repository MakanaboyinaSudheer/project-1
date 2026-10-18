"""Synthesis node. Owner: Member 2 (Week 4)."""

import json
import re
from pathlib import Path

from discovery_agent.config import settings
from discovery_agent.graph.state import ResearchState, trace_event
from discovery_agent.synthesis.latex import render_abstract, render_bibliography, render_paper


def _markdown_escape(value: str) -> str:
    return re.sub(r"([\\`*_{}\[\]()#+\-.!|>])", r"\\\1", value)


def synthesizer_node(state: ResearchState) -> dict:
    """Return `abstract_latex` and `draft_markdown` rendered via synthesis.latex."""
    hypothesis = state.get("hypothesis")
    if hypothesis is None:
        raise ValueError("Cannot synthesize a paper without a hypothesis")
    abstract = render_abstract(state)
    paper = render_paper(state)
    bibliography = render_bibliography(hypothesis.citations)
    attempts = state.get("attempts", [])
    final_metrics = attempts[-1].result.metrics if attempts else {}
    critic_report = state.get("critic_report")
    markdown = "\n".join(
        [
            f"# {_markdown_escape(hypothesis.statement)}",
            "",
            "## Abstract",
            _markdown_escape(hypothesis.rationale + " " + hypothesis.expected_outcome),
            "",
            "## Methods",
            _markdown_escape(hypothesis.test_plan),
            "",
            "## Results",
            "```json",
            json.dumps(final_metrics, indent=2),
            "```",
            "",
            "## Statistical review",
            _markdown_escape(critic_report.summary if critic_report else "No critic report."),
            "",
            "## References",
            *[
                f"- {_markdown_escape(citation.title)} ({_markdown_escape(citation.paper_id)}): "
                f'"{_markdown_escape(citation.quote)}"'
                for citation in hypothesis.citations
            ],
        ]
    )
    run_id = state.get("thread_id")
    if run_id:
        output_dir = Path(settings.runs_dir) / run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "abstract.tex").write_text(abstract, encoding="utf-8")
        (output_dir / "paper.tex").write_text(paper, encoding="utf-8")
        (output_dir / "refs.bib").write_text(bibliography, encoding="utf-8")
        (output_dir / "draft.md").write_text(markdown, encoding="utf-8")
    return {
        "abstract_latex": abstract,
        "draft_markdown": markdown,
        "trace": [trace_event("synthesizer", "Rendered LaTeX and Markdown research drafts.")],
    }
