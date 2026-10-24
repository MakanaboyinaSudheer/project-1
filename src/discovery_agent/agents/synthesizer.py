"""Render LaTeX and Markdown drafts from the completed research state."""

import json
import re
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from discovery_agent.config import settings
from discovery_agent.graph.state import ResearchState, trace_event
from discovery_agent.synthesis.latex import (
    _best_attempt,
    _citation_context,
    render_abstract,
    render_bibliography,
    render_paper,
)

_SAFE_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_MARKDOWN_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+.!|>~-])")
_TEMPLATE_DIR = Path(__file__).parents[1] / "synthesis" / "templates"
_ENV = Environment(
    loader=FileSystemLoader(_TEMPLATE_DIR),
    autoescape=False,
    keep_trailing_newline=True,
)


def _escape_markdown(value: str) -> str:
    return _MARKDOWN_SPECIAL.sub(r"\\\1", value)


def _markdown_fence(content: str) -> str:
    longest = max((len(match.group()) for match in re.finditer(r"`+", content)), default=0)
    return "`" * max(3, longest + 1)


def _render_markdown(state: ResearchState) -> str:
    hypothesis = state["hypothesis"]
    best_attempt = _best_attempt(state)
    metrics = best_attempt.result.metrics if best_attempt else {}
    critic_report = state.get("critic_report")
    metrics_json = json.dumps(metrics, ensure_ascii=False, indent=2)
    citations = [
        {
            "title": _escape_markdown(citation["title"]),
            "paper_id": _escape_markdown(citation["paper_id"]),
            "quote": _escape_markdown(citation["quote"]),
        }
        for citation in _citation_context(hypothesis.citations)
    ]
    return _ENV.get_template("draft.md.j2").render(
        title=_escape_markdown(hypothesis.statement),
        rationale=_escape_markdown(hypothesis.rationale),
        expected_outcome=_escape_markdown(hypothesis.expected_outcome),
        test_plan=_escape_markdown(hypothesis.test_plan),
        metrics_json=metrics_json,
        metrics_fence=_markdown_fence(metrics_json),
        critic_summary=_escape_markdown(critic_report.summary if critic_report else "No critic report."),
        citations=citations,
    )


def synthesizer_node(state: ResearchState) -> dict:
    """Render final research artifacts and return the graph's synthesis state."""
    if state.get("hypothesis") is None:
        raise ValueError("Cannot synthesize a paper without a hypothesis")

    abstract = render_abstract(state)
    paper = render_paper(state)
    citations = state["hypothesis"].citations
    bibliography = render_bibliography(citations)
    markdown = _render_markdown(state)

    run_id = state.get("thread_id")
    if run_id:
        if not _SAFE_RUN_ID.fullmatch(run_id):
            raise ValueError("thread_id must contain only letters, numbers, dots, underscores, and hyphens")
        output_dir = Path(settings.runs_dir) / run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        artifacts = {
            "abstract.tex": abstract,
            "paper.tex": paper,
            "refs.bib": bibliography,
            "draft.md": markdown,
        }
        for filename, content in artifacts.items():
            (output_dir / filename).write_text(content, encoding="utf-8")

    return {
        "abstract_latex": abstract,
        "draft_markdown": markdown,
        "trace": [trace_event("synthesizer", "Rendered LaTeX and Markdown research drafts.")],
    }
