"""Render research abstract / draft from Jinja2 templates in ./templates. Owner: Member 2 (Week 4)."""

import re
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from discovery_agent.graph.state import ResearchState
from discovery_agent.schemas import Citation

_TEMPLATE_DIR = Path(__file__).with_name("templates")
_ENV = Environment(
    loader=FileSystemLoader(_TEMPLATE_DIR),
    autoescape=False,
    keep_trailing_newline=True,
)
_LATEX_ESCAPES = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def _escape(value: str) -> str:
    return "".join(_LATEX_ESCAPES.get(char, char) for char in value)


def _citations(state: ResearchState) -> list[Citation]:
    hypothesis = state.get("hypothesis")
    return hypothesis.citations if hypothesis else []


def citation_key(paper_id: str) -> str:
    key = re.sub(r"[^A-Za-z0-9]+", "", paper_id)
    if not key:
        raise ValueError(f"Paper id cannot produce a BibTeX key: {paper_id!r}")
    return f"paper{key}"


def render_bibliography(citations: list[Citation]) -> str:
    entries = []
    seen = set()
    for citation in citations:
        key = citation_key(citation.paper_id)
        if key in seen:
            continue
        seen.add(key)
        title = _escape(citation.title)
        entries.append(
            f"@misc{{{key},\n  title = {{{title}}},\n  note = {{{_escape(citation.paper_id)}}}\n}}"
        )
    return "\n\n".join(entries) + ("\n" if entries else "")


def render_abstract(state: ResearchState) -> str:
    hypothesis = state.get("hypothesis")
    if hypothesis is None:
        raise ValueError("Cannot render an abstract without a hypothesis")
    return _ENV.get_template("abstract.tex.j2").render(
        statement=_escape(hypothesis.statement),
        rationale=_escape(hypothesis.rationale),
        expected_outcome=_escape(hypothesis.expected_outcome),
        citation_keys=[citation_key(item.paper_id) for item in _citations(state)],
    )


def render_paper(state: ResearchState) -> str:
    """Full simulated paper (.tex) with a bibliography built from the hypothesis citations."""
    hypothesis = state.get("hypothesis")
    if hypothesis is None:
        raise ValueError("Cannot render a paper without a hypothesis")
    attempts = state.get("attempts", [])
    last_attempt = attempts[-1] if attempts else None
    critic_report = state.get("critic_report")
    literature = state.get("literature", [])
    return _ENV.get_template("paper.tex.j2").render(
        statement=_escape(hypothesis.statement),
        rationale=_escape(hypothesis.rationale),
        test_plan=_escape(hypothesis.test_plan),
        expected_outcome=_escape(hypothesis.expected_outcome),
        summary=_escape(state.get("literature_summary", "")),
        literature=[
            {
                "paper_id": _escape(item.paper_id),
                "title": _escape(item.metadata.get("title", item.paper_id)),
                "text": _escape(item.text),
            }
            for item in literature
        ],
        metrics=_escape(str(last_attempt.result.metrics)) if last_attempt else "No execution result.",
        verdict=critic_report.verdict if critic_report else "unavailable",
        issues=[_escape(issue) for issue in critic_report.issues] if critic_report else [],
        citation_keys=[citation_key(item.paper_id) for item in _citations(state)],
        bibliography=render_bibliography(_citations(state)),
    )
