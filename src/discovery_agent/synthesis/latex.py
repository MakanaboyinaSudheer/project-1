"""Render research drafts and bibliographies from Jinja2 templates."""

import hashlib
import re
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from discovery_agent.graph.state import ResearchState
from discovery_agent.schemas import Citation, CodeAttempt

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


def _escape_latex(value: str) -> str:
    return "".join(_LATEX_ESCAPES.get(char, char) for char in value)


def citation_key(paper_id: str) -> str:
    """Create a stable BibTeX key from a paper identifier."""
    key = re.sub(r"[^A-Za-z0-9]+", "", paper_id)
    if not key:
        raise ValueError(f"Paper id cannot produce a BibTeX key: {paper_id!r}")
    return f"paper{key}"


def _citation_context(citations: list[Citation]) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    seen: dict[str, str] = {}
    for citation in citations:
        key = citation_key(citation.paper_id)
        if seen.get(key) == citation.paper_id:
            continue
        if key in seen:
            suffix = hashlib.sha256(citation.paper_id.encode("utf-8")).hexdigest()[:10]
            key = f"{key}{suffix}"
            while key in seen and seen[key] != citation.paper_id:
                suffix = hashlib.sha256((citation.paper_id + suffix).encode("utf-8")).hexdigest()[:10]
                key = f"{citation_key(citation.paper_id)}{suffix}"
        seen[key] = citation.paper_id
        entries.append(
            {
                "key": key,
                "paper_id": _escape_latex(citation.paper_id),
                "title": _escape_latex(citation.title),
                "quote": _escape_latex(citation.quote),
            }
        )
    return entries


def _citation_keys(citations: list[Citation]) -> list[str]:
    return [entry["key"] for entry in _citation_context(citations)]


def _best_attempt(state: ResearchState) -> CodeAttempt | None:
    attempts = state.get("attempts", [])
    return max(attempts, key=lambda attempt: (attempt.reward, attempt.iteration), default=None)


def render_bibliography(citations: list[Citation]) -> str:
    """Render unique citations as BibTeX entries."""
    return _ENV.get_template("refs.bib.j2").render(citations=_citation_context(citations))


def render_abstract(state: ResearchState) -> str:
    """Render the LaTeX abstract; a hypothesis is required for a meaningful draft."""
    hypothesis = state.get("hypothesis")
    if hypothesis is None:
        raise ValueError("Cannot render an abstract without a hypothesis")
    return _ENV.get_template("abstract.tex.j2").render(
        statement=_escape_latex(hypothesis.statement),
        rationale=_escape_latex(hypothesis.rationale),
        expected_outcome=_escape_latex(hypothesis.expected_outcome),
        citation_keys=_citation_keys(hypothesis.citations),
    )


def render_paper(state: ResearchState) -> str:
    """Render the full simulated research paper as LaTeX."""
    hypothesis = state.get("hypothesis")
    if hypothesis is None:
        raise ValueError("Cannot render a paper without a hypothesis")

    best_attempt = _best_attempt(state)
    critic_report = state.get("critic_report")
    literature = state.get("literature", [])
    return _ENV.get_template("paper.tex.j2").render(
        statement=_escape_latex(hypothesis.statement),
        rationale=_escape_latex(hypothesis.rationale),
        test_plan=_escape_latex(hypothesis.test_plan),
        expected_outcome=_escape_latex(hypothesis.expected_outcome),
        summary=_escape_latex(state.get("literature_summary", "")),
        literature=[
            {
                "paper_id": _escape_latex(item.paper_id),
                "title": _escape_latex(str(item.metadata.get("title", item.paper_id))),
                "text": _escape_latex(item.text),
            }
            for item in literature
        ],
        metrics=_escape_latex(str(best_attempt.result.metrics)) if best_attempt else "No execution result.",
        verdict=critic_report.verdict if critic_report else "unavailable",
        issues=[_escape_latex(issue) for issue in critic_report.issues] if critic_report else [],
        citation_keys=_citation_keys(hypothesis.citations),
    )
