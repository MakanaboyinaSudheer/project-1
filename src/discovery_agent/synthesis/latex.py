"""Render research abstract / draft from Jinja2 templates in ./templates. Owner: Member 2 (Week 4)."""

from discovery_agent.graph.state import ResearchState


def render_abstract(state: ResearchState) -> str:
    raise NotImplementedError


def render_paper(state: ResearchState) -> str:
    """Full simulated paper (.tex) with a bibliography built from the hypothesis citations."""
    raise NotImplementedError
