"""Hypothesis Generator node, grounded in peer-reviewed literature via RAG. Owner: Member 2 (Week 2)."""

from discovery_agent.graph.state import ResearchState


def hypothesis_generator_node(state: ResearchState) -> dict:
    """Produce a structured Hypothesis whose every claim cites a retrieved peer-reviewed chunk.
    Reject/regenerate if any citation's paper_id is not in state['literature']."""
    raise NotImplementedError
