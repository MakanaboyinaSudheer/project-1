"""Literature Reviewer node. Owner: Member 2 (Week 1)."""

from discovery_agent.graph.state import ResearchState


def literature_reviewer_node(state: ResearchState) -> dict:
    """Retrieve literature for state['topic']; return `literature` + `literature_summary`."""
    raise NotImplementedError
