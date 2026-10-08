"""Self-debugging node: rewrites failed code from stderr + attempt history. Owner: Member 3 (Week 2)."""

from discovery_agent.graph.state import ResearchState


def debugger_node(state: ResearchState) -> dict:
    """Use the previous attempts (code, stderr, reward) to produce corrected `code`;
    increment `debug_iteration`."""
    raise NotImplementedError
