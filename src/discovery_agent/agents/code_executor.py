"""Code Executor node: writes experiment code and runs it in the sandbox. Owner: Member 3 (Week 1-2)."""

from discovery_agent.graph.state import ResearchState


def code_executor_node(state: ResearchState) -> dict:
    """Generate code for state['hypothesis'].test_plan, run via sandbox.runner, append a CodeAttempt.
    Generated code must write its results to /output/metrics.json."""
    raise NotImplementedError
