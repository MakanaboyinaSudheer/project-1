"""Context-window management for long, multi-day runs. Owner: Member 2 (Week 3).

- The LangGraph SQLite checkpointer persists state per thread_id (resume after crash/restart).
- Rolling summary: when a prompt would exceed the token budget, compress older
  attempts/trace into state['memory_summary'] and keep only the last N items verbatim.
"""

from discovery_agent.graph.state import ResearchState


def build_context(state: ResearchState, budget_tokens: int = 12000) -> str:
    raise NotImplementedError


def compress(state: ResearchState) -> dict:
    raise NotImplementedError
