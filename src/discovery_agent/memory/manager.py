"""Context-window management for long, multi-day runs. Owner: Member 2 (Week 3).

- The LangGraph SQLite checkpointer persists state per thread_id (resume after crash/restart).
- Rolling summary: when a prompt would exceed the token budget, compress older
  attempts/trace into state['memory_summary'] and keep only the last N items verbatim.
"""

from discovery_agent.graph.state import ResearchState


def build_context(state: ResearchState, budget_tokens: int = 12000) -> str:
    if budget_tokens < 1:
        raise ValueError("budget_tokens must be positive")
    parts = []
    if state.get("memory_summary"):
        parts.append(f"Prior run summary:\n{state['memory_summary']}")
    if state.get("topic"):
        parts.append(f"Research topic:\n{state['topic']}")
    literature = state.get("literature", [])
    if literature:
        excerpts = [
            f"[{item.paper_id}] {item.title if hasattr(item, 'title') else item.id}: {item.text}"
            for item in literature[:8]
        ]
        parts.append("Relevant literature:\n" + "\n".join(excerpts))
    attempts = state.get("attempts", [])
    if attempts:
        recent = attempts[-3:]
        parts.append(
            "Recent execution attempts:\n"
            + "\n".join(
                f"Attempt {item.iteration}: exit={item.result.exit_code}, reward={item.reward:.2f}, "
                f"stderr={item.result.stderr[-500:]}"
                for item in recent
            )
        )
    context = "\n\n".join(parts)
    max_chars = budget_tokens * 4
    if len(context) > max_chars:
        context = context[-max_chars:]
    return context


def compress(state: ResearchState) -> dict:
    attempts = state.get("attempts", [])
    older_attempts = attempts[:-3]
    old_trace = state.get("trace", [])[:-8]
    summary_parts = [state.get("memory_summary", "").strip()]
    if older_attempts:
        successful = sum(1 for item in older_attempts if item.result.exit_code == 0)
        summary_parts.append(
            f"Compressed {len(older_attempts)} earlier attempts; {successful} completed successfully."
        )
    if old_trace:
        nodes = [str(item.get("node", "unknown")) for item in old_trace]
        summary_parts.append("Earlier workflow nodes: " + " -> ".join(nodes))
    return {"memory_summary": "\n".join(part for part in summary_parts if part)}
