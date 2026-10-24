"""Context-window management for long, multi-day runs. Owner: Member 2 (Week 3).

- The LangGraph SQLite checkpointer persists state per thread_id (resume after crash/restart).
- Rolling summary: when a prompt would exceed the token budget, compress older
  attempts/trace into state['memory_summary'] and keep only the last N items verbatim.
"""

from discovery_agent.graph.state import ResearchState


def _estimate_tokens(text: str) -> int:
    """Estimate token count using a standard ~4 characters per token heuristic."""
    if not text:
        return 0
    return len(str(text)) // 4


def build_context(state: ResearchState, budget_tokens: int = 12000) -> str:
    """Construct a token-budgeted context string from ResearchState.

    Prioritizes critical core fields (topic, hypothesis, feedback) and truncates
    literature context and execution history to strictly stay within budget_tokens.
    """
    topic = state.get("research_topic", "")
    hypothesis = state.get("hypothesis") or {}
    summary = state.get("literature_summary", "")
    feedback = state.get("final_report", "None")
    code_attempts = state.get("code_attempts", [])

    core_header = f"Topic: {topic}\n\nHypothesis:\n{hypothesis}\n\nCritic Feedback:\n{feedback}\n\n"
    core_tokens = _estimate_tokens(core_header)

    remaining_budget = budget_tokens - core_tokens
    if remaining_budget <= 0:
        return core_header[: budget_tokens * 4]

    summary_tokens = _estimate_tokens(summary)
    if summary_tokens > remaining_budget // 2:
        max_summary_chars = (remaining_budget // 2) * 4
        summary_text = f"Literature Summary:\n{summary[:max_summary_chars]}...\n\n"
    else:
        summary_text = f"Literature Summary:\n{summary}\n\n"

    remaining_budget -= _estimate_tokens(summary_text)

    attempts_text = ""
    if code_attempts:
        attempts_str = f"Code Attempts History ({len(code_attempts)} attempts):\n{code_attempts[-2:]}\n"
        if _estimate_tokens(attempts_str) <= remaining_budget:
            attempts_text = attempts_str
        else:
            max_attempt_chars = max(0, remaining_budget * 4)
            attempts_text = f"Code Attempts History:\n{str(code_attempts[-1])[:max_attempt_chars]}...\n"

    return core_header + summary_text + attempts_text


def compress(state: ResearchState) -> dict:
    """Compress older attempts/trace into state['memory_summary']."""
    code_attempts = state.get("code_attempts", [])
    current_summary = state.get("memory_summary", "")

    if len(code_attempts) <= 2:
        return {}

    attempts_to_compress = code_attempts[:-2]
    recent_attempts = code_attempts[-2:]
    compressed_text = "\n".join(
        [f"Archived Attempt: {str(attempt)[:300]}..." for attempt in attempts_to_compress]
    )
    new_summary = current_summary + "\n" + compressed_text if current_summary else compressed_text

    return {"memory_summary": new_summary.strip(), "code_attempts": recent_attempts}
