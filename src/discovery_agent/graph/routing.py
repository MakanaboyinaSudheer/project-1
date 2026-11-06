"""Routing policies and safety fallbacks for the LangGraph state machine."""

from typing import Literal

from discovery_agent.graph.state import ResearchState

MAX_REFINEMENT_STEPS = 3


def route_after_critic(state: ResearchState) -> Literal["execution", "hypothesis_generator", "end"]:
    """
    Evaluates critic feedback and routes the graph.
    Enforces a strict iteration budget to prevent infinite refinement loops.
    """
    critic_feedback = state.get("final_report", "")
    iteration = state.get("refinement_count", 0)

    # Safety fallback: Force progression if budget is exhausted
    if iteration >= MAX_REFINEMENT_STEPS:
        return "execution"

    # If the Critic approves, move to execution
    if "APPROVED" in critic_feedback.upper():
        return "execution"

    # Otherwise, loop back to refine the hypothesis
    return "hypothesis_generator"
