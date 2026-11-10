"""Parallel execution logic for generating multiple hypotheses simultaneously."""

from typing import Any

from langgraph.types import Send

from discovery_agent.graph.state import ResearchState


def generate_multiple_angles(state: ResearchState) -> dict[str, Any]:
    """Generates seed data for parallel hypothesis agents."""
    # Create 3 distinct angles for the hypothesis generation phase
    angles = ["conservative", "optimistic", "novel"]
    return {"hypothesis_angles": angles}


def route_to_parallel_agents(state: ResearchState) -> list[Send]:
    """Fan-out function to trigger multiple agents in parallel."""
    angles = state.get("hypothesis_angles", [])

    # Map each angle to a separate instance of the hypothesis_generator node
    return [Send("hypothesis_generator", {"current_angle": angle}) for angle in angles]
