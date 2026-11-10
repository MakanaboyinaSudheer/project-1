"""Updated state schema to support parallel agent execution."""

from typing import Annotated, Any, TypedDict


def add_items(existing: list, new_items: list) -> list:
    """Safe reducer to combine parallel hypotheses without throwing TypeErrors on None."""
    if not existing:
        return new_items
    if not new_items:
        return existing
    return existing + new_items


class ParallelResearchState(TypedDict):
    """State schema for parallel hypothesis generation."""

    research_topic: str
    hypothesis_angles: list[str]

    # Explicitly define this so the Send API payload is not dropped
    current_angle: str

    # Use the safe custom reducer to aggregate parallel branch outputs
    hypotheses: Annotated[list[dict[str, Any]], add_items]

    final_report: str
    refinement_count: int
