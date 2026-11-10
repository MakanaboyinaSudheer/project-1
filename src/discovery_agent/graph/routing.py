"""Routing helpers shared by the workflow graph."""

from __future__ import annotations

from typing import Any


def route_after_critic(state: dict[str, Any]) -> str:
    """Decide whether non-significant results should be refined or the run should fail."""
    report = state.get("critic_report")
    if report is None:
        return "failed"

    verdict = getattr(report, "verdict", None)
    if verdict == "significant":
        return "synthesize"
    if verdict in {"not_significant", "invalid"}:
        refinement_iteration = int(state.get("refinement_iteration", 0) or 0)
        return "failed" if refinement_iteration >= 3 else "refine"
    return "failed"
