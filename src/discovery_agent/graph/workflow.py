"""Core orchestration workflow for the Discovery Agent."""

from __future__ import annotations

import uuid
from contextlib import suppress
from typing import Any

from discovery_agent.agents.code_executor import code_executor_node
from discovery_agent.agents.critic import critic_node
from discovery_agent.agents.hypothesis_generator import hypothesis_generator_node
from discovery_agent.agents.literature_reviewer import literature_reviewer_node
from discovery_agent.agents.synthesizer import synthesizer_node
from discovery_agent.config import settings
from discovery_agent.graph.routing import route_after_critic as _route_after_critic
from discovery_agent.schemas import CriticReport, Hypothesis

# Backward-compatible aliases expected by the wider codebase.
execute_code = code_executor_node
evaluate_results = critic_node
generate_hypothesis = hypothesis_generator_node
review_literature = literature_reviewer_node
synthesize_report = synthesizer_node

_RUN_STATES: dict[str, dict[str, Any]] = {}


def route_after_critic(state: dict[str, Any]) -> str:
    """Compatibility wrapper that delegates to the shared routing helper."""
    return _route_after_critic(state)


def route_after_execution(state: dict[str, Any]) -> str:
    """Decide whether a completed code attempt should continue to the critic or debug loop."""
    attempts = state.get("attempts") or []
    if not attempts:
        return "failed"

    latest = attempts[-1]
    result = getattr(latest, "result", None)
    if isinstance(result, dict):
        exit_code = result.get("exit_code", 1)
        timed_out = bool(result.get("timed_out", False))
    else:
        exit_code = getattr(result, "exit_code", 1)
        timed_out = bool(getattr(result, "timed_out", False))

    if exit_code == 0 and not timed_out:
        return "critic"

    debug_iteration = int(state.get("debug_iteration", 0) or 0)
    if debug_iteration >= settings.max_debug_iterations:
        return "failed"
    return "debug"


def get_run_status(thread_id: str) -> dict[str, Any] | None:
    """Return the most recent in-memory state for a run thread."""
    return _RUN_STATES.get(thread_id)


def run_discovery(topic: str, thread_id: str | None = None) -> dict[str, Any]:
    """Run the core research loop from literature review through synthesis."""
    thread_id = thread_id or uuid.uuid4().hex
    state: dict[str, Any] = {
        "topic": topic,
        "research_topic": topic,
        "thread_id": thread_id,
        "status": "running",
        "trace": [],
        "attempts": [],
        "refinement_iteration": 0,
        "debug_iteration": 0,
    }

    while True:
        state.update(literature_reviewer_node(state))
        state.update(hypothesis_generator_node(state))

        hypothesis = state.get("hypothesis")
        if isinstance(hypothesis, dict):
            with suppress(Exception):
                state["hypothesis"] = Hypothesis.model_validate(hypothesis)

        state.update(code_executor_node(state))
        state.update(critic_node(state))

        critique = state.get("critic_report")
        if isinstance(critique, dict):
            with suppress(Exception):
                critique = CriticReport.model_validate(critique)
                state["critic_report"] = critique

        verdict = getattr(critique, "verdict", None)
        if verdict == "significant":
            state.update(synthesizer_node(state))
            state["status"] = "succeeded"
            _RUN_STATES[thread_id] = state
            return state

        if verdict in {"not_significant", "invalid"}:
            state["refinement_iteration"] = int(state.get("refinement_iteration", 0)) + 1
            state["final_report"] = getattr(critique, "summary", str(critique))
            if state["refinement_iteration"] >= 3:
                state["status"] = "failed"
                _RUN_STATES[thread_id] = state
                return state
            continue

        state["status"] = "failed"
        _RUN_STATES[thread_id] = state
        return state
