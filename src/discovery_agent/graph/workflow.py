"""LangGraph wiring of the agents.

    literature_reviewer -> hypothesis_generator -> code_executor
                                                     |      ^
                                       (error)       v      | (rewrite, < MAX_DEBUG_ITERATIONS)
                                                   debugger-+
                                                     |
                                       (success)     v
                                                  critic -> synthesizer -> END
                         (not significant / invalid) |
                                                     +-> hypothesis_generator (refine)

Owner: Member 2. Node implementations live in discovery_agent.agents.*.
"""

import json
import sqlite3
import uuid
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from pydantic import BaseModel

from discovery_agent.agents.code_executor import code_executor_node
from discovery_agent.agents.critic import critic_node
from discovery_agent.agents.debugger import debugger_node
from discovery_agent.agents.hypothesis_generator import hypothesis_generator_node
from discovery_agent.agents.literature_reviewer import literature_reviewer_node
from discovery_agent.agents.synthesizer import synthesizer_node
from discovery_agent.config import settings
from discovery_agent.graph.state import ResearchState, trace_event
from discovery_agent.telemetry.costs import CostTracker

MAX_REFINEMENTS = 3


def _finish_success(state: ResearchState) -> dict:
    return {
        "status": "succeeded",
        "trace": [trace_event("finish", "Research workflow completed.")],
    }


def _finish_failure(state: ResearchState) -> dict:
    return {
        "status": "failed",
        "trace": [trace_event("finish", "Retry or refinement budget exhausted.")],
    }


def _prepare_refinement(state: ResearchState) -> dict:
    return {
        "refinement_iteration": state.get("refinement_iteration", 0) + 1,
        "trace": [
            trace_event(
                "refinement",
                f"Preparing refinement {state.get('refinement_iteration', 0) + 1}.",
            )
        ],
    }


def _compile_graph(checkpointer):
    workflow = StateGraph(ResearchState)
    workflow.add_node("literature_reviewer", literature_reviewer_node)
    workflow.add_node("hypothesis_generator", hypothesis_generator_node)
    workflow.add_node("code_executor", code_executor_node)
    workflow.add_node("debugger", debugger_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("synthesizer", synthesizer_node)
    workflow.add_node("prepare_refinement", _prepare_refinement)
    workflow.add_node("finish_success", _finish_success)
    workflow.add_node("finish_failure", _finish_failure)
    workflow.set_entry_point("literature_reviewer")
    workflow.add_edge("literature_reviewer", "hypothesis_generator")
    workflow.add_edge("hypothesis_generator", "code_executor")
    workflow.add_conditional_edges(
        "code_executor",
        route_after_execution,
        {"debug": "debugger", "critic": "critic", "failed": "finish_failure"},
    )
    workflow.add_edge("debugger", "code_executor")
    workflow.add_conditional_edges(
        "critic",
        route_after_critic,
        {"synthesize": "synthesizer", "refine": "prepare_refinement", "failed": "finish_failure"},
    )
    workflow.add_edge("prepare_refinement", "hypothesis_generator")
    workflow.add_edge("synthesizer", "finish_success")
    workflow.add_edge("finish_success", END)
    workflow.add_edge("finish_failure", END)
    return workflow.compile(checkpointer=checkpointer)


def build_graph():
    """Return the compiled StateGraph with a SQLite checkpointer (resumable runs)."""
    checkpoint_path = Path(settings.checkpoint_db)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(checkpoint_path, check_same_thread=False)
    return _compile_graph(SqliteSaver(connection))


def route_after_execution(state: ResearchState) -> str:
    """'debug' if the last attempt failed and budget remains, 'critic' if it succeeded, else 'fail'."""
    attempts = state.get("attempts", [])
    if not attempts:
        return "failed"
    result = attempts[-1].result
    if result.exit_code == 0 and not result.timed_out:
        return "critic"
    return "debug" if state.get("debug_iteration", 0) < settings.max_debug_iterations else "failed"


def route_after_critic(state: ResearchState) -> str:
    """'synthesize' if significant, otherwise 'refine' (back to hypothesis generator)."""
    report = state.get("critic_report")
    if report is None:
        return "failed"
    if report.verdict == "significant":
        return "synthesize"
    return "refine" if state.get("refinement_iteration", 0) < MAX_REFINEMENTS else "failed"


def run_discovery(topic: str, thread_id: str | None = None) -> ResearchState:
    topic = topic.strip()
    if not topic:
        raise ValueError("topic cannot be empty")
    run_id = thread_id or str(uuid.uuid4())
    if not run_id or Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError("thread_id must be a simple directory name")
    checkpoint_path = Path(settings.checkpoint_db)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(checkpoint_path, check_same_thread=False)
    graph = _compile_graph(SqliteSaver(connection))
    config = {
        "configurable": {"thread_id": run_id},
        "callbacks": [CostTracker(run_id=run_id)],
    }
    try:
        snapshot = graph.get_state(config)
        if snapshot.next:
            saved_topic = snapshot.values.get("topic")
            if saved_topic and saved_topic != topic:
                raise ValueError(f"Run {run_id!r} belongs to topic {saved_topic!r}, not {topic!r}")
            result = graph.invoke(None, config=config)
        elif snapshot.values:
            raise ValueError(f"Run {run_id!r} has already completed; use a new thread_id")
        else:
            result = graph.invoke(
                {
                    "topic": topic,
                    "thread_id": run_id,
                    "status": "running",
                    "debug_iteration": 0,
                    "refinement_iteration": 0,
                },
                config=config,
            )
    finally:
        connection.close()

    run_dir = Path(settings.runs_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "state.json", result)
    _write_json(run_dir / "trace.json", {"trace": result.get("trace", [])})
    _write_json(run_dir / "critic_report.json", result.get("critic_report", {}))
    _write_json(run_dir / "attempts.json", {"attempts": result.get("attempts", [])})
    _write_json(run_dir / "literature.json", {"literature": result.get("literature", [])})
    return result


def _write_json(path: Path, value: object) -> None:
    def serialize(item: object):
        if isinstance(item, BaseModel):
            return item.model_dump(mode="json")
        if hasattr(item, "isoformat"):
            return item.isoformat()
        raise TypeError(f"Object of type {type(item).__name__} is not JSON serializable")

    path.write_text(json.dumps(value, default=serialize, indent=2), encoding="utf-8")
