"""LangGraph orchestration: the single authoritative discovery workflow.

literature -> hypothesis -> execute -(fail)-> debug -> execute ...
                               \\-(ok)-> critic -(significant)-> synthesize -> END
                                               \\-(otherwise)-> refine -> hypothesis (max 3) | failed

Each run has its own thread_id: LangGraph checkpoints go to SQLite (resumable) and a JSON
snapshot is written to runs/<thread_id>/state.json after every node for the dashboard.
"""

from __future__ import annotations

import json
import re
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from discovery_agent.agents.code_executor import code_executor_node
from discovery_agent.agents.critic import critic_node
from discovery_agent.agents.debugger import debugger_node
from discovery_agent.agents.hypothesis_generator import hypothesis_generator_node
from discovery_agent.agents.literature_reviewer import literature_reviewer_node
from discovery_agent.agents.synthesizer import synthesizer_node
from discovery_agent.config import settings
from discovery_agent.graph.routing import route_after_critic
from discovery_agent.graph.state import ResearchState, trace_event

_SAFE_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
STAGES = ["literature", "hypothesis", "execute", "debug", "critic", "refine", "synthesize"]


def route_after_execution(state: dict[str, Any]) -> str:
    """Send successful attempts to the critic, failed ones to the debugger within budget."""
    attempts = state.get("attempts") or []
    if not attempts:
        return "failed"
    result = attempts[-1].result
    if result.exit_code == 0 and not result.timed_out:
        return "critic"
    if int(state.get("debug_iteration", 0) or 0) >= settings.max_debug_iterations:
        return "failed"
    return "debug"


def _refine(state: ResearchState) -> dict:
    iteration = int(state.get("refinement_iteration", 0) or 0) + 1
    return {
        "refinement_iteration": iteration,
        "code": "",
        "debug_iteration": 0,
        "debug_strategy": None,
        "trace": [trace_event("refine", f"Refinement {iteration}: regenerating hypothesis from critique.")],
    }


def _finish(status: str):
    def node(state: ResearchState) -> dict:
        return {"status": status, "trace": [trace_event(status, f"Run {status}.")]}

    return node


def build_graph(checkpointer=None):
    """Compile the discovery graph. Node functions are looked up at build time (patchable in tests)."""
    g = StateGraph(ResearchState)
    g.add_node("literature", literature_reviewer_node)
    g.add_node("hypothesis", hypothesis_generator_node)
    g.add_node("execute", code_executor_node)
    g.add_node("debug", debugger_node)
    g.add_node("critic", critic_node)
    g.add_node("refine", _refine)
    g.add_node("synthesize", synthesizer_node)
    g.add_node("succeeded", _finish("succeeded"))
    g.add_node("failed", _finish("failed"))

    g.add_edge(START, "literature")
    g.add_edge("literature", "hypothesis")
    g.add_edge("hypothesis", "execute")
    g.add_conditional_edges("execute", route_after_execution, ["critic", "debug", "failed"])
    g.add_edge("debug", "execute")
    g.add_conditional_edges(
        "critic", route_after_critic, {"synthesize": "synthesize", "refine": "refine", "failed": "failed"}
    )
    g.add_edge("refine", "hypothesis")
    g.add_edge("synthesize", "succeeded")
    g.add_edge("succeeded", END)
    g.add_edge("failed", END)
    return g.compile(checkpointer=checkpointer)


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    return value


def _write_snapshot(run_dir: Path, state: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    data = _jsonable(state)
    data["updated_at"] = datetime.now(UTC).isoformat()
    tmp = run_dir / "state.json.tmp"
    tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    tmp.replace(run_dir / "state.json")


def get_run_status(thread_id: str) -> dict[str, Any] | None:
    """Return the persisted state snapshot for a run (survives process restarts)."""
    path = Path(settings.runs_dir) / thread_id / "state.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def run_discovery(topic: str, thread_id: str | None = None, peer_reviewed_only: bool = False) -> dict:
    """Run (or resume, if thread_id has a checkpoint) the discovery graph; return the final state."""
    thread_id = thread_id or f"{datetime.now(UTC):%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
    if not _SAFE_RUN_ID.fullmatch(thread_id):
        raise ValueError("thread_id must contain only letters, numbers, dots, underscores, and hyphens")
    run_dir = Path(settings.runs_dir) / thread_id
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 100}

    db = settings.checkpoint_db
    if db != ":memory:":
        Path(db).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db, check_same_thread=False)
    try:
        graph = build_graph(SqliteSaver(conn))
        existing = graph.get_state(config)
        if existing.next:  # interrupted earlier: resume from the last checkpoint
            inputs, state = None, dict(existing.values)
        else:
            inputs = {
                "topic": topic,
                "thread_id": thread_id,
                "peer_reviewed_only": peer_reviewed_only,
                "status": "running",
                "refinement_iteration": 0,
                "debug_iteration": 0,
                "trace": [trace_event("start", f"Run started for topic: {topic}")],
            }
            state = dict(inputs)
        _write_snapshot(run_dir, state)
        try:
            for state in graph.stream(inputs, config, stream_mode="values"):
                _write_snapshot(run_dir, state)
        except Exception as exc:
            state = dict(graph.get_state(config).values or state)
            state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            _write_snapshot(run_dir, state)
            raise
        return state
    finally:
        conn.close()
