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

import os
import sqlite3
import uuid

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

from discovery_agent.agents.code_executor import code_executor_node
from discovery_agent.agents.critic import critic_node
from discovery_agent.agents.debugger import debugger_node
from discovery_agent.agents.hypothesis_generator import hypothesis_generator_node
from discovery_agent.agents.literature_reviewer import literature_reviewer_node
from discovery_agent.agents.synthesizer import synthesizer_node
from discovery_agent.graph.state import ResearchState

DEFAULT_DB_PATH = os.path.abspath(os.path.join(os.getcwd(), "checkpoints.sqlite"))


def route_after_execution(state: ResearchState) -> str:
    """Route to debugger on error, or critic on success."""
    code_attempts = state.get("code_attempts", [])

    if not code_attempts:
        return "critic"

    last_attempt = code_attempts[-1]

    if isinstance(last_attempt, dict) and (last_attempt.get("error") or not last_attempt.get("success")):
        if len(code_attempts) >= 5:
            return "critic"
        return "debugger"

    return "critic"


def route_after_critic(state: ResearchState) -> str:
    """Route based on critic evaluation."""
    if state.get("is_significant") or state.get("passed_checks"):
        return "synthesizer"
    if len(state.get("code_attempts", [])) >= 5:
        return "synthesizer"

    return "hypothesis_generator"


def build_graph(db_path: str = DEFAULT_DB_PATH):
    """Return the compiled StateGraph with a SQLite checkpointer."""
    workflow = StateGraph(ResearchState)

    workflow.add_node("literature_reviewer", literature_reviewer_node)
    workflow.add_node("hypothesis_generator", hypothesis_generator_node)
    workflow.add_node("code_executor", code_executor_node)
    workflow.add_node("debugger", debugger_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("synthesizer", synthesizer_node)

    workflow.set_entry_point("literature_reviewer")
    workflow.add_edge("literature_reviewer", "hypothesis_generator")
    workflow.add_edge("hypothesis_generator", "code_executor")

    workflow.add_conditional_edges("code_executor", route_after_execution)
    workflow.add_edge("debugger", "code_executor")

    workflow.add_conditional_edges("critic", route_after_critic)
    workflow.add_edge("synthesizer", END)

    if db_path != ":memory:":
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

    conn = sqlite3.connect(db_path, check_same_thread=False)
    memory = SqliteSaver(conn)
    memory.setup()

    return workflow.compile(checkpointer=memory)


def run_discovery(
    topic: str,
    thread_id: str | None = None,
    db_path: str = DEFAULT_DB_PATH,
) -> dict:
    """Execute the discovery workflow for a given topic."""
    if not thread_id:
        thread_id = str(uuid.uuid4())

    graph = build_graph(db_path=db_path)
    config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}

    initial_state = {"research_topic": topic}

    return graph.invoke(initial_state, config=config)


def get_run_status(
    thread_id: str,
    db_path: str = DEFAULT_DB_PATH,
) -> dict | None:
    """Retrieve the latest state of a specific run for recovery monitoring."""
    graph = build_graph(db_path=db_path)
    config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}

    state_snapshot = graph.get_state(config)
    if state_snapshot and hasattr(state_snapshot, "values"):
        return state_snapshot.values
    return None


def resume_discovery(
    thread_id: str,
    db_path: str = DEFAULT_DB_PATH,
) -> dict:
    """Resume a suspended or crashed discovery workflow."""
    graph = build_graph(db_path=db_path)
    config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}

    return graph.invoke(None, config=config)
