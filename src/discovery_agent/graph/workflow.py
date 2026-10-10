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

from discovery_agent.graph.state import ResearchState


def literature_reviewer_stub(state: ResearchState):
    return {"literature_context": []}


def hypothesis_generator_stub(state: ResearchState):
    return {"hypothesis": None}


def code_executor_stub(state: ResearchState):
    return {"code_attempts": []}


def debugger_stub(state: ResearchState):
    return {}


def critic_stub(state: ResearchState):
    return {"final_report": None}


def synthesizer_stub(state: ResearchState):
    return {}


def build_graph():
    """Return the compiled StateGraph with a SQLite checkpointer (resumable runs)."""
    workflow = StateGraph(ResearchState)

    workflow.add_node("literature_reviewer", literature_reviewer_stub)
    workflow.add_node("hypothesis_generator", hypothesis_generator_stub)
    workflow.add_node("code_executor", code_executor_stub)
    workflow.add_node("debugger", debugger_stub)
    workflow.add_node("critic", critic_stub)
    workflow.add_node("synthesizer", synthesizer_stub)

    workflow.set_entry_point("literature_reviewer")
    workflow.add_edge("literature_reviewer", "hypothesis_generator")
    workflow.add_edge("hypothesis_generator", "code_executor")

    workflow.add_conditional_edges("code_executor", route_after_execution)
    workflow.add_edge("debugger", "code_executor")

    workflow.add_conditional_edges("critic", route_after_critic)
    workflow.add_edge("synthesizer", END)

    os.makedirs("volumes", exist_ok=True)
    conn = sqlite3.connect("volumes/checkpoints.sqlite", check_same_thread=False)
    memory = SqliteSaver(conn)

    return workflow.compile(checkpointer=memory)


def route_after_execution(state: ResearchState) -> str:
    """'debug' if the last attempt failed and budget remains, 'critic' if it succeeded, else 'fail'."""
    return "critic"


def route_after_critic(state: ResearchState) -> str:
    """'synthesize' if significant, otherwise 'refine' (back to hypothesis generator)."""
    return "synthesizer"


def run_discovery(topic: str, thread_id: str | None = None) -> dict:
    """Execute the discovery workflow for a given topic."""
    if not thread_id:
        thread_id = str(uuid.uuid4())

    graph = build_graph()
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {"research_topic": topic}

    return graph.invoke(initial_state, config=config)
