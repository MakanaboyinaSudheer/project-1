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
from discovery_agent.agents.critic import critic_node, route_after_critic
from discovery_agent.agents.debugger import debugger_node, route_after_execution
from discovery_agent.agents.hypothesis_generator import hypothesis_generator_node
from discovery_agent.agents.literature_reviewer import literature_reviewer_node
from discovery_agent.agents.synthesizer import synthesizer_node
from discovery_agent.graph.state import ResearchState


def build_graph():
    """Return the compiled StateGraph with a SQLite checkpointer (resumable runs)."""
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

    os.makedirs("volumes", exist_ok=True)
    conn = sqlite3.connect("volumes/checkpoints.sqlite", check_same_thread=False)
    memory = SqliteSaver(conn)

    return workflow.compile(checkpointer=memory)


def run_discovery(topic: str, thread_id: str | None = None) -> dict:
    """Execute the discovery workflow for a given topic."""
    if not thread_id:
        thread_id = str(uuid.uuid4())

    graph = build_graph()
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {"research_topic": topic}

    return graph.invoke(initial_state, config=config)
