"""Core multi-agent workflow graph for the Discovery Agent."""

import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from discovery_agent.agents.code_executor import execute_code
from discovery_agent.agents.critic import evaluate_results
from discovery_agent.agents.hypothesis_generator import generate_hypothesis

# Import agent nodes
from discovery_agent.agents.literature_reviewer import review_literature
from discovery_agent.agents.synthesizer import synthesize_report
from discovery_agent.graph.parallel import route_to_parallel_agents
from discovery_agent.graph.routing import route_after_critic

# Import your state and routing logic
from discovery_agent.graph.state_parallel import ParallelResearchState


def create_graph(memory_db_path: str = "checkpoints.sqlite"):
    """Compiles the multi-agent state machine."""
    builder = StateGraph(ParallelResearchState)

    # 1. Add all core nodes
    builder.add_node("literature_reviewer", review_literature)
    builder.add_node("hypothesis_generator", generate_hypothesis)
    builder.add_node("code_executor", execute_code)
    builder.add_node("statistical_critic", evaluate_results)
    builder.add_node("synthesizer", synthesize_report)

    # 2. Define standard flow
    builder.add_edge(START, "literature_reviewer")

    # 3. Parallel Fan-out (M2.7/M2.8)
    builder.add_conditional_edges(
        "literature_reviewer",
        route_to_parallel_agents,
        ["hypothesis_generator"],  # Required for LangGraph v1.0 mapping
    )

    builder.add_edge("hypothesis_generator", "code_executor")
    builder.add_edge("code_executor", "statistical_critic")

    # 4. Refinement Loop with Safety Fallbacks (M2.5)
    builder.add_conditional_edges(
        "statistical_critic",
        route_after_critic,
        {
            "execution": "synthesizer",  # Approved or Max Iterations Reached
            "hypothesis_generator": "hypothesis_generator",  # Rejected: Loop back
            "end": END,
        },
    )

    builder.add_edge("synthesizer", END)

    # 5. Compile with SQLite Checkpointer for resume-after-kill persistence
    conn = sqlite3.connect(memory_db_path, check_same_thread=False)
    memory = SqliteSaver(conn)

    return builder.compile(checkpointer=memory)
