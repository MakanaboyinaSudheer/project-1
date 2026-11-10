from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from discovery_agent.graph.state_parallel import ParallelResearchState


# Mock nodes for the test
def mock_reviewer(state: ParallelResearchState):
    return {"hypothesis_angles": ["conservative", "optimistic", "novel"]}


def mock_hypothesis_generator(state: dict):
    # This node receives the specific angle mapped by the Send API
    angle = state.get("current_angle", "unknown")
    return {"hypotheses": [{"angle": angle, "content": f"{angle} hypothesis generated."}]}


def test_parallel_graph_compilation():
    """Verify that the Send API successfully fans out and reduces parallel outputs."""
    builder = StateGraph(ParallelResearchState)

    builder.add_node("literature_reviewer", mock_reviewer)
    builder.add_node("hypothesis_generator", mock_hypothesis_generator)

    builder.add_edge(START, "literature_reviewer")

    # Define a local router to safely extract angles and dispatch Send objects
    def router(state: ParallelResearchState):
        angles = state.get("hypothesis_angles", [])
        return [Send("hypothesis_generator", {"current_angle": a}) for a in angles]

    # LangGraph REQUIRES the destination node array for Send topology mapping
    builder.add_conditional_edges("literature_reviewer", router, ["hypothesis_generator"])

    builder.add_edge("hypothesis_generator", END)

    graph = builder.compile()

    # Run the graph
    initial_state = {"research_topic": "AI in Edge Computing"}
    final_state = graph.invoke(initial_state)

    # Safely retrieve the aggregated hypotheses
    result_list = final_state.get("hypotheses", [])

    # Assert the reducer successfully combined all 3 parallel outputs
    assert len(result_list) == 3

    angles_produced = [h["angle"] for h in result_list]
    assert "conservative" in angles_produced
    assert "optimistic" in angles_produced
    assert "novel" in angles_produced
