from langgraph.types import Send

from discovery_agent.graph.parallel import route_to_parallel_agents


def test_route_to_parallel_agents():
    state = {"hypothesis_angles": ["conservative", "optimistic", "novel"]}
    sends = route_to_parallel_agents(state)

    # Assert it created exactly 3 parallel branches
    assert len(sends) == 3
    assert all(isinstance(s, Send) for s in sends)

    # Assert the nodes and arguments are mapped correctly
    assert sends[0].node == "hypothesis_generator"
    assert sends[0].arg == {"current_angle": "conservative"}
    assert sends[1].arg == {"current_angle": "optimistic"}
    assert sends[2].arg == {"current_angle": "novel"}
