from discovery_agent.graph.routing import MAX_REFINEMENT_STEPS, route_after_critic


def test_no_deadlock_on_continuous_rejection():
    """
    Simulates a worst-case scenario where the Critic continuously rejects the hypothesis.
    Verifies that the state machine eventually breaks out to the execution node.
    """
    state = {"final_report": "REJECTED: Methodological flaws.", "refinement_count": 0}

    execution_path = []

    # Simulate the LangGraph loop hitting the routing edge
    for _ in range(MAX_REFINEMENT_STEPS + 2):
        next_node = route_after_critic(state)
        execution_path.append(next_node)

        if next_node == "execution":
            break

        # Increment as the hypothesis_generator node would
        state["refinement_count"] += 1

    # The graph must eventually reach 'execution' to avoid a deadlock
    assert "execution" in execution_path

    # It should take exactly MAX_REFINEMENT_STEPS loops before the safety fallback triggers
    assert execution_path.count("hypothesis_generator") == MAX_REFINEMENT_STEPS
