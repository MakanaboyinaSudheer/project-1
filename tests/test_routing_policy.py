from discovery_agent.graph.routing import route_after_critic


def test_route_after_critic_approval():
    """Test standard progression on critic approval."""
    state = {
        "final_report": "The methodology is mathematically sound. STATUS: APPROVED.",
        "refinement_count": 0,
    }
    assert route_after_critic(state) == "execution"


def test_route_after_critic_rejection():
    """Test loop-back on critic rejection."""
    state = {
        "final_report": "Missing control variables. Please revise.",
        "refinement_count": 1,
    }
    assert route_after_critic(state) == "hypothesis_generator"


def test_route_after_critic_max_iterations():
    """Test safety fallback when max refinement iterations are reached."""
    state = {
        "final_report": "Still missing p-value corrections.",
        "refinement_count": 3,
    }
    # Should force progression to avoid deadlock, despite rejection
    assert route_after_critic(state) == "execution"
