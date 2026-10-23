"""Tests for M2.3 Hypothesis Generator node. Owner: Member 2."""

from unittest.mock import MagicMock, patch

from discovery_agent.agents.hypothesis_generator import Hypothesis, hypothesis_generator_node


@patch("discovery_agent.agents.hypothesis_generator.HYPOTHESIS_PROMPT")
@patch("discovery_agent.agents.hypothesis_generator.get_llm")
def test_reject_hallucinated_citations(mock_get_llm, mock_prompt):
    """M2.3: Verify the generator rejects hallucinated citations and retries."""

    state = {
        "research_topic": "AI in healthcare",
        "literature_summary": "Summary here",
        "literature_context": [{"id": "valid_paper_1"}, {"id": "valid_paper_2"}],
    }

    hallucinated_output = Hypothesis(
        title="Attempt 1: Bad",
        reasoning="Includes a fake citation",
        proposed_methodology="Mock methodology",
        citations=["valid_paper_1", "hallucinated_paper_99"],
    )

    valid_output = Hypothesis(
        title="Attempt 2: Good",
        reasoning="Only valid citations",
        proposed_methodology="Mock methodology",
        citations=["valid_paper_2"],
    )

    mock_chain = MagicMock()
    mock_chain.invoke.side_effect = [hallucinated_output, valid_output]
    mock_prompt.__or__.return_value = mock_chain

    result = hypothesis_generator_node(state)

    assert mock_chain.invoke.call_count == 2
    assert result["hypothesis"]["title"] == "Attempt 2: Good"
    assert "hallucinated_paper_99" not in result["hypothesis"]["citations"]


@patch("discovery_agent.agents.hypothesis_generator.HYPOTHESIS_PROMPT")
@patch("discovery_agent.agents.hypothesis_generator.get_llm")
def test_hypothesis_refinement_loop(mock_get_llm, mock_prompt):
    """M2.5: Verify the hypothesis generator incorporates previous hypothesis and critic feedback."""
    state = {
        "research_topic": "Quantum machine learning",
        "literature_summary": "QML overview.",
        "literature_context": [{"id": "qml_paper_1"}],
        "hypothesis": {
            "title": "Old Hypothesis",
            "reasoning": "...",
            "proposed_methodology": "...",
            "citations": [],
        },
        "final_report": "Critic Feedback: Methodology is too vague.",
    }

    mock_output = Hypothesis(
        title="Refined Hypothesis",
        reasoning="Addressed critic feedback",
        proposed_methodology="Specific QML methodology",
        citations=["qml_paper_1"],
    )

    mock_chain = MagicMock()
    mock_chain.invoke.return_value = mock_output
    mock_prompt.__or__.return_value = mock_chain

    result = hypothesis_generator_node(state)

    invoke_args = mock_chain.invoke.call_args[0][0]
    assert invoke_args["previous_hypothesis"] == state["hypothesis"]
    assert invoke_args["feedback"] == state["final_report"]

    assert result["hypothesis"]["title"] == "Refined Hypothesis"
