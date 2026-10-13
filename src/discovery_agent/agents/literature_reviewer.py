"""Literature Reviewer node. Owner: Member 2 (Week 1)."""

from discovery_agent.agents.prompts import (
    LITERATURE_REVIEW_PROMPT,
)
from discovery_agent.graph.state import ResearchState
from discovery_agent.llm import get_llm


def literature_reviewer_node(state: ResearchState) -> dict:
    """Retrieve literature for state['topic']; return `literature` + `literature_summary`."""
    topic = state.get("research_topic", "")
    llm = get_llm()

    mock_context_text = "Mocked literature context pending Day 6 RAG module."

    chain = LITERATURE_REVIEW_PROMPT | llm
    response = chain.invoke({"topic": topic, "context": mock_context_text})

    return {
        "literature_context": [],
        "literature_summary": response.content,
    }
