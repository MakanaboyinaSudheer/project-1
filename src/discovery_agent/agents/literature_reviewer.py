"""Literature Reviewer node. Owner: Member 2 (Week 1)."""

from langchain_core.prompts import ChatPromptTemplate

from discovery_agent.graph.state import ResearchState
from discovery_agent.llm import get_llm


def literature_reviewer_node(state: ResearchState) -> dict:
    """Retrieve literature for state['topic']; return `literature` + `literature_summary`."""
    topic = state.get("research_topic", "")
    mock_context_text = "Mocked literature context pending Day 6 RAG module."

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert AI research scientist. "
                "Synthesize the provided literature into a concise foundational review.",
            ),
            ("user", "Topic: {topic}\n\nLiterature Context:\n{context}"),
        ]
    )

    llm = get_llm()
    chain = prompt | llm

    response = chain.invoke({"topic": topic, "context": mock_context_text})
    return {"literature_context": [], "literature_summary": response.content}
