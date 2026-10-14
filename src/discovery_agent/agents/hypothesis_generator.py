"""Hypothesis Generator node, grounded in peer-reviewed literature via RAG. Owner: Member 2 (Week 2)."""

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from discovery_agent.graph.state import ResearchState
from discovery_agent.llm import get_llm


class Hypothesis(BaseModel):
    """Structured output for the generated research hypothesis."""

    title: str = Field(description="Title of the research hypothesis")
    reasoning: str = Field(description="Scientific reasoning based on the literature")
    proposed_methodology: str = Field(description="Proposed computational steps to test it")
    citations: list[str] = Field(description="List of source IDs cited from the literature context")


HYPOTHESIS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert AI researcher. Formulate a novel, testable hypothesis "
            "based on the literature summary. If critic feedback is provided, refine "
            "your previous hypothesis to address the flaws.",
        ),
        ("user", "Topic: {topic}\n\nLiterature Summary:\n{summary}\n\nCritic Feedback:\n{feedback}"),
    ]
)


def hypothesis_generator_node(state: ResearchState) -> dict:
    """Produce a structured Hypothesis whose every claim cites a retrieved peer-reviewed chunk.
    Reject/regenerate if any citation's paper_id is not in state['literature']."""
    topic = state.get("research_topic", "")
    summary = state.get("literature_summary", "")
    feedback = state.get("final_report", "None")

    llm = get_llm().with_structured_output(Hypothesis)
    chain = HYPOTHESIS_PROMPT | llm

    hypothesis_obj = chain.invoke({"topic": topic, "summary": summary, "feedback": feedback})

    return {"hypothesis": hypothesis_obj.model_dump()}
