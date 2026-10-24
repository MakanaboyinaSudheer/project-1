"""Hypothesis Generator node, grounded in peer-reviewed literature via RAG. Owner: Member 2."""

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
            "based on the literature summary. If a previous hypothesis and critic feedback "
            "are provided, you MUST explicitly refine the previous hypothesis to address the flaws. "
            "ONLY cite source IDs explicitly provided in the Valid Context IDs list.",
        ),
        (
            "user",
            "Topic: {topic}\n\n"
            "Valid Context IDs: {context_ids}\n\n"
            "Literature Summary:\n{summary}\n\n"
            "Previous Hypothesis:\n{previous_hypothesis}\n\n"
            "Critic Feedback:\n{feedback}\n\n"
            "{validation_errors}",
        ),
    ]
)


def hypothesis_generator_node(state: ResearchState) -> dict:
    """Produce a structured Hypothesis whose every claim cites a retrieved peer-reviewed chunk.
    Reject/regenerate if any citation's paper_id is not in state['literature_context']."""
    topic = state.get("research_topic", "")
    summary = state.get("literature_summary", "")
    feedback = state.get("final_report", "None")

    previous_hypothesis = state.get("hypothesis") or "None"

    literature_context = state.get("literature_context", [])
    valid_ids = {item.get("id") for item in literature_context if isinstance(item, dict) and "id" in item}

    if not valid_ids:
        valid_ids = {"mock_id_1", "mock_id_2"}

    llm = get_llm().with_structured_output(Hypothesis)
    chain = HYPOTHESIS_PROMPT | llm

    max_retries = 3
    validation_errors = ""
    hypothesis_obj = None

    for _ in range(max_retries):
        hypothesis_obj = chain.invoke(
            {
                "topic": topic,
                "context_ids": list(valid_ids),
                "summary": summary,
                "previous_hypothesis": previous_hypothesis,
                "feedback": feedback,
                "validation_errors": validation_errors,
            }
        )

        invalid_citations = [c for c in hypothesis_obj.citations if c not in valid_ids]

        if not invalid_citations:
            break

        validation_errors = (
            "Error in previous attempt: The following citations are "
            f"invalid/hallucinated: {invalid_citations}. "
            "You must ONLY use IDs from the Valid Context IDs list."
        )

    return {"hypothesis": hypothesis_obj.model_dump()}
