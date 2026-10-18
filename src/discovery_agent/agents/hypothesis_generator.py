"""Hypothesis Generator node, grounded in peer-reviewed literature via RAG. Owner: Member 2 (Week 2)."""

import re

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from discovery_agent.graph.state import ResearchState, trace_event
from discovery_agent.llm import get_llm
from discovery_agent.schemas import Citation, Hypothesis, RetrievedChunk


class _CitationQuote(BaseModel):
    paper_id: str
    quote: str


class _Generation(BaseModel):
    statement: str = Field(description="A specific, falsifiable scientific hypothesis")
    rationale: str
    citations: list["_CitationQuote"]
    test_plan: str
    expected_outcome: str
    novelty_score: float | None = None


_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Create a testable research hypothesis grounded in the supplied peer-reviewed literature. "
            "Every citation must use a supplied paper_id. Address the critic feedback when present. "
            "Do not invent evidence or citation identifiers.",
        ),
        (
            "user",
            "Topic: {topic}\n\nLiterature summary:\n{summary}\n\n"
            "Allowed peer-reviewed papers:\n{papers}\n\nCritic feedback:\n{feedback}\n"
            "Previous citation validation errors:\n{validation_errors}",
        ),
    ]
)


def hypothesis_generator_node(state: ResearchState) -> dict:
    """Produce a structured Hypothesis whose every claim cites a retrieved peer-reviewed chunk.
    Reject/regenerate if any citation's paper_id is not in state['literature']."""
    literature = state.get("literature", [])
    allowed: dict[str, RetrievedChunk] = {}
    source_texts: dict[str, list[str]] = {}
    for chunk in literature:
        if chunk.metadata.get("peer_reviewed") is True and chunk.metadata.get("source") != "arxiv":
            allowed.setdefault(chunk.paper_id, chunk)
            source_texts.setdefault(chunk.paper_id, []).append(chunk.text)
    if not allowed:
        raise ValueError("Cannot generate a grounded hypothesis without peer-reviewed literature")
    papers = "\n".join(
        f"{paper_id}: {chunk.metadata.get('title', paper_id)} — {chunk.text[:1200]}"
        for paper_id, chunk in allowed.items()
    )
    report = state.get("critic_report")
    feedback = "; ".join(report.issues) if report else "None"
    model = get_llm().with_structured_output(_Generation)
    errors = ""
    normalized_sources = {
        paper_id: re.sub(r"\s+", " ", " ".join(texts)).casefold() for paper_id, texts in source_texts.items()
    }
    for _ in range(3):
        generated = _Generation.model_validate(
            (_PROMPT | model).invoke(
                {
                    "topic": state.get("topic", ""),
                    "summary": state.get("literature_summary", ""),
                    "papers": papers,
                    "feedback": feedback,
                    "validation_errors": errors,
                }
            )
        )
        valid_citations = [
            citation
            for citation in generated.citations
            if citation.paper_id in allowed
            and citation.quote.strip()
            and re.sub(r"\s+", " ", citation.quote).strip().casefold()
            in normalized_sources[citation.paper_id]
        ]
        if valid_citations and len(valid_citations) == len(generated.citations):
            hypothesis = Hypothesis(
                statement=generated.statement,
                rationale=generated.rationale,
                citations=[
                    Citation(
                        paper_id=item.paper_id,
                        title=allowed[item.paper_id].metadata.get("title", item.paper_id),
                        quote=item.quote,
                    )
                    for item in valid_citations
                ],
                test_plan=generated.test_plan,
                expected_outcome=generated.expected_outcome,
                novelty_score=generated.novelty_score,
            )
            return {
                "hypothesis": hypothesis,
                "code": "",
                "debug_strategy": "",
                "refinement_iteration": state.get("refinement_iteration", 0),
                "trace": [
                    trace_event(
                        "hypothesis_generator",
                        "Generated a cited, structured hypothesis.",
                    )
                ],
            }
        errors = (
            "Each citation must identify an allowed paper and include a non-empty quotation. "
            "At least one valid citation is required."
        )
    raise ValueError(f"Hypothesis generation failed citation validation after three attempts: {errors}")
