"""LangGraph state shared by every agent node."""

import operator
from datetime import UTC, datetime
from typing import Annotated, Literal, TypedDict

from discovery_agent.schemas import (
    CodeAttempt,
    CostEvent,
    CriticReport,
    Hypothesis,
    RetrievedChunk,
)


class ResearchState(TypedDict, total=False):
    # Input
    topic: str
    thread_id: str
    peer_reviewed_only: bool

    # Literature Reviewer
    literature: list[RetrievedChunk]
    literature_summary: str

    # Hypothesis Generator
    hypothesis: Hypothesis
    hypotheses: Annotated[list[Hypothesis], operator.add]  # refinement history

    # Code Executor + self-debug loop
    code: str
    attempts: Annotated[list[CodeAttempt], operator.add]
    debug_iteration: int
    debug_strategy: str
    refinement_iteration: int

    # Critic
    critic_report: CriticReport

    # Synthesis
    abstract_latex: str
    draft_markdown: str
    latex_compilation: dict[str, str]

    # Bookkeeping (dashboard reads these)
    status: Literal["running", "succeeded", "failed"]
    error: str
    trace: Annotated[list[dict], operator.add]  # {"node", "summary", "at"}
    costs: Annotated[list[CostEvent], operator.add]
    memory_summary: str  # rolling compressed context (Week 3)


def trace_event(node: str, summary: str) -> dict[str, str]:
    return {"node": node, "summary": summary, "at": datetime.now(UTC).isoformat()}
