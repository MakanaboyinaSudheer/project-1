"""LangGraph state shared by every agent node."""

import operator
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

    # Literature Reviewer
    literature: list[RetrievedChunk]
    literature_summary: str

    # Hypothesis Generator
    hypothesis: Hypothesis

    # Code Executor + self-debug loop
    code: str
    attempts: Annotated[list[CodeAttempt], operator.add]
    debug_iteration: int

    # Critic
    critic_report: CriticReport

    # Synthesis
    abstract_latex: str
    draft_markdown: str

    # Bookkeeping (dashboard reads these)
    status: Literal["running", "succeeded", "failed"]
    trace: Annotated[list[dict], operator.add]  # {"node", "summary", "at"}
    costs: Annotated[list[CostEvent], operator.add]
    memory_summary: str  # rolling compressed context (Week 3)
