"""Shared data contracts between modules.

Changing a model here affects every teammate: do it in its own PR, tag all four
members for review, and update MODULES.md in the same PR.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class Paper(BaseModel):
    id: str  # "arxiv:2401.01234" or "pubmed:38123456"
    source: Literal["arxiv", "pubmed"]
    title: str
    abstract: str
    authors: list[str] = Field(default_factory=list)
    published: datetime | None = None
    url: str = ""
    pdf_url: str | None = None
    peer_reviewed: bool = False  # PubMed journal articles = True; ArXiv preprints = False


class Chunk(BaseModel):
    """A unit stored in Milvus. Text chunks and (Week 3) figure descriptions."""

    id: str
    paper_id: str
    text: str
    kind: Literal["abstract", "body", "figure"] = "abstract"
    metadata: dict = Field(default_factory=dict)


class RetrievedChunk(Chunk):
    score: float


class Citation(BaseModel):
    paper_id: str
    title: str
    quote: str


class Hypothesis(BaseModel):
    statement: str
    rationale: str
    citations: list[Citation]
    test_plan: str  # how the experiment will test it
    expected_outcome: str
    novelty_score: float | None = None


class ExecutionResult(BaseModel):
    exit_code: int
    stdout: str
    stderr: str
    duration_s: float
    timed_out: bool = False
    artifacts: dict[str, str] = Field(default_factory=dict)  # filename -> path on host
    metrics: dict = Field(default_factory=dict)  # parsed from /output/metrics.json


class CodeAttempt(BaseModel):
    iteration: int
    code: str
    result: ExecutionResult
    reward: float  # see sandbox/feedback.py for the reward definition


class CriticReport(BaseModel):
    verdict: Literal["significant", "not_significant", "invalid"]
    p_values: dict[str, float] = Field(default_factory=dict)
    confidence_intervals: dict[str, tuple[float, float]] = Field(default_factory=dict)
    effect_sizes: dict[str, float] = Field(default_factory=dict)
    issues: list[str] = Field(default_factory=list)
    summary: str = ""


class CostEvent(BaseModel):
    node: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    usd: float = 0.0
    at: datetime = Field(default_factory=datetime.utcnow)
