"""Evaluate a completed discovery run against a fixed literature holdout set."""

import json
import math
import os
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Any

import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from discovery_agent.evaluation.holdout import DEFAULT_HOLDOUT_PATH, load_holdout
from discovery_agent.llm import get_llm
from discovery_agent.retrieval.embeddings import embed_texts
from discovery_agent.schemas import Paper
from discovery_agent.telemetry.costs import CostTracker

_JUDGE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Evaluate whether the research hypothesis is supported by or meaningfully related to "
            "the paper's reported findings. Score evidence alignment from 0 (none) to 5 (strong); "
            "do not reward mere keyword overlap. Explain the score using only the supplied text.",
        ),
        (
            "human",
            "Hypothesis:\n{hypothesis}\n\nPaper title:\n{title}\n\nPaper abstract:\n{abstract}",
        ),
    ]
)


class _JudgeScore(BaseModel):
    score: int = Field(ge=0, le=5, description="Evidence alignment from 0 (none) to 5 (strong)")
    rationale: str = Field(min_length=1)


def _read_run_state(run_path: Path) -> dict[str, Any]:
    state_path = run_path / "state.json"
    if not state_path.is_file():
        raise FileNotFoundError(f"Run state not found: {state_path}")
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read valid run state from {state_path}: {exc}") from exc
    if not isinstance(state, dict):
        raise ValueError(f"Run state must contain a JSON object: {state_path}")
    hypothesis = state.get("hypothesis")
    if not isinstance(hypothesis, dict) or not isinstance(hypothesis.get("statement"), str):
        raise ValueError("Run state does not contain a generated hypothesis statement")
    return state


def _hypothesis_text(hypothesis: dict[str, Any]) -> str:
    return "\n".join(
        f"{label}: {hypothesis.get(key, '')}"
        for label, key in (
            ("Statement", "statement"),
            ("Rationale", "rationale"),
            ("Expected outcome", "expected_outcome"),
        )
    )


def _embedding_similarities(hypothesis: str, papers: list[Paper]) -> list[float]:
    vectors = np.asarray(
        embed_texts([hypothesis, *(f"{paper.title}\n{paper.abstract}" for paper in papers)]),
        dtype=float,
    )
    if vectors.ndim != 2 or vectors.shape[0] != len(papers) + 1 or vectors.shape[1] == 0:
        raise ValueError(
            "Embedding helper returned an unexpected shape for the hypothesis and holdout papers"
        )
    if not np.isfinite(vectors).all():
        raise ValueError("Embedding helper returned non-finite values")

    hypothesis_vector = vectors[0]
    paper_vectors = vectors[1:]
    norms = np.linalg.norm(vectors, axis=1)
    denominator = norms[0] * norms[1:]
    similarities = np.zeros(len(papers), dtype=float)
    nonzero = denominator > 0
    similarities[nonzero] = (paper_vectors[nonzero] @ hypothesis_vector) / denominator[nonzero]
    return np.clip(similarities, -1.0, 1.0).tolist()


def _peer_reviewed_literature(state: dict[str, Any]) -> dict[str, str]:
    sources: dict[str, str] = {}
    for item in state.get("literature", []):
        if not isinstance(item, dict):
            continue
        metadata = item.get("metadata")
        if not isinstance(metadata, dict):
            continue
        source = str(metadata.get("source", "")).casefold()
        peer_reviewed = metadata.get("peer_reviewed") is True or source == "pubmed"
        paper_id = item.get("paper_id")
        text = item.get("text")
        if peer_reviewed and source != "arxiv" and isinstance(paper_id, str) and isinstance(text, str):
            sources[paper_id] = f"{sources.get(paper_id, '')} {text}".strip()
    return sources


def _citation_validity(state: dict[str, Any]) -> tuple[float, int, int]:
    hypothesis = state["hypothesis"]
    citations = hypothesis.get("citations", [])
    if not isinstance(citations, list):
        raise ValueError("Hypothesis citations must be a list")
    literature = _peer_reviewed_literature(state)

    valid_count = 0
    for citation in citations:
        if not isinstance(citation, dict):
            continue
        paper_id = citation.get("paper_id")
        quote = citation.get("quote")
        if not isinstance(paper_id, str) or not isinstance(quote, str) or not quote.strip():
            continue
        source = literature.get(paper_id, "")
        normalized_quote = " ".join(quote.casefold().split())
        normalized_source = " ".join(source.casefold().split())
        if normalized_source and normalized_quote in normalized_source:
            valid_count += 1
    rate = valid_count / len(citations) if citations else 0.0
    return rate, valid_count, len(citations)


def _experiment_success(state: dict[str, Any]) -> tuple[float, int, int]:
    attempts = state.get("attempts", [])
    if not isinstance(attempts, list):
        raise ValueError("Run attempts must be a list")
    successes = 0
    for attempt in attempts:
        if not isinstance(attempt, dict):
            continue
        result = attempt.get("result")
        if not isinstance(result, dict):
            continue
        if result.get("exit_code") == 0 and result.get("timed_out") is not True:
            successes += 1
    return (successes / len(attempts) if attempts else 0.0, successes, len(attempts))


def _total_cost(run_path: Path) -> float:
    costs_path = run_path / "costs.jsonl"
    if not costs_path.exists():
        return 0.0
    total = 0.0
    try:
        lines = costs_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"Could not read cost records from {costs_path}: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
            amount = float(event["usd"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid cost record on line {line_number} in {costs_path}: {exc}") from exc
        if not math.isfinite(amount) or amount < 0:
            raise ValueError(f"Cost on line {line_number} in {costs_path} must be finite and non-negative")
        total += amount
    return total


def _write_report(run_path: Path, report: dict[str, Any]) -> None:
    descriptor, temp_name = tempfile.mkstemp(prefix=".grade-", suffix=".tmp", dir=run_path)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, run_path / "grade.json")
    except Exception:
        with suppress(FileNotFoundError):
            os.unlink(temp_name)
        raise


def grade_run(
    run_dir: str | Path,
    holdout_path: str | Path = DEFAULT_HOLDOUT_PATH,
) -> dict[str, Any]:
    """Compute holdout, citation, execution, statistical, and cost metrics for one run."""
    run_path = Path(run_dir)
    if not run_path.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_path}")
    state = _read_run_state(run_path)
    holdout = load_holdout(holdout_path)
    if not holdout:
        raise ValueError("Holdout dataset is empty; cannot calculate evaluation metrics")

    hypothesis_text = _hypothesis_text(state["hypothesis"])
    similarities = _embedding_similarities(hypothesis_text, holdout)
    judge = get_llm().with_structured_output(_JudgeScore)
    prompt = _JUDGE_PROMPT
    cost_tracker = CostTracker(run_id=run_path.name, runs_dir=run_path.parent)
    per_paper: list[dict[str, Any]] = []
    for paper, similarity in zip(holdout, similarities, strict=True):
        judged = (prompt | judge).invoke(
            {
                "hypothesis": hypothesis_text,
                "title": paper.title,
                "abstract": paper.abstract,
            },
            config={"callbacks": [cost_tracker]},
        )
        score = _JudgeScore.model_validate(judged)
        per_paper.append(
            {
                "paper_id": paper.id,
                "title": paper.title,
                "embedding_similarity": similarity,
                "llm_judge_score": score.score,
                "llm_judge_rationale": score.rationale,
            }
        )

    citation_rate, valid_citations, citation_count = _citation_validity(state)
    success_rate, successful_attempts, attempt_count = _experiment_success(state)
    critic_report = state.get("critic_report")
    verdict = critic_report.get("verdict") if isinstance(critic_report, dict) else None
    report: dict[str, Any] = {
        "run_id": run_path.name,
        "holdout_count": len(holdout),
        "mean_embedding_similarity": float(np.mean(similarities)),
        "mean_llm_judge_score": float(np.mean([item["llm_judge_score"] for item in per_paper])),
        "llm_judge_score_scale": {"minimum": 0, "maximum": 5},
        "citation_validity_rate": citation_rate,
        "valid_citation_count": valid_citations,
        "citation_count": citation_count,
        "experiment_success_rate": success_rate,
        "successful_attempt_count": successful_attempts,
        "attempt_count": attempt_count,
        "statistical_validity": verdict in {"significant", "not_significant"},
        "critic_verdict": verdict,
        "cost_usd": _total_cost(run_path),
        "per_paper": per_paper,
    }
    _write_report(run_path, report)
    return report
