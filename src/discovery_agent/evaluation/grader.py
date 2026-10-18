"""Grade agent output against the holdout set. Owner: Member 1 (Week 4).

Metrics: hypothesis overlap with holdout findings (embedding similarity + LLM judge),
citation validity rate, experiment success rate, statistical validity, cost per run.
"""

import json
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

from discovery_agent.evaluation.holdout import load_holdout
from discovery_agent.llm import get_llm
from discovery_agent.retrieval.embeddings import embed_texts
from discovery_agent.telemetry.costs import CostTracker


class _JudgeScore(BaseModel):
    score: int = Field(ge=0, le=5, description="Evidence alignment from 0 (none) to 5 (strong)")
    rationale: str


def grade_run(run_dir: str | Path, holdout_path: str | Path = "data/holdout.jsonl") -> dict:
    run_path = Path(run_dir)
    state_path = run_path / "state.json"
    if not state_path.is_file():
        raise FileNotFoundError(f"Run state not found: {state_path}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    hypothesis = state.get("hypothesis")
    if not isinstance(hypothesis, dict) or not hypothesis.get("statement"):
        raise ValueError("Run state does not contain a generated hypothesis")
    holdout = load_holdout(holdout_path)
    if not holdout:
        raise ValueError("Holdout dataset is empty; cannot calculate evaluation metrics")

    hypothesis_text = " ".join(
        str(hypothesis.get(field, "")) for field in ("statement", "rationale", "expected_outcome")
    )
    paper_texts = [f"{paper.title}\n{paper.abstract}" for paper in holdout]
    vectors = np.asarray(embed_texts([hypothesis_text, *paper_texts]), dtype=float)
    similarities = vectors[1:] @ vectors[0]
    judge = get_llm().with_structured_output(_JudgeScore)
    cost_tracker = CostTracker(run_id=run_path.name, runs_dir=run_path.parent)
    per_paper = []
    for paper, similarity in zip(holdout, similarities, strict=True):
        result = _JudgeScore.model_validate(
            judge.invoke(
                [
                    (
                        "system",
                        "Judge whether the research hypothesis is supported by or meaningfully related "
                        "to the paper's reported findings. Score evidence alignment from 0 to 5; do not "
                        "reward mere keyword overlap.",
                    ),
                    (
                        "human",
                        f"Hypothesis: {hypothesis_text}\n\n"
                        f"Paper title: {paper.title}\nAbstract: {paper.abstract[:5000]}",
                    ),
                ],
                config={"callbacks": [cost_tracker]},
            )
        )
        per_paper.append(
            {
                "paper_id": paper.id,
                "embedding_similarity": float(similarity),
                "llm_judge_score": result.score,
                "llm_judge_rationale": result.rationale,
            }
        )

    literature = state.get("literature", [])
    retrieved_text: dict[str, str] = {}
    for item in literature:
        if not isinstance(item, dict):
            continue
        metadata = item.get("metadata")
        if (
            isinstance(metadata, dict)
            and item.get("paper_id")
            and metadata.get("peer_reviewed") is True
            and metadata.get("source") != "arxiv"
        ):
            paper_id = item["paper_id"]
            retrieved_text[paper_id] = retrieved_text.get(paper_id, "") + " " + str(item.get("text", ""))
    citations = hypothesis.get("citations", [])
    citation_validity_count = 0
    citation_count = 0
    for item in citations:
        if not isinstance(item, dict):
            continue
        citation_count += 1
        source = retrieved_text.get(str(item.get("paper_id", "")), "")
        quote = " ".join(str(item.get("quote", "")).casefold().split())
        if source and quote and quote in " ".join(source.casefold().split()):
            citation_validity_count += 1
    citation_validity = citation_validity_count / citation_count if citation_count else 0.0
    attempts = state.get("attempts", [])
    experiment_success = (
        sum(
            1
            for attempt in attempts
            if isinstance(attempt, dict)
            and isinstance(attempt.get("result"), dict)
            and attempt["result"].get("exit_code") == 0
            and not attempt["result"].get("timed_out", False)
        )
        / len(attempts)
        if attempts
        else 0.0
    )
    report = state.get("critic_report") or {}
    costs_path = run_path / "costs.jsonl"
    total_cost = 0.0
    if costs_path.exists():
        for line_number, line in enumerate(costs_path.read_text(encoding="utf-8").splitlines(), start=1):
            try:
                item = json.loads(line)
                total_cost += float(item["usd"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid cost record on line {line_number}: {exc}") from exc
    report = {
        "run_id": run_path.name,
        "holdout_count": len(holdout),
        "mean_embedding_similarity": float(np.mean(similarities)),
        "mean_llm_judge_score": float(np.mean([item["llm_judge_score"] for item in per_paper])),
        "citation_validity_rate": citation_validity,
        "experiment_success_rate": experiment_success,
        "statistical_validity": report.get("verdict") in {"significant", "not_significant"},
        "cost_usd": total_cost,
        "per_paper": per_paper,
    }
    (run_path / "grade.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
