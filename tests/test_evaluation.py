"""Tests for the holdout loader, embedding wrapper, and run grader."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from langchain_core.runnables import RunnableLambda

from discovery_agent.evaluation import grader, holdout
from discovery_agent.evaluation.holdout import exclude_holdout_papers, load_holdout
from discovery_agent.retrieval import embeddings
from discovery_agent.schemas import Paper


def _paper(paper_id: str = "pubmed:1", title: str = "Study") -> Paper:
    return Paper(
        id=paper_id,
        source="pubmed",
        title=title,
        abstract="The treatment improved the outcome.",
        peer_reviewed=True,
    )


def test_load_holdout_validates_jsonl_and_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = tmp_path / "holdout.jsonl"
    record = _paper().model_dump(mode="json")
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    papers = load_holdout(path)

    assert [paper.id for paper in papers] == ["pubmed:1"]
    path.write_text(json.dumps(record) + "\n" + json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate holdout"):
        load_holdout(path)


def test_load_holdout_reports_invalid_records_and_missing_file(tmp_path: Path) -> None:
    path = tmp_path / "holdout.jsonl"
    path.write_text('{"source":"unknown"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="line 1"):
        load_holdout(path)
    with pytest.raises(FileNotFoundError, match="Holdout dataset not found"):
        load_holdout(tmp_path / "missing.jsonl")


def test_exclude_holdout_papers_filters_by_exact_id(tmp_path: Path) -> None:
    path = tmp_path / "holdout.jsonl"
    path.write_text(json.dumps(_paper().model_dump(mode="json")) + "\n", encoding="utf-8")

    retained = exclude_holdout_papers(
        [_paper(), _paper("pubmed:10", "Another study")],
        holdout_path=path,
    )

    assert [paper.id for paper in retained] == ["pubmed:10"]


def test_default_holdout_contains_ten_unique_recent_papers() -> None:
    papers = load_holdout()

    assert len(papers) == 10
    assert len({paper.id for paper in papers}) == 10
    assert all(paper.source == "arxiv" and not paper.peer_reviewed for paper in papers)
    assert all(paper.published is not None and paper.url for paper in papers)


def test_default_holdout_falls_back_to_installed_data_files(tmp_path: Path, monkeypatch) -> None:
    installed = tmp_path / "share" / "discovery-agent" / "holdout.jsonl"
    installed.parent.mkdir(parents=True)
    installed.write_text(json.dumps(_paper().model_dump(mode="json")) + "\n", encoding="utf-8")
    monkeypatch.setattr(holdout, "DEFAULT_HOLDOUT_PATH", Path("missing-data") / "holdout.jsonl")
    monkeypatch.setattr(holdout.sys, "prefix", str(tmp_path))

    assert [paper.id for paper in load_holdout(holdout.DEFAULT_HOLDOUT_PATH)] == ["pubmed:1"]


def test_embed_texts_loads_model_once_and_returns_normalized_vectors(monkeypatch) -> None:
    class FakeModel:
        def __init__(self) -> None:
            self.calls = []

        def encode(self, texts, **kwargs):
            self.calls.append((texts, kwargs))
            return np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=float)

    model = FakeModel()
    monkeypatch.setattr(embeddings, "_embedding_model", lambda: model)
    monkeypatch.setattr(embeddings.settings, "embedding_dim", 2)

    result = embeddings.embed_texts(["one", "two"])

    assert result == [[1.0, 0.0], [0.0, 1.0]]
    assert model.calls[0][1]["normalize_embeddings"] is True
    assert embeddings.embed_texts([]) == []
    with pytest.raises(TypeError, match="only strings"):
        embeddings.embed_texts(["ok", 3])


def test_embed_texts_rejects_wrong_model_dimension(monkeypatch) -> None:
    class FakeModel:
        def encode(self, texts, **kwargs):
            return np.ones((len(texts), 3), dtype=float)

    monkeypatch.setattr(embeddings, "_embedding_model", lambda: FakeModel())
    monkeypatch.setattr(embeddings.settings, "embedding_dim", 2)

    with pytest.raises(ValueError, match="expected 2"):
        embeddings.embed_texts(["text"])


def test_grade_run_computes_metrics_and_persists_report(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "run-1"
    run_dir.mkdir()
    state = {
        "hypothesis": {
            "statement": "Treatment improves outcome",
            "rationale": "Based on prior evidence",
            "expected_outcome": "A positive effect",
            "citations": [
                {"paper_id": "pubmed:1", "quote": "Treatment improved the outcome."},
                {"paper_id": "arxiv:2", "quote": "A non-peer-reviewed quote"},
            ],
        },
        "literature": [
            {
                "paper_id": "pubmed:1",
                "text": "Treatment improved the outcome.",
                "metadata": {"peer_reviewed": True, "source": "pubmed"},
            },
            {
                "paper_id": "arxiv:2",
                "text": "A non-peer-reviewed quote",
                "metadata": {"peer_reviewed": False, "source": "arxiv"},
            },
        ],
        "attempts": [
            {"result": {"exit_code": 0, "timed_out": False}},
            {"result": {"exit_code": 1, "timed_out": False}},
        ],
        "critic_report": {"verdict": "not_significant"},
    }
    (run_dir / "state.json").write_text(json.dumps(state), encoding="utf-8")
    (run_dir / "costs.jsonl").write_text('{"usd":0.25}\n', encoding="utf-8")

    papers = [
        _paper("arxiv:10", "Holdout one"),
        Paper(
            id="arxiv:11",
            source="arxiv",
            title="Holdout two",
            abstract="A second finding.",
        ),
    ]
    monkeypatch.setattr(grader, "load_holdout", lambda _: papers)
    monkeypatch.setattr(
        grader,
        "embed_texts",
        lambda _: [[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]],
    )
    judged_prompts = []

    def judge_response(prompt):
        judged_prompts.append(prompt)
        return grader._JudgeScore(score=4, rationale="The findings align.")

    fake_llm = SimpleNamespace(
        with_structured_output=lambda _: RunnableLambda(judge_response),
    )
    monkeypatch.setattr(grader, "get_llm", lambda: fake_llm)

    report = grader.grade_run(run_dir, tmp_path / "holdout.jsonl")

    assert report["holdout_count"] == 2
    assert report["mean_embedding_similarity"] == pytest.approx(0.4)
    assert report["mean_llm_judge_score"] == 4
    assert report["citation_validity_rate"] == 0.5
    assert report["valid_citation_count"] == 1
    assert report["experiment_success_rate"] == 0.5
    assert report["statistical_validity"] is True
    assert report["critic_verdict"] == "not_significant"
    assert report["cost_usd"] == 0.25
    assert len(judged_prompts) == 2
    saved = json.loads((run_dir / "grade.json").read_text(encoding="utf-8"))
    assert saved == report


def test_grade_run_rejects_missing_state_and_empty_holdout(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="Run state not found"):
        grader.grade_run(run_dir)

    (run_dir / "state.json").write_text(
        json.dumps({"hypothesis": {"statement": "A claim"}}),
        encoding="utf-8",
    )
    empty_holdout = tmp_path / "empty.jsonl"
    empty_holdout.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="Holdout dataset is empty"):
        grader.grade_run(run_dir, empty_holdout)


def test_grade_run_rejects_invalid_cost_entries(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "state.json").write_text(
        json.dumps({"hypothesis": {"statement": "Claim", "citations": []}}),
        encoding="utf-8",
    )
    (run_dir / "costs.jsonl").write_text('{"usd":"bad"}\n', encoding="utf-8")
    empty_holdout = tmp_path / "papers.jsonl"
    empty_holdout.write_text(json.dumps(_paper().model_dump(mode="json")) + "\n", encoding="utf-8")
    monkeypatch.setattr(grader, "embed_texts", lambda _: [[1.0], [1.0]])
    monkeypatch.setattr(
        grader,
        "get_llm",
        lambda: SimpleNamespace(
            with_structured_output=lambda _: RunnableLambda(
                lambda _: grader._JudgeScore(score=1, rationale="Weak relation")
            )
        ),
    )

    with pytest.raises(ValueError, match="Invalid cost record"):
        grader.grade_run(run_dir, empty_holdout)
