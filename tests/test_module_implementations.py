import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import fitz
import pytest
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from discovery_agent.agents import code_executor as code_executor_module
from discovery_agent.agents import critic as critic_module
from discovery_agent.agents import debugger as debugger_module
from discovery_agent.agents import hypothesis_generator as hypothesis_module
from discovery_agent.agents import literature_reviewer as literature_reviewer_module
from discovery_agent.agents import synthesizer as synthesizer_module
from discovery_agent.evaluation import grader
from discovery_agent.graph import workflow
from discovery_agent.ingestion import arxiv_client, figure_parser, pdf_parser, pipeline, pubmed_client
from discovery_agent.memory import manager
from discovery_agent.retrieval import benchmark, embeddings, retriever, vector_store
from discovery_agent.sandbox import feedback
from discovery_agent.schemas import (
    Chunk,
    CodeAttempt,
    CriticReport,
    ExecutionResult,
    Hypothesis,
    Paper,
    RetrievedChunk,
)
from discovery_agent.synthesis import latex
from discovery_agent.telemetry import costs


def test_fetch_arxiv_maps_categories_and_sorts_newest_first(monkeypatch) -> None:
    older = MagicMock(
        get_short_id=lambda: "2601.00001",
        title="Older",
        summary="A",
        authors=[],
        published=datetime(2026, 1, 1, tzinfo=UTC),
        entry_id="https://arxiv.org/abs/2601.00001",
        pdf_url="https://arxiv.org/pdf/2601.00001",
        categories=["cs.LG"],
    )
    newer = MagicMock(
        get_short_id=lambda: "2601.00002",
        title="Newer",
        summary="B",
        authors=[],
        published=datetime(2026, 2, 1, tzinfo=UTC),
        entry_id="https://arxiv.org/abs/2601.00002",
        pdf_url=None,
        categories=["cs.LG"],
    )
    unrelated = MagicMock(categories=["math.AG"])
    client = MagicMock()
    client.results.return_value = [older, unrelated, newer]
    monkeypatch.setattr(arxiv_client.arxiv, "Client", lambda: client)

    papers = arxiv_client.fetch_arxiv("machine learning", ["cs.LG"], 10)

    assert [paper.id for paper in papers] == ["arxiv:2601.00002", "arxiv:2601.00001"]
    assert all(paper.peer_reviewed is False for paper in papers)


def test_fetch_pubmed_requires_email_and_maps_records(monkeypatch) -> None:
    monkeypatch.setattr(pubmed_client.settings, "ncbi_email", "research@example.org")
    monkeypatch.setattr(pubmed_client.settings, "ncbi_api_key", "")
    monkeypatch.setattr(pubmed_client.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        pubmed_client.Entrez,
        "esearch",
        lambda **_: MagicMock(close=lambda: None),
    )
    monkeypatch.setattr(
        pubmed_client.Entrez,
        "efetch",
        lambda **_: MagicMock(close=lambda: None),
    )
    records = iter(
        [
            {"IdList": ["123"]},
            {
                "PubmedArticle": [
                    {
                        "MedlineCitation": {
                            "PMID": "123",
                            "Article": {
                                "ArticleTitle": "Study",
                                "Abstract": {"AbstractText": ["Part one", "Part two"]},
                                "AuthorList": [{"ForeName": "Ada", "LastName": "Lovelace"}],
                                "PublicationTypeList": [{"PublicationType": "Journal Article"}],
                                "Journal": {
                                    "JournalIssue": {"PubDate": {"Year": "2025", "Month": "Jan", "Day": "2"}}
                                },
                            },
                        }
                    }
                ]
            },
        ]
    )
    monkeypatch.setattr(
        pubmed_client.Entrez,
        "read",
        lambda _: next(records),
    )

    paper = pubmed_client.fetch_pubmed("science", 1)[0]

    assert paper.id == "pubmed:123"
    assert paper.peer_reviewed is True
    assert paper.abstract == "Part one Part two"
    assert paper.authors == ["Ada Lovelace"]
    assert paper.published == datetime(2025, 1, 2, tzinfo=UTC)


def test_fetch_pubmed_fails_clearly_when_email_is_missing(monkeypatch) -> None:
    monkeypatch.setattr(pubmed_client.settings, "ncbi_email", "")
    with pytest.raises(ValueError, match="NCBI_EMAIL"):
        pubmed_client.fetch_pubmed("science")


def test_embeddings_use_batch_normalized_model(monkeypatch) -> None:
    captured = {}

    class FakeModel:
        def encode(self, texts, **kwargs):
            captured["texts"] = texts
            captured.update(kwargs)
            return MagicMock(tolist=lambda: [[1.0, 0.0] for _ in texts])

    monkeypatch.setattr(embeddings, "_model", lambda _: FakeModel())

    assert embeddings.embed_texts(["one", "two"]) == [[1.0, 0.0], [1.0, 0.0]]
    assert captured["normalize_embeddings"] is True
    assert captured["batch_size"] == 32
    assert embeddings.embed_texts([]) == []


def test_vector_store_upsert_and_search_round_trip(monkeypatch) -> None:
    class FakeClient:
        def has_collection(self, _):
            return True

        def upsert(self, **kwargs):
            self.rows = kwargs["data"]

        def get(self, **kwargs):
            return []

        def search(self, **kwargs):
            self.search_args = kwargs
            row = self.rows[0]
            return [[{"id": row["id"], "distance": 0.8, "entity": row}]]

        def close(self):
            pass

    store = vector_store.VectorStore.__new__(vector_store.VectorStore)
    store.uri = "unused"
    store.collection = "test"
    store.client = FakeClient()
    monkeypatch.setattr(vector_store, "embed_texts", lambda texts: [[0.5, 0.5] for _ in texts])
    chunk = Chunk(
        id="pubmed:1:abstract",
        paper_id="pubmed:1",
        text="Study abstract",
        metadata={
            "source": "pubmed",
            "peer_reviewed": True,
            "title": "Study",
            "url": "https://example.org",
            "published": datetime(2025, 1, 1, tzinfo=UTC),
        },
    )

    assert store.upsert([chunk]) == 1
    found = store.search("study", k=2, filter_expr="peer_reviewed == true")

    assert found[0].paper_id == "pubmed:1"
    assert found[0].metadata["peer_reviewed"] is True
    assert found[0].metadata["title"] == "Study"
    assert store.client.search_args["filter"] == "peer_reviewed == true"


def test_ingestion_is_idempotent_by_chunk_id(monkeypatch, caplog) -> None:
    paper = Paper(
        id="arxiv:1",
        source="arxiv",
        title="Paper",
        abstract="Abstract",
        peer_reviewed=False,
    )
    inserted = set()

    class FakeStore:
        def get_existing_ids(self, ids):
            return set(ids) & inserted

        def upsert(self, chunks):
            inserted.update(chunk.id for chunk in chunks)
            return len(chunks)

        def close(self):
            pass

    monkeypatch.setattr(pipeline, "fetch_arxiv", lambda *args: [paper])
    monkeypatch.setattr(pipeline, "fetch_pubmed", lambda *args: [])
    monkeypatch.setattr(pipeline, "VectorStore", FakeStore)
    monkeypatch.setattr(pipeline.settings, "ncbi_email", "")

    assert pipeline.run_ingestion("topic", 1) == 1
    assert pipeline.run_ingestion("topic", 1) == 0


def test_retrieve_filters_peer_reviewed_and_applies_mmr(monkeypatch) -> None:
    candidates = [
        RetrievedChunk(
            id="one",
            paper_id="pubmed:1",
            text="relevant",
            score=0.9,
            metadata={"source": "pubmed", "peer_reviewed": True},
        ),
        RetrievedChunk(
            id="two",
            paper_id="arxiv:2",
            text="similar relevant",
            score=0.89,
            metadata={"source": "arxiv", "peer_reviewed": False},
        ),
    ]
    fake_store = MagicMock()
    fake_store.search.return_value = candidates
    monkeypatch.setattr(retriever, "VectorStore", lambda: fake_store)
    monkeypatch.setattr(
        retriever,
        "embed_texts",
        lambda texts: [[1.0, 0.0] for _ in texts],
    )

    result = retriever.retrieve("query", k=2, peer_reviewed_only=True)

    assert [item.paper_id for item in result] == ["pubmed:1"]
    assert fake_store.search.call_args.kwargs["filter_expr"] == "peer_reviewed == true"
    fake_store.close.assert_called_once()


def test_retrieval_benchmark_reports_recall_and_rank() -> None:
    item = RetrievedChunk(
        id="chunk-1",
        paper_id="pubmed:1",
        text="evidence",
        score=0.9,
    )
    report = benchmark.benchmark_retrieval(
        [{"query": "topic", "relevant_paper_ids": ["pubmed:1", "pubmed:2"]}],
        k=1,
        retrieve_fn=lambda *_args, **_kwargs: [item],
    )

    assert report["query_count"] == 1
    assert report["recall_at_k"] == 0.5
    assert report["mean_reciprocal_rank"] == 1.0


def test_extract_text_chunks_respects_size_and_overlap(tmp_path: Path) -> None:
    pdf_path = tmp_path / "paper.pdf"
    document = fitz.open()
    for _ in range(3):
        page = document.new_page()
        page.insert_textbox(fitz.Rect(30, 30, 500, 700), " ".join(["science"] * 100))
    document.save(pdf_path)
    document.close()
    paper = Paper(
        id="pubmed:12",
        source="pubmed",
        title="Paper",
        abstract="Abstract",
        peer_reviewed=True,
    )

    chunks = pdf_parser.extract_text_chunks(paper, pdf_path, chunk_size=400)

    assert len(chunks) > 1
    assert all(chunk.kind == "body" and len(chunk.text) <= 400 for chunk in chunks)
    assert set(chunks[0].text.split()) & set(chunks[1].text.split())


def test_describe_figure_calls_vision_model(monkeypatch, tmp_path: Path) -> None:
    image_path = tmp_path / "figure.png"
    image_path.write_bytes(b"png bytes")
    model = MagicMock()
    model.invoke.return_value.content = "A line chart increases over time."
    monkeypatch.setattr(figure_parser, "get_vlm", lambda: model)

    chunk = figure_parser.describe_figure(image_path, "pubmed:20", "Figure 1")

    assert chunk.kind == "figure"
    assert "line chart" in chunk.text
    assert chunk.paper_id == "pubmed:20"
    model.invoke.assert_called_once()


def test_holdout_loads_jsonl_and_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = tmp_path / "holdout.jsonl"
    paper = {
        "id": "pubmed:1",
        "source": "pubmed",
        "title": "Study",
        "abstract": "Summary",
    }
    path.write_text(json.dumps(paper) + "\n", encoding="utf-8")

    assert len(grader.load_holdout(path)) == 1
    path.write_text(json.dumps(paper) + "\n" + json.dumps(paper) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate holdout"):
        grader.load_holdout(path)


def test_feedback_classifies_rewards_and_persists_bandit(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(feedback, "_bandit_path", lambda: tmp_path / "bandit.json")
    result = ExecutionResult(
        exit_code=0,
        stdout="",
        stderr="",
        duration_s=1,
        metrics={"p_value": 0.02, "test": "t-test", "effect_size": 0.4},
    )

    assert feedback.classify_error("ModuleNotFoundError: missing") == "import"
    assert feedback.compute_reward(result, iteration=2) == pytest.approx(0.4)
    feedback.update_strategy("minimal_patch", 0.7)
    feedback.update_strategy("minimal_patch", 0.3)
    assert json.loads((tmp_path / "bandit.json").read_text())["minimal_patch"] == {
        "count": 2,
        "mean_reward": pytest.approx(0.5),
    }


def test_memory_budget_compression_and_graph_routes() -> None:
    context = manager.build_context({"topic": "x", "memory_summary": "old"}, budget_tokens=2)
    compressed = manager.compress({"memory_summary": "start", "attempts": [], "trace": []})
    assert len(context) <= 8
    assert compressed["memory_summary"] == "start"
    assert workflow.route_after_execution({"attempts": []}) == "failed"
    assert workflow.route_after_critic({"critic_report": CriticReport(verdict="significant")}) == "synthesize"
    assert (
        workflow.route_after_critic(
            {"critic_report": CriticReport(verdict="invalid"), "refinement_iteration": 3}
        )
        == "failed"
    )


def test_workflow_wires_nodes_checkpointer_and_run_artifacts(monkeypatch, tmp_path: Path) -> None:
    hypothesis = Hypothesis(
        statement="Claim",
        rationale="Reason",
        citations=[{"paper_id": "pubmed:1", "title": "Study", "quote": "Evidence"}],
        test_plan="Test",
        expected_outcome="Expected",
    )
    result = ExecutionResult(
        exit_code=0,
        stdout="",
        stderr="",
        duration_s=0.2,
        metrics={"p_value": 0.01, "test": "t-test", "effect_size": 0.4},
    )
    monkeypatch.setattr(workflow.settings, "checkpoint_db", str(tmp_path / "checkpoints.sqlite"))
    monkeypatch.setattr(workflow.settings, "runs_dir", str(tmp_path / "runs"))
    monkeypatch.setattr(
        workflow,
        "literature_reviewer_node",
        lambda _: {"literature_summary": "Review", "trace": [{"node": "review"}]},
    )
    monkeypatch.setattr(
        workflow,
        "hypothesis_generator_node",
        lambda _: {"hypothesis": hypothesis, "trace": [{"node": "hypothesis"}]},
    )
    monkeypatch.setattr(
        workflow,
        "code_executor_node",
        lambda _: {
            "attempts": [CodeAttempt(iteration=1, code="pass", result=result, reward=0.5)],
            "trace": [{"node": "execute"}],
        },
    )
    monkeypatch.setattr(
        workflow,
        "critic_node",
        lambda _: {
            "critic_report": CriticReport(verdict="significant"),
            "trace": [{"node": "critic"}],
        },
    )
    monkeypatch.setattr(
        workflow,
        "synthesizer_node",
        lambda _: {"draft_markdown": "# draft", "trace": [{"node": "synthesizer"}]},
    )

    state = workflow.run_discovery("topic", thread_id="run-1")

    assert state["status"] == "succeeded"
    assert (tmp_path / "runs" / "run-1" / "state.json").is_file()
    assert (tmp_path / "runs" / "run-1" / "trace.json").is_file()
    with pytest.raises(ValueError, match="already completed"):
        workflow.run_discovery("topic", thread_id="run-1")


def test_cost_tracker_persists_model_usage(tmp_path: Path) -> None:
    tracker = costs.CostTracker(
        run_id="run-1",
        runs_dir=tmp_path,
        pricing={"model": (2.0, 4.0)},
    )
    response = MagicMock(
        llm_output={
            "model_name": "model",
            "token_usage": {"prompt_tokens": 1000, "completion_tokens": 500},
        }
    )

    tracker.on_llm_end(response, metadata={"langgraph_node": "hypothesis_generator"})

    saved = json.loads((tmp_path / "run-1" / "costs.jsonl").read_text(encoding="utf-8"))
    assert saved["node"] == "hypothesis_generator"
    assert saved["usd"] == 0.004


def test_latex_renderer_escapes_text_and_emits_citations() -> None:
    hypothesis = Hypothesis(
        statement="A & B",
        rationale="Result is 50%",
        citations=[{"paper_id": "pubmed:1", "title": "Paper_1", "quote": "exact quote"}],
        test_plan="Compare groups",
        expected_outcome="p < 0.05",
    )

    abstract = latex.render_abstract({"hypothesis": hypothesis})
    paper = latex.render_paper({"hypothesis": hypothesis})
    bibliography = latex.render_bibliography(hypothesis.citations)

    assert r"A \& B" in abstract
    assert r"50\%" in abstract
    assert r"\cite{paperpubmed1}" in abstract
    assert r"\bibliography{refs}" in paper
    assert "paperpubmed1" in bibliography


def test_hypothesis_generator_rejects_unverified_quotes(monkeypatch) -> None:
    from discovery_agent.schemas import RetrievedChunk

    chunk = RetrievedChunk(
        id="c1",
        paper_id="pubmed:1",
        text="Evidence supports the approach.",
        score=0.9,
        metadata={"source": "pubmed", "peer_reviewed": True, "title": "Study"},
    )
    generated = iter(
        [
            hypothesis_module._Generation(
                statement="Claim",
                rationale="Reason",
                citations=[{"paper_id": "pubmed:1", "quote": "invented quotation"}],
                test_plan="Test",
                expected_outcome="Outcome",
            ),
            hypothesis_module._Generation(
                statement="Claim",
                rationale="Reason",
                citations=[{"paper_id": "pubmed:1", "quote": "Evidence supports the approach."}],
                test_plan="Test",
                expected_outcome="Outcome",
            ),
        ]
    )
    llm = MagicMock(with_structured_output=lambda _: RunnableLambda(lambda _: next(generated)))
    monkeypatch.setattr(hypothesis_module, "get_llm", lambda: llm)

    result = hypothesis_module.hypothesis_generator_node({"literature": [chunk], "topic": "test"})

    assert result["hypothesis"].citations[0].paper_id == "pubmed:1"
    assert result["code"] == ""
    assert result["debug_strategy"] == ""


def test_literature_reviewer_expands_queries_and_summarizes(monkeypatch) -> None:
    chunk = RetrievedChunk(
        id="one",
        paper_id="pubmed:1",
        text="A finding.",
        score=0.8,
        metadata={"title": "Study"},
    )
    responses = iter(
        [
            AIMessage(content="query one\nquery two\nquery three"),
            AIMessage(content="Findings and gaps."),
        ]
    )
    monkeypatch.setattr(
        literature_reviewer_module,
        "get_llm",
        lambda: RunnableLambda(lambda _: next(responses)),
    )
    searched = []
    monkeypatch.setattr(
        literature_reviewer_module,
        "retrieve",
        lambda query, **_: searched.append(query) or [chunk],
    )

    result = literature_reviewer_module.literature_reviewer_node({"topic": "topic"})

    assert searched == ["topic", "query one", "query two", "query three"]
    assert result["literature_summary"] == "Findings and gaps."
    assert len(result["literature"]) == 1


def test_code_executor_runs_generated_code_and_records_attempt(monkeypatch, tmp_path: Path) -> None:
    hypothesis = Hypothesis(
        statement="Claim",
        rationale="Reason",
        citations=[{"paper_id": "pubmed:1", "title": "Study", "quote": "Evidence"}],
        test_plan="Test plan",
        expected_outcome="Expected",
    )
    monkeypatch.setattr(
        code_executor_module,
        "_CODE_PROMPT",
        ChatPromptTemplate.from_template("{statement}"),
    )
    monkeypatch.setattr(
        code_executor_module,
        "get_llm",
        lambda: RunnableLambda(lambda _: AIMessage(content="print('test')")),
    )
    execution = ExecutionResult(
        exit_code=0,
        stdout="done",
        stderr="",
        duration_s=0.5,
        metrics={"p_value": 0.01, "test": "t-test", "effect_size": 0.5},
    )
    monkeypatch.setattr(code_executor_module, "run_code", lambda code, path: execution)
    monkeypatch.setattr(code_executor_module.settings, "runs_dir", str(tmp_path))

    result = code_executor_module.code_executor_node(
        {"hypothesis": hypothesis, "thread_id": "test-run", "attempts": []}
    )

    assert result["code"] == "print('test')"
    assert result["attempts"][0].result is execution
    assert result["attempts"][0].reward == 0.5


def test_debugger_uses_bandit_strategy_and_rewrites_code(monkeypatch) -> None:
    previous = CodeAttempt(
        iteration=1,
        code="raise RuntimeError()",
        result=ExecutionResult(exit_code=1, stdout="", stderr="RuntimeError", duration_s=0.2),
        reward=-0.5,
    )
    monkeypatch.setattr(debugger_module, "choose_strategy", lambda: "simplify")
    monkeypatch.setattr(
        debugger_module,
        "get_llm",
        lambda: RunnableLambda(lambda _: AIMessage(content="print('fixed')")),
    )

    result = debugger_module.debugger_node({"attempts": [previous], "debug_iteration": 0})

    assert result["code"] == "print('fixed')"
    assert result["debug_iteration"] == 1
    assert result["debug_strategy"] == "simplify"


def test_critic_adds_methodology_review_without_changing_numeric_verdict(monkeypatch) -> None:
    attempt = CodeAttempt(
        iteration=1,
        code="",
        result=ExecutionResult(
            exit_code=0,
            stdout="",
            stderr="",
            duration_s=0.2,
            metrics={
                "p_value": 0.01,
                "test": "t-test",
                "effect_size": 0.5,
                "sample_size": 50,
            },
        ),
        reward=0.5,
    )
    hypothesis = Hypothesis(
        statement="Claim",
        rationale="Reason",
        citations=[{"paper_id": "pubmed:1", "title": "Study", "quote": "Evidence"}],
        test_plan="Test",
        expected_outcome="Expected",
    )
    monkeypatch.setattr(
        critic_module,
        "get_llm",
        lambda: SimpleNamespace(
            with_structured_output=lambda _: RunnableLambda(
                lambda _: {"issues": ["Possible confounder"], "summary": "Review"}
            )
        ),
    )

    report = critic_module.critic_node({"attempts": [attempt], "hypothesis": hypothesis})["critic_report"]

    assert report.verdict == "significant"
    assert report.p_values == {"primary": 0.01}
    assert "Possible confounder" in report.issues


def test_synthesizer_writes_draft_artifacts(monkeypatch, tmp_path: Path) -> None:
    hypothesis = Hypothesis(
        statement="Claim",
        rationale="Reason",
        citations=[{"paper_id": "pubmed:1", "title": "Study", "quote": "Evidence"}],
        test_plan="Test",
        expected_outcome="Expected",
    )
    monkeypatch.setattr(synthesizer_module.settings, "runs_dir", str(tmp_path))

    result = synthesizer_module.synthesizer_node({"hypothesis": hypothesis, "thread_id": "run-2"})

    run_dir = tmp_path / "run-2"
    assert set(path.name for path in run_dir.iterdir()) == {
        "abstract.tex",
        "paper.tex",
        "refs.bib",
        "draft.md",
    }
    assert "Claim" in result["draft_markdown"]


def test_grader_reports_holdout_metrics_and_persists_output(monkeypatch, tmp_path: Path) -> None:
    run_dir = tmp_path / "run-1"
    run_dir.mkdir()
    state = {
        "hypothesis": {
            "statement": "A testable claim",
            "rationale": "Reason",
            "expected_outcome": "Outcome",
            "citations": [{"paper_id": "pubmed:1", "quote": "Quoted evidence"}],
        },
        "literature": [
            {
                "paper_id": "pubmed:1",
                "text": "Quoted evidence",
                "metadata": {"peer_reviewed": True, "source": "pubmed"},
            }
        ],
        "attempts": [{"result": {"exit_code": 0, "timed_out": False}}],
        "critic_report": {"verdict": "significant"},
    }
    (run_dir / "state.json").write_text(json.dumps(state), encoding="utf-8")
    (run_dir / "costs.jsonl").write_text('{"usd":0.25}\n', encoding="utf-8")
    paper = Paper(id="pubmed:1", source="pubmed", title="Study", abstract="Findings")
    monkeypatch.setattr(grader, "load_holdout", lambda _: [paper])
    monkeypatch.setattr(grader, "embed_texts", lambda _: [[1.0, 0.0], [0.8, 0.6]])
    judge = MagicMock()
    judge.invoke.return_value = grader._JudgeScore(score=4, rationale="Strongly related")
    monkeypatch.setattr(grader, "get_llm", lambda: MagicMock(with_structured_output=lambda _: judge))

    report = grader.grade_run(run_dir, tmp_path / "holdout.jsonl")

    assert report["citation_validity_rate"] == 1.0
    assert report["experiment_success_rate"] == 1.0
    assert report["cost_usd"] == 0.25
    assert (run_dir / "grade.json").is_file()
