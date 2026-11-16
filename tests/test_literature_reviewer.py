from unittest.mock import MagicMock, patch

import pytest

from discovery_agent.agents import literature_reviewer as lr
from discovery_agent.schemas import RetrievedChunk


def _chunk(cid: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(id=cid, paper_id=cid.split(":abs")[0], text="t", score=score)


def _llm(queries):
    llm = MagicMock()
    llm.with_structured_output.return_value = MagicMock(
        invoke=MagicMock(return_value=lr.SubQueries(queries=queries))
    )
    llm.invoke.return_value = MagicMock(content="summary")
    return llm


def test_expands_queries_and_dedupes_chunks(tmp_path, monkeypatch):
    monkeypatch.setattr(lr.CostTracker, "__init__", lambda self, **kw: None)
    hits = {"topic": [_chunk("p1:abs", 0.5)], "q1": [_chunk("p1:abs", 0.9), _chunk("p2:abs", 0.4)]}
    calls = []

    def fake_retrieve(query, k, peer_reviewed_only):
        calls.append((query, peer_reviewed_only))
        if peer_reviewed_only:
            return [_chunk("pubmed:9:abs", 0.1)] if query == "q2" else []
        return hits.get(query, [])

    with (
        patch.object(lr, "get_llm", return_value=_llm(["q1", "q2"])),
        patch.object(lr, "retrieve", side_effect=fake_retrieve),
        patch.object(lr, "LITERATURE_REVIEW_PROMPT") as prompt,
        patch.object(lr, "SUB_QUERY_PROMPT") as sub,
    ):
        sub.__or__.side_effect = lambda other: other
        prompt.__or__.return_value = MagicMock(invoke=MagicMock(return_value=MagicMock(content="summary")))
        out = lr.literature_reviewer_node({"topic": "topic", "thread_id": "t"})

    assert [q for q, _ in calls] == ["topic", "topic", "q1", "q1", "q2", "q2"]
    # peer-reviewed evidence is kept even though it scores lower than the preprints
    assert [c.id for c in out["literature"]] == ["pubmed:9:abs", "p1:abs", "p2:abs"]
    assert out["literature"][1].score == 0.9
    assert out["literature_context"] == out["literature"]
    assert out["literature_summary"] == "summary"


def test_empty_index_fails_loudly(monkeypatch):
    monkeypatch.setattr(lr.CostTracker, "__init__", lambda self, **kw: None)
    with (
        patch.object(lr, "get_llm", return_value=_llm([])),
        patch.object(lr, "retrieve", return_value=[]),
        patch.object(lr, "SUB_QUERY_PROMPT") as sub,
        pytest.raises(RuntimeError, match="No literature"),
    ):
        sub.__or__.side_effect = lambda other: other
        lr.literature_reviewer_node({"topic": "topic"})
