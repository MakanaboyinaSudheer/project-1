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
        calls.append(query)
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

    assert calls == ["topic", "q1", "q2"]
    assert [c.id for c in out["literature"]] == ["p1:abs", "p2:abs"]
    assert out["literature"][0].score == 0.9
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
