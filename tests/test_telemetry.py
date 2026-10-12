"""Tests for LLM cost telemetry."""

import json

import pytest
from langchain_core.outputs import LLMResult

from discovery_agent.telemetry.costs import CostTracker


def test_cost_tracker_records_provider_usage_and_run_metadata(tmp_path) -> None:
    tracker = CostTracker(run_id="run-1", runs_dir=tmp_path)
    result = LLMResult(
        generations=[[]],
        llm_output={
            "model_name": "test-model",
            "token_usage": {"prompt_tokens": 12, "completion_tokens": 7},
        },
    )

    tracker.on_llm_end(
        result,
        metadata={"langgraph_node": "literature_reviewer"},
    )

    event = tracker.events[0]
    assert event.node == "literature_reviewer"
    assert event.model == "test-model"
    assert event.input_tokens == 12
    assert event.output_tokens == 7
    assert event.usd == 0
    saved_event = json.loads((tmp_path / "run-1" / "costs.jsonl").read_text(encoding="utf-8"))
    assert saved_event["node"] == "literature_reviewer"
    assert saved_event["model"] == "test-model"
    assert saved_event["input_tokens"] == 12
    assert saved_event["output_tokens"] == 7
    assert saved_event["usd"] == 0


def test_cost_tracker_uses_defaults_when_usage_is_unavailable(tmp_path, caplog) -> None:
    tracker = CostTracker(run_id="run-2", runs_dir=tmp_path)

    tracker.on_llm_end(LLMResult(generations=[[]]))

    event = tracker.events[0]
    assert event.node == "llm"
    assert event.model == "unknown"
    assert event.input_tokens == 0
    assert event.output_tokens == 0
    assert event.usd == 0
    assert "No token pricing configured for model 'unknown'" in caplog.text


def test_cost_tracker_estimates_priced_and_local_model_costs(tmp_path) -> None:
    tracker = CostTracker(run_id="prices", runs_dir=tmp_path)
    result = LLMResult(
        generations=[[]],
        llm_output={
            "model_name": "gpt-4o-2024-08-06",
            "token_usage": {"prompt_tokens": 1_000_000, "completion_tokens": 500_000},
        },
    )

    tracker.on_llm_end(result)
    tracker.on_llm_end(
        LLMResult(
            generations=[[]],
            llm_output={"model_name": "ollama:llama3.1", "token_usage": {"prompt_tokens": 100}},
        )
    )

    assert tracker.events[0].usd == 7.5
    assert tracker.events[1].usd == 0
    lines = (tmp_path / "prices" / "costs.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert [json.loads(line)["usd"] for line in lines] == [7.5, 0]


@pytest.mark.parametrize("run_id", ["../outside", "subdir/run", r"subdir\run", ""])
def test_cost_tracker_rejects_unsafe_run_ids(tmp_path, run_id: str) -> None:
    with pytest.raises(ValueError, match="run_id"):
        CostTracker(run_id=run_id, runs_dir=tmp_path)


def test_cost_tracker_rejects_invalid_pricing(tmp_path) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        CostTracker(run_id="invalid-prices", runs_dir=tmp_path, pricing={"model": (-1, 2)})
