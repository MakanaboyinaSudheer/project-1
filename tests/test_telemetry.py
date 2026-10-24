"""Tests for LLM cost telemetry."""

import json

import pytest
from langchain_core.outputs import LLMResult

from discovery_agent.telemetry.costs import CostTracker


def test_cost_tracker_persists_usage_and_run_metadata(tmp_path) -> None:
    tracker = CostTracker(run_id="run-1", runs_dir=tmp_path)
    response = LLMResult(
        generations=[[]],
        llm_output={
            "model_name": "gpt-4o",
            "token_usage": {"prompt_tokens": 12, "completion_tokens": 7},
        },
    )

    tracker.on_llm_end(response, metadata={"langgraph_node": "literature_reviewer"})

    event = tracker.events[0]
    assert event.node == "literature_reviewer"
    assert event.model == "gpt-4o"
    assert event.input_tokens == 12
    assert event.output_tokens == 7
    assert event.usd == pytest.approx((12 * 2.5 + 7 * 10) / 1_000_000)

    saved_event = json.loads((tmp_path / "run-1" / "costs.jsonl").read_text(encoding="utf-8"))
    assert saved_event["node"] == "literature_reviewer"
    assert saved_event["model"] == "gpt-4o"
    assert saved_event["input_tokens"] == 12
    assert saved_event["output_tokens"] == 7
    assert saved_event["usd"] == event.usd
    assert "at" in saved_event


def test_cost_tracker_uses_local_and_unknown_model_pricing(tmp_path, caplog) -> None:
    tracker = CostTracker(run_id="run-2", runs_dir=tmp_path)

    tracker.on_llm_end(
        LLMResult(
            generations=[[]],
            llm_output={
                "model_name": "ollama:llama3.1",
                "token_usage": {"prompt_tokens": 100, "completion_tokens": 50},
            },
        )
    )
    tracker.on_llm_end(LLMResult(generations=[[]]))

    assert tracker.events[0].usd == 0
    assert tracker.events[1].model == "unknown"
    assert tracker.events[1].input_tokens == 0
    assert tracker.events[1].output_tokens == 0
    assert "No token pricing configured for model 'unknown'" in caplog.text
    assert len((tmp_path / "run-2" / "costs.jsonl").read_text(encoding="utf-8").splitlines()) == 2


@pytest.mark.parametrize("run_id", ["../outside", "subdir/run", r"subdir\run", ""])
def test_cost_tracker_rejects_unsafe_run_ids(tmp_path, run_id: str) -> None:
    with pytest.raises(ValueError, match="run_id"):
        CostTracker(run_id=run_id, runs_dir=tmp_path)


@pytest.mark.parametrize("rates", [(-1, 2), (float("inf"), 1), (1,)])
def test_cost_tracker_rejects_invalid_pricing(tmp_path, rates: tuple[float, ...]) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        CostTracker(run_id="invalid-prices", runs_dir=tmp_path, pricing={"model": rates})


def test_cost_tracker_rejects_negative_token_usage(tmp_path) -> None:
    tracker = CostTracker(run_id="negative-usage", runs_dir=tmp_path)
    response = LLMResult(
        generations=[[]],
        llm_output={"model_name": "gpt-4o", "token_usage": {"prompt_tokens": -1}},
    )

    with pytest.raises(ValueError, match="cannot be negative"):
        tracker.on_llm_end(response)
