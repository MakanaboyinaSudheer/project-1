"""Tests for LLM cost telemetry."""

from langchain_core.outputs import LLMResult

from discovery_agent.schemas import CostEvent
from discovery_agent.telemetry.costs import CostTracker


def test_cost_tracker_records_provider_usage_and_run_metadata() -> None:
    tracker = CostTracker()
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

    assert tracker.events == [
        CostEvent(
            node="literature_reviewer",
            model="test-model",
            input_tokens=12,
            output_tokens=7,
        )
    ]


def test_cost_tracker_uses_defaults_when_usage_is_unavailable() -> None:
    tracker = CostTracker()

    tracker.on_llm_end(LLMResult(generations=[[]]))

    event = tracker.events[0]
    assert event.node == "llm"
    assert event.model == "unknown"
    assert event.input_tokens == 0
    assert event.output_tokens == 0
    assert event.usd == 0
