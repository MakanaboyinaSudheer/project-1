"""Unit tests for the sandbox-backed code-executor node."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

from discovery_agent.agents import code_executor
from discovery_agent.schemas import ExecutionResult, Hypothesis


class FakeChain:
    def __init__(self, response: str) -> None:
        self.response = response
        self.inputs = None
        self.config = None

    def invoke(self, inputs, config=None):
        self.inputs = inputs
        self.config = config
        return AIMessage(content=self.response)


class FakePrompt:
    def __init__(self, chain: FakeChain) -> None:
        self.chain = chain
        self.model = None

    def __or__(self, model):
        self.model = model
        return self.chain


def _hypothesis() -> Hypothesis:
    return Hypothesis(
        statement="Treatment improves outcome",
        rationale="A prior observation supports the question.",
        citations=[],
        test_plan="Compare treatment and control samples.",
        expected_outcome="The treatment group has a lower p-value.",
    )


def test_executor_generates_code_runs_it_and_records_attempt(monkeypatch, tmp_path: Path) -> None:
    chain = FakeChain("```python\nprint('experiment')\n```")
    monkeypatch.setattr(code_executor, "_CODE_PROMPT", FakePrompt(chain))
    monkeypatch.setattr(code_executor, "get_llm", lambda: object())
    monkeypatch.setattr(code_executor.settings, "runs_dir", str(tmp_path))
    monkeypatch.setattr(code_executor, "CostTracker", lambda **_: "tracker")
    execution = ExecutionResult(
        exit_code=0,
        stdout="finished",
        stderr="",
        duration_s=0.5,
        metrics={"test": "welch_t_test", "p_value": 0.01},
    )
    calls = []
    monkeypatch.setattr(code_executor, "run_code", lambda code, path: calls.append((code, path)) or execution)

    result = code_executor.code_executor_node(
        {
            "hypothesis": _hypothesis(),
            "thread_id": "run-1",
            "attempts": [],
        }
    )

    assert calls[0][0] == "print('experiment')"
    assert calls[0][1] == tmp_path / "run-1" / "attempt_1"
    assert chain.inputs["statement"] == "Treatment improves outcome"
    assert chain.config["callbacks"] == ["tracker"]
    attempt = result["attempts"][0]
    assert attempt.iteration == 1
    assert attempt.code == "print('experiment')"
    assert attempt.result is execution
    assert attempt.reward == 0.5
    assert result["trace"][0]["node"] == "code_executor"


def test_executor_runs_debugger_code_without_calling_llm(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(code_executor.settings, "runs_dir", str(tmp_path))
    monkeypatch.setattr(code_executor, "get_llm", lambda: pytest.fail("LLM should not be called"))
    monkeypatch.setattr(
        code_executor, "CostTracker", lambda **_: pytest.fail("Tracker should not be created")
    )
    calls = []

    def fake_run_code(code: str, path: Path) -> ExecutionResult:
        calls.append((code, path))
        return ExecutionResult(
            exit_code=0,
            stdout="",
            stderr="",
            duration_s=0.1,
            metrics={"test": "t_test"},
        )

    monkeypatch.setattr(code_executor, "run_code", fake_run_code)

    result = code_executor.code_executor_node(
        {
            "hypothesis": _hypothesis(),
            "thread_id": "run-2",
            "code": "print('repaired')",
            "attempts": [SimpleNamespace()],
        }
    )

    assert result["code"] == "print('repaired')"
    assert result["attempts"][0].iteration == 2
    assert calls == [("print('repaired')", tmp_path / "run-2" / "attempt_2")]


@pytest.mark.parametrize(
    ("state", "message"),
    [
        ({}, "without a hypothesis"),
        ({"hypothesis": _hypothesis(), "thread_id": "../outside"}, "thread_id"),
        ({"hypothesis": _hypothesis(), "thread_id": "nested/run"}, "thread_id"),
    ],
)
def test_executor_rejects_missing_hypothesis_or_unsafe_run_id(state, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        code_executor.code_executor_node(state)


def test_executor_rejects_empty_generated_code(monkeypatch) -> None:
    chain = FakeChain("```python\n```")
    monkeypatch.setattr(code_executor, "_CODE_PROMPT", FakePrompt(chain))
    monkeypatch.setattr(code_executor, "get_llm", lambda: object())
    monkeypatch.setattr(code_executor, "CostTracker", lambda **_: "tracker")

    with pytest.raises(ValueError, match="no executable source"):
        code_executor.code_executor_node({"hypothesis": _hypothesis()})
