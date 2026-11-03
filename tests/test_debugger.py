from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from discovery_agent.agents import debugger
from discovery_agent.schemas import CodeAttempt, ExecutionResult, Hypothesis


class FakeChain:
    def __init__(self, content: str) -> None:
        self.content = content
        self.inputs = None
        self.config = None

    def invoke(self, inputs, config=None):
        self.inputs = inputs
        self.config = config
        return AIMessage(content=self.content)


class FakePrompt:
    def __init__(self, chain: FakeChain) -> None:
        self.chain = chain

    def __or__(self, model):
        assert model is not None
        return self.chain


def _failed_attempt() -> CodeAttempt:
    return CodeAttempt(
        iteration=1,
        code="print(undefined)",
        result=ExecutionResult(
            exit_code=1,
            stdout="",
            stderr="NameError: name 'undefined' is not defined",
            duration_s=0.1,
        ),
        reward=-0.5,
    )


def test_debugger_generates_repair_and_records_strategy(monkeypatch, tmp_path: Path) -> None:
    chain = FakeChain("```python\nprint('fixed')\n```")
    monkeypatch.setattr(debugger, "_DEBUG_PROMPT", FakePrompt(chain))
    monkeypatch.setattr(debugger, "get_llm", lambda: object())
    monkeypatch.setattr(debugger, "CostTracker", lambda **_: "tracker")
    monkeypatch.setattr(debugger.settings, "runs_dir", str(tmp_path))
    selected = []
    monkeypatch.setattr(debugger, "select_strategy", lambda path: selected.append(path) or "minimal_patch")

    state = {
        "hypothesis": Hypothesis(
            statement="Treatment improves outcome",
            rationale="Prior evidence",
            citations=[],
            test_plan="Compare groups",
            expected_outcome="Improvement",
        ),
        "thread_id": "run-1",
        "attempts": [_failed_attempt()],
    }

    result = debugger.debugger_node(state)

    assert result["code"] == "print('fixed')"
    assert result["debug_iteration"] == 1
    assert result["debug_strategy"] == "minimal_patch"
    assert selected == [tmp_path / "bandit.json"]
    assert chain.inputs["error_class"] == "runtime"
    assert chain.inputs["strategy_guidance"]
    assert chain.config["callbacks"] == ["tracker"]
    assert result["trace"][0]["node"] == "debugger"


@pytest.mark.parametrize(
    ("state", "message"),
    [
        ({}, "without a previous code attempt"),
        (
            {
                "attempts": [
                    CodeAttempt(
                        iteration=1,
                        code="pass",
                        result=ExecutionResult(exit_code=0, stdout="", stderr="", duration_s=0.1),
                        reward=0.5,
                    )
                ]
            },
            "successful code attempt",
        ),
        ({"attempts": [_failed_attempt()]}, "without a hypothesis"),
    ],
)
def test_debugger_rejects_invalid_state(state, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        debugger.debugger_node(state)
