import pytest

from discovery_agent.graph import workflow
from discovery_agent.schemas import CodeAttempt, ExecutionResult


def _attempt(*, exit_code: int, timed_out: bool = False) -> CodeAttempt:
    return CodeAttempt(
        iteration=1,
        code="pass",
        result=ExecutionResult(
            exit_code=exit_code,
            stdout="",
            stderr="",
            duration_s=0.1,
            timed_out=timed_out,
        ),
        reward=0.5,
    )


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        ({}, "failed"),
        ({"attempts": [_attempt(exit_code=0)]}, "critic"),
        ({"attempts": [_attempt(exit_code=1)]}, "debug"),
        ({"attempts": [_attempt(exit_code=0, timed_out=True)]}, "debug"),
    ],
)
def test_route_after_execution(state, expected: str) -> None:
    assert workflow.route_after_execution(state) == expected


def test_route_after_execution_stops_after_debug_budget(monkeypatch) -> None:
    monkeypatch.setattr(workflow.settings, "max_debug_iterations", 2)

    assert (
        workflow.route_after_execution({"attempts": [_attempt(exit_code=1)], "debug_iteration": 2})
        == "failed"
    )
