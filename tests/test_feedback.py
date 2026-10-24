"""Tests for reward shaping and error classification used by the code executor."""

import pytest

from discovery_agent.sandbox.feedback import classify_error, compute_reward
from discovery_agent.schemas import ExecutionResult


@pytest.mark.parametrize(
    ("stderr", "expected"),
    [
        ("SyntaxError: invalid syntax", "syntax"),
        ("IndentationError: expected an indented block", "syntax"),
        ("ModuleNotFoundError: No module named 'x'", "import"),
        ("ImportError: cannot import name 'x'", "import"),
        ("MemoryError: unable to allocate", "oom"),
        ("operation timed out", "timeout"),
        ("RuntimeError: failed", "runtime"),
        ("", "other"),
    ],
)
def test_classify_error(stderr: str, expected: str) -> None:
    assert classify_error(stderr) == expected


def test_compute_reward_uses_metrics_and_iteration_penalty() -> None:
    valid = ExecutionResult(
        exit_code=0,
        stdout="",
        stderr="",
        duration_s=0.1,
        metrics={"test": "t_test", "p_value": 0.03},
    )
    missing_metrics = ExecutionResult(exit_code=0, stdout="", stderr="", duration_s=0.1)
    syntax_error = ExecutionResult(
        exit_code=1,
        stdout="",
        stderr="SyntaxError: invalid syntax",
        duration_s=0.1,
    )

    assert compute_reward(valid, 0) == pytest.approx(0.5)
    assert compute_reward(valid, 2) == pytest.approx(0.4)
    assert compute_reward(missing_metrics, 0) == pytest.approx(0.0)
    assert compute_reward(syntax_error, 0) == pytest.approx(-1.0)
    with pytest.raises(ValueError, match="non-negative"):
        compute_reward(valid, -1)
