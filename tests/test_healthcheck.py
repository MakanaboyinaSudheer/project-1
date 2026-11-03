import pytest

from discovery_agent.healthcheck import _sandbox_execution
from discovery_agent.schemas import ExecutionResult


def test_sandbox_execution_check_runs_a_real_smoke_program(monkeypatch) -> None:
    calls = []

    def fake_run_code(code, path, timeout_s):
        calls.append((code, path.name, timeout_s))
        return ExecutionResult(
            exit_code=0,
            stdout="sandbox-ready\n",
            stderr="",
            duration_s=0.25,
        )

    monkeypatch.setattr("discovery_agent.healthcheck.run_code", fake_run_code)
    monkeypatch.setattr("discovery_agent.healthcheck.settings.sandbox_timeout_s", 120)

    assert _sandbox_execution() == "execution completed in 0.25s"
    assert calls == [("print('sandbox-ready')", "attempt", 15)]


def test_sandbox_execution_check_reports_failed_execution(monkeypatch) -> None:
    monkeypatch.setattr(
        "discovery_agent.healthcheck.run_code",
        lambda *_, **__: ExecutionResult(
            exit_code=1,
            stdout="",
            stderr="container failed",
            duration_s=0.1,
        ),
    )

    with pytest.raises(RuntimeError, match="container failed"):
        _sandbox_execution()
