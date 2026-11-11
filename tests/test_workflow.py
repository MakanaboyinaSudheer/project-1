"""Orchestration tests: the compiled LangGraph runs end to end with deterministic node stubs."""

import json

import pytest

from discovery_agent.graph import workflow
from discovery_agent.schemas import CodeAttempt, CriticReport, ExecutionResult


def _attempt(iteration: int, exit_code: int) -> CodeAttempt:
    result = ExecutionResult(exit_code=exit_code, stdout="", stderr="boom", duration_s=0.1)
    return CodeAttempt(iteration=iteration, code="pass", result=result, reward=0.5)


@pytest.fixture
def stubbed(monkeypatch, tmp_path):
    monkeypatch.setattr(workflow.settings, "checkpoint_db", str(tmp_path / "graph.sqlite"))
    monkeypatch.setattr(workflow.settings, "runs_dir", str(tmp_path / "runs"))
    calls = {"execute": 0, "critic": 0}
    plan = {"exit_codes": [0], "verdicts": ["significant"]}

    def execute(state):
        calls["execute"] += 1
        codes = plan["exit_codes"]
        return {"attempts": [_attempt(calls["execute"], codes[min(calls["execute"], len(codes)) - 1])]}

    def critic(state):
        calls["critic"] += 1
        verdicts = plan["verdicts"]
        return {"critic_report": CriticReport(verdict=verdicts[min(calls["critic"], len(verdicts)) - 1])}

    def debug(state):
        return {"debug_iteration": state.get("debug_iteration", 0) + 1}

    monkeypatch.setattr(workflow, "literature_reviewer_node", lambda s: {"literature_summary": "lit"})
    monkeypatch.setattr(workflow, "hypothesis_generator_node", lambda s: {"code": ""})
    monkeypatch.setattr(workflow, "code_executor_node", execute)
    monkeypatch.setattr(workflow, "debugger_node", debug)
    monkeypatch.setattr(workflow, "critic_node", critic)
    monkeypatch.setattr(workflow, "synthesizer_node", lambda s: {"draft_markdown": "draft"})
    return calls, plan, tmp_path


def test_end_to_end_success_persists_snapshot(stubbed):
    _, _, tmp_path = stubbed
    state = workflow.run_discovery("Federated learning", thread_id="run-ok")

    assert state["status"] == "succeeded"
    assert state["draft_markdown"] == "draft"
    saved = json.loads((tmp_path / "runs" / "run-ok" / "state.json").read_text())
    assert saved["status"] == "succeeded"
    assert saved["trace"][-1]["node"] == "succeeded"
    assert workflow.get_run_status("run-ok")["topic"] == "Federated learning"


def test_failed_execution_routes_through_debugger(stubbed):
    calls, plan, _ = stubbed
    plan["exit_codes"] = [1, 1, 0]
    state = workflow.run_discovery("topic", thread_id="run-debug")

    assert calls["execute"] == 3
    assert state["debug_iteration"] == 2
    assert state["status"] == "succeeded"


def test_debug_budget_stops_endless_loops(stubbed, monkeypatch):
    calls, plan, _ = stubbed
    monkeypatch.setattr(workflow.settings, "max_debug_iterations", 2)
    plan["exit_codes"] = [1]
    state = workflow.run_discovery("topic", thread_id="run-budget")

    assert state["status"] == "failed"
    assert calls["execute"] == 3


def test_non_significant_results_refine_then_fail_after_budget(stubbed):
    calls, plan, _ = stubbed
    plan["verdicts"] = ["not_significant"]
    state = workflow.run_discovery("topic", thread_id="run-refine")

    assert state["status"] == "failed"
    assert state["refinement_iteration"] == 3
    assert calls["critic"] == 4


def test_crash_is_recorded_then_resumable_and_runs_are_isolated(stubbed, monkeypatch):
    calls, _, _ = stubbed

    def boom(state):
        raise RuntimeError("LLM down")

    monkeypatch.setattr(workflow, "critic_node", boom)
    with pytest.raises(RuntimeError):
        workflow.run_discovery("topic", thread_id="run-crash")
    assert workflow.get_run_status("run-crash")["status"] == "failed"
    assert "LLM down" in workflow.get_run_status("run-crash")["error"]

    monkeypatch.setattr(
        workflow, "critic_node", lambda s: {"critic_report": CriticReport(verdict="significant")}
    )
    state = workflow.run_discovery("topic", thread_id="run-crash")
    assert state["status"] == "succeeded"
    assert calls["execute"] == 1  # resumed at the critic; the experiment was not re-run

    other = workflow.run_discovery("other", thread_id="run-other")
    assert other["topic"] == "other"
    assert workflow.get_run_status("run-crash")["topic"] == "topic"


def test_rejects_unsafe_thread_id(stubbed):
    with pytest.raises(ValueError):
        workflow.run_discovery("topic", thread_id="../escape")
