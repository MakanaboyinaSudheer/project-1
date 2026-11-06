"""End-to-end orchestration test using deterministic agent stubs."""

from discovery_agent.graph import workflow
from discovery_agent.schemas import CodeAttempt, CriticReport, ExecutionResult


def test_end_to_end_stub_run(monkeypatch, tmp_path):
    """Verify the graph runs through every node without external services."""
    test_topic = "Federated learning over multimodal clinical data"

    monkeypatch.setattr(workflow.settings, "checkpoint_db", ":memory:")
    monkeypatch.setattr(workflow.settings, "runs_dir", str(tmp_path))
    monkeypatch.setattr(
        workflow,
        "literature_reviewer_node",
        lambda state: {"literature_summary": "Stub literature summary."},
    )
    monkeypatch.setattr(
        workflow,
        "hypothesis_generator_node",
        lambda state: {"hypothesis": {"statement": "Stub hypothesis."}},
    )
    monkeypatch.setattr(
        workflow,
        "code_executor_node",
        lambda state: {
            "attempts": [
                CodeAttempt(
                    iteration=1,
                    code="pass",
                    result=ExecutionResult(
                        exit_code=0,
                        stdout="",
                        stderr="",
                        duration_s=0.0,
                    ),
                    reward=0.5,
                )
            ]
        },
    )
    monkeypatch.setattr(
        workflow,
        "critic_node",
        lambda state: {"critic_report": CriticReport(verdict="significant")},
    )
    monkeypatch.setattr(
        workflow,
        "synthesizer_node",
        lambda state: {"draft_markdown": "Stub draft."},
    )

    final_state = workflow.run_discovery(topic=test_topic)

    assert final_state is not None
    assert final_state["topic"] == test_topic
    assert final_state["status"] == "succeeded"
