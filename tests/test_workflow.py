"""End-to-end workflow integration tests. Owner: Member 2 (Week 1)."""

import uuid

from discovery_agent.graph.workflow import run_discovery


def test_end_to_end_stub_run():
    """M2.1: Verify the stub graph executes E2E through all required nodes."""
    test_topic = "Federated learning over multimodal clinical data"
    thread_id = str(uuid.uuid4())

    final_state = run_discovery(topic=test_topic, thread_id=thread_id)

    assert final_state is not None
    assert final_state.get("research_topic") == test_topic

    assert "literature_context" in final_state
    assert "literature_summary" in final_state
    assert "hypothesis" in final_state
    assert "code_attempts" in final_state
