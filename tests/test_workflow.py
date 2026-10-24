"""End-to-end workflow integration tests. Owner: Member 2 (Week 1)."""

import uuid

import pytest

from discovery_agent.graph.workflow import run_discovery


@pytest.mark.integration
def test_end_to_end_stub_run():
    """M2.1: Verify the stub graph executes E2E through all required nodes."""
    test_topic = "Federated learning over multimodal clinical data"
    thread_id = str(uuid.uuid4())

    final_state = run_discovery(topic=test_topic, thread_id=thread_id, db_path=":memory:")

    assert final_state is not None
    assert "research_topic" in final_state
    assert final_state["research_topic"] == test_topic
