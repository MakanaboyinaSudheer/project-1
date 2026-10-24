"""Tests for M2.4 Memory Manager and SQLite Persistence. Owner: Member 2."""

import sqlite3
import uuid

from langgraph.checkpoint.sqlite import SqliteSaver

from discovery_agent.memory.manager import compress


def test_rolling_compression():
    """M2.4: Verify older attempts are compressed into memory_summary."""
    mock_state = {"code_attempts": ["Attempt 1", "Attempt 2", "Attempt 3", "Attempt 4"], "memory_summary": ""}

    update = compress(mock_state)

    assert len(update["code_attempts"]) == 2
    assert update["code_attempts"] == ["Attempt 3", "Attempt 4"]

    assert "Attempt 1" in update["memory_summary"]
    assert "Attempt 2" in update["memory_summary"]


def test_resume_after_kill():
    """M2.4: Verify SQLite checkpointer saves and retrieves thread state."""
    thread_id = str(uuid.uuid4())

    conn = sqlite3.connect(":memory:", check_same_thread=False)
    memory = SqliteSaver(conn)
    memory.setup()

    config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
    checkpoint_data = {
        "v": 1,
        "ts": "2026-10-24T00:00:00.000000+00:00",
        "id": str(uuid.uuid4()),
        "channel_values": {"test_key": "persisted_value"},
    }

    memory.put(config, checkpoint_data, {"source": "input", "step": 1, "writes": {}}, {})

    retrieved = memory.get_tuple(config)
    assert retrieved is not None
    assert retrieved.checkpoint["channel_values"]["test_key"] == "persisted_value"
