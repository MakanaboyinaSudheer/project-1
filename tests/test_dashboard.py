import json
import os
import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard.app import _serialize_run, _trace_dataframe, discover_runs  # noqa: E402


def test_serialize_run_exports_run_data_as_json() -> None:
    run = {"thread_id": "run-1", "updated_at": Path("artifact").resolve()}

    exported = json.loads(_serialize_run(run))

    assert exported == {"thread_id": "run-1", "updated_at": str(run["updated_at"])}


def test_dashboard_exposes_refresh_and_run_exports(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "runs" / "sample-run"
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text(json.dumps({"status": "succeeded"}), encoding="utf-8")
    (run_dir / "draft.md").write_text("# Sample draft", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    app = AppTest.from_file(str(ROOT / "dashboard" / "app.py")).run()

    assert not app.exception
    assert any(item.label == "Auto-refresh" for item in app.selectbox)
    assert any(item.label == "Refresh now" for item in app.button)
    download_labels = [item.label for item in app.sidebar if item.type == "download_button"]
    assert download_labels == ["Export run JSON", "Export paper draft"]


def test_discover_runs_handles_missing_directory(tmp_path: Path) -> None:
    missing_runs = tmp_path / "no-runs"

    runs = discover_runs(missing_runs)

    assert runs == []


def test_discover_runs_lists_metadata_and_sorts_by_latest_artifact(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    older_run = runs_dir / "older"
    newer_run = runs_dir / "newer"
    older_run.mkdir(parents=True)
    newer_run.mkdir()

    (older_run / "state.json").write_text(
        json.dumps({"status": "succeeded", "attempts": [{"iteration": 1}]}),
        encoding="utf-8",
    )
    (older_run / "costs.jsonl").write_text('{"usd": 0.25}\n', encoding="utf-8")
    (newer_run / "state.json").write_text(json.dumps({"status": "running"}), encoding="utf-8")
    (newer_run / "costs.jsonl").write_text('{"usd": 0.5}\n', encoding="utf-8")

    os.utime(older_run / "state.json", (100, 100))
    os.utime(older_run / "costs.jsonl", (110, 110))
    os.utime(newer_run / "state.json", (120, 120))
    os.utime(newer_run / "costs.jsonl", (130, 130))
    os.utime(older_run, (90, 90))
    os.utime(newer_run, (100, 100))

    runs = discover_runs(runs_dir)

    assert [run["thread_id"] for run in runs] == ["newer", "older"]
    assert runs[0]["status"] == "running"
    assert runs[1]["status"] == "succeeded"
    assert runs[1]["attempts"] == [{"iteration": 1}]
    assert runs[0]["cost_total_usd"] == 0.5
    assert runs[1]["cost_total_usd"] == 0.25
    assert runs[0]["updated_at"] == 130


def test_discover_runs_ignores_malformed_cost_records(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "partial"
    run_dir.mkdir(parents=True)
    (run_dir / "costs.jsonl").write_text(
        '{"usd": 0.2}\nnot-json\n[]\n{"usd": "invalid"}\n{"usd": 0.3}\n',
        encoding="utf-8",
    )
    (run_dir / "run.lock").touch()
    idle_run_dir = tmp_path / "runs" / "idle"
    idle_run_dir.mkdir()
    (idle_run_dir / "state.json").write_text("{invalid", encoding="utf-8")

    runs = discover_runs(run_dir.parent)

    by_id = {run["thread_id"]: run for run in runs}
    assert by_id["partial"]["status"] == "running"
    assert by_id["partial"]["cost_total_usd"] == 0.5
    assert by_id["partial"]["costs"] == [{"usd": 0.2}, {"usd": "invalid"}, {"usd": 0.3}]
    assert by_id["idle"]["status"] == "ready"


def test_trace_dataframe_preserves_step_order_and_handles_incomplete_events() -> None:
    trace = [
        {"node": "literature_reviewer", "summary": "Found papers", "at": "2026-10-14T10:00:00Z"},
        {"node": "hypothesis_generator", "summary": "Generated hypothesis"},
        "unexpected trace entry",
    ]

    frame = _trace_dataframe(trace)

    assert list(frame.columns) == ["Step", "Node", "Summary", "Time"]
    assert frame.to_dict("records") == [
        {
            "Step": 1,
            "Node": "literature_reviewer",
            "Summary": "Found papers",
            "Time": "2026-10-14T10:00:00Z",
        },
        {"Step": 2, "Node": "hypothesis_generator", "Summary": "Generated hypothesis", "Time": ""},
        {"Step": 3, "Node": "unknown", "Summary": "unexpected trace entry", "Time": ""},
    ]
