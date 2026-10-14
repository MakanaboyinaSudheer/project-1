"""Streamlit supervisor dashboard for the discovery-agent project.

The dashboard is intentionally resilient to a missing runs/ directory so it can be
launched during early project stages and still provide a useful empty-state view.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

DEFAULT_RUNS_DIR = Path("runs")


def _read_json(path: Path) -> Any:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _read_costs(path: Path) -> tuple[float, list[dict[str, Any]]]:
    if not path.exists():
        return 0.0, []

    records: list[dict[str, Any]] = []
    total = 0.0
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(item, dict):
                continue
            records.append(item)
            try:
                total += float(item.get("usd", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
    except (json.JSONDecodeError, OSError):
        return 0.0, []
    return total, records


def discover_runs(base_dir: str | Path = DEFAULT_RUNS_DIR) -> list[dict[str, Any]]:
    """Return discovered runs under the project runs directory.

    Each entry exposes enough metadata for the dashboard to render a run list even
    before the full graph and agents are implemented.
    """
    root = Path(base_dir)
    if not root.exists():
        return []

    runs: list[dict[str, Any]] = []
    for run_dir in root.iterdir():
        if not run_dir.is_dir():
            continue

        try:
            updated_at = run_dir.stat().st_mtime
        except FileNotFoundError:
            continue
        for artifact in run_dir.iterdir():
            try:
                updated_at = max(updated_at, artifact.stat().st_mtime)
            except FileNotFoundError:
                continue

        state = _read_json(run_dir / "state.json")
        trace = state.get("trace", []) if isinstance(state, dict) else []
        if not trace:
            trace = _read_json(run_dir / "trace.json")
        if isinstance(trace, dict):
            trace = trace.get("trace", [])

        critic_report = state.get("critic_report", {}) if isinstance(state, dict) else {}
        if not critic_report:
            critic_report = _read_json(run_dir / "critic_report.json")

        attempts = state.get("attempts", []) if isinstance(state, dict) else []
        if not attempts:
            attempts = _read_json(run_dir / "attempts.json")
        if isinstance(attempts, dict):
            attempts = attempts.get("attempts", [])

        literature = state.get("literature", []) if isinstance(state, dict) else []
        if not literature:
            literature = _read_json(run_dir / "literature.json")
        if isinstance(literature, dict):
            literature = literature.get("literature", [])

        cost_total, cost_rows = _read_costs(run_dir / "costs.jsonl")
        status = state.get("status", "unknown") if isinstance(state, dict) else "unknown"
        if status == "unknown":
            status = "running" if (run_dir / "run.lock").exists() else "ready"

        runs.append(
            {
                "thread_id": run_dir.name,
                "path": str(run_dir),
                "status": status,
                "updated_at": updated_at,
                "trace": trace if isinstance(trace, list) else [],
                "attempts": attempts if isinstance(attempts, list) else [],
                "critic_report": critic_report if isinstance(critic_report, dict) else {},
                "literature": literature if isinstance(literature, list) else [],
                "costs": cost_rows,
                "cost_total_usd": cost_total,
            }
        )

    return sorted(runs, key=lambda run: run["updated_at"], reverse=True)


def _run_summary(run: dict[str, Any]) -> str:
    return (
        f"{run.get('thread_id', 'unknown')} · {run.get('status', 'ready')} · "
        f"${run.get('cost_total_usd', 0.0):.3f}"
    )


def _trace_dataframe(trace: list[Any]) -> pd.DataFrame:
    rows = []
    for step, event in enumerate(trace, start=1):
        if isinstance(event, dict):
            rows.append(
                {
                    "Step": step,
                    "Node": event.get("node", "unknown"),
                    "Summary": event.get("summary", ""),
                    "Time": event.get("at", ""),
                }
            )
        else:
            rows.append({"Step": step, "Node": "unknown", "Summary": str(event), "Time": ""})
    return pd.DataFrame(rows, columns=["Step", "Node", "Summary", "Time"])


def main() -> None:
    st.set_page_config(page_title="Discovery Agent", layout="wide")
    st.title("Autonomous Scientific Discovery Agent")
    st.caption("Browse discovery runs and review their status and summary metrics.")

    runs = discover_runs()

    if not runs:
        st.info(
            "No run artifacts were found yet. Launch the discovery workflow to populate the runs/ directory."
        )
        with st.expander("Expected run layout"):
            st.code(
                """
                runs/
                ├── 2024-01-01T12-00-00/
                │   ├── state.json
                │   ├── trace.json
                │   ├── critic_report.json
                │   ├── costs.jsonl
                │   └── attempts.json
                """.strip()
            )
        return

    run_df = pd.DataFrame(
        [
            {
                "Run": run["thread_id"],
                "Status": run["status"],
                "Attempts": len(run.get("attempts", [])),
                "Cost ($)": round(float(run.get("cost_total_usd", 0.0)), 3),
                "Updated": pd.to_datetime(run["updated_at"], unit="s").strftime("%Y-%m-%d %H:%M:%S"),
            }
            for run in runs
        ]
    )

    left, middle, right, end = st.columns(4)
    left.metric("Runs", len(runs))
    middle.metric("Successful", sum(1 for r in runs if r.get("status") == "succeeded"))
    right.metric("Running", sum(1 for r in runs if r.get("status") == "running"))
    end.metric("Total spend", f"${sum(float(r.get('cost_total_usd', 0.0)) for r in runs):.3f}")

    st.subheader("Run overview")
    st.dataframe(run_df, use_container_width=True, hide_index=True)

    selected_id = st.sidebar.selectbox("Select run", [run["thread_id"] for run in runs])
    selected_run = next(run for run in runs if run["thread_id"] == selected_id)
    st.sidebar.write(_run_summary(selected_run))

    tabs = st.tabs(["Trace", "Citations", "Attempts", "Critic", "Costs"])

    with tabs[0]:
        trace = selected_run.get("trace", [])
        if not trace:
            st.info("No trace data is available for this run yet.")
        else:
            st.dataframe(_trace_dataframe(trace), use_container_width=True, hide_index=True)
            with st.expander("Raw trace data"):
                st.json(trace)

    with tabs[1]:
        literature = selected_run.get("literature", [])
        if not literature:
            st.info("No literature citations were recorded for this run yet.")
        else:
            st.dataframe(pd.json_normalize(literature), use_container_width=True)

    with tabs[2]:
        attempts = selected_run.get("attempts", [])
        if not attempts:
            st.info("No code attempts are available for this run yet.")
        else:
            st.dataframe(pd.json_normalize(attempts), use_container_width=True)

    with tabs[3]:
        critic_report = selected_run.get("critic_report", {})
        if not critic_report:
            st.info("No critic report has been written for this run yet.")
        else:
            st.json(critic_report)

    with tabs[4]:
        cost_rows = selected_run.get("costs", [])
        if not cost_rows:
            st.info("No cost records are available for this run yet.")
        else:
            st.dataframe(pd.json_normalize(cost_rows), use_container_width=True)


if __name__ == "__main__":
    main()
