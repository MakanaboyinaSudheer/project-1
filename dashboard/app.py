"""Streamlit supervisor dashboard for the discovery-agent project.

The dashboard is intentionally resilient to a missing runs/ directory so it can be
launched during early project stages and still provide a useful empty-state view.
"""

from __future__ import annotations

import difflib
import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

# Ensure the src directory is accessible for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

# Import graph execution function
try:
    from discovery_agent.graph.workflow import run_discovery

    WORKFLOW_AVAILABLE = True
except ImportError:
    WORKFLOW_AVAILABLE = False

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
    """Return discovered runs under the project runs directory."""
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
                "hypothesis": state.get("hypothesis", {}) if isinstance(state, dict) else {},
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


def _serialize_run(run: dict[str, Any]) -> str:
    return json.dumps(run, indent=2, default=str)


def _citation_dataframe(hypothesis: Any, literature: list[Any]) -> pd.DataFrame:
    citations = hypothesis.get("citations", []) if isinstance(hypothesis, dict) else []
    sources: dict[str, dict[str, Any]] = {}
    for item in literature:
        if isinstance(item, dict) and isinstance(item.get("paper_id"), str):
            sources.setdefault(item["paper_id"], item)

    rows = []
    for citation in citations:
        if not isinstance(citation, dict):
            continue
        paper_id = citation.get("paper_id", "")
        source = sources.get(paper_id, {})
        metadata = source.get("metadata", {}) if isinstance(source, dict) else {}
        metadata = metadata if isinstance(metadata, dict) else {}
        rows.append(
            {
                "Title": citation.get("title") or metadata.get("title") or paper_id,
                "Paper ID": paper_id,
                "Quote": citation.get("quote", ""),
                "URL": metadata.get("url", ""),
            }
        )
    return pd.DataFrame(rows, columns=["Title", "Paper ID", "Quote", "URL"])


def _attempt_rewards(attempts: list[Any]) -> pd.DataFrame:
    rows = []
    for index, attempt in enumerate(attempts, start=1):
        if not isinstance(attempt, dict):
            continue
        iteration = attempt.get("iteration", index)
        reward = attempt.get("reward")
        if isinstance(reward, int | float) and not isinstance(reward, bool):
            rows.append({"Iteration": iteration, "Reward": reward})
    return pd.DataFrame(rows, columns=["Iteration", "Reward"])


def _attempt_diffs(attempts: list[Any]) -> list[dict[str, Any]]:
    diffs = []
    previous_code: str | None = None
    previous_iteration: Any = None
    for index, attempt in enumerate(attempts, start=1):
        if not isinstance(attempt, dict):
            continue
        code = attempt.get("code")
        if not isinstance(code, str):
            continue
        iteration = attempt.get("iteration", index)
        if previous_code is not None:
            diff = "\n".join(
                difflib.unified_diff(
                    previous_code.splitlines(),
                    code.splitlines(),
                    fromfile=f"attempt-{previous_iteration}",
                    tofile=f"attempt-{iteration}",
                    lineterm="",
                )
            )
            diffs.append({"from": previous_iteration, "to": iteration, "diff": diff})
        previous_code = code
        previous_iteration = iteration
    return diffs


def _render_runs(runs: list[dict[str, Any]]) -> None:
    if not runs:
        st.info(
            "No run artifacts were found yet. Launch the discovery workflow to populate the runs/ directory."
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

    st.sidebar.download_button(
        "Export run JSON",
        data=_serialize_run(selected_run),
        file_name=f"{selected_id}.json",
        mime="application/json",
    )
    draft_path = Path(selected_run["path"]) / "draft.md"
    if draft_path.is_file():
        st.sidebar.download_button(
            "Export paper draft",
            data=draft_path.read_text(encoding="utf-8"),
            file_name=f"{selected_id}-draft.md",
            mime="text/markdown",
        )

    tabs = st.tabs(["Trace", "Citations", "Attempts", "Critic", "Costs"])

    with tabs[0]:
        trace = selected_run.get("trace", [])
        if not trace:
            st.info("No trace data is available for this run yet.")
        else:
            st.dataframe(_trace_dataframe(trace), use_container_width=True, hide_index=True)

    with tabs[1]:
        citation_frame = _citation_dataframe(
            selected_run.get("hypothesis", {}),
            selected_run.get("literature", []),
        )
        if citation_frame.empty:
            st.info("No hypothesis citations were recorded for this run yet.")
        else:
            st.dataframe(
                citation_frame,
                use_container_width=True,
                hide_index=True,
                column_config={"URL": st.column_config.LinkColumn("Paper")},
            )

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


def main() -> None:
    st.set_page_config(page_title="Discovery Agent", page_icon="🔬", layout="wide")
    st.title("🔬 Autonomous Scientific Discovery Agent")
    st.caption("Generate hypotheses and browse discovery runs.")

    # Combined Sidebar: Trigger workflow + Auto-refresh
    with st.sidebar:
        st.header("Launch Discovery")
        topic = st.text_input("Research Topic:", "How do shipping times impact e-commerce sales?")

        if st.button("Run Workflow"):
            if not WORKFLOW_AVAILABLE:
                st.error("⚠️ Could not import your workflow graph function.")
            else:
                with st.status("Executing LangGraph Pipeline...", expanded=True) as status:
                    try:
                        run_discovery(topic=topic, thread_id="dashboard_session_1")
                        status.update(label="Pipeline Complete!", state="complete", expanded=False)
                        st.success("Analysis complete! The run is now available.")
                    except Exception as e:
                        status.update(label="Pipeline Failed", state="error")
                        st.error(f"Execution Error: {str(e)}")

        st.divider()

        refresh_intervals = {"Off": None, "5 seconds": 5, "15 seconds": 15}
        refresh_interval = st.selectbox(
            "Auto-refresh Dashboard",
            options=list(refresh_intervals),
            index=2,
        )
        if st.button("Refresh now"):
            st.rerun()

    @st.fragment(run_every=refresh_intervals[refresh_interval])
    def render_dashboard() -> None:
        _render_runs(discover_runs())

    render_dashboard()


if __name__ == "__main__":
    main()
