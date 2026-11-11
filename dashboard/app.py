"""Streamlit research workspace for the discovery agent.

Every view reads persisted artifacts (runs/<id>/state.json, costs.jsonl, data/papers.json); runs are
launched as detached `discovery-agent run` processes so they survive dashboard restarts.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import os
import subprocess
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from discovery_agent.config import settings  # noqa: E402
from discovery_agent.ingestion.catalog import load_catalog  # noqa: E402

DEFAULT_RUNS_DIR = Path(settings.runs_dir)
STAGES = [
    ("literature_reviewer", "Literature"),
    ("hypothesis_generator", "Hypothesis"),
    ("code_executor", "Execution"),
    ("debugger", "Debugging"),
    ("critic", "Statistics"),
    ("refine", "Refinement"),
    ("synthesizer", "Synthesis"),
]
PAGES = [
    "Overview",
    "New discovery",
    "Live run",
    "Literature",
    "Hypotheses",
    "Experiments",
    "Reports",
    "Settings",
]
STATUS_ICON = {"succeeded": "🟢", "running": "🔵", "failed": "🔴", "ready": "⚪", "unknown": "⚪"}
CSS = """<style>
.block-container{padding-top:2rem;max-width:1200px}
h1,h2,h3{letter-spacing:-.01em;font-weight:600}
[data-testid="stMetricValue"]{font-size:1.6rem}
.stage{display:inline-block;padding:.25rem .7rem;margin:0 .3rem .3rem 0;border-radius:999px;
 border:1px solid rgba(128,128,128,.35);font-size:.82rem;opacity:.55}
.stage.done{opacity:1;border-color:#2e9e6a}.stage.now{opacity:1;border-color:#3b82f6;font-weight:600}
</style>"""


# ----------------------------------------------------------------------------- data access
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
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return 0.0, []
    for line in lines:
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
    return total, records


def discover_runs(base_dir: str | Path = DEFAULT_RUNS_DIR) -> list[dict[str, Any]]:
    """Return discovered runs under the runs directory, newest first."""
    root = Path(base_dir)
    if not root.exists():
        return []
    runs: list[dict[str, Any]] = []
    for run_dir in root.iterdir():
        if not run_dir.is_dir():
            continue
        try:
            updated_at = max([run_dir.stat().st_mtime, *(a.stat().st_mtime for a in run_dir.iterdir())])
        except FileNotFoundError:
            continue
        state = _read_json(run_dir / "state.json")
        state = state if isinstance(state, dict) else {}
        cost_total, cost_rows = _read_costs(run_dir / "costs.jsonl")
        status = state.get("status", "unknown")
        if status == "unknown":
            status = "running" if (run_dir / "run.lock").exists() else "ready"
        runs.append(
            {
                "thread_id": run_dir.name,
                "path": str(run_dir),
                "status": status,
                "updated_at": updated_at,
                "topic": state.get("topic", ""),
                "error": state.get("error", ""),
                "trace": state.get("trace") or [],
                "attempts": state.get("attempts") or [],
                "critic_report": state.get("critic_report") or {},
                "hypothesis": state.get("hypothesis") or {},
                "hypotheses": state.get("hypotheses") or [],
                "literature": state.get("literature") or [],
                "literature_summary": state.get("literature_summary", ""),
                "costs": cost_rows,
                "cost_total_usd": cost_total,
            }
        )
    return sorted(runs, key=lambda run: run["updated_at"], reverse=True)


def _serialize_run(run: dict[str, Any]) -> str:
    return json.dumps(run, indent=2, default=str)


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


def _citation_dataframe(hypothesis: Any, literature: list[Any]) -> pd.DataFrame:
    citations = hypothesis.get("citations", []) if isinstance(hypothesis, dict) else []
    sources = {item["paper_id"]: item for item in literature if isinstance(item, dict) and "paper_id" in item}
    rows = []
    for citation in citations:
        if not isinstance(citation, dict):
            continue
        paper_id = citation.get("paper_id", "")
        metadata = sources.get(paper_id, {}).get("metadata") or {}
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
    rows = [
        {"Iteration": a.get("iteration", i), "Reward": a["reward"]}
        for i, a in enumerate(attempts, start=1)
        if isinstance(a, dict)
        and isinstance(a.get("reward"), int | float)
        and not isinstance(a["reward"], bool)
    ]
    return pd.DataFrame(rows, columns=["Iteration", "Reward"])


def _attempt_diffs(attempts: list[Any]) -> list[dict[str, Any]]:
    diffs, previous = [], None
    for index, attempt in enumerate(attempts, start=1):
        if not isinstance(attempt, dict) or not isinstance(attempt.get("code"), str):
            continue
        current = (attempt.get("iteration", index), attempt["code"])
        if previous is not None:
            diff = "\n".join(
                difflib.unified_diff(
                    previous[1].splitlines(),
                    current[1].splitlines(),
                    fromfile=f"attempt-{previous[0]}",
                    tofile=f"attempt-{current[0]}",
                    lineterm="",
                )
            )
            diffs.append({"from": previous[0], "to": current[0], "diff": diff})
        previous = current
    return diffs


def _stage_index(trace: list[Any]) -> tuple[set[str], str | None]:
    done = {e.get("node") for e in trace if isinstance(e, dict)}
    last = next((e.get("node") for e in reversed(trace) if isinstance(e, dict)), None)
    return done, last


def _stage_class(node: str, done: set[str], last: str | None, status: str) -> str:
    if node == last and status == "running":
        return "now"
    return "done" if node in done else ""


def launch_run(topic: str, ingest: int, peer_reviewed_only: bool, model: str) -> str:
    """Start `discovery-agent run` as a detached process logging to runs/<id>/run.log."""
    run_id = f"{datetime.now(UTC):%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
    run_dir = DEFAULT_RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "discovery_agent.cli", "run", "--topic", topic, "--thread-id", run_id]
    if ingest:
        cmd += ["--ingest", str(ingest)]
    if peer_reviewed_only:
        cmd.append("--peer-reviewed-only")
    _spawn(cmd, run_dir, {"LLM_MODEL": model} if model else {})
    return run_id


def _spawn(cmd: list[str], run_dir: Path, extra_env: dict[str, str]) -> None:
    env = {**os.environ, **extra_env, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}
    kwargs: dict[str, Any] = (
        {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS}
        if os.name == "nt"
        else {"start_new_session": True}
    )
    with (run_dir / "run.log").open("ab") as log:
        subprocess.Popen(
            cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env, **kwargs
        )


@st.cache_data(ttl=60, show_spinner=False)
def _service_health() -> dict[str, str]:
    health: dict[str, str] = {}
    try:
        from pymilvus import MilvusClient

        client = MilvusClient(uri=settings.milvus_uri, timeout=3)
        health["Milvus"] = f"online · {len(client.list_collections())} collections"
        client.close()
    except Exception as exc:  # noqa: BLE001 - surfaced to the user
        health["Milvus"] = f"offline · {type(exc).__name__}"
    try:
        import docker

        client = docker.from_env(timeout=3)
        client.images.get(settings.sandbox_image)
        health["Docker sandbox"] = f"ready · {settings.sandbox_image}"
        client.close()
    except Exception as exc:  # noqa: BLE001
        health["Docker sandbox"] = f"unavailable · {type(exc).__name__}"
    provider = settings.llm_model.split(":", 1)[0]
    key = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}.get(provider)
    configured = key is None or bool(os.environ.get(key) or _dotenv_has(key))
    health["LLM"] = f"{settings.llm_model} · {'key configured' if configured else 'API key missing'}"
    return health


def _dotenv_has(key: str) -> bool:
    try:
        return any(
            line.split("=", 1)[0].strip() == key and line.split("=", 1)[1].strip()
            for line in Path(".env").read_text(encoding="utf-8").splitlines()
            if "=" in line
        )
    except OSError:
        return False


# ----------------------------------------------------------------------------- pages
def _pick_run(runs: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not runs:
        st.info("No research runs yet. Start one from **New discovery**.")
        return None
    ids = [r["thread_id"] for r in runs]
    current = st.session_state.get("run_id")
    choice = st.sidebar.selectbox(
        "Run",
        ids,
        index=ids.index(current) if current in ids else 0,
        format_func=lambda i: (
            f"{STATUS_ICON.get(next(r for r in runs if r['thread_id'] == i)['status'], '⚪')} {i}"
        ),
    )
    st.session_state.run_id = choice
    run = next(r for r in runs if r["thread_id"] == choice)
    st.sidebar.download_button(
        "Export run JSON", _serialize_run(run), f"{choice}.json", "application/json", width="stretch"
    )
    return run


def page_overview(runs: list[dict[str, Any]]) -> None:
    st.title("Research workspace")
    st.caption("Autonomous multimodal scientific discovery — literature → hypothesis → experiment → paper.")
    counts = {s: sum(r["status"] == s for r in runs) for s in ("running", "succeeded", "failed")}
    papers = load_catalog()
    cols = st.columns(5)
    cols[0].metric("Runs", len(runs))
    cols[1].metric("Active", counts["running"])
    cols[2].metric("Discoveries", counts["succeeded"])
    cols[3].metric("Papers indexed", len(papers))
    cols[4].metric("LLM spend", f"${sum(r['cost_total_usd'] for r in runs):.2f}")

    left, right = st.columns([2, 1], gap="large")
    with left:
        st.subheader("Recent runs")
        if runs:
            st.dataframe(
                pd.DataFrame(
                    {
                        "Status": f"{STATUS_ICON.get(r['status'], '⚪')} {r['status']}",
                        "Run": r["thread_id"],
                        "Topic": r["topic"],
                        "Attempts": len(r["attempts"]),
                        "Verdict": (r["critic_report"] or {}).get("verdict", "—"),
                        "Cost": f"${r['cost_total_usd']:.3f}",
                    }
                    for r in runs[:10]
                ),
                hide_index=True,
                width="stretch",
            )
        else:
            st.info("No runs yet.")
        if st.button("Start a new discovery", type="primary"):
            st.session_state.goto = "New discovery"
            st.rerun()
    with right:
        st.subheader("System health")
        for name, status in _service_health().items():
            ok = any(w in status for w in ("online", "ready", "configured"))
            st.markdown(
                f"{'🟢' if ok else '🔴'} **{name}**  \n<small>{status}</small>", unsafe_allow_html=True
            )
        if st.button("Re-check"):
            _service_health.clear()
            st.rerun()


def page_new_discovery() -> None:
    st.title("New discovery")
    st.caption("The run executes in a background process; follow it under **Live run**.")
    with st.form("new-run"):
        topic = st.text_area(
            "Research topic or question",
            placeholder="e.g. Does label noise reduce the calibration of gradient-boosted classifiers?",
        )
        c1, c2 = st.columns(2)
        ingest = c1.number_input("Fetch new papers per source (0 = use existing index)", 0, 100, 10)
        model = c2.text_input("LLM model", settings.llm_model, help="provider:model, e.g. ollama:llama3.1")
        peer = st.checkbox("Peer-reviewed literature only (PubMed)")
        submitted = st.form_submit_button("Start discovery", type="primary")
    if submitted:
        topic = topic.strip()
        if not 10 <= len(topic) <= 500:
            st.error("Topic must be between 10 and 500 characters.")
        elif ":" not in model:
            st.error("Model must look like provider:model.")
        else:
            st.session_state.run_id = launch_run(topic, int(ingest), peer, model.strip())
            st.session_state.goto = "Live run"
            st.rerun()


def page_live(runs: list[dict[str, Any]]) -> None:
    run = _pick_run(runs)
    if run is None:
        return
    live = st.sidebar.toggle("Auto-refresh (5 s)", value=run["status"] == "running")

    @st.fragment(run_every=5 if live else None)
    def body() -> None:
        current = next((r for r in discover_runs() if r["thread_id"] == run["thread_id"]), run)
        st.title(current["topic"] or current["thread_id"])
        st.caption(
            f"{STATUS_ICON.get(current['status'], '⚪')} {current['status']} · run `{current['thread_id']}`"
        )
        done, last = _stage_index(current["trace"])
        st.markdown(
            "".join(
                f'<span class="stage {_stage_class(node, done, last, current["status"])}">{label}</span>'
                for node, label in STAGES
            ),
            unsafe_allow_html=True,
        )
        if current["error"]:
            st.error(current["error"])
        c = st.columns(4)
        c[0].metric("Papers", len({x.get("paper_id") for x in current["literature"] if isinstance(x, dict)}))
        c[1].metric("Code attempts", len(current["attempts"]))
        c[2].metric("Hypothesis versions", len(current["hypotheses"]))
        c[3].metric("Cost", f"${current['cost_total_usd']:.3f}")

        st.subheader("Agent activity")
        trace = _trace_dataframe(current["trace"])
        st.dataframe(trace, hide_index=True, width="stretch")
        log = Path(current["path"]) / "run.log"
        if log.is_file():
            with st.expander("Process log"):
                st.code(log.read_text(encoding="utf-8", errors="replace")[-6000:] or "(empty)")
        if current["costs"]:
            with st.expander("LLM calls"):
                st.dataframe(pd.DataFrame(current["costs"]), hide_index=True, width="stretch")

    body()
    if run["status"] == "failed" and st.sidebar.button("Resume from checkpoint", width="stretch"):
        cmd = [
            sys.executable,
            "-m",
            "discovery_agent.cli",
            "run",
            "--topic",
            run["topic"],
            "--thread-id",
            run["thread_id"],
        ]
        _spawn(cmd, Path(run["path"]), {})
        st.toast("Resuming run from its last checkpoint…")


def page_literature() -> None:
    st.title("Literature library")
    catalog = load_catalog()
    if not catalog:
        st.info(
            "No papers ingested yet. Start a discovery with paper fetching, or run `discovery-agent ingest`."
        )
        return
    c1, c2, c3 = st.columns([3, 1, 1])
    query = c1.text_input("Search", placeholder="Title, author, abstract…").lower()
    source = c2.selectbox("Source", ["All", "arxiv", "pubmed"])
    peer = c3.selectbox("Review", ["All", "Peer-reviewed", "Preprint"])
    papers = [
        p
        for p in catalog.values()
        if (source == "All" or p.get("source") == source)
        and (peer == "All" or p.get("peer_reviewed") == (peer == "Peer-reviewed"))
        and (
            not query
            or query in " ".join([p.get("title", ""), p.get("abstract", ""), *p.get("authors", [])]).lower()
        )
    ]
    papers.sort(key=lambda p: p.get("published") or "", reverse=True)
    st.caption(f"{len(papers)} of {len(catalog)} papers")
    for p in papers[:100]:
        chunks = p.get("chunks", {})
        with st.expander(f"{p.get('title', p['id'])}  ·  {(p.get('published') or '')[:10]}"):
            st.markdown(
                f"**{', '.join(p.get('authors', [])[:6])}**  \n`{p['id']}` · {p.get('source')} · "
                f"{'peer-reviewed' if p.get('peer_reviewed') else 'preprint'} · indexed chunks: "
                + (", ".join(f"{k} {v}" for k, v in chunks.items()) or "none")
            )
            if p.get("url"):
                st.markdown(f"[Open paper]({p['url']})")
            st.write(p.get("abstract", ""))
            fig_dir = Path(settings.ingest_figure_dir) / hashlib.sha256(p["id"].encode()).hexdigest()[:16]
            images = sorted(fig_dir.glob("*.png"))[:4] if fig_dir.is_dir() else []
            if images:
                st.image([str(i) for i in images], width=220)


def page_hypotheses(runs: list[dict[str, Any]]) -> None:
    run = _pick_run(runs)
    if run is None:
        return
    st.title("Hypotheses")
    h = run["hypothesis"]
    if not h:
        st.info("This run has not produced a hypothesis yet.")
        return
    st.subheader(h.get("statement", ""))
    st.markdown(f"**Rationale.** {h.get('rationale', '')}")
    st.markdown(f"**Methodology.** {h.get('test_plan', '')}")
    st.markdown(f"**Expected outcome.** {h.get('expected_outcome', '')}")
    st.subheader("Supporting literature")
    st.dataframe(
        _citation_dataframe(h, run["literature"]),
        hide_index=True,
        width="stretch",
        column_config={"URL": st.column_config.LinkColumn()},
    )
    report = run["critic_report"]
    if report:
        st.subheader("Critic feedback")
        st.markdown(f"Verdict: **{report.get('verdict')}** — {report.get('summary', '')}")
        for issue in report.get("issues", []):
            st.markdown(f"- {issue}")
    if len(run["hypotheses"]) > 1:
        st.subheader("Refinement history")
        for i, old in enumerate(run["hypotheses"], start=1):
            st.markdown(f"**v{i}.** {old.get('statement', '')}")
    if run["literature_summary"]:
        with st.expander("Literature review"):
            st.markdown(run["literature_summary"])


def page_experiments(runs: list[dict[str, Any]]) -> None:
    run = _pick_run(runs)
    if run is None:
        return
    st.title("Experiments")
    attempts = run["attempts"]
    if not attempts:
        st.info("No experiment attempts recorded for this run yet.")
        return
    rewards = _attempt_rewards(attempts)
    if len(rewards) > 1:
        st.line_chart(rewards, x="Iteration", y="Reward", height=180)
    for a in reversed(attempts):
        r = a.get("result", {})
        ok = r.get("exit_code") == 0 and not r.get("timed_out")
        title = (
            f"{'✅' if ok else '❌'} Attempt {a.get('iteration')} · exit {r.get('exit_code')} · "
            f"{r.get('duration_s', 0):.1f}s · reward {a.get('reward', 0):.2f}"
            + (f" · strategy {a['strategy']}" if a.get("strategy") else "")
        )
        with st.expander(title, expanded=a is attempts[-1]):
            tabs = st.tabs(["Metrics", "Code", "stdout", "stderr", "Outputs"])
            tabs[0].json(r.get("metrics") or {})
            tabs[1].code(a.get("code", ""), language="python")
            tabs[2].code(r.get("stdout", "")[-5000:] or "(empty)")
            tabs[3].code(r.get("stderr", "")[-5000:] or "(empty)")
            with tabs[4]:
                for name, path in (r.get("artifacts") or {}).items():
                    p = Path(path)
                    if p.suffix.lower() in {".png", ".jpg", ".jpeg"} and p.is_file():
                        st.image(str(p), caption=name)
                    else:
                        st.text(name)
    if run["critic_report"]:
        st.subheader("Statistical validation")
        cr = run["critic_report"]
        st.dataframe(
            pd.DataFrame(
                {
                    "Test": name,
                    "Adjusted p": p,
                    "Effect size": (cr.get("effect_sizes") or {}).get(name),
                    "95% CI": (cr.get("confidence_intervals") or {}).get(name),
                }
                for name, p in (cr.get("p_values") or {}).items()
            ),
            hide_index=True,
            width="stretch",
        )
    for d in _attempt_diffs(attempts):
        with st.expander(f"Debug diff {d['from']} → {d['to']}"):
            st.code(d["diff"], language="diff")


def page_reports(runs: list[dict[str, Any]]) -> None:
    run = _pick_run(runs)
    if run is None:
        return
    st.title("Research report")
    run_dir = Path(run["path"])
    files = {
        name: run_dir / name for name in ("draft.md", "paper.tex", "abstract.tex", "refs.bib", "paper.pdf")
    }
    if not files["draft.md"].is_file():
        st.info("Reports are generated when a run reaches synthesis (statistically significant result).")
    else:
        st.caption(f"Generated {datetime.fromtimestamp(files['draft.md'].stat().st_mtime):%Y-%m-%d %H:%M}")
        cols = st.columns(len(files))
        for col, (name, path) in zip(cols, files.items(), strict=True):
            if path.is_file():
                col.download_button(name, path.read_bytes(), f"{run['thread_id']}-{name}", width="stretch")
        md, tex, bib = st.tabs(["Markdown", "LaTeX", "Bibliography"])
        md.markdown(files["draft.md"].read_text(encoding="utf-8"))
        tex.code(
            files["paper.tex"].read_text(encoding="utf-8") if files["paper.tex"].is_file() else "", "latex"
        )
        bib.code(
            files["refs.bib"].read_text(encoding="utf-8") if files["refs.bib"].is_file() else "", "bibtex"
        )

    st.subheader("Evaluation")
    if st.button("Grade against literature holdout"):
        try:
            from discovery_agent.evaluation.grader import grade_run

            st.json(grade_run(run_dir, "data/holdout.jsonl"))
        except Exception as exc:  # noqa: BLE001
            st.error(f"Grading failed: {exc}")


def page_settings() -> None:
    st.title("Settings")
    st.caption(
        "Effective configuration loaded from environment variables and `.env`. Edit `.env` and restart."
    )
    rows = []
    for name, value in settings.model_dump().items():
        secret = any(k in name for k in ("key", "token", "secret", "password"))
        rows.append(
            {"Setting": name.upper(), "Value": ("••••" if value else "(unset)") if secret else str(value)}
        )
    for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        rows.append(
            {"Setting": key, "Value": "••••" if os.environ.get(key) or _dotenv_has(key) else "(unset)"}
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def main() -> None:
    st.set_page_config(page_title="Discovery Agent", page_icon="🔬", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.sidebar.markdown("### 🔬 Discovery Agent")
    if "goto" in st.session_state:
        st.session_state.page = st.session_state.pop("goto")
    page = st.sidebar.radio("Navigate", PAGES, key="page", label_visibility="collapsed")
    st.sidebar.divider()
    runs = discover_runs()
    {
        "Overview": lambda: page_overview(runs),
        "New discovery": page_new_discovery,
        "Live run": lambda: page_live(runs),
        "Literature": page_literature,
        "Hypotheses": lambda: page_hypotheses(runs),
        "Experiments": lambda: page_experiments(runs),
        "Reports": lambda: page_reports(runs),
        "Settings": page_settings,
    }[page]()


if __name__ == "__main__":
    main()
