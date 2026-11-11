# Final implementation audit

Baseline: `main` @ `9a470aa` (develop is fully merged into it). Work branch: `final/integration`.
Unit baseline before changes: 153 passed / 3 skipped. After: 159 passed / 3 skipped; ruff clean.

## Status matrix

| Module | Before | After |
|---|---|---|
| ingestion (arxiv, pubmed, pdf, figures, pipeline) | Implemented, tested | Unchanged + persistent paper catalog `data/papers.json` (titles, authors, URLs, chunk counts by kind) |
| retrieval (embeddings, Milvus store, MMR + recency) | Implemented, tested | Results enriched with real titles/URLs from the catalog (Milvus schema stores only `paper_id`) |
| literature_reviewer | **Broken**: wrote `literature_context`, wasted an unused sub-query LLM call, no trace | Writes `literature`, fails loudly on an empty index, cost-tracked, traced |
| hypothesis_generator | **Broken**: own `Hypothesis` schema incompatible with executor/critic/synthesizer; mock-ID citation fallback; validation never ran (checked dicts, got models) | Uses canonical `schemas.Hypothesis`; citations must match retrieved `paper_id`s (3 retries then error); titles taken from source; non-verbatim quotes replaced by real source text; refinement history kept |
| code_executor, debugger, bandit, feedback, sandbox runner | Implemented, tested | Unchanged (debugger was never called before; now wired into the graph) |
| critic (deterministic stats, Holm–Bonferroni, raw-data recompute) | Implemented, tested | Unchanged; LLM only appends methodology issues |
| synthesizer / LaTeX / BibTeX / PDF | Implemented, tested | Unchanged |
| orchestration | **Broken**: plain `while` loop, no debugger step, no checkpoints, status in a process-local dict | Single LangGraph `StateGraph` with SQLite checkpointing, debug loop bounded by `MAX_DEBUG_ITERATIONS`, refinement bounded at 3, per-node `runs/<id>/state.json` snapshots, crash recording and resume by `--thread-id` |
| dashboard | **Broken**: imported non-existent `create_graph`, sent wrong state key, fixed thread id `dashboard_session_1`, blocked the UI thread | 8-page workspace; runs launched as detached CLI processes (survive restarts); every view reads persisted artifacts |
| llm factory | **Broken for default model**: sent `temperature` to Claude 5.x models, which reject it | Temperature only sent to non-Anthropic providers |
| memory manager | Implemented, tested, not wired (uses legacy keys) | Unchanged; not needed by the graph |
| evaluation grader | Implemented, tested | Exposed in the Reports page |

## Removed (dead or competing code, recoverable from git history)
- `src/discovery_agent/app.py`: a second Streamlit app on the broken loop.
- `graph/parallel.py`, `graph/state_parallel.py`, `ui/components.py` and their tests: a standalone parallel-angle demo never wired into the workflow; its "Promote" button had no effect.
- `tools/milvus_tools.py`, `utils/*.py`, `test_agent_tool.py`: scripts hard-coded to Docker-internal hosts and a different collection and embedding size (768 vs 384).

## Verification performed here
- `pytest -m "not integration"`: 159 passed, 3 skipped. Graph tests cover success, debug loop, debug budget, refinement budget, crash then resume without re-execution, run isolation, and unsafe IDs. The dashboard test renders every page from a persisted run.
- `streamlit run dashboard/app.py`: server starts, `/_stcore/health` returns `ok`.
- `discovery-agent run` with Milvus down: fails with a clear error, persisted to `state.json` with `status: failed`.

## Blocked in this environment (not verified)
- Docker daemon not running, so no sandbox execution or image build.
- Milvus not running, so no ingestion or retrieval against a real store.
- No `.env` or `ANTHROPIC_API_KEY`, so no live LLM calls and no real end-to-end run.
- Integration tests (`pytest -m integration`) need all of the above.

## Remaining risks
- The dashboard binds to 127.0.0.1 (`.streamlit/config.toml`). Exposing it more widely lets anyone start runs, which spend LLM budget and launch Docker containers, so put authentication in front first.
- Each run is one OS process with no global concurrency limit.
- A `running` snapshot whose process was killed stays `running`; only crashes caught in Python are marked `failed`.
