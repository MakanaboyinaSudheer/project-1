# Execution Plan (20 working days, 3 members)

Module IDs and the **per-person daily commit schedule (20 commits each)** are in
[MODULES.md](MODULES.md). **Every member pushes at least one meaningful commit every day.**
The commit history is the productivity metric for evaluation.

| Week | Days | Theme |
|---|---|---|
| 1 | 1–5 | Foundation & data ingestion |
| 2 | 6–10 | RAG, sandboxing & self-correction → **Mid-Project Review (Day 10)** |
| 3 | 11–15 | Multimodality & statistical rigor |
| 4 | 16–20 | Synthesis, dashboard & final evaluation |

## Milestones

| Day | Milestone | Exit criteria |
|---|---|---|
| 1 | Environment ready | All 3 members pass `discovery-agent check` (Milvus, sandbox image, LLM) |
| 5 | Foundation | Ingestion fills Milvus from ArXiv and PubMed; LangGraph runs end-to-end on stub nodes; sandbox runs generated code |
| **10** | **Mid-Project Review (MVP)** | (1) multi-agent framework plus a live vector DB, (2) the sandbox contains every escape test, (3) a failing script triggers the debugger and is fixed automatically, with rewards logged. Merge `develop` → `main`, tag `v0.1-mvp` |
| 15 | Multimodal + rigor | Figure descriptions retrievable; the critic gates synthesis; a run resumes after a process restart |
| 20 | Final delivery | End-to-end pipeline, critic agent, dashboard, holdout evaluation, architectural report, simulated paper. Tag `v1.0` |

## Week-by-week overview

| Week | Member 1: Data & Retrieval | Member 2: Agents & Orchestration | Member 3: Execution & Platform |
|---|---|---|---|
| 1 (D1–5) | ArXiv + PubMed clients, embeddings, Milvus store, idempotent ingestion | LangGraph skeleton + checkpointer, Literature Reviewer | CI, hardened sandbox runner, Code Executor v0 |
| 2 (D6–10) | Peer-reviewed RAG retriever, MMR/recency, retrieval eval | Grounded Hypothesis Generator, citation validation, wire the real graph | Sandbox escape tests, reward shaping, debugger, bandit |
| 3 (D11–15) | PDF text + figure extraction, LLaVA figure chunks | Statistical Critic, memory/context management | Cost telemetry, stress test, dashboard start |
| 4 (D16–20) | Holdout set, grader, evaluation results | LaTeX synthesis, simulated paper | Dashboard completion, runbook, demo |

**Integration points:** Days 5, 10, 15 and 20. Merge feature branches into `develop`
and run the full pipeline together. Day 20 is a freeze: bug fixes only.

## Daily routine
1. `git checkout develop && git pull`, then rebase your feature branch on it.
2. Post a 3-line stand-up in the team chat (yesterday / today / blockers).
3. Make the day's planned commit from MODULES.md. **Push before the end of the day**, even if it's WIP.
4. Open a PR as soon as a module is reviewable; review teammates' PRs within 24h.

## Risks
| Risk | Mitigation |
|---|---|
| LLM API costs or rate limits | Cheap/local model (`ollama:`) for development; stronger model for final runs; CostTracker |
| PubMed/ArXiv rate limits | `tenacity` backoff, NCBI API key, cache raw responses in `data/raw/` |
| LLaVA too heavy for laptops | Run it on one teammate's machine and point `OLLAMA_BASE_URL` at it, or use a hosted VLM via `VLM_MODEL` |
| Docker Desktop differences (Windows/macOS) | Integration tests in CI on Linux; setup scripts for both |
| Interface drift between members | Changes to `schemas.py`/`state.py` need all 3 reviewers (CODEOWNERS) |
| A member falls behind schedule | Blocked work uses mocks of `schemas.py` types; re-plan at the next integration day |
