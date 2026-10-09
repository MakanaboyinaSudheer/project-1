# Modules, Ownership & 20-Day Commit Schedule

Four work streams, each owning a vertical slice across the 4 team members:

| Stream | Owner | Area |
|---|---|---|
| **M1: Data & Retrieval** | Member 1 (`VARDHAN`) | ArXiv/PubMed ingestion, Milvus, RAG, multimodal PDF/figure parsing, vector index optimization |
| **M2: Core Agents & Orchestration** | Member 2 (`AADIL`) | LangGraph workflow state machine, Literature Reviewer, Hypothesis Generator, memory management & state recovery |
| **M3: Execution & Platform** | Member 3 (`MANSI`) | Docker sandbox, security test suite, Code Executor, self-debug RL loop (bandit & feedback), platform infra & CI |
| **M4: Evaluation, Synthesis & Dashboard** | Member 4 (`SUDHEER`) | Statistical Critic, Jinja2 LaTeX synthesizer, Holdout paper evaluation & LLM-judge grader, Cost telemetry, Streamlit dashboard |

**Timeline:** 20 working days = 4 weeks × 5 days.
Week 1 = Days 1–5 · Week 2 = Days 6–10 (**Mid-Project Review on Day 10**) · Week 3 = Days 11–15 · Week 4 = Days 16–20.

**Rule:** one planned commit per person per day, so **20 commits each** (80 for the team).
Each row below is the minimum you push that day. Extra commits are welcome.
Use the module ID in branch names (`feature/M1.2-pubmed-client`) and commit scopes (`feat(M1.2): ...`).
Interfaces are already stubbed in `src/`. Implement each one behind its existing signature.

---

## Daily commit schedule

### Member 1: Data & Retrieval (VARDHAN)

| Day | Module | Commit |
|---|---|---|
| 1 | M1.1 | `feat(M1.1): fetch ArXiv preprints and map to Paper schema` |
| 2 | M1.1 | `test(M1.1): mocked ArXiv tests with retry/backoff via tenacity` |
| 3 | M1.2 | `feat(M1.2): PubMed esearch/efetch client with XML parsing` |
| 4 | M1.3 | `feat(M1.3): sentence-transformer embeddings and Milvus VectorStore` |
| 5 | M1.4 | `feat(M1.4): idempotent ingestion pipeline and ingest CLI` |
| 6 | M1.5 | `feat(M1.5): RAG retriever with peer-reviewed filter` |
| 7 | M1.5 | `feat(M1.5): MMR diversification and recency boost` |
| 8 | M1.4 | `test(M1.4): ingest+retrieve integration tests; bulk-ingest seed topics` |
| 9 | M1.5 | `test(M1.5): retrieval eval on 20 queries (recall@k), tune params` |
| 10 | MVP | `docs(M1): vector DB milestone evidence and scheduling guide` |
| 11 | M1.6 | `feat(M1.6): PDF download with local cache` |
| 12 | M1.6 | `feat(M1.6): body-text chunking with overlap` |
| 13 | M1.6 | `feat(M1.6): figure and caption extraction with PyMuPDF` |
| 14 | M1.7 | `feat(M1.7): LLaVA figure description to figure chunks` |
| 15 | M1.7 | `feat(M1.7): ingest figure chunks; multimodal retrieval test` |
| 16 | M1.8 | `feat(M1.8): integration of multimodal retrieval in vector store` |
| 17 | M1.8 | `test(M1.8): multimodal chunk indexing & query performance benchmarks` |
| 18 | M1.8 | `docs(M1.8): data pipeline performance report and chunk schema docs` |
| 19 | Report | `docs: architecture report - data pipeline and retrieval architecture` |
| 20 | Release | `fix: final data pipeline E2E fixes for v1.0` |

### Member 2: Core Agents & Orchestration (AADIL)

| Day | Module | Commit |
|---|---|---|
| 1 | M2.1 | `feat(M2.1): LangGraph skeleton with stub nodes and routing` |
| 2 | M2.1 | `feat(M2.1): SQLite checkpointer, run_discovery and run CLI` |
| 3 | M2.2 | `feat(M2.2): literature reviewer query expansion (mocked retrieval)` |
| 4 | M2.2 | `feat(M2.2): literature summary prompt and prompts library` |
| 5 | M2.1 | `test(M2.1): end-to-end stub run through every node` |
| 6 | M2.3 | `feat(M2.3): hypothesis generator with structured output` |
| 7 | M2.3 | `feat(M2.3): citation validation against retrieved literature with retries` |
| 8 | M2.3 | `test(M2.3): reject hallucinated and non-peer-reviewed citations` |
| 9 | M2.1 | `feat(M2.1): wire real agent nodes; E2E topic to code run` |
| 10 | MVP | `docs(M2): mid-project demo script; release v0.1-mvp` |
| 11 | M2.4 | `feat(M2.4): token-budgeted build_context in memory manager` |
| 12 | M2.4 | `feat(M2.4): rolling compression and resume-after-kill test` |
| 13 | M2.4 | `feat(M2.4): long-run agent state persistence and recovery` |
| 14 | M2.5 | `feat(M2.5): hypothesis refinement loop incorporating critic feedback` |
| 15 | M2.5 | `test(M2.5): multi-turn refinement stress test (up to 3 iterations)` |
| 16 | M2.5 | `feat(M2.5): agent routing policy optimization and safety fallbacks` |
| 17 | M2.6 | `test(M2.6): multi-agent graph state transition & deadlock verification` |
| 18 | M2.6 | `docs(M2.6): state machine specification and graph visualizer` |
| 19 | Report | `docs: architecture report - multi-agent graph & memory management` |
| 20 | Release | `chore: tag v1.0 agent core & final workflow release` |

### Member 3: Execution & Platform (MANSI)

| Day | Module | Commit |
|---|---|---|
| 1 | M3.1 | `chore(M3.1): CI integration-test job and compose health checks` |
| 2 | M3.2 | `feat(M3.2): sandbox runner executes code and captures output` |
| 3 | M3.2 | `feat(M3.2): harden container (no network, read-only, caps, non-root)` |
| 4 | M3.2 | `feat(M3.2): wall-clock timeout kill and /output metrics collection` |
| 5 | M3.3 | `feat(M3.3): code executor agent v0 (generate, extract, run)` |
| 6 | M3.4 | `test(M3.4): sandbox escape tests - network and filesystem` |
| 7 | M3.4 | `test(M3.4): fork bomb, memory bomb, infinite loop, setuid` |
| 8 | M3.5 | `feat(M3.5): reward shaping and error classification` |
| 9 | M3.5 | `feat(M3.5): debugger node and route_after_execution` |
| 10 | MVP | `feat(M3.5): bandit over repair strategies; self-debug demo` |
| 11 | M3.5 | `test(M3.5): RL bandit convergence tests across synthetic bug suite` |
| 12 | M3.2 | `test(M3.2): 50-attempt long-run stress test; sandbox image tuning` |
| 13 | M3.3 | `feat(M3.3): code executor retry strategy and dependency isolation` |
| 14 | M3.5 | `feat(M3.5): bandit persistence (runs/bandit.json) and online updates` |
| 15 | M3.2 | `feat(M3.2): sandbox resource monitoring and memory profiling` |
| 16 | M3.6 | `docs(M3.6): sandbox security audit report and escape benchmarks` |
| 17 | M3.6 | `docs(M3.6): platform runbook and clean-install verification` |
| 18 | M3.6 | `test(M3.6): full platform environment smoke test` |
| 19 | Report | `docs: architecture report - sandbox, execution engine & RL loop` |
| 20 | Release | `chore: platform final release checks & CI green confirmation` |

### Member 4: Evaluation, Synthesis & Dashboard (SUDHEER)

| Day | Module | Commit |
|---|---|---|
| 1 | M4.1 | `feat(M4.1): telemetry CostTracker callback stub & cost schema` |
| 2 | M4.1 | `feat(M4.1): model pricing table and costs.jsonl output` |
| 3 | M4.2 | `feat(M4.2): Streamlit app skeleton and run list view` |
| 4 | M4.2 | `feat(M4.2): dashboard logic-tree (trace) view` |
| 5 | M4.3 | `feat(M4.3): critic deterministic stage - parse and validate metrics` |
| 6 | M4.3 | `feat(M4.3): Holm-Bonferroni, small-n and missing-test checks + tests` |
| 7 | M4.3 | `feat(M4.3): LLM methodology review and route_after_critic` |
| 8 | M4.2 | `feat(M4.2): dashboard code attempts with diffs and reward curve` |
| 9 | M4.2 | `feat(M4.2): citations, critic report and cost views` |
| 10 | MVP | `docs(M4): dashboard MVP walkthrough and cost analytics demo` |
| 11 | M4.4 | `feat(M4.4): Jinja2 LaTeX templates for abstract, paper and bib` |
| 12 | M4.4 | `feat(M4.4): synthesizer node with LaTeX escaping` |
| 13 | M4.4 | `feat(M4.4): compile simulated paper from best run` |
| 14 | M4.5 | `feat(M4.5): holdout paper set and ingestion exclusion` |
| 15 | M4.5 | `feat(M4.5): grader metrics (similarity, citation validity, success rate)` |
| 16 | M4.5 | `feat(M4.5): LLM-judge scoring; run holdout evaluation` |
| 17 | M4.2 | `feat(M4.2): live auto-refresh and export features in dashboard` |
| 18 | M4.5 | `docs(M4.5): evaluation methodology and holdout benchmark results` |
| 19 | Report | `docs: architecture report - critic, synthesis, dashboard & eval` |
| 20 | Release | `chore: final dashboard polish, paper templates tag v1.0` |

---

## Module specifications

### M1: Data & Retrieval (Member 1: VARDHAN)

**M1.1 ArXiv client (Days 1–2)**
- `ingestion/arxiv_client.py` → `fetch_arxiv(query, categories, max_results) -> list[Paper]`
- Newest first; `Paper(id="arxiv:<id>", source="arxiv", peer_reviewed=False, pdf_url=...)`; 3 s between calls; `tenacity` retries.
- Done when: mocked unit test passes and an integration test fetches ≥10 papers.

**M1.2 PubMed client (Day 3)**
- `ingestion/pubmed_client.py` → `fetch_pubmed(query, max_results) -> list[Paper]`
- `Entrez.esearch` (sort=pub_date) then `efetch` (XML); `Entrez.email`/`api_key` from settings; `peer_reviewed=True` for "Journal Article".

**M1.3 Embeddings + Milvus (Day 4)**
- `retrieval/embeddings.py`, `retrieval/vector_store.py`. Load the model once and batch-encode normalized vectors.
- `ensure_collection()` creates the schema from ARCHITECTURE §4.1 with a COSINE index; `search()` supports `filter_expr`.

**M1.4 Ingestion pipeline (Days 5, 8)**
- `ingestion/pipeline.py` → `run_ingestion(query, max_results)`; CLI `discovery-agent ingest`.
- Dedupe by `Paper.id`. Done when a second run on the same query adds 0 chunks. Document scheduling (Task Scheduler / cron).

**M1.5 RAG retriever (Days 6–7, 9)**
- `retrieval/retriever.py` → `retrieve(query, k, peer_reviewed_only)`, with MMR and a recency boost.
- Done when a test proves `peer_reviewed_only=True` never returns ArXiv chunks.

**M1.6 PDF parser (Days 11–13)**
- `ingestion/pdf_parser.py`: download, text chunks (≈1000 chars with overlap, `kind="body"`), figures plus captions.

**M1.7 Figure parser, multimodal (Days 14–15)**
- `ingestion/figure_parser.py` → `describe_figure(...) -> Chunk(kind="figure")`; LLaVA describes chart type, axes, units, trends and key values.
- Done when figures from 5 sample papers are retrievable by semantic query.

**M1.8 Vector Store Index & Benchmarking (Days 16–18)**
- Multimodal chunk indexing optimization, retrieval benchmark tests, collection index tuning & data pipeline docs.

### M2: Core Agents & Orchestration (Member 2: AADIL)

**M2.1 LangGraph workflow (Days 1–2, 5, 9)**
- `graph/workflow.py`: `build_graph()` with all nodes and conditional edges, compiled with `SqliteSaver`.
- `run_discovery(topic, thread_id)` streams the graph and saves to `runs/<thread_id>/`. Stub nodes first, so the graph runs before the real agents exist.

**M2.2 Literature Reviewer (Days 3–4)**
- `agents/literature_reviewer.py`: 2–4 sub-queries, retrieve, summarize findings and gaps. Mock `retrieve` until M1.5 lands.

**M2.3 Hypothesis Generator, RAG-grounded (Days 6–8)**
- `llm.with_structured_output(Hypothesis)`; retrieves with `peer_reviewed_only=True`; validates citations against `state["literature"]`, max 3 retries; on refine, includes the critic's issues.

**M2.4 Memory manager & long runs (Days 11–13)**
- `memory/manager.py`: `build_context`, `compress`. Token budget control, rolling compression, long-run state persistence and recovery after process restarts.

**M2.5 Refinement Loop & Policy Optimization (Days 14–16)**
- Hypothesis refinement loop with critic feedback integration, multi-turn stress testing (up to 3 iterations), agent routing policy optimization and safety fallbacks.

**M2.6 State Machine Verification & Specs (Days 17–18)**
- End-to-end graph state transition verification, deadlock testing, state machine specification & visualization.

### M3: Execution & Platform (Member 3: MANSI)

**M3.1 CI & Infrastructure (Days 1, 18)**
- Integration test jobs in GitHub Actions, docker-compose health checks, setup scripts, and environment verification.

**M3.2 Sandbox runner (Days 2–4, 12, 15)**
- `sandbox/runner.py`, `sandbox/Dockerfile`: controls from ARCHITECTURE §4.3; collect stdout/stderr, exit code, timeout flag, `/output` artifacts and `metrics.json`. CPU-only torch support, 50-attempt stress testing, container resource monitoring & profiling.

**M3.3 Code Executor agent (Days 5, 13)**
- `agents/code_executor.py`: prompt with hypothesis, test plan, sandbox libraries, and `metrics.json` contract; run via M3.2; append `CodeAttempt`. Retry strategies and dependency isolation.

**M3.4 Sandbox security tests (Days 6–7, 16)**
- `tests/test_sandbox_security.py`: network, writes outside `/output`, host reads, fork bomb, memory bomb, infinite loop, `setuid(0)`. Write security audit report.

**M3.5 Self-debug RL loop (Days 8–11, 14)**
- `agents/debugger.py`, `sandbox/feedback.py`, `route_after_execution`; epsilon-greedy bandit over repair strategies in `runs/bandit.json`. Convergence testing, persistence, and online reward updates.

**M3.6 Platform Runbook & Verification (Days 17)**
- Clean-install verification, platform runbook documentation, full platform smoke testing.

### M4: Evaluation, Synthesis & Dashboard (Member 4: SUDHEER)

**M4.1 Cost telemetry (Days 1–2)**
- `telemetry/costs.py`: callback writing `CostEvent`s to `runs/<id>/costs.jsonl`, with per-model pricing table.

**M4.2 Streamlit dashboard (Days 3–4, 8–10, 17)**
- `dashboard/app.py`: run list, logic tree, citations, attempts with diffs and reward curve, critic report, costs, live auto-refresh & export features.

**M4.3 Statistical Critic (Days 5–7)**
- `agents/critic.py`: two-stage design from ARCHITECTURE §4.5.
- Stage 1 deterministic parsing/validation (scipy, Holm-Bonferroni, small-n, missing test). Stage 2 LLM methodology review.

**M4.4 Synthesizer + LaTeX (Days 11–13)**
- `agents/synthesizer.py`, `synthesis/latex.py`, `synthesis/templates/*.j2` → `abstract.tex`, `paper.tex`, `refs.bib`, `draft.md`. LaTeX compilation verification.

**M4.5 Holdout set & grader (Days 14–16, 18)**
- `evaluation/holdout.py`, `evaluation/grader.py`, `data/holdout.jsonl`. 10 recent papers excluded from ingestion. LLM-judge scoring, metrics (`runs/<id>/grade.json`), and evaluation report.

---

## Dependency map

```
M1.3 ─► M1.4 ─► M1.5 ─► M2.2 / M2.3          (mock retrieve until Day 6)
M3.2 ─► M3.3 ─► M3.5 ─► M4.3 ─► M4.4         (mock run_code until Day 4)
M2.1 is the backbone every agent node plugs into (stub graph by Day 2)
M4.1 ─► M4.2
M1.6 ─► M1.7 ─► richer M2.2 retrieval
M4.3 & M4.4 plug into M2.1 graph routing
```
Until an upstream module lands, code against `schemas.py` and use mocks.

