# Modules, Ownership & 20-Day Commit Schedule

Three work streams, each owning a vertical slice. Replace the placeholder names once you've agreed on roles.

| Stream | Owner | Area |
|---|---|---|
| **M1: Data & Retrieval** | Member 1 (`<name>`) | ArXiv/PubMed ingestion, Milvus, RAG, multimodal PDF/figure parsing, evaluation |
| **M2: Agents & Orchestration** | Member 2 (`<name>`) | LangGraph workflow, Literature Reviewer, Hypothesis Generator, Critic, memory, synthesis |
| **M3: Execution & Platform** | Member 3 (`<name>`) | Docker sandbox, Code Executor, self-debug RL loop, telemetry, dashboard, CI/infra |

**Timeline:** 20 working days = 4 weeks × 5 days.
Week 1 = Days 1–5 · Week 2 = Days 6–10 (**Mid-Project Review on Day 10**) · Week 3 = Days 11–15 · Week 4 = Days 16–20.

**Rule:** one planned commit per person per day, so **20 commits each** (60 for the team).
Each row below is the minimum you push that day. Extra commits are welcome.
Use the module ID in branch names (`feature/M1.2-pubmed-client`) and commit scopes (`feat(M1.2): ...`).
Interfaces are already stubbed in `src/`. Implement each one behind its existing signature.

---

## Daily commit schedule

### Member 1: Data & Retrieval

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
| 16 | M1.8 | `feat(M1.8): holdout paper set and ingestion exclusion` |
| 17 | M1.8 | `feat(M1.8): grader metrics (similarity, citation validity, success rate)` |
| 18 | M1.8 | `feat(M1.8): LLM-judge scoring; run holdout evaluation` |
| 19 | Report | `docs: architecture report - data pipeline and evaluation results` |
| 20 | Release | `fix: final E2E fixes for v1.0` |

### Member 2: Agents & Orchestration

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
| 11 | M2.4 | `feat(M2.4): critic deterministic stage - parse and validate metrics` |
| 12 | M2.4 | `feat(M2.4): Holm-Bonferroni, small-n and missing-test checks + tests` |
| 13 | M2.4 | `feat(M2.4): LLM methodology review and route_after_critic` |
| 14 | M2.5 | `feat(M2.5): token-budgeted build_context` |
| 15 | M2.5 | `feat(M2.5): rolling compression and resume-after-kill test` |
| 16 | M2.6 | `feat(M2.6): Jinja2 LaTeX templates for abstract, paper and bib` |
| 17 | M2.6 | `feat(M2.6): synthesizer node with LaTeX escaping` |
| 18 | M2.6 | `feat(M2.6): compile simulated paper from best run` |
| 19 | Report | `docs: architecture report - agents, memory and critic` |
| 20 | Release | `chore: tag v1.0 and final delivery package` |

### Member 3: Execution & Platform

| Day | Module | Commit |
|---|---|---|
| 1 | M3.7 | `chore(M3.7): CI integration-test job and compose health checks` |
| 2 | M3.1 | `feat(M3.1): sandbox runner executes code and captures output` |
| 3 | M3.1 | `feat(M3.1): harden container (no network, read-only, caps, non-root)` |
| 4 | M3.1 | `feat(M3.1): wall-clock timeout kill and /output metrics collection` |
| 5 | M3.3 | `feat(M3.3): code executor agent v0 (generate, extract, run)` |
| 6 | M3.2 | `test(M3.2): sandbox escape tests - network and filesystem` |
| 7 | M3.2 | `test(M3.2): fork bomb, memory bomb, infinite loop, setuid` |
| 8 | M3.4 | `feat(M3.4): reward shaping and error classification` |
| 9 | M3.4 | `feat(M3.4): debugger node and route_after_execution` |
| 10 | MVP | `feat(M3.4): bandit over repair strategies; self-debug demo` |
| 11 | M3.5 | `feat(M3.5): CostTracker LangChain callback` |
| 12 | M3.5 | `feat(M3.5): model pricing table and costs.jsonl output` |
| 13 | M3.1 | `test(M3.1): 50-attempt long-run stress test; sandbox image tuning` |
| 14 | M3.6 | `feat(M3.6): dashboard run list and logic-tree view` |
| 15 | M3.6 | `feat(M3.6): code attempts with diffs and reward curve` |
| 16 | M3.6 | `feat(M3.6): citations and critic report views` |
| 17 | M3.6 | `feat(M3.6): cost breakdown and live refresh` |
| 18 | M3.7 | `docs(M3.7): runbook and clean-install verification` |
| 19 | Report | `docs: architecture report - sandbox, RL loop and dashboard` |
| 20 | Release | `chore: demo recording assets and final CI green` |

---

## Module specifications

### M1: Data & Retrieval (Member 1)

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

**M1.8 Holdout set + grader (Days 16–18)**
- `evaluation/holdout.py`, `evaluation/grader.py`, `data/holdout.jsonl`. About 10 recent human-authored papers, excluded from ingestion.
- Metrics: ARCHITECTURE §4.9. Writes `runs/<id>/grade.json` plus a summary table.

### M2: Agents & Orchestration (Member 2)

**M2.1 LangGraph workflow (Days 1–2, 5, 9)**
- `graph/workflow.py`: `build_graph()` with all nodes and conditional edges, compiled with `SqliteSaver`.
- `run_discovery(topic, thread_id)` streams the graph and saves to `runs/<thread_id>/`. Stub nodes first, so the graph runs before the real agents exist.

**M2.2 Literature Reviewer (Days 3–4)**
- `agents/literature_reviewer.py`: 2–4 sub-queries, retrieve, summarize findings and gaps. Mock `retrieve` until M1.5 lands.

**M2.3 Hypothesis Generator, RAG-grounded (Days 6–8)**
- `llm.with_structured_output(Hypothesis)`; retrieves with `peer_reviewed_only=True`; validates citations against `state["literature"]`, max 3 retries; on refine, includes the critic's issues.

**M2.4 Statistical Critic (Days 11–13)**
- `agents/critic.py`: two-stage design from ARCHITECTURE §4.5.
- Tests: significant, non-significant, missing test, multiple comparisons, tiny n.

**M2.5 Memory manager (Days 14–15)**
- `memory/manager.py`: `build_context`, `compress`. Done when a 50-attempt run stays under budget and resumes after the process is killed.

**M2.6 Synthesizer + LaTeX (Days 16–18)**
- `agents/synthesizer.py`, `synthesis/latex.py`, `synthesis/templates/*.j2` → `abstract.tex`, `paper.tex`, `refs.bib`, `draft.md`.
- Done when `paper.tex` compiles (Overleaf or `tectonic`).

### M3: Execution & Platform (Member 3)

**M3.1 Sandbox runner (Days 2–4, 13)**
- `sandbox/runner.py`, `sandbox/Dockerfile`: every control in ARCHITECTURE §4.3; collect stdout/stderr, exit code, timeout flag, `/output` artifacts and `metrics.json`. Add CPU-only torch if experiments need it.

**M3.2 Sandbox security tests (Days 6–7)**
- `tests/test_sandbox_security.py` (marked `integration`): network, writes outside `/output`, host file reads, fork bomb, memory bomb, infinite loop, `os.setuid(0)`. All must be contained.

**M3.3 Code Executor agent (Day 5)**
- `agents/code_executor.py`: prompt with the hypothesis, test plan, sandbox libraries and the `metrics.json` contract; run via M3.1; append a `CodeAttempt`.

**M3.4 Self-debug RL loop (Days 8–10)**
- `agents/debugger.py`, `sandbox/feedback.py`, `route_after_execution`; epsilon-greedy bandit over repair strategies in `runs/bandit.json`.
- Done when seeded broken scripts (syntax, import, runtime, timeout) are fixed automatically, with rewards logged.

**M3.5 Cost telemetry (Days 11–12)**
- `telemetry/costs.py`: callback writing `CostEvent`s to `runs/<id>/costs.jsonl`, with a per-model pricing table.

**M3.6 Streamlit dashboard (Days 14–17)**
- `dashboard/app.py`: run list, logic tree, citations, attempts with diffs and a reward curve, critic report, costs, live refresh.

**M3.7 CI & infrastructure (Days 1, 18, then ongoing)**
- Keep CI green, maintain `docker-compose.yml` and the setup scripts, and help teammates with environment problems.

---

## Dependency map

```
M1.3 ─► M1.4 ─► M1.5 ─► M2.2 / M2.3          (mock retrieve until Day 6)
M3.1 ─► M3.3 ─► M3.4 ─► M2.4 ─► M2.6         (mock run_code until Day 4)
M2.1 is the backbone every agent node plugs into (stub graph by Day 2)
M3.5 ─► M3.6
M1.6 ─► M1.7 ─► richer M2.2 retrieval
```
Until an upstream module lands, code against `schemas.py` and use mocks.
