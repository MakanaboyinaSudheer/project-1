# Final Presentation Guide: Autonomous Scientific Discovery Agent

This guide is written against the implementation currently in the repository. It
is intended to help explain the project clearly without presenting planned or
partially connected features as completed end-to-end behavior.

## 1. The project in one sentence

The Autonomous Scientific Discovery Agent is a research-assistant prototype
that searches scientific literature, proposes a testable hypothesis, generates
and safely runs an experiment, checks its reported results, and presents the
run in a dashboard.

## 2. Short opening pitch

> Researchers spend significant time finding relevant papers, turning evidence
> into a testable idea, writing experiments, and checking results. Our project
> brings those steps into one AI-assisted workflow. It combines literature
> retrieval with language-model agents, executes generated Python in a
> restricted Docker container, applies statistical and methodological checks,
> and provides a dashboard for inspecting the evidence and run artifacts. It is
> a prototype to assist researchers, not a replacement for expert scientific
> judgment.

## 3. What problem are we addressing?

Scientific discovery involves several connected tasks:

1. Find relevant, trustworthy prior work.
2. Identify a gap or relationship worth investigating.
3. Turn that idea into a hypothesis and an experiment.
4. Execute the experiment reproducibly and safely.
5. Check whether the results support the claim.
6. Preserve enough evidence to review what happened.

Doing these steps manually is time-consuming. Having an LLM do them without
controls is also risky: it may cite irrelevant papers, generate broken code,
report incorrect statistics, or make claims that the evidence does not support.
The project addresses this by separating the work into modules and adding
retrieval, validation, sandboxing, evaluation, and observability.

## 4. Architecture and data flow

### Literature preparation

```text
ArXiv + PubMed
      |
      v
Holdout exclusion and deduplication
      |
      v
Abstracts + PDF text + figure descriptions
      |
      v
Embeddings and Milvus vector store
```

### Current discovery-run path

```text
Research topic
      |
      v
Literature reviewer -- query expansion, retrieval, evidence summary
      |
      v
Hypothesis generator -- structured hypothesis and citations
      |
      v
Code executor -- generate Python and run it in Docker
      |
      v
Critic -- metric checks and methodology review
      |
      +---- not significant / invalid: refine and try again (up to 3 cycles)
      |
      +---- significant: synthesize a draft
```

The current `run_discovery` path is a Python loop. Some graph-routing,
debugging, memory, and parallel-execution helpers exist, but are not all
connected to this path. See [Important implementation boundaries](#10-important-implementation-boundaries)
before describing the workflow as fully autonomous or resumable.

### Supporting services

```text
Milvus       stores and searches embedded literature chunks
Docker       isolates generated experiment code
LLM / VLM    powers summaries, hypotheses, code, critique, and figure descriptions
Streamlit    displays run artifacts, citations, attempts, critique, and costs
```

## 5. Main modules and why they exist

### M1 — Data and retrieval

This workstream supplies the evidence the agents use. It fetches papers from
ArXiv and PubMed, avoids putting evaluation holdout papers into the search
collection, extracts text and figures from PDFs, creates embeddings, and
retrieves relevant chunks from Milvus. Retrieval can filter for peer-reviewed
papers and uses diversification and recency signals so the context is not just
several near-duplicates.

### M2 — Core agents and orchestration

This workstream turns a research question into a sequence of agent tasks. The
literature reviewer expands the query and summarizes retrieved evidence. The
hypothesis generator produces a structured idea and test plan. The workflow
passes results to execution and critique, and can request another hypothesis
after a weak or invalid result.

### M3 — Execution and platform

LLM-generated code must be treated as untrusted. The sandbox runs experiments
inside Docker with controls such as no network, a read-only root filesystem,
non-root execution, resource limits, a timeout, and bounded output collection.
The platform workstream also includes error classification, reward calculation,
repair-strategy selection, setup checks, and CI integration tests.

### M4 — Evaluation, synthesis, and dashboard

This workstream checks experiment metrics and methodology, records estimated
model costs, evaluates completed runs against a fixed holdout set, formats
research drafts from templates, and displays available run data in Streamlit.

The team plan assigns M1 to Vardhan, M2 to Aadil, M3 to Mansi, and M4 to
Sudheer. The plan is useful for explaining ownership; it should not be used as
evidence that every planned integration or benchmark is complete.

## 6. Function and file tour

These are the main entry points and responsibilities. Private formatting,
parsing, and validation helpers support them but are not all listed separately.

| Area | Key function or file | What it does and why it exists |
|---|---|---|
| Contracts | [`schemas.py`](src/discovery_agent/schemas.py) | Pydantic models such as `Paper`, `Chunk`, `RetrievedChunk`, `Hypothesis`, `CodeAttempt`, `CriticReport`, and `CostEvent` keep data formats consistent between modules. |
| Configuration | [`config.py`](src/discovery_agent/config.py) | Loads settings for model providers, Milvus, PDF processing, and sandbox limits from the environment. |
| Language models | [`llm.py`](src/discovery_agent/llm.py) | `get_llm` and `get_vlm` create the configured text and vision model clients so agents do not each build their own client. |
| ArXiv ingestion | [`arxiv_client.py`](src/discovery_agent/ingestion/arxiv_client.py) | `fetch_arxiv` searches ArXiv and maps results to the shared `Paper` contract. ArXiv records are marked as preprints, not peer-reviewed journal articles. |
| PubMed ingestion | [`pubmed_client.py`](src/discovery_agent/ingestion/pubmed_client.py) | `fetch_pubmed` searches PubMed, parses article metadata, and maps it to `Paper`. |
| PDF processing | [`pdf_parser.py`](src/discovery_agent/ingestion/pdf_parser.py) | `download_pdf` uses bounded downloads and a local cache; `extract_text_chunks` creates overlapping body-text chunks; `extract_figures` extracts embedded or rendered figures and associates nearby captions. |
| Figure descriptions | [`figure_parser.py`](src/discovery_agent/ingestion/figure_parser.py) | `describe_figure` asks the configured vision model to describe an image and returns a searchable figure `Chunk`. |
| Ingestion orchestration | [`pipeline.py`](src/discovery_agent/ingestion/pipeline.py) | `run_ingestion` connects source fetching, holdout exclusion, deduplication, PDF/figure processing, and vector-store upserts. Per-paper failures are logged so one bad PDF need not stop all ingestion. |
| Embeddings | [`embeddings.py`](src/discovery_agent/retrieval/embeddings.py) | `embed_texts` converts text to vectors for semantic search. A deterministic fallback may keep development paths running, but it does not provide equivalent semantic quality. |
| Vector storage | [`vector_store.py`](src/discovery_agent/retrieval/vector_store.py) | `VectorStore` manages the Milvus collection, embeddings, upserts, and searches. It stores metadata such as paper ID and peer-review status along with vectors. |
| Retrieval | [`retriever.py`](src/discovery_agent/retrieval/retriever.py) | `retrieve` searches for candidate chunks, optionally filters to peer-reviewed work, diversifies results with MMR, and applies a recency boost. |
| Literature agent | [`literature_reviewer.py`](src/discovery_agent/agents/literature_reviewer.py) | `literature_reviewer_node` expands the research question into search angles, retrieves evidence, prioritizes relevant literature, and summarizes findings and gaps. |
| Hypothesis agent | [`hypothesis_generator.py`](src/discovery_agent/agents/hypothesis_generator.py) | `hypothesis_generator_node` returns a structured hypothesis, rationale, citations, test plan, and expected outcome, with citation checks and retry logic. |
| Experiment agent | [`code_executor.py`](src/discovery_agent/agents/code_executor.py) | `code_executor_node` asks the model for experiment code, extracts it, runs it through the sandbox, and records the result as an attempt. |
| Debugger helper | [`debugger.py`](src/discovery_agent/agents/debugger.py) | `debugger_node` analyzes execution errors and proposes repaired code. It is a component available to orchestration, not a guaranteed step in the current `run_discovery` loop. |
| Statistical critic | [`critic.py`](src/discovery_agent/agents/critic.py) | `statistical_checks` performs deterministic metric checks, including multiple-comparison handling and raw-data recomputation when suitable data is present. `critic_node` then adds an LLM methodology review. |
| Draft generation | [`synthesizer.py`](src/discovery_agent/agents/synthesizer.py) and [`synthesis/templates/`](src/discovery_agent/synthesis/templates/) | `synthesizer_node` renders Markdown and LaTeX/BibTeX templates from the run's available evidence and results. A generated draft is a starting point for human review, not a publication-ready paper. |
| Workflow | [`workflow.py`](src/discovery_agent/graph/workflow.py) | `run_discovery` coordinates the current reviewer → hypothesis → executor → critic loop. `route_after_execution` and `route_after_critic` provide routing helpers; `get_run_status` reports in-process run status. |
| Shared run state | [`state.py`](src/discovery_agent/graph/state.py) | Defines the state and trace-event data used to pass topic, literature, hypothesis, attempts, reports, and errors between workflow components. |
| Parallel helpers | [`parallel.py`](src/discovery_agent/graph/parallel.py), [`state_parallel.py`](src/discovery_agent/graph/state_parallel.py), and [`ui/components.py`](src/discovery_agent/ui/components.py) | Define fan-out helpers and a side-by-side hypothesis UI. They are separate from the current main workflow path. |
| Memory helper | [`manager.py`](src/discovery_agent/memory/manager.py) | `build_context` and `compress` support compacting run context. The current main workflow does not call these helpers. |
| Sandbox | [`runner.py`](src/discovery_agent/sandbox/runner.py) | `run_code` creates and runs the container with runtime restrictions, enforces time/resource/output bounds, and collects stdout, stderr, metrics, and validated artifacts. |
| Feedback | [`feedback.py`](src/discovery_agent/sandbox/feedback.py) | `compute_reward` scores an execution result and `classify_error` categorizes common failure types for repair logic. |
| Strategy selection | [`bandit.py`](src/discovery_agent/sandbox/bandit.py) | `select_strategy` chooses a repair strategy and `update_strategy` updates its reward estimates. |
| Holdout evaluation | [`holdout.py`](src/discovery_agent/evaluation/holdout.py) and [`grader.py`](src/discovery_agent/evaluation/grader.py) | `load_holdout` and `exclude_holdout_papers` protect the fixed 10-paper holdout from ingestion; `grade_run` scores a compatible saved run using similarity, citation validity, experiment success, critic verdict, and cost-related data. |
| Cost tracking | [`costs.py`](src/discovery_agent/telemetry/costs.py) | `CostTracker` records model/token usage and estimated USD cost to run artifacts. It is an estimate, not the provider's final invoice. |
| Dashboard | [`dashboard/app.py`](dashboard/app.py) | `discover_runs` locates available run artifacts and `main` renders the Streamlit workflow launcher, run summary, refresh controls, and Trace, Citations, Attempts, Critic, and Costs tabs. |
| Environment checks | [`healthcheck.py`](src/discovery_agent/healthcheck.py) | `run_checks` tests Milvus, the sandbox image, sandbox execution, and LLM connectivity before a demo or run. |
| CLI | [`cli.py`](src/discovery_agent/cli.py) | Exposes `ingest`, `run`, `grade`, and `check` commands for the main workflows. |

## 7. Why these design choices?

- **Separate agents:** Each stage has a focused responsibility and a defined
  input/output, making failures easier to locate than in one giant prompt.
- **Structured schemas:** Typed data makes it easier to pass papers,
  hypotheses, attempts, and reports between agents and to validate outputs.
- **Retrieval-augmented generation:** The literature reviewer can use stored
  evidence instead of relying only on the model's training data.
- **Milvus:** A vector database supports similarity search over abstracts,
  paper-body chunks, and figure descriptions.
- **Holdout exclusion:** A fixed set of 10 papers is excluded from normal
  ingestion so the evaluator can use it as a separate check.
- **MMR and recency:** MMR reduces repetitive search results; recency can help
  surface newer work without replacing relevance.
- **Docker sandbox:** Generated code is untrusted, so it should not execute
  directly on the developer's host with unrestricted network and filesystem
  access.
- **Two-stage critique:** Deterministic checks are more repeatable for
  numerical issues; the LLM stage can flag methodological concerns that are
  harder to encode as simple rules.
- **Dashboard and traces:** A research result is more reviewable when the
  evidence, attempts, costs, and criticism can be inspected rather than shown
  only as a final paragraph.

## 8. Suggested 10-slide presentation

For a 7–10 minute presentation, spend about 40–60 seconds per slide. Replace
the bracketed prompts with your screenshots or team details.

### Slide 1 — Title and team

**Show:** Project title, team members, course/project context.

**Say:** “We built an AI-assisted scientific discovery prototype that connects
literature search, hypothesis generation, safe experiment execution,
evaluation, and run visualization.”

### Slide 2 — The research workflow problem

**Show:** A simple chain of paper search → idea → experiment → analysis.

**Say:** “These steps are usually separate and time-consuming. Our goal was to
connect them while keeping evidence and checks visible.”

### Slide 3 — What the system is intended to do

**Show:** The architecture/data-flow diagram from section 4.

**Say:** “Papers are prepared and indexed first. At run time, agents retrieve
evidence, propose a test, execute code in a sandbox, and ask the critic to
assess the result.”

### Slide 4 — Literature ingestion and retrieval

**Show:** ArXiv/PubMed, PDF text/figure chunks, and Milvus. Optionally show a
retrieved citation in the dashboard.

**Say:** “We ingest abstracts and, when available, PDF body text and figure
descriptions. Retrieval can diversify results and filter for peer-reviewed
papers. ArXiv preprints are labeled separately.”

### Slide 5 — Agents and the experiment loop

**Show:** Literature reviewer → hypothesis generator → code executor → critic.

**Say:** “The hypothesis is structured so it includes a rationale and a test
plan. The generated experiment runs in Docker. The critic checks available
metrics and adds a methodology review.”

### Slide 6 — Safety and reliability

**Show:** A small sandbox diagram or the controls table from
[`SECURITY_AUDIT.md`](SECURITY_AUDIT.md).

**Say:** “Model-generated code is treated as untrusted. The container has no
network, a read-only root, a non-root user, resource limits, a timeout, and
bounded artifact collection. These controls reduce risk, but Docker is not a
formal guarantee against every host or kernel vulnerability.”

### Slide 7 — Evaluation and reporting

**Show:** Critic output, holdout evaluation fields, and a sample generated draft
if available.

**Say:** “The holdout grader checks several dimensions rather than calling a
run a scientific discovery. Statistical checks can recompute results when raw
data is available; otherwise some reported metrics cannot be independently
verified.”

### Slide 8 — Dashboard demonstration

**Show:** The live dashboard at `http://127.0.0.1:8501` or a prepared screenshot.

**Say:** “The dashboard is intended to make runs inspectable: the tabs cover
trace events, citations, code attempts, the critic report, and cost records.
The data shown depends on what the workflow has actually persisted.”

### Slide 9 — Current implementation status and limitations

**Show:** Two columns: “Implemented components” and “Integration still needed.”

**Say:** “The ingestion, retrieval, sandbox, critique, synthesis, and dashboard
components exist. Some planned capabilities are not fully wired into the main
run path yet, especially persistent graph checkpoints, automatic debugger
routing, citation grounding, and complete saved-run state. We are presenting
this as a prototype and being explicit about that boundary.”

### Slide 10 — Conclusion and next steps

**Show:** Three next steps: integrate/persist workflow state, harden citation
grounding, run end-to-end evaluation with documented evidence.

**Say:** “The main contribution is a modular pipeline with safety and
inspectability built into the design. The next step is to close the remaining
integration gaps and validate the full workflow on representative topics.”

## 9. Demo checklist and safe commands

### Before the presentation

1. Confirm Docker is running and the environment variables/model credentials
   are configured. Do not display secrets from `.env`.
2. Start the shared services:

   ```bash
   docker compose up -d --wait --wait-timeout 180
   ```

3. Build the sandbox image if it is not already present:

   ```bash
   docker build -t discovery-sandbox:latest sandbox/
   ```

4. Check the dependencies before launching a workflow:

   ```bash
   discovery-agent check
   ```

5. If the vector collection needs papers, ingest them before the demo:

   ```bash
   discovery-agent ingest --query "your research topic"
   ```

6. Start the dashboard when needed:

   ```bash
   streamlit run dashboard/app.py
   ```

### During the demo

- Start with a prepared topic and a run whose artifacts are already available.
- Show the topic, evidence, attempt output, critic report, and costs as
  separate parts of the process.
- Avoid running a fresh workflow unless the LLM, Milvus, and Docker checks pass.
  A live run can require external services, take time, and incur model costs.
- Do not describe a generated hypothesis as a proven discovery.
- The current local preview may list old run data. State that it is sample/local
  run history unless you have verified that the displayed run was created in
  the current demo.

### CLI reference

```bash
discovery-agent check
discovery-agent ingest --query "your research topic"
discovery-agent run --topic "your research topic"
discovery-agent grade --run-dir runs/your-run-id
```

The `--thread-id` help text currently says “Resume a previous run,” but durable
resume is not implemented in the current `run_discovery` path. Do not use this
option as evidence of crash-safe resume. The grader also expects a completed
run directory with the saved state fields it reads; the current runner does
not persist the full state in that format automatically.

## 10. Important implementation boundaries

These points are important for an accurate final presentation:

1. **The main runner is not currently a compiled, checkpointed LangGraph
   workflow.** `run_discovery` uses a direct Python loop. The repository has
   graph-related helpers and documentation describing a graph/checkpointer,
   but the current path does not provide durable SQLite resume after a process
   restart.
2. **Automatic debugging is not wired into the main loop.** `debugger_node`
   and `route_after_execution` exist, but the current runner calls the executor
   and critic directly. Do not claim that every failed script is automatically
   repaired and rerun.
3. **Citation grounding needs integration work.** The retrieval path returns
   Pydantic `RetrievedChunk` objects, while the hypothesis citation filter
   currently recognizes dictionary-shaped context entries. This can leave the
   valid-ID set empty and allow fallback/mock citation IDs. The generated
   citations should not be described as guaranteed verified evidence.
4. **Dashboard persistence is incomplete.** The dashboard reads run artifacts,
   including saved state when present, while the current runner keeps its status
   in process memory and does not persist the complete state as a resumable
   checkpoint. A dashboard entry can therefore be missing, incomplete, or
   stale.
5. **Memory and parallel-agent support are helpers, not the active main path.**
   The memory manager and parallel hypothesis modules are not called by
   `run_discovery`.
6. **Statistical validation depends on available data.** Deterministic
   recomputation is useful when the experiment provides raw observations in
   the expected format. It cannot verify values that are not present.
7. **The holdout grader is an evaluation tool, not proof of scientific
   validity.** Similarity and an LLM judge can estimate alignment; neither
   establishes novelty, causality, or reproducibility.
8. **Operational components still need their configured services.** A complete
   run can depend on a working LLM/VLM provider, Milvus, Docker, and downloaded
   literature. The UI alone does not demonstrate that all backends are live.

### Recommended way to frame this

Say: “The system demonstrates the major stages and provides a foundation for
an integrated research assistant. Some reliability features are implemented
as components but still need to be connected and validated end-to-end.”

Avoid saying: “It autonomously discovers proven scientific facts,” “every
citation is verified,” “failed code is always repaired,” or “runs resume after
a restart.”

## 11. Likely questions and suggested answers

**What does the project do?**

It assists with a research workflow: literature retrieval, hypothesis
generation, experiment-code execution in a sandbox, critique, and run review.

**Where does its evidence come from?**

The ingestion pipeline supports ArXiv and PubMed and stores abstract, body, and
figure-description chunks in Milvus. Source and peer-review metadata are
retained, but source metadata does not guarantee that a paper is correct.

**How do you prevent generated code from harming the host?**

It runs inside a hardened Docker container with network disabled, a read-only
root filesystem, a non-root user, resource limits, a timeout, and constrained
artifact collection. This is defense in depth, not a claim of perfect
isolation.

**Does the system prove that a hypothesis is true or novel?**

No. It proposes a testable idea and analyzes the experiment output. Human
review and independent scientific validation are still required.

**Does it automatically debug every failed experiment?**

The debugger and strategy-selection components exist, but the current main
workflow does not route failures through them. That integration is future work.

**Can it resume a run after the application restarts?**

Not reliably in the current main path. Persistent checkpointing is a planned
integration, not a demonstrated capability.

**What does the dashboard show?**

When run artifacts are available, it can show overview metrics, trace events,
citations, code attempts, critic output, and recorded cost estimates.

**How do you know the system works?**

The repository includes unit tests and CI jobs for linting, Python 3.11/3.12
unit tests, sandbox-image builds, and Docker/Milvus integration tests. Describe
the latest test result only after checking the run associated with the current
commit; do not rely on an older green run.

## 12. Final takeaway

The strongest presentation is honest and concrete: this project connects
important research-assistant stages and adds evidence retrieval, isolated code
execution, evaluation hooks, and observability. It is a prototype with
substantial components in place; full end-to-end grounding, automatic repair,
and durable run recovery remain integration work.
