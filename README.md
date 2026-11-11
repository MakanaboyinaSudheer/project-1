# Autonomous Multimodal Scientific Discovery Agent

A self-directed multi-agent AI researcher. It ingests ArXiv/PubMed literature, generates
grounded hypotheses, runs experiments in a secure Docker sandbox with self-debugging, checks
the statistics, and writes LaTeX research drafts.

- **[ARCHITECTURE.md](ARCHITECTURE.md):** system design
- **[PLAN.md](PLAN.md):** 4-week day-by-day schedule
- **[MODULES.md](MODULES.md):** module specs and ownership
- **[CONTRIBUTING.md](CONTRIBUTING.md):** branching, commits, PRs
- **[PLATFORM_RUNBOOK.md](PLATFORM_RUNBOOK.md):** platform setup and verification
- **[SECURITY_AUDIT.md](SECURITY_AUDIT.md):** sandbox controls and residual risks
- **[M4_EVALUATION_REPORT.md](M4_EVALUATION_REPORT.md):** evaluation, synthesis, and dashboard report

## Prerequisites
- Python **3.12**
- Docker Desktop (running)
- Git
- An LLM: an API key (Anthropic / OpenAI) **or** [Ollama](https://ollama.com) installed locally
- Optional: an NCBI API key (https://www.ncbi.nlm.nih.gov/account/)

## Setup (each teammate, once)
```bash
git clone <repo-url> && cd <repo>
git checkout develop
```
**Windows**
```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
.\.venv\Scripts\Activate.ps1
```
**macOS / Linux**
```bash
bash scripts/setup.sh
source .venv/bin/activate
```
Then edit `.env` (API keys, `NCBI_EMAIL`, `LLM_MODEL`) and verify:
```bash
discovery-agent check
```

The setup script creates `.venv`, installs dependencies, installs pre-commit hooks, builds
the `discovery-sandbox` image and starts Milvus. You can browse Milvus with Attu at http://localhost:8000.

## Everyday commands
```bash
docker compose up -d                       # start Milvus
discovery-agent ingest --query "protein folding"
discovery-agent run --topic "..." [--ingest 10] [--peer-reviewed-only] [--thread-id ID]  # full loop; reusing an ID resumes it
discovery-agent grade --run-dir runs/<id>   # evaluate a completed run against holdout papers
streamlit run dashboard/app.py             # research workspace: launch, monitor, browse literature/hypotheses/experiments/reports
pytest -m "not integration"                # unit tests
pytest -m integration                      # needs Docker + Milvus
docker compose down                        # stop Milvus (data persists in ./volumes)
```

ArXiv PDFs are cached under `data/raw/pdfs/` and extracted figures under
`data/raw/figures/`; set `INGEST_PDF_DIR`, `INGEST_FIGURE_DIR`, or
`INGEST_PDF_MAX_BYTES` in `.env` to change those defaults. Schedule ingestion
with Task Scheduler or cron as appropriate, and configure a vision-capable
`VLM_MODEL` if figure descriptions should be generated.
Milvus is required for persistent ingestion by default. Set
`MILVUS_ALLOW_MEMORY_FALLBACK=true` only for disposable local development;
in-memory vector data is lost when the process exits.

## Troubleshooting
- **Milvus not ready:** it takes about 60–90 s on first start. Check `docker compose ps` until it shows `healthy`.
- **`pip install` fails on Windows:** make sure you're using Python 3.12 (`py -3.12 --version`).
- **Sandbox image missing:** `docker build -t discovery-sandbox:latest sandbox/`.
