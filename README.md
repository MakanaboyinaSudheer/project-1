# Autonomous Multimodal Scientific Discovery Agent

A self-directed multi-agent AI researcher. It ingests ArXiv/PubMed literature, generates
grounded hypotheses, runs experiments in a secure Docker sandbox with self-debugging, checks
the statistics, and writes LaTeX research drafts.

- **[ARCHITECTURE.md](ARCHITECTURE.md):** system design
- **[PLAN.md](PLAN.md):** 4-week day-by-day schedule
- **[MODULES.md](MODULES.md):** module specs and ownership
- **[CONTRIBUTING.md](CONTRIBUTING.md):** branching, commits, PRs

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
discovery-agent run --topic "..."          # full discovery loop
streamlit run dashboard/app.py             # supervisor dashboard
pytest -m "not integration"                # unit tests
pytest -m integration                      # needs Docker + Milvus
docker compose down                        # stop Milvus (data persists in ./volumes)
```

## Troubleshooting
- **Milvus not ready:** it takes about 60–90 s on first start. Check `docker compose ps` until it shows `healthy`.
- **`pip install` fails on Windows:** make sure you're using Python 3.12 (`py -3.12 --version`).
- **Sandbox image missing:** `docker build -t discovery-sandbox:latest sandbox/`.
