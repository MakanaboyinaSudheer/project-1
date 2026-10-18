# Platform runbook

## Local setup

1. Install Python 3.12, Git, and Docker Desktop; start Docker.
2. From the repository root, run `powershell -ExecutionPolicy Bypass -File scripts\setup.ps1`.
3. Configure `.env`. `NCBI_EMAIL` is required for PubMed ingestion; provider credentials are required for live LLM/VLM calls.
4. Verify the local services with `discovery-agent check` and `docker compose ps`.

## Routine validation

```powershell
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\ruff.exe format --check .
.\.venv\Scripts\pytest.exe -m "not integration" -q
docker compose config --quiet
```

Run `.\.venv\Scripts\pytest.exe -m integration -q` only when Docker and Milvus are available.
Build the isolated experiment image with `docker build -t discovery-sandbox:latest sandbox\`.

## Running the application

```powershell
discovery-agent ingest --query "research topic"
discovery-agent run --topic "research topic"
streamlit run dashboard\app.py
```

Use the generated run ID with `discovery-agent evaluate --run-dir runs\<thread-id>` after
populating `data\holdout.jsonl` with the project's ten held-out papers.

## Operational boundaries

- Runs, checkpoints, downloaded papers, figures, and Milvus state are local, ignored artifacts.
- Do not add credentials to the repository. The sandbox must be built locally before running generated experiments.
- The current repository does not include the required holdout corpus or evidence of a clean-install, live-provider, or live-Docker acceptance run. Do not treat those checks as passed until they have been performed in the target environment.
