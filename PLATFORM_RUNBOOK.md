# Platform Runbook

## Requirements

- Python 3.12 (the package supports Python 3.11 and 3.12)
- Docker Desktop on Windows or Docker Engine with Compose v2 on Linux/macOS
- An LLM provider configured in `.env` for `discovery-agent check`
- A healthy Milvus service; in-memory vector fallback is disabled by default

## First-time setup

From the repository root, run one setup script:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
```

```bash
bash scripts/setup.sh
```

The script installs the project and development tools, creates `.env` from
`.env.example`, builds `discovery-sandbox:latest`, and starts Milvus. Fill in the
provider settings in `.env` before running the complete environment check.

## Start and verify the platform

```bash
docker compose up -d --wait --wait-timeout 180
docker compose ps
docker build -t discovery-sandbox:latest sandbox/
discovery-agent check
```

The check verifies Milvus connectivity, sandbox image availability, a real
container execution, and the configured LLM. A nonzero exit means at least one
check failed; resolve the printed `[FAIL]` item before starting a discovery run.

Run unit and integration suites separately:

```bash
pytest -m "not integration" -q
pytest -m integration -q
ruff check .
ruff format --check .
```

Integration tests require Docker and include sandbox escape-control and repeated
execution checks. Milvus integration tests additionally require the Compose
services to be healthy.

## Sandbox operation

Generated code runs in an isolated container with networking disabled, a
read-only root filesystem, a read-only `/workspace`, a size-limited `/output`
tmpfs, a non-root UID, dropped Linux capabilities, no-new-privileges, and
configured CPU, memory, process, and wall-clock limits. Output files are
validated and safely copied to `runs/<run-id>/attempt_<n>/output/`. Set
`SANDBOX_OUTPUT_LIMIT_MB` to change the default 128 MiB cap. Resource counters
are included in each execution result; a `resource_usage_error` field indicates
when Docker could not provide those counters.

The repair-strategy bandit is shared across runs at `runs/bandit.json`. Preserve
this file to retain learned strategy averages. Back it up before manually
editing it; malformed or unsupported state is reported as an error rather than
silently reset.

## Troubleshooting and shutdown

- **Milvus is not healthy:** inspect `docker compose ps` and
  `docker compose logs milvus`; allow up to three minutes on a cold start.
- **Sandbox image missing or stale:** run
  `docker build -t discovery-sandbox:latest sandbox/`.
- **Docker unavailable:** start Docker Desktop/Engine, then rerun the image and
  Compose checks.
- **Sandbox execution check fails:** inspect the Docker daemon logs and confirm
  the configured image is available and the Docker API permits archive retrieval.
- **LLM check fails:** verify `.env`, provider credentials, model name, and
  network/provider availability.

Stop shared services when finished:

```bash
docker compose down
```

Milvus data is retained under the ignored `volumes/` directory.
