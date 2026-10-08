# One-shot dev environment setup for Windows. Run from repo root:
#   powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
$ErrorActionPreference = "Stop"

Write-Host "==> Creating virtualenv (.venv, Python 3.12)"
if (-not (Test-Path .venv)) { py -3.12 -m venv .venv }
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -e ".[dev]"

Write-Host "==> Installing pre-commit hooks"
& .\.venv\Scripts\pre-commit.exe install

if (-not (Test-Path .env)) {
    Copy-Item .env.example .env
    Write-Host "==> Created .env from .env.example - fill in your API keys"
}

Write-Host "==> Building sandbox image"
docker build -t discovery-sandbox:latest sandbox/

Write-Host "==> Starting Milvus"
docker compose up -d

Write-Host "Done. Activate with: .\.venv\Scripts\Activate.ps1 ; then run: discovery-agent check"
