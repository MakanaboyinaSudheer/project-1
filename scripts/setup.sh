#!/usr/bin/env bash
# One-shot dev environment setup for macOS/Linux. Run from repo root: bash scripts/setup.sh
set -euo pipefail

echo "==> Creating virtualenv (.venv, Python 3.12)"
[ -d .venv ] || python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e ".[dev]"

echo "==> Installing pre-commit hooks"
.venv/bin/pre-commit install

if [ ! -f .env ]; then
  cp .env.example .env
  echo "==> Created .env from .env.example - fill in your API keys"
fi

echo "==> Building sandbox image"
docker build -t discovery-sandbox:latest sandbox/

echo "==> Starting Milvus"
docker compose up -d

echo "Done. Activate with: source .venv/bin/activate ; then run: discovery-agent check"
