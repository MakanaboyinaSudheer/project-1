"""`discovery-agent check` — confirms a teammate's environment matches the team setup."""

import tempfile
from pathlib import Path

import docker
from pymilvus import MilvusClient

from discovery_agent.config import settings
from discovery_agent.llm import get_llm
from discovery_agent.sandbox.runner import run_code


def _check(name: str, fn) -> bool:
    try:
        detail = fn()
        print(f"[ OK ] {name}{f' - {detail}' if detail else ''}")
        return True
    except Exception as exc:  # noqa: BLE001 - report any failure
        print(f"[FAIL] {name} - {exc}")
        return False


def _milvus() -> str:
    client = MilvusClient(uri=settings.milvus_uri)
    return f"collections: {client.list_collections()}"


def _docker() -> str:
    client = docker.from_env()
    try:
        client.images.get(settings.sandbox_image)
    finally:
        client.close()
    return f"image {settings.sandbox_image} present"


def _sandbox_execution() -> str:
    with tempfile.TemporaryDirectory(prefix="discovery-agent-check-") as temp_dir:
        result = run_code(
            "print('sandbox-ready')",
            Path(temp_dir) / "attempt",
            timeout_s=min(settings.sandbox_timeout_s, 15),
        )
    if result.exit_code != 0 or result.timed_out or "sandbox-ready" not in result.stdout:
        raise RuntimeError(
            f"Sandbox smoke execution failed (exit={result.exit_code}, "
            f"timed_out={result.timed_out}): {result.stderr}"
        )
    return f"execution completed in {result.duration_s:.2f}s"


def _llm() -> str:
    reply = get_llm().invoke("Reply with the single word: pong")
    return f"{settings.llm_model} -> {str(reply.content).strip()[:20]}"


def run_checks() -> None:
    results = [
        _check("Milvus", _milvus),
        _check("Docker sandbox image", _docker),
        _check("Docker sandbox execution", _sandbox_execution),
        _check("LLM", _llm),
    ]
    raise SystemExit(0 if all(results) else 1)
