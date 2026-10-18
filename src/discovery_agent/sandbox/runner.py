"""Execute generated Python code in a disposable Docker container."""

import json
import os
import time
from contextlib import suppress
from pathlib import Path

import docker
from requests.exceptions import Timeout as DockerTimeout

from discovery_agent.config import settings
from discovery_agent.schemas import ExecutionResult


def _decode_logs(container, *, stdout: bool) -> str:
    logs = container.logs(stdout=stdout, stderr=not stdout)
    return logs.decode("utf-8", errors="replace") if isinstance(logs, bytes) else str(logs)


def _collect_artifacts(output_dir: Path) -> tuple[dict[str, str], dict]:
    artifacts: dict[str, str] = {}
    metrics: dict = {}

    for path in output_dir.rglob("*"):
        if not path.is_file():
            continue
        resolved = path.resolve()
        if not resolved.is_relative_to(output_dir.resolve()):
            continue
        artifacts[path.relative_to(output_dir).as_posix()] = str(resolved)

    metrics_path = output_dir / "metrics.json"
    if metrics_path.is_file():
        try:
            parsed_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            parsed_metrics = {}
        if isinstance(parsed_metrics, dict):
            metrics = parsed_metrics

    return artifacts, metrics


def _collect_resource_usage(container) -> dict[str, float | int]:
    stats = container.stats(stream=False)
    memory = stats.get("memory_stats", {})
    cpu = stats.get("cpu_stats", {})
    previous_cpu = stats.get("precpu_stats", {})
    memory_usage = memory.get("max_usage", memory.get("usage"))
    memory_limit = memory.get("limit")
    cpu_usage = cpu.get("cpu_usage", {}).get("total_usage")
    previous_usage = previous_cpu.get("cpu_usage", {}).get("total_usage")
    system_usage = cpu.get("system_cpu_usage")
    previous_system_usage = previous_cpu.get("system_cpu_usage")
    online_cpus = cpu.get("online_cpus") or len(cpu.get("cpu_usage", {}).get("percpu_usage", []))

    resource_usage: dict[str, float | int] = {}
    if isinstance(memory_usage, int) and isinstance(memory_limit, int):
        resource_usage["memory_peak_bytes"] = memory_usage
        resource_usage["memory_limit_bytes"] = memory_limit
    if all(
        isinstance(value, int)
        for value in (cpu_usage, previous_usage, system_usage, previous_system_usage, online_cpus)
    ):
        cpu_delta = cpu_usage - previous_usage
        system_delta = system_usage - previous_system_usage
        if system_delta > 0:
            resource_usage["cpu_percent"] = round(
                (cpu_delta / system_delta) * online_cpus * 100,
                2,
            )
    return resource_usage


def run_code(code: str, run_dir: Path, timeout_s: int | None = None) -> ExecutionResult:
    """Run code and collect its output, exit status, and files written to ``/output``."""
    timeout = settings.sandbox_timeout_s if timeout_s is None else timeout_s
    if timeout <= 0:
        raise ValueError("timeout_s must be greater than zero")

    attempt_dir = Path(run_dir).resolve()
    workspace_dir = attempt_dir / "workspace"
    output_dir = attempt_dir / "output"
    workspace_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(output_dir, 0o777)
    (workspace_dir / "main.py").write_text(code, encoding="utf-8")

    client = docker.from_env()
    container = None
    started_at = time.monotonic()
    timed_out = False
    try:
        container = client.containers.run(
            settings.sandbox_image,
            command=["/workspace/main.py"],
            detach=True,
            volumes={
                str(workspace_dir): {"bind": "/workspace", "mode": "ro"},
                str(output_dir): {"bind": "/output", "mode": "rw"},
            },
            network_disabled=True,
            read_only=True,
            tmpfs={"/tmp": "rw,noexec,nosuid,size=64m"},
            cap_drop=["ALL"],
            security_opt=["no-new-privileges:true"],
            user="10001:10001",
            pids_limit=128,
            mem_limit=settings.sandbox_mem_limit,
            nano_cpus=int(settings.sandbox_cpus * 1_000_000_000),
        )
        try:
            result = container.wait(timeout=timeout)
            exit_code = int(result["StatusCode"])
        except DockerTimeout:
            timed_out = True
            container.kill()
            result = container.wait(timeout=10)
            exit_code = int(result["StatusCode"])

        stdout = _decode_logs(container, stdout=True)
        stderr = _decode_logs(container, stdout=False)
        artifacts, metrics = _collect_artifacts(output_dir)
        resource_usage = _collect_resource_usage(container)
        return ExecutionResult(
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            duration_s=time.monotonic() - started_at,
            timed_out=timed_out,
            artifacts=artifacts,
            metrics=metrics,
            resource_usage=resource_usage,
        )
    finally:
        if container is not None:
            with suppress(docker.errors.NotFound):
                container.remove(force=True)
        client.close()
