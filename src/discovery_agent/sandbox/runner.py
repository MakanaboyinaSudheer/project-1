"""Execute generated Python code in a disposable Docker container."""

import base64
import binascii
import io
import json
import logging
import os
import secrets
import tarfile
import tempfile
import time
from contextlib import suppress
from pathlib import Path, PurePosixPath

import docker
from docker.errors import DockerException
from requests.exceptions import ConnectionError as DockerConnectionError
from requests.exceptions import RequestException
from requests.exceptions import Timeout as DockerTimeout

from discovery_agent.config import settings
from discovery_agent.schemas import ExecutionResult

logger = logging.getLogger(__name__)


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
            logger.warning("Unable to read valid metrics JSON from %s", metrics_path, exc_info=True)
            parsed_metrics = {}
        if isinstance(parsed_metrics, dict):
            metrics = parsed_metrics
        else:
            logger.warning("Metrics file must contain a JSON object: %s", metrics_path)

    return artifacts, metrics


def _collect_output_archive(archive_data: bytes, output_dir: Path, max_bytes: int) -> None:
    """Safely copy the size-limited container tmpfs output into the attempt directory."""
    archive_limit = max_bytes + 10 * 1024 * 1024
    if len(archive_data) > archive_limit:
        raise ValueError("Sandbox output archive exceeds the configured size limit")

    with tarfile.open(fileobj=io.BytesIO(archive_data), mode="r:*") as archive:
        members = archive.getmembers()
        if len(members) > 10_000:
            raise ValueError("Sandbox output archive contains too many entries")
        total_size = 0
        for member in members:
            parts = PurePosixPath(member.name).parts
            if parts and parts[0] == "output":
                parts = parts[1:]
            if not parts or any(part in {"", ".", ".."} for part in parts):
                if member.isdir() and parts == ():
                    continue
                raise ValueError(f"Unsafe path in sandbox output archive: {member.name!r}")

            target = output_dir.joinpath(*parts)
            if not target.resolve().is_relative_to(output_dir.resolve()):
                raise ValueError(f"Sandbox output path escapes the output directory: {member.name!r}")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise ValueError(f"Unsupported sandbox output entry: {member.name!r}")

            total_size += member.size
            if total_size > max_bytes:
                raise ValueError("Sandbox output exceeds the configured size limit")
            if target.is_symlink():
                raise ValueError(f"Sandbox output cannot replace a symlink: {member.name!r}")
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"Unable to read sandbox output file: {member.name!r}")
            with source, target.open("wb") as destination:
                while block := source.read(1024 * 1024):
                    destination.write(block)


def _launcher_script(archive_token: str) -> str:
    return f"""
import base64
import io
from pathlib import Path
import subprocess
import sys
import tarfile

result = subprocess.run(
    [sys.executable, "/workspace/main.py"],
)
archive_buffer = io.BytesIO()
output_dir = Path("/output")
with tarfile.open(fileobj=archive_buffer, mode="w") as archive:
    for path in sorted(output_dir.rglob("*")):
        archive.add(
            path,
            arcname=f"output/{{path.relative_to(output_dir).as_posix()}}",
            recursive=False,
        )
encoded_archive = base64.b64encode(archive_buffer.getvalue())
sys.stdout.buffer.write(b"__DISCOVERY_OUTPUT_BEGIN_{archive_token}__")
sys.stdout.buffer.write(encoded_archive)
sys.stdout.buffer.write(b"__DISCOVERY_OUTPUT_END_{archive_token}__")
exit_code = result.returncode if result.returncode >= 0 else 128 - result.returncode
sys.exit(exit_code)
"""


def _split_output_archive(stdout: bytes, archive_token: str) -> tuple[bytes, bytes | None]:
    begin = f"__DISCOVERY_OUTPUT_BEGIN_{archive_token}__".encode("ascii")
    end = f"__DISCOVERY_OUTPUT_END_{archive_token}__".encode("ascii")
    start_index = stdout.rfind(begin)
    if start_index < 0:
        return stdout, None
    end_index = stdout.find(end, start_index + len(begin))
    if end_index < 0:
        raise ValueError("Sandbox output archive marker is incomplete")
    encoded_archive = stdout[start_index + len(begin) : end_index]
    try:
        archive = base64.b64decode(encoded_archive, validate=True)
    except binascii.Error as exc:
        raise ValueError("Sandbox output archive is not valid base64") from exc
    return stdout[:start_index], archive


def _collect_resource_usage(container) -> tuple[dict[str, int | float], str | None]:
    try:
        stats = container.stats(stream=False)
    except (DockerException, RequestException) as exc:
        return {}, f"{type(exc).__name__}: {exc}"

    memory = stats.get("memory_stats", {})
    cpu = stats.get("cpu_stats", {})
    pre_cpu = stats.get("precpu_stats", {})
    cpu_usage = cpu.get("cpu_usage", {})
    pre_cpu_usage = pre_cpu.get("cpu_usage", {})
    usage: dict[str, int | float] = {}
    for key, value in (
        ("memory_usage_bytes", memory.get("usage")),
        ("memory_limit_bytes", memory.get("limit")),
        ("memory_peak_bytes", memory.get("max_usage")),
        ("cpu_total_usage_ns", cpu_usage.get("total_usage")),
    ):
        if isinstance(value, (int, float)) and value >= 0:
            usage[key] = value

    cpu_delta = cpu_usage.get("total_usage", 0) - pre_cpu_usage.get("total_usage", 0)
    system_delta = cpu.get("system_cpu_usage", 0) - pre_cpu.get("system_cpu_usage", 0)
    online_cpus = cpu.get("online_cpus") or len(cpu_usage.get("percpu_usage", []))
    if cpu_delta >= 0 and system_delta > 0 and online_cpus > 0:
        usage["cpu_percent"] = round(cpu_delta / system_delta * online_cpus * 100, 2)
    return usage, None


def _create_attempt_directory(path: Path, attempt_dir: Path) -> Path:
    if path.is_symlink():
        raise ValueError(f"Sandbox mount directory must not be a symlink: {path}")
    path.mkdir(parents=True, exist_ok=True)
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(attempt_dir):
        raise ValueError(f"Sandbox mount directory must remain inside the attempt directory: {path}")
    if not resolved.is_dir():
        raise ValueError(f"Sandbox mount path must be a directory: {path}")
    return resolved


def run_code(code: str, run_dir: Path, timeout_s: int | None = None) -> ExecutionResult:
    """Run code and collect its output, exit status, and files written to ``/output``."""
    timeout = settings.sandbox_timeout_s if timeout_s is None else timeout_s
    if timeout <= 0:
        raise ValueError("timeout_s must be greater than zero")

    attempt_dir = Path(run_dir).resolve()
    attempt_dir.mkdir(parents=True, exist_ok=True)
    workspace_dir = _create_attempt_directory(attempt_dir / "workspace", attempt_dir)
    output_dir = _create_attempt_directory(attempt_dir / "output", attempt_dir)
    if any(output_dir.iterdir()):
        raise ValueError(f"Sandbox output directory must be empty before a run: {output_dir}")
    output_limit_mb = settings.sandbox_output_limit_mb
    if output_limit_mb <= 0:
        raise ValueError("sandbox_output_limit_mb must be greater than zero")
    output_limit_bytes = output_limit_mb * 1024 * 1024
    archive_token = secrets.token_hex(16)
    script_path = workspace_dir / "main.py"
    temporary_script: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=workspace_dir,
            prefix=".main-",
            suffix=".tmp",
            delete=False,
        ) as script:
            temporary_script = Path(script.name)
            script.write(code)
        os.chmod(temporary_script, 0o644)
        temporary_script.replace(script_path)
    finally:
        if temporary_script is not None and temporary_script.exists():
            temporary_script.unlink()

    client = docker.from_env()
    container = None
    started_at = time.monotonic()
    timed_out = False
    try:
        container = client.containers.run(
            settings.sandbox_image,
            command=["-c", _launcher_script(archive_token)],
            detach=True,
            volumes={
                str(workspace_dir): {"bind": "/workspace", "mode": "ro"},
            },
            network_disabled=True,
            read_only=True,
            tmpfs={
                "/tmp": "rw,noexec,nosuid,size=64m",
                "/output": (f"rw,noexec,nosuid,nodev,size={output_limit_mb}m,uid=10001,gid=10001,mode=0700"),
            },
            cap_drop=["ALL"],
            security_opt=["no-new-privileges:true"],
            user="10001:10001",
            pids_limit=128,
            mem_limit=settings.sandbox_mem_limit,
            nano_cpus=int(settings.sandbox_cpus * 1_000_000_000),
        )
        wait_started_at = time.monotonic()
        try:
            result = container.wait(timeout=timeout)
            exit_code = int(result["StatusCode"])
        except DockerTimeout:
            timed_out = True
        except DockerConnectionError:
            if time.monotonic() - wait_started_at < timeout:
                raise
            timed_out = True

        if timed_out:
            container.kill()
            result = container.wait(timeout=10)
            exit_code = int(result["StatusCode"])

        stdout_bytes = container.logs(stdout=True, stderr=False)
        stdout_bytes, archive_data = _split_output_archive(stdout_bytes, archive_token)
        stderr_bytes = container.logs(stdout=False, stderr=True)
        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")
        if archive_data is not None:
            _collect_output_archive(archive_data, output_dir, output_limit_bytes)
        elif not timed_out and exit_code == 0:
            raise RuntimeError("Sandbox completed without returning its output archive")
        artifacts, metrics = _collect_artifacts(output_dir)
        resource_usage, resource_usage_error = _collect_resource_usage(container)
        return ExecutionResult(
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            duration_s=time.monotonic() - started_at,
            timed_out=timed_out,
            artifacts=artifacts,
            metrics=metrics,
            resource_usage=resource_usage,
            resource_usage_error=resource_usage_error,
        )
    finally:
        if container is not None:
            with suppress(docker.errors.NotFound):
                container.remove(force=True)
        client.close()
