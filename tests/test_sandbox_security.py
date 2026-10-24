"""Defensive regression tests for the sandbox's container isolation controls."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from discovery_agent.sandbox import runner


def test_runner_passes_required_isolation_and_resource_limits(monkeypatch, tmp_path: Path) -> None:
    container = MagicMock()
    container.wait.return_value = {"StatusCode": 0}
    container.logs.return_value = b""
    client = MagicMock()
    client.containers.run.return_value = container
    monkeypatch.setattr(runner.docker, "from_env", lambda: client)

    runner.run_code("pass", tmp_path / "attempt", timeout_s=5)

    kwargs = client.containers.run.call_args.kwargs
    assert kwargs["network_disabled"] is True
    assert kwargs["read_only"] is True
    assert kwargs["volumes"][str((tmp_path / "attempt" / "workspace").resolve())] == {
        "bind": "/workspace",
        "mode": "ro",
    }
    assert kwargs["volumes"][str((tmp_path / "attempt" / "output").resolve())] == {
        "bind": "/output",
        "mode": "rw",
    }
    assert kwargs["cap_drop"] == ["ALL"]
    assert kwargs["security_opt"] == ["no-new-privileges:true"]
    assert kwargs["user"] == "10001:10001"
    assert kwargs["pids_limit"] == 128
    assert kwargs["mem_limit"] == runner.settings.sandbox_mem_limit
    assert kwargs["nano_cpus"] == int(runner.settings.sandbox_cpus * 1_000_000_000)
    assert kwargs["tmpfs"]["/tmp"] == "rw,noexec,nosuid,size=64m"


@pytest.mark.integration
def test_container_blocks_host_writes_and_privilege_escalation(tmp_path: Path) -> None:
    output_dir = tmp_path / "attempt"
    code = """
import json
import os
from pathlib import Path

def is_read_only(path):
    try:
        Path(path).write_text("must not be written")
    except OSError:
        return True
    return False

result = {
    "non_root": os.geteuid() != 0,
    "workspace_read_only": is_read_only("/workspace/security-test.txt"),
    "rootfs_read_only": is_read_only("/sandbox-security-test.txt"),
}
try:
    os.setuid(0)
    result["setuid_blocked"] = os.geteuid() != 0
except PermissionError:
    result["setuid_blocked"] = True

Path("/output/metrics.json").write_text(json.dumps(result), encoding="utf-8")
"""

    result = runner.run_code(code, output_dir, timeout_s=10)

    assert result.exit_code == 0, result.stderr
    assert result.metrics == {
        "non_root": True,
        "workspace_read_only": True,
        "rootfs_read_only": True,
        "setuid_blocked": True,
    }
    assert not (output_dir / "workspace" / "security-test.txt").exists()
    assert not (output_dir / "sandbox-security-test.txt").exists()


@pytest.mark.integration
def test_network_disabled_container_has_no_default_route(tmp_path: Path) -> None:
    code = """
import json
from pathlib import Path

route_lines = Path("/proc/net/route").read_text(encoding="utf-8").splitlines()[1:]
has_default_route = any(
    len(line.split()) > 1 and line.split()[1] == "00000000"
    for line in route_lines
)
Path("/output/metrics.json").write_text(
    json.dumps({"has_default_route": has_default_route}),
    encoding="utf-8",
)
"""

    result = runner.run_code(code, tmp_path / "attempt", timeout_s=10)

    assert result.exit_code == 0, result.stderr
    assert result.metrics == {"has_default_route": False}


@pytest.mark.integration
def test_timed_out_container_is_killed(tmp_path: Path) -> None:
    result = runner.run_code("while True:\n    pass\n", tmp_path / "attempt", timeout_s=1)

    assert result.timed_out is True
    assert result.exit_code != 0
