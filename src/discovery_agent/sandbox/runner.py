"""Secure Docker execution of generated code. Owner: Member 3 (Week 2).

Required container hardening (verified by tests/test_sandbox_security.py):
  network_disabled, read_only rootfs, tmpfs /tmp, cap_drop=["ALL"],
  security_opt=["no-new-privileges"], non-root user, pids_limit, mem_limit,
  nano_cpus, wall-clock timeout -> kill, code mounted read-only, only /output writable.
"""

from pathlib import Path

from discovery_agent.schemas import ExecutionResult


def run_code(code: str, run_dir: Path, timeout_s: int | None = None) -> ExecutionResult:
    raise NotImplementedError
