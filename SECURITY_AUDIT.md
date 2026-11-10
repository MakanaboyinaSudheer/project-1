# Sandbox Security Review

## Scope and status

This review covers generated-code execution through
`src/discovery_agent/sandbox/runner.py`, the image in `sandbox/Dockerfile`, and
the corresponding automated tests. It records the implemented defense-in-depth
controls; it is not a formal third-party penetration test or a guarantee that
arbitrary code is safe.

## Controls

| Area | Control | Verification |
|---|---|---|
| Network | Docker network disabled | Integration test checks there is no default route |
| Filesystem | Read-only container root and source mount; `/output` uses a bounded tmpfs | Integration test checks writes are denied |
| Host boundary | Only the per-attempt workspace is mounted; output is copied from the container archive | Integration test confirms an unmounted host marker is not visible |
| Privileges | UID 10001, all Linux capabilities dropped, no-new-privileges | Integration test checks non-root execution and blocked `setuid` |
| Resources | Docker memory, CPU, and PID limits; private `/tmp` | Runner configuration tests and resource-limit integration tests |
| Runtime | Wall-clock timeout kills the container; container is removed afterward | Integration timeout test |
| Artifacts | Archive paths and entry types are validated; total output is capped before copying | Runner unit tests |
| Mount preparation | Workspace mount cannot be a symlink or resolve outside the attempt directory; source replacement is atomic | Runner unit tests |

The CI integration job builds the sandbox image before running Docker-dependent
tests. Review its test result on every change to the sandbox image or runner.

## Residual risks and operational requirements

- Container isolation shares the Docker host kernel. A kernel, Docker Engine, or
  runtime vulnerability could weaken the boundary.
- A user with Docker daemon access effectively has host-level authority. Do not
  expose an unauthenticated Docker socket or grant untrusted users access to
  the daemon.
- Collected artifacts remain untrusted. Do not execute them on the host.
- Resource limits constrain individual containers but do not protect the host
  from an incorrectly configured daemon or unrelated workloads.
- Keep Docker Engine and the base image updated; rebuild the image after
  dependency or base-image changes.

Run the security-specific checks with:

```bash
pytest -m integration tests/test_sandbox_security.py -q
```
