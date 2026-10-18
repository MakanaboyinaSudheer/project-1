from pathlib import Path

import pytest
from requests.exceptions import Timeout as DockerTimeout

from discovery_agent.sandbox import runner


class FakeContainer:
    def __init__(self, wait_results: list[dict | Exception], output_dir: Path):
        self.wait_results = wait_results
        self.output_dir = output_dir
        self.wait_calls = 0
        self.killed = False
        self.removed = False

    def wait(self, **kwargs):
        result = self.wait_results[self.wait_calls]
        self.wait_calls += 1
        if isinstance(result, Exception):
            raise result
        return result

    def logs(self, *, stdout: bool, stderr: bool) -> bytes:
        return b"standard output\n" if stdout else b"standard error\n"

    def stats(self, *, stream: bool) -> dict:
        assert stream is False
        return {
            "memory_stats": {"max_usage": 1024, "limit": 2048},
            "cpu_stats": {
                "cpu_usage": {"total_usage": 200, "percpu_usage": [100, 100]},
                "system_cpu_usage": 2000,
                "online_cpus": 2,
            },
            "precpu_stats": {
                "cpu_usage": {"total_usage": 100, "percpu_usage": [50, 50]},
                "system_cpu_usage": 1000,
            },
        }

    def kill(self) -> None:
        self.killed = True

    def remove(self, *, force: bool) -> None:
        self.removed = force


class FakeContainers:
    def __init__(self, container: FakeContainer):
        self.container = container
        self.kwargs = None

    def run(self, image, **kwargs):
        self.kwargs = {"image": image, **kwargs}
        output_dir = self.container.output_dir
        output_dir.joinpath("metrics.json").write_text('{"p_value": 0.01}', encoding="utf-8")
        output_dir.joinpath("plot.png").write_bytes(b"plot")
        return self.container


class FakeClient:
    def __init__(self, container: FakeContainer):
        self.containers = FakeContainers(container)
        self.closed = False

    def close(self) -> None:
        self.closed = True


def install_fake_docker(monkeypatch, container: FakeContainer) -> FakeClient:
    client = FakeClient(container)
    monkeypatch.setattr(runner.docker, "from_env", lambda: client)
    return client


def test_run_code_collects_logs_metrics_and_artifacts(tmp_path, monkeypatch) -> None:
    attempt_dir = tmp_path / "attempt_1"
    output_dir = attempt_dir / "output"
    container = FakeContainer([{"StatusCode": 0}], output_dir)
    client = install_fake_docker(monkeypatch, container)

    result = runner.run_code("print('hello')", attempt_dir, timeout_s=5)

    assert result.exit_code == 0
    assert result.stdout == "standard output\n"
    assert result.stderr == "standard error\n"
    assert result.timed_out is False
    assert result.metrics == {"p_value": 0.01}
    assert result.resource_usage == {
        "memory_peak_bytes": 1024,
        "memory_limit_bytes": 2048,
        "cpu_percent": 20.0,
    }
    assert set(result.artifacts) == {"metrics.json", "plot.png"}
    assert container.removed is True
    assert client.closed is True
    assert client.containers.kwargs["command"] == ["/workspace/main.py"]
    assert client.containers.kwargs["volumes"][str(attempt_dir / "workspace")]["bind"] == "/workspace"
    assert client.containers.kwargs["network_disabled"] is True
    assert client.containers.kwargs["read_only"] is True
    assert client.containers.kwargs["tmpfs"] == {"/tmp": "rw,noexec,nosuid,size=64m"}
    assert client.containers.kwargs["cap_drop"] == ["ALL"]
    assert client.containers.kwargs["security_opt"] == ["no-new-privileges:true"]
    assert client.containers.kwargs["user"] == "10001:10001"
    assert client.containers.kwargs["pids_limit"] == 128
    assert client.containers.kwargs["mem_limit"] == runner.settings.sandbox_mem_limit
    assert client.containers.kwargs["nano_cpus"] == int(runner.settings.sandbox_cpus * 1_000_000_000)
    assert (attempt_dir / "workspace" / "main.py").read_text(encoding="utf-8") == "print('hello')"


def test_run_code_kills_and_reports_timeout(tmp_path, monkeypatch) -> None:
    output_dir = tmp_path / "attempt" / "output"
    container = FakeContainer(
        [DockerTimeout("time limit"), {"StatusCode": 137}],
        output_dir,
    )
    install_fake_docker(monkeypatch, container)

    result = runner.run_code("while True: pass", tmp_path / "attempt", timeout_s=1)

    assert result.exit_code == 137
    assert result.timed_out is True
    assert container.killed is True
    assert container.removed is True


def test_run_code_rejects_non_positive_timeout(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(runner.docker, "from_env", lambda: pytest.fail("Docker should not be initialized"))

    with pytest.raises(ValueError, match="greater than zero"):
        runner.run_code("pass", tmp_path / "attempt", timeout_s=0)
