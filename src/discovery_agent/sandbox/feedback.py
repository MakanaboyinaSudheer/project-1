"""Reward signal for the self-debugging RL loop. Owner: Member 3 (Week 2).

Starting reward shaping (tune it, but keep this docstring in sync):
  -1.0   syntax error / import error
  -0.5   runtime exception, timeout or OOM
   0.0   runs, but metrics.json missing or malformed
  +0.5   runs and metrics.json is valid
  +1.0   critic verdict is 'significant'
  -0.05  per debug iteration (efficiency penalty)
Rewards are logged per attempt and drive the choice of debugging strategy
(epsilon-greedy bandit over repair prompts).
"""

import math
import re
from collections.abc import Mapping

from discovery_agent.schemas import ExecutionResult


def compute_reward(result: ExecutionResult, iteration: int) -> float:
    """Return the documented shaped reward for one sandbox execution."""
    if iteration < 0:
        raise ValueError("iteration must be non-negative")
    if result.timed_out or result.exit_code != 0:
        error = classify_error(result.stderr)
        reward = -1.0 if error in {"syntax", "import"} else -0.5
    elif not _has_valid_metrics(result.metrics):
        reward = 0.0
    else:
        reward = 0.5
    return reward - 0.05 * iteration


def _has_valid_metrics(metrics: object) -> bool:
    if not isinstance(metrics, Mapping):
        return False

    raw_p_values = metrics.get("p_values")
    if raw_p_values is None:
        raw_p_values = {"primary": metrics.get("p_value")}
    if not isinstance(raw_p_values, Mapping) or not raw_p_values:
        return False
    if any(
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or not 0 <= value <= 1
        for value in raw_p_values.values()
    ):
        return False

    raw_effect_sizes = metrics.get("effect_sizes")
    if raw_effect_sizes is None:
        raw_effect_sizes = {"primary": metrics.get("effect_size")}
    if not isinstance(raw_effect_sizes, Mapping) or not raw_effect_sizes:
        return False
    if any(
        not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value)
        for value in raw_effect_sizes.values()
    ):
        return False

    raw_sample_sizes = metrics.get("sample_sizes")
    if raw_sample_sizes is None:
        raw_sample_sizes = {"primary": metrics.get("sample_size", metrics.get("n"))}
    if not isinstance(raw_sample_sizes, Mapping) or not raw_sample_sizes:
        return False
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 1
        for value in raw_sample_sizes.values()
    ):
        return False

    tests = metrics.get("tests")
    if tests is None:
        tests = {}
    if not isinstance(tests, Mapping):
        return False
    tests = dict(tests)
    test = metrics.get("test", metrics.get("test_name"))
    if test is not None:
        tests.setdefault("primary", test)
    return any(isinstance(test, str) and test.strip() for test in tests.values())


def classify_error(stderr: str) -> str:
    """Map stderr to 'syntax', 'import', 'runtime', 'timeout', 'oom' or 'other'."""
    text = stderr.casefold()
    if any(marker in text for marker in ("timed out", "timeout", "time limit exceeded")):
        return "timeout"
    if any(marker in text for marker in ("out of memory", "memoryerror", "oom", "cannot allocate memory")):
        return "oom"
    if any(
        marker in text
        for marker in ("syntaxerror", "indentationerror", "invalid syntax", "unexpected eof while parsing")
    ):
        return "syntax"
    if any(
        marker in text
        for marker in ("modulenotfounderror", "importerror", "no module named", "cannot import name")
    ):
        return "import"
    if re.search(r"\b(?:[a-z_][a-z0-9_]*(?:error|exception)):", text):
        return "runtime"
    return "other"
