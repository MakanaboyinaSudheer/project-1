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

import json
import random
import threading
from pathlib import Path

from discovery_agent.config import settings
from discovery_agent.schemas import ExecutionResult

STRATEGIES = ("minimal_patch", "rewrite", "simplify")
_bandit_lock = threading.Lock()


def _bandit_path() -> Path:
    return Path(settings.runs_dir) / "bandit.json"


def compute_reward(result: ExecutionResult, iteration: int) -> float:
    if iteration < 0:
        raise ValueError("iteration cannot be negative")
    if result.timed_out or result.exit_code != 0:
        error_class = classify_error(result.stderr)
        reward = -1.0 if error_class in {"syntax", "import"} else -0.5
    elif not _has_valid_metrics(result.metrics):
        reward = 0.0
    else:
        reward = 0.5
    return reward - 0.05 * iteration


def _has_valid_metrics(metrics: dict) -> bool:
    p_values = metrics.get("p_values", {"primary": metrics.get("p_value")})
    tests = metrics.get("tests", {"primary": metrics.get("test", metrics.get("test_name"))})
    effects = metrics.get("effect_sizes", {"primary": metrics.get("effect_size")})
    if not isinstance(p_values, dict) or not isinstance(tests, dict) or not isinstance(effects, dict):
        return False
    if not p_values or set(p_values) - set(tests) or set(p_values) - set(effects):
        return False
    return all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and 0 <= value <= 1
        and isinstance(tests[name], str)
        and bool(tests[name].strip())
        and isinstance(effects[name], (int, float))
        and not isinstance(effects[name], bool)
        for name, value in p_values.items()
    )


def classify_error(stderr: str) -> str:
    """Map stderr to 'syntax', 'import', 'runtime', 'timeout', 'oom' or 'other'."""
    message = stderr.casefold()
    if any(token in message for token in ("out of memory", "memoryerror", "oomkilled")):
        return "oom"
    if "timeout" in message or "timed out" in message:
        return "timeout"
    if "syntaxerror" in message or "indentationerror" in message:
        return "syntax"
    if "modulenotfounderror" in message or "importerror" in message:
        return "import"
    if "traceback" in message or "exception" in message or "error:" in message:
        return "runtime"
    return "other"


def choose_strategy(epsilon: float = 0.1) -> str:
    if not 0 <= epsilon <= 1:
        raise ValueError("epsilon must be between 0 and 1")
    path = _bandit_path()
    with _bandit_lock:
        values = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if random.random() < epsilon:
        return random.choice(STRATEGIES)
    return max(
        STRATEGIES,
        key=lambda strategy: values.get(strategy, {}).get("mean_reward", 0.0),
    )


def update_strategy(strategy: str, reward: float) -> None:
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown repair strategy: {strategy}")
    if not isinstance(reward, (int, float)):
        raise ValueError("reward must be numeric")
    path = _bandit_path()
    with _bandit_lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        values = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        record = values.setdefault(strategy, {"count": 0, "mean_reward": 0.0})
        count = int(record["count"])
        mean = float(record["mean_reward"])
        count += 1
        record["count"] = count
        record["mean_reward"] = mean + (float(reward) - mean) / count
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(values, indent=2), encoding="utf-8")
        temporary.replace(path)
