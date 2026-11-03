"""Persistent epsilon-greedy selection for experiment repair strategies."""

import json
import math
import random
import tempfile
import threading
from pathlib import Path
from typing import Any

STRATEGIES = ("minimal_patch", "rewrite", "simplify")
_LOCK = threading.RLock()
_VERSION = 1


def _empty_state() -> dict[str, Any]:
    return {
        "version": _VERSION,
        "strategies": {strategy: {"count": 0, "total_reward": 0.0} for strategy in STRATEGIES},
    }


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        return _empty_state()
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read bandit state at {path}: {exc}") from exc

    if not isinstance(state, dict) or state.get("version") != _VERSION:
        raise ValueError(f"Unsupported or invalid bandit state at {path}")
    arms = state.get("strategies")
    if not isinstance(arms, dict) or set(arms) != set(STRATEGIES):
        raise ValueError(f"Bandit state at {path} must define exactly {STRATEGIES}")
    for strategy in STRATEGIES:
        arm = arms[strategy]
        if (
            not isinstance(arm, dict)
            or not isinstance(arm.get("count"), int)
            or arm["count"] < 0
            or not isinstance(arm.get("total_reward"), (int, float))
            or not math.isfinite(arm["total_reward"])
        ):
            raise ValueError(f"Invalid bandit statistics for {strategy!r} in {path}")
    return state


def _save(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
            temporary_path = Path(stream.name)
            json.dump(state, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def select_strategy(path: str | Path, epsilon: float = 0.1) -> str:
    """Choose an exploration arm or the arm with the highest mean reward."""
    if not math.isfinite(epsilon) or not 0 <= epsilon <= 1:
        raise ValueError("epsilon must be between 0 and 1")
    bandit_path = Path(path)
    with _LOCK:
        state = _load(bandit_path)
        _save(bandit_path, state)
        if random.random() < epsilon:
            return random.choice(STRATEGIES)

        arms = state["strategies"]
        means = {
            strategy: (
                arms[strategy]["total_reward"] / arms[strategy]["count"] if arms[strategy]["count"] else 0.0
            )
            for strategy in STRATEGIES
        }
        best_mean = max(means.values())
        return random.choice([strategy for strategy, mean in means.items() if mean == best_mean])


def update_strategy(path: str | Path, strategy: str, reward: float) -> None:
    """Record an observed reward for the strategy used by an experiment attempt."""
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown repair strategy: {strategy!r}")
    if not math.isfinite(reward):
        raise ValueError("reward must be finite")

    bandit_path = Path(path)
    with _LOCK:
        state = _load(bandit_path)
        arm = state["strategies"][strategy]
        arm["count"] += 1
        arm["total_reward"] += reward
        _save(bandit_path, state)
