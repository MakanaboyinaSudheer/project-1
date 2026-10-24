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

import re

from discovery_agent.schemas import ExecutionResult


def compute_reward(result: ExecutionResult, iteration: int) -> float:
    """Return the documented shaped reward for one sandbox execution."""
    if iteration < 0:
        raise ValueError("iteration must be non-negative")
    if result.timed_out or result.exit_code != 0:
        error = classify_error(result.stderr)
        reward = -1.0 if error in {"syntax", "import"} else -0.5
    elif not isinstance(result.metrics, dict) or not result.metrics:
        reward = 0.0
    else:
        reward = 0.5
    return reward - 0.05 * iteration


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
