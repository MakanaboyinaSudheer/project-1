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

from discovery_agent.schemas import ExecutionResult


def compute_reward(result: ExecutionResult, iteration: int) -> float:
    raise NotImplementedError


def classify_error(stderr: str) -> str:
    """Map stderr to 'syntax', 'import', 'runtime', 'timeout', 'oom' or 'other'."""
    raise NotImplementedError
