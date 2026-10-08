"""Statistical Critic Agent. Owner: Member 2 (Week 3).

Deterministic checks (scipy) run first, LLM review second: the LLM must never invent p-values.
"""

from discovery_agent.graph.state import ResearchState
from discovery_agent.schemas import CriticReport, ExecutionResult


def statistical_checks(result: ExecutionResult, alpha: float = 0.05) -> CriticReport:
    """Validate p-values / CIs / effect sizes in result.metrics; flag missing tests,
    tiny samples, and multiple comparisons (apply Holm-Bonferroni)."""
    raise NotImplementedError


def critic_node(state: ResearchState) -> dict:
    raise NotImplementedError
