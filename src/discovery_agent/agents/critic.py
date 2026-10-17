"""Statistical Critic Agent. Owner: Member 4 (M4.3)."""

import math
from collections.abc import Mapping

from discovery_agent.graph.state import ResearchState
from discovery_agent.schemas import CriticReport, ExecutionResult


def _finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _named_values(
    metrics: Mapping[str, object],
    plural_key: str,
    singular_key: str,
) -> tuple[dict[str, object], list[str]]:
    issues: list[str] = []
    values: dict[str, object] = {}

    plural = metrics.get(plural_key)
    if plural is not None:
        if isinstance(plural, Mapping):
            values.update(plural)
        else:
            issues.append(f"`{plural_key}` must be an object keyed by outcome name.")

    singular = metrics.get(singular_key)
    if singular is not None:
        values.setdefault("primary", singular)

    return values, issues


def statistical_checks(result: ExecutionResult, alpha: float = 0.05) -> CriticReport:
    """Validate p-values, confidence intervals, and effect sizes in execution metrics."""
    if not _finite_number(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be a finite number strictly between 0 and 1.")

    issues: list[str] = []
    if result.timed_out or result.exit_code != 0:
        issues.append("Execution did not complete successfully.")

    metrics = result.metrics
    if not isinstance(metrics, Mapping):
        return CriticReport(verdict="invalid", issues=issues + ["Metrics must be an object."])

    raw_p_values, parse_issues = _named_values(metrics, "p_values", "p_value")
    issues.extend(parse_issues)
    p_values: dict[str, float] = {}
    for name, value in raw_p_values.items():
        if not isinstance(name, str) or not name.strip():
            issues.append("Each p-value must have a non-empty outcome name.")
        elif not _finite_number(value) or not 0 <= value <= 1:
            issues.append(f"P-value for `{name}` must be a finite number from 0 to 1.")
        else:
            p_values[name] = float(value)

    raw_intervals, parse_issues = _named_values(metrics, "confidence_intervals", "confidence_interval")
    issues.extend(parse_issues)
    confidence_intervals: dict[str, tuple[float, float]] = {}
    for name, interval in raw_intervals.items():
        if (
            not isinstance(name, str)
            or not name.strip()
            or not isinstance(interval, (list, tuple))
            or len(interval) != 2
            or not all(_finite_number(bound) for bound in interval)
        ):
            issues.append(f"Confidence interval for `{name}` must contain two finite bounds.")
            continue
        lower, upper = float(interval[0]), float(interval[1])
        if lower > upper:
            issues.append(f"Confidence interval for `{name}` has a lower bound above its upper bound.")
            continue
        confidence_intervals[name] = (lower, upper)

    raw_effect_sizes, parse_issues = _named_values(metrics, "effect_sizes", "effect_size")
    issues.extend(parse_issues)
    effect_sizes: dict[str, float] = {}
    for name, value in raw_effect_sizes.items():
        if not isinstance(name, str) or not name.strip():
            issues.append("Each effect size must have a non-empty outcome name.")
        elif not _finite_number(value):
            issues.append(f"Effect size for `{name}` must be a finite number.")
        else:
            effect_sizes[name] = float(value)

    if not p_values:
        issues.append("Metrics must include at least one valid p-value.")

    invalid = bool(issues)
    verdict = (
        "invalid"
        if invalid
        else "significant"
        if any(value < alpha for value in p_values.values())
        else "not_significant"
    )
    summary = {
        "significant": f"At least one p-value is below alpha={alpha:g}.",
        "not_significant": f"No p-value is below alpha={alpha:g}.",
        "invalid": "Metrics failed deterministic validation.",
    }[verdict]

    return CriticReport(
        verdict=verdict,
        p_values=p_values,
        confidence_intervals=confidence_intervals,
        effect_sizes=effect_sizes,
        issues=issues,
        summary=summary,
    )


def critic_node(state: ResearchState) -> dict:
    """Evaluate the most recent execution attempt and publish its deterministic report."""
    attempts = state.get("attempts", [])
    if not attempts:
        report = CriticReport(
            verdict="invalid",
            issues=["No execution attempts are available for review."],
            summary="Cannot evaluate metrics without an execution attempt.",
        )
    else:
        report = statistical_checks(attempts[-1].result)
    return {"critic_report": report}
