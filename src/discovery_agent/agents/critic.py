"""Statistical Critic Agent. Owner: Member 4 (M4.3)."""

import math
from collections.abc import Mapping
from typing import Literal, TypeGuard

from pydantic import BaseModel, Field

from discovery_agent.graph.state import ResearchState, trace_event
from discovery_agent.llm import get_llm
from discovery_agent.sandbox.feedback import update_strategy
from discovery_agent.schemas import CriticReport, ExecutionResult


def _finite_number(value: object) -> TypeGuard[int | float]:
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
        primary = values.get("primary")
        if "primary" in values:
            if (
                plural_key == "confidence_intervals"
                and isinstance(primary, (list, tuple))
                and isinstance(singular, (list, tuple))
            ):
                differs = tuple(primary) != tuple(singular)
            else:
                differs = primary != singular
            if differs:
                issues.append(
                    f"`{plural_key}.primary` conflicts with `{singular_key}`; "
                    "the named primary value is retained."
                )
        values.setdefault("primary", singular)

    return values, issues


def _holm_bonferroni(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running_max = 0.0
    count = len(ordered)
    for index, (name, p_value) in enumerate(ordered):
        running_max = max(running_max, (count - index) * p_value)
        adjusted[name] = min(1.0, running_max)
    return {name: adjusted[name] for name in p_values}


def statistical_checks(result: ExecutionResult, alpha: float = 0.05) -> CriticReport:
    """Validate metrics, apply Holm-Bonferroni, and classify the execution result.

    `tests` and `effect_sizes` map outcome names to values. `sample_sizes` is an
    optional outcome-to-count mapping; `test`, `test_name`, `sample_size`, `n`,
    `effect_size`, and singular metric fields support a single primary outcome.
    The report's p-values are Holm-Bonferroni-adjusted when multiple are present.
    """
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

    raw_tests = metrics.get("tests", metrics.get("test_names"))
    if raw_tests is None:
        raw_tests = {}
    if not isinstance(raw_tests, Mapping):
        issues.append("`tests` must be an object mapping outcome names to test names.")
        raw_tests = {}
    tests = dict(raw_tests)
    singular_test = metrics.get("test", metrics.get("test_name"))
    if singular_test is not None:
        tests.setdefault("primary", singular_test)
    for name in p_values:
        test_name = tests.get(name)
        if not isinstance(test_name, str) or not test_name.strip():
            issues.append(f"P-value for `{name}` is missing a named statistical test.")

    raw_sample_sizes = metrics.get("sample_sizes")
    if raw_sample_sizes is None:
        raw_sample_sizes = {}
    if not isinstance(raw_sample_sizes, Mapping):
        issues.append("`sample_sizes` must be an object mapping outcome names to counts.")
        raw_sample_sizes = {}
    sample_sizes = dict(raw_sample_sizes)
    singular_sample_size = metrics.get("sample_size", metrics.get("n"))
    if singular_sample_size is not None:
        sample_sizes.setdefault("primary", singular_sample_size)
    for name, sample_size in sample_sizes.items():
        if not isinstance(name, str) or not name.strip():
            issues.append("Each sample size must have a non-empty outcome name.")
        elif not isinstance(sample_size, int) or isinstance(sample_size, bool) or sample_size < 1:
            issues.append(f"Sample size for `{name}` must be a positive integer.")
        elif sample_size < 30:
            issues.append(f"Small sample size for `{name}` (n={sample_size} < 30).")

    for name in p_values:
        if name not in effect_sizes:
            issues.append(f"Effect size for `{name}` is missing.")

    adjusted_p_values = _holm_bonferroni(p_values)
    invalid = bool(issues)
    verdict: Literal["significant", "not_significant", "invalid"] = (
        "invalid"
        if invalid
        else "significant"
        if any(value < alpha for value in adjusted_p_values.values())
        else "not_significant"
    )
    summary = {
        "significant": f"At least one Holm-Bonferroni-adjusted p-value is below alpha={alpha:g}.",
        "not_significant": f"No Holm-Bonferroni-adjusted p-value is below alpha={alpha:g}.",
        "invalid": "Metrics failed deterministic validation.",
    }[verdict]

    return CriticReport(
        verdict=verdict,
        p_values=adjusted_p_values,
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
    hypothesis = state.get("hypothesis")
    if attempts and hypothesis is not None:

        class MethodologyReview(BaseModel):
            issues: list[str] = Field(description="Methodology flaws or an empty list")
            summary: str = Field(description="Concise methodology review")

        review_result = (
            get_llm()
            .with_structured_output(MethodologyReview)
            .invoke(
                [
                    (
                        "system",
                        "Review the experiment for data leakage, confounding, sample selection, and "
                        "whether the statistical test matches the hypothesis. Do not recalculate or "
                        "change any reported p-values, confidence intervals, effect sizes, or verdict.",
                    ),
                    (
                        "human",
                        f"Hypothesis: {hypothesis.model_dump_json()}\n"
                        f"Execution metrics: {attempts[-1].result.metrics!r}\n"
                        f"Deterministic critic: {report.model_dump_json()}",
                    ),
                ]
            )
        )
        review = MethodologyReview.model_validate(review_result)
        report = report.model_copy(
            update={
                "issues": [*report.issues, *review.issues],
                "summary": f"{report.summary} Methodology review: {review.summary}",
            }
        )
    strategy = state.get("debug_strategy")
    if attempts and strategy and attempts[-1].result.exit_code == 0 and not attempts[-1].result.timed_out:
        score = {"significant": 1.0, "not_significant": 0.5, "invalid": 0.0}[report.verdict]
        update_strategy(strategy, score - 0.05 * (len(attempts) - 1))
    return {
        "critic_report": report,
        "trace": [trace_event("critic", f"Deterministic verdict: {report.verdict}.")],
    }
