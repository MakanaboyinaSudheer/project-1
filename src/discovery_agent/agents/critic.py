"""Deterministic statistical validation followed by an LLM methodology review."""

import math
from collections.abc import Mapping
from typing import Literal, TypeGuard

from pydantic import BaseModel, Field
from scipy import stats

from discovery_agent.graph.state import ResearchState, trace_event
from discovery_agent.llm import get_llm
from discovery_agent.schemas import CriticReport, ExecutionResult


class MethodologyReview(BaseModel):
    issues: list[str] = Field(description="Methodology flaws or an empty list")
    summary: str = Field(description="Concise methodology review")


def _finite_number(value: object) -> TypeGuard[int | float]:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _named_values(
    metrics: Mapping[str, object],
    plural_key: str,
    singular_key: str,
) -> tuple[dict[str, object], list[str]]:
    values: dict[str, object] = {}
    issues: list[str] = []

    plural = metrics.get(plural_key)
    if plural is not None:
        if isinstance(plural, Mapping):
            values.update(plural)
        else:
            issues.append(f"`{plural_key}` must be an object keyed by outcome name.")

    singular = metrics.get(singular_key)
    if singular is not None:
        primary = values.get("primary")
        if plural_key == "confidence_intervals" and isinstance(primary, (list, tuple)) and isinstance(
            singular, (list, tuple)
        ):
            differs = tuple(primary) != tuple(singular)
        else:
            differs = "primary" in values and primary != singular
        if "primary" in values and differs:
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


def _recompute_from_raw_data(
    outcome: str,
    raw_data: object,
    test_name: object,
    alpha: float,
) -> tuple[float, tuple[float, float], int]:
    if not isinstance(raw_data, Mapping):
        raise ValueError(f"Raw data for `{outcome}` must contain `control` and `treatment` arrays.")

    control = raw_data.get("control")
    treatment = raw_data.get("treatment")
    if (
        not isinstance(control, (list, tuple))
        or not isinstance(treatment, (list, tuple))
        or len(control) < 2
        or len(treatment) < 2
        or not all(_finite_number(value) for value in (*control, *treatment))
    ):
        raise ValueError(
            f"Raw data for `{outcome}` must have at least two finite numeric values "
            "in both `control` and `treatment`."
        )

    control_values = [float(value) for value in control]
    treatment_values = [float(value) for value in treatment]
    if not isinstance(test_name, str):
        raise ValueError(f"Raw data for `{outcome}` requires a named supported statistical test.")
    if test_name in {"welch_t_test", "welch_ttest", "independent_t_test"}:
        test_result = stats.ttest_ind(treatment_values, control_values, equal_var=False)
        control_variance = stats.tvar(control_values)
        treatment_variance = stats.tvar(treatment_values)
        control_count = len(control_values)
        treatment_count = len(treatment_values)
        standard_error_squared = control_variance / control_count + treatment_variance / treatment_count
        if standard_error_squared <= 0:
            raise ValueError(f"Raw data for `{outcome}` has no variance; a t-test is undefined.")
        degrees_of_freedom = standard_error_squared**2 / (
            (control_variance / control_count) ** 2 / (control_count - 1)
            + (treatment_variance / treatment_count) ** 2 / (treatment_count - 1)
        )
        difference = stats.tmean(treatment_values) - stats.tmean(control_values)
        margin = stats.t.ppf(1 - alpha / 2, degrees_of_freedom) * math.sqrt(standard_error_squared)
        confidence_interval = (difference - margin, difference + margin)
        sample_size = min(control_count, treatment_count)
    elif test_name in {"paired_t_test", "paired_ttest"}:
        if len(control_values) != len(treatment_values):
            raise ValueError(f"Paired raw data for `{outcome}` must have equal-sized groups.")
        differences = [
            treatment - control for control, treatment in zip(control_values, treatment_values, strict=True)
        ]
        test_result = stats.ttest_rel(treatment_values, control_values)
        standard_error = stats.tstd(differences) / math.sqrt(len(differences))
        if standard_error <= 0:
            raise ValueError(f"Paired raw data for `{outcome}` has no variance; a t-test is undefined.")
        difference = stats.tmean(differences)
        margin = stats.t.ppf(1 - alpha / 2, len(differences) - 1) * standard_error
        confidence_interval = (difference - margin, difference + margin)
        sample_size = len(differences)
    else:
        raise ValueError(
            f"Raw-data recomputation for `{outcome}` does not support test `{test_name}`; "
            "use `welch_t_test` or `paired_t_test`."
        )

    p_value = float(test_result.pvalue)
    if not math.isfinite(p_value) or not all(math.isfinite(value) for value in confidence_interval):
        raise ValueError(f"Raw data for `{outcome}` did not produce finite statistical results.")
    return p_value, confidence_interval, sample_size


def statistical_checks(result: ExecutionResult, alpha: float = 0.05) -> CriticReport:
    """Validate reported metrics, recompute supported raw-data tests, and adjust p-values."""
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
    raw_intervals, parse_issues = _named_values(metrics, "confidence_intervals", "confidence_interval")
    issues.extend(parse_issues)
    raw_effect_sizes, parse_issues = _named_values(metrics, "effect_sizes", "effect_size")
    issues.extend(parse_issues)

    raw_tests = metrics.get("tests", metrics.get("test_names"))
    if raw_tests is None:
        raw_tests = {}
    if not isinstance(raw_tests, Mapping):
        issues.append("`tests` must be an object mapping outcome names to test names.")
        raw_tests = {}
    tests = dict(raw_tests)
    singular_test = metrics.get("test", metrics.get("test_name"))
    if singular_test is not None:
        if "primary" in tests and tests["primary"] != singular_test:
            issues.append("`tests.primary` conflicts with `test`; the named primary test is retained.")
        tests.setdefault("primary", singular_test)

    raw_sample_sizes = metrics.get("sample_sizes", {})
    if raw_sample_sizes is None:
        raw_sample_sizes = {}
    singular_sample_size = metrics.get("sample_size", metrics.get("n"))
    if singular_sample_size is not None and isinstance(raw_sample_sizes, Mapping):
        raw_sample_sizes = dict(raw_sample_sizes)
        if "primary" in raw_sample_sizes and raw_sample_sizes["primary"] != singular_sample_size:
            issues.append(
                "`sample_sizes.primary` conflicts with `sample_size`; the named primary value is retained."
            )
        raw_sample_sizes.setdefault("primary", singular_sample_size)
    raw_data = metrics.get("raw_data")
    if raw_data is not None:
        if not isinstance(raw_data, Mapping):
            issues.append("`raw_data` must be an object keyed by outcome name.")
        else:
            for outcome, outcome_data in raw_data.items():
                if not isinstance(outcome, str) or not outcome.strip():
                    issues.append("Each raw-data outcome must have a non-empty name.")
                    continue
                try:
                    recomputed_p, recomputed_interval, sample_size = _recompute_from_raw_data(
                        outcome, outcome_data, tests.get(outcome), alpha
                    )
                except ValueError as exc:
                    issues.append(str(exc))
                    continue

                reported_p = raw_p_values.get(outcome)
                if reported_p is not None and (
                    not _finite_number(reported_p) or not math.isclose(reported_p, recomputed_p, abs_tol=1e-6)
                ):
                    issues.append(
                        f"Reported p-value for `{outcome}` differs from the value recomputed from raw data."
                    )
                raw_p_values[outcome] = recomputed_p

                reported_interval = raw_intervals.get(outcome)
                if reported_interval is not None and (
                    not isinstance(reported_interval, (list, tuple))
                    or len(reported_interval) != 2
                    or not all(_finite_number(bound) for bound in reported_interval)
                    or not all(
                        math.isclose(float(reported_interval[i]), recomputed_interval[i], abs_tol=1e-6)
                        for i in range(2)
                    )
                ):
                    issues.append(
                        f"Reported confidence interval for `{outcome}` differs from the value "
                        "recomputed from raw data."
                    )
                raw_intervals[outcome] = recomputed_interval

                if isinstance(raw_sample_sizes, Mapping):
                    raw_sample_sizes = dict(raw_sample_sizes)
                    raw_sample_sizes.setdefault(outcome, sample_size)

    p_values: dict[str, float] = {}
    for name, value in raw_p_values.items():
        if not isinstance(name, str) or not name.strip():
            issues.append("Each p-value must have a non-empty outcome name.")
        elif not _finite_number(value) or not 0 <= value <= 1:
            issues.append(f"P-value for `{name}` must be a finite number from 0 to 1.")
        else:
            p_values[name] = float(value)

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

    for name in p_values:
        test_name = tests.get(name)
        if not isinstance(test_name, str) or not test_name.strip():
            issues.append(f"P-value for `{name}` is missing a named statistical test.")
        if name not in effect_sizes:
            issues.append(f"Effect size for `{name}` is missing.")

    if not isinstance(raw_sample_sizes, Mapping):
        issues.append("`sample_sizes` must be an object mapping outcome names to counts.")
    else:
        for name, sample_size in raw_sample_sizes.items():
            if not isinstance(name, str) or not name.strip():
                issues.append("Each sample size must have a non-empty outcome name.")
            elif not isinstance(sample_size, int) or isinstance(sample_size, bool) or sample_size < 1:
                issues.append(f"Sample size for `{name}` must be a positive integer.")
            elif sample_size < 30:
                issues.append(f"Small sample size for `{name}` (n={sample_size} < 30).")

    adjusted_p_values = _holm_bonferroni(p_values)
    verdict: Literal["significant", "not_significant", "invalid"] = (
        "invalid"
        if issues
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


def critic_node(state: ResearchState) -> dict[str, object]:
    """Evaluate the latest attempt and append methodology feedback without changing statistics."""
    attempts = state.get("attempts", [])
    if not attempts:
        report = CriticReport(
            verdict="invalid",
            issues=["No execution attempts are available for review."],
            summary="Cannot evaluate metrics without an execution attempt.",
        )
    else:
        latest_result = attempts[-1].result
        report = statistical_checks(latest_result)
        hypothesis = state.get("hypothesis")
        if hypothesis is not None:
            review = (
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
                            f"Execution metrics: {latest_result.metrics!r}\n"
                            f"Deterministic critic: {report.model_dump_json()}",
                        ),
                    ]
                )
            )
            review = MethodologyReview.model_validate(review)
            report = report.model_copy(
                update={
                    "issues": [*report.issues, *review.issues],
                    "summary": f"{report.summary} Methodology review: {review.summary}",
                }
            )

    return {
        "critic_report": report,
        "trace": [trace_event("critic", f"Deterministic verdict: {report.verdict}.")],
    }
