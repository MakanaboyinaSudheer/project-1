import math

import pytest

from discovery_agent.agents.critic import critic_node, statistical_checks
from discovery_agent.graph.state import ResearchState
from discovery_agent.schemas import CodeAttempt, ExecutionResult


def _result(metrics: dict, *, exit_code: int = 0, timed_out: bool = False) -> ExecutionResult:
    return ExecutionResult(
        exit_code=exit_code,
        stdout="",
        stderr="",
        duration_s=0.1,
        timed_out=timed_out,
        metrics=metrics,
    )


def test_statistical_checks_parses_and_classifies_named_metrics() -> None:
    report = statistical_checks(
        _result(
            {
                "p_values": {"primary": 0.01, "secondary": 0.2},
                "tests": {"primary": "welch_t_test", "secondary": "mann_whitney_u"},
                "confidence_intervals": {"primary": [0.2, 0.8]},
                "effect_sizes": {"primary": 0.5, "secondary": 0.2},
                "sample_sizes": {"primary": 40, "secondary": 35},
            }
        )
    )

    assert report.verdict == "significant"
    assert report.p_values == {"primary": 0.02, "secondary": 0.2}
    assert report.confidence_intervals == {"primary": (0.2, 0.8)}
    assert report.effect_sizes == {"primary": 0.5, "secondary": 0.2}
    assert report.issues == []


def test_statistical_checks_supports_single_metric_fields() -> None:
    report = statistical_checks(
        _result(
            {
                "p_value": 0.05,
                "test": "one_sample_t_test",
                "confidence_interval": [-1, 1],
                "effect_size": 0.0,
                "sample_size": 30,
            }
        )
    )

    assert report.verdict == "not_significant"
    assert report.p_values == {"primary": 0.05}
    assert report.confidence_intervals == {"primary": (-1.0, 1.0)}
    assert report.effect_sizes == {"primary": 0.0}


@pytest.mark.parametrize(
    ("metrics", "issue_fragment"),
    [
        ({"p_values": {"primary": 1.1}}, "P-value"),
        ({"p_values": {"primary": math.nan}}, "P-value"),
        ({"p_values": []}, "`p_values`"),
        ({"p_value": 0.01, "confidence_interval": [2, 1]}, "lower bound"),
        ({"p_value": 0.01, "effect_size": math.inf}, "finite number"),
        ({}, "at least one valid p-value"),
    ],
)
def test_statistical_checks_rejects_invalid_metrics(metrics: dict, issue_fragment: str) -> None:
    report = statistical_checks(_result(metrics))

    assert report.verdict == "invalid"
    assert any(issue_fragment in issue for issue in report.issues)


@pytest.mark.parametrize("alpha", [0, 1, -0.1, math.nan, math.inf])
def test_statistical_checks_requires_valid_alpha(alpha: float) -> None:
    with pytest.raises(ValueError, match="alpha"):
        statistical_checks(_result({"p_value": 0.01}), alpha=alpha)


def test_statistical_checks_rejects_failed_or_timed_out_execution() -> None:
    metrics = {"p_value": 0.01, "test": "t_test", "effect_size": 0.5}
    assert statistical_checks(_result(metrics, exit_code=1)).verdict == "invalid"
    assert statistical_checks(_result(metrics, timed_out=True)).verdict == "invalid"


def test_critic_node_reports_missing_attempt() -> None:
    report = critic_node(ResearchState())["critic_report"]

    assert report.verdict == "invalid"
    assert report.issues == ["No execution attempts are available for review."]


def test_critic_node_reviews_latest_attempt() -> None:
    state = ResearchState(
        attempts=[
            CodeAttempt(
                iteration=1,
                code="",
                result=_result({"p_value": 0.01, "test": "t_test", "effect_size": 0.5}),
                reward=0.0,
            ),
            CodeAttempt(
                iteration=2,
                code="",
                result=_result({"p_value": 0.5, "test": "t_test", "effect_size": 0.5}),
                reward=0.0,
            ),
        ]
    )

    assert critic_node(state)["critic_report"].verdict == "not_significant"


def test_statistical_checks_applies_holm_bonferroni_before_verdict() -> None:
    report = statistical_checks(
        _result(
            {
                "p_values": {"first": 0.02, "second": 0.04, "third": 0.03},
                "tests": {
                    "first": "t_test",
                    "second": "t_test",
                    "third": "t_test",
                },
                "effect_sizes": {"first": 0.5, "second": 0.2, "third": 0.3},
            }
        )
    )

    assert report.verdict == "not_significant"
    assert report.p_values == {"first": 0.06, "second": 0.06, "third": 0.06}
    assert "Holm-Bonferroni-adjusted" in report.summary


def test_statistical_checks_flags_small_sample_sizes() -> None:
    report = statistical_checks(
        _result(
            {
                "p_value": 0.001,
                "test": "t_test",
                "effect_size": 0.8,
                "sample_size": 29,
            }
        )
    )

    assert report.verdict == "invalid"
    assert any("Small sample size" in issue for issue in report.issues)


@pytest.mark.parametrize(
    ("metadata", "issue_fragment"),
    [
        ({"effect_size": 0.4}, "missing a named statistical test"),
        ({"test": "t_test"}, "Effect size"),
    ],
)
def test_statistical_checks_requires_test_name_and_effect_size(metadata: dict, issue_fragment: str) -> None:
    report = statistical_checks(_result({"p_value": 0.01, **metadata}))

    assert report.verdict == "invalid"
    assert any(issue_fragment in issue for issue in report.issues)
