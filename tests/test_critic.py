import math

import pytest
from scipy import stats

from discovery_agent.agents import critic
from discovery_agent.agents.critic import critic_node, statistical_checks
from discovery_agent.graph.state import ResearchState
from discovery_agent.graph.workflow import route_after_critic
from discovery_agent.schemas import CodeAttempt, CriticReport, ExecutionResult, Hypothesis


def _result(metrics: dict, *, exit_code: int = 0, timed_out: bool = False) -> ExecutionResult:
    return ExecutionResult(
        exit_code=exit_code,
        stdout="",
        stderr="",
        duration_s=0.1,
        timed_out=timed_out,
        metrics=metrics,
    )


def test_statistical_checks_adjusts_multiple_p_values_with_holm_bonferroni() -> None:
    report = statistical_checks(
        _result(
            {
                "p_values": {"first": 0.02, "second": 0.04, "third": 0.03},
                "tests": {"first": "t_test", "second": "t_test", "third": "t_test"},
                "effect_sizes": {"first": 0.5, "second": 0.2, "third": 0.3},
            }
        )
    )

    assert report.verdict == "not_significant"
    assert report.p_values == {"first": 0.06, "second": 0.06, "third": 0.06}
    assert "Holm-Bonferroni-adjusted" in report.summary


@pytest.mark.parametrize(
    ("metrics", "expected_issue"),
    [
        ({"p_value": 0.01, "effect_size": 0.4}, "missing a named statistical test"),
        ({"p_value": 0.01, "test": "t_test"}, "Effect size"),
        ({"p_value": math.nan, "test": "t_test", "effect_size": 0.2}, "finite number"),
        (
            {"p_value": 0.01, "test": "t_test", "effect_size": 0.2, "sample_size": 29},
            "Small sample size",
        ),
        (
            {
                "p_value": 0.01,
                "test": "t_test",
                "effect_size": 0.2,
                "confidence_interval": [2, 1],
            },
            "lower bound",
        ),
    ],
)
def test_statistical_checks_rejects_invalid_metrics(metrics: dict, expected_issue: str) -> None:
    report = statistical_checks(_result(metrics))

    assert report.verdict == "invalid"
    assert any(expected_issue in issue for issue in report.issues)


@pytest.mark.parametrize("alpha", [0, 1, -0.1, math.nan, math.inf])
def test_statistical_checks_requires_valid_alpha(alpha: float) -> None:
    with pytest.raises(ValueError, match="alpha"):
        statistical_checks(_result({"p_value": 0.01}), alpha=alpha)


@pytest.mark.parametrize(("exit_code", "timed_out"), [(1, False), (0, True)])
def test_statistical_checks_rejects_failed_execution(exit_code: int, timed_out: bool) -> None:
    report = statistical_checks(
        _result(
            {"p_value": 0.01, "test": "t_test", "effect_size": 0.5},
            exit_code=exit_code,
            timed_out=timed_out,
        )
    )

    assert report.verdict == "invalid"
    assert "Execution did not complete successfully." in report.issues


def test_statistical_checks_recomputes_welch_test_from_raw_data() -> None:
    control = [2, 3, 4, 5, 6, 8] * 6
    treatment = [4, 5, 7, 8, 10, 12] * 6
    expected = stats.ttest_ind(treatment, control, equal_var=False).pvalue
    metrics = {
        "tests": {"primary": "welch_t_test"},
        "raw_data": {"primary": {"control": control, "treatment": treatment}},
        "effect_sizes": {"primary": 0.5},
    }

    report = statistical_checks(_result(metrics))

    assert report.verdict == "significant"
    assert report.p_values["primary"] == pytest.approx(expected)
    assert len(report.confidence_intervals["primary"]) == 2
    assert report.issues == []
    assert "sample_sizes" not in metrics


def test_statistical_checks_flags_reported_p_value_that_disagrees_with_raw_data() -> None:
    report = statistical_checks(
        _result(
            {
                "p_value": 0.9,
                "test": "welch_t_test",
                "effect_size": 0.5,
                "raw_data": {
                    "primary": {
                        "control": [2, 3, 4, 5, 6, 8] * 6,
                        "treatment": [4, 5, 7, 8, 10, 12] * 6,
                    }
                },
            }
        )
    )

    assert report.verdict == "invalid"
    assert any("differs from the value recomputed" in issue for issue in report.issues)
    assert report.p_values["primary"] < 0.05


def test_critic_node_runs_methodology_review_without_changing_statistics(monkeypatch) -> None:
    class FakeModel:
        def with_structured_output(self, schema):
            assert schema.__name__ == "MethodologyReview"
            return self

        def invoke(self, messages):
            assert "Do not recalculate or change" in messages[0][1]
            return {"issues": ["Potential selection bias."], "summary": "Review flagged selection bias."}

    monkeypatch.setattr(critic, "get_llm", lambda: FakeModel())
    state = ResearchState(
        hypothesis=Hypothesis(
            statement="Treatment affects outcomes",
            rationale="Prior studies suggest an effect",
            citations=[],
            test_plan="Compare treatment and control groups",
            expected_outcome="Higher outcome for treatment",
        ),
        attempts=[
            CodeAttempt(
                iteration=1,
                code="",
                result=_result({"p_value": 0.01, "test": "t_test", "effect_size": 0.5}),
                reward=0.5,
            )
        ],
    )

    update = critic_node(state)

    assert update["critic_report"].verdict == "significant"
    assert update["critic_report"].p_values == {"primary": 0.01}
    assert update["critic_report"].issues == ["Potential selection bias."]
    assert update["trace"][0]["node"] == "critic"


def test_critic_node_without_attempt_returns_invalid_report() -> None:
    update = critic_node(ResearchState())

    assert update["critic_report"].verdict == "invalid"
    assert update["critic_report"].issues == ["No execution attempts are available for review."]


@pytest.mark.parametrize("verdict", ["not_significant", "invalid"])
def test_route_after_critic_allows_refinement_for_non_significant_results(verdict: str) -> None:
    state = ResearchState(
        critic_report=CriticReport(verdict=verdict),
        refinement_iteration=2,
    )

    assert route_after_critic(state) == "refine"


def test_route_after_critic_synthesizes_significant_results() -> None:
    state = ResearchState(critic_report=CriticReport(verdict="significant"))

    assert route_after_critic(state) == "synthesize"


def test_route_after_critic_stops_after_three_refinements() -> None:
    state = ResearchState(
        critic_report=CriticReport(verdict="not_significant"),
        refinement_iteration=3,
    )

    assert route_after_critic(state) == "failed"


def test_route_after_critic_fails_without_a_critic_report() -> None:
    assert route_after_critic(ResearchState()) == "failed"
