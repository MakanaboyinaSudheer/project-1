"""Smoke tests: package imports and shared contracts are valid. Must always pass on main/develop."""

import importlib

import pytest

MODULES = [
    "discovery_agent.config",
    "discovery_agent.llm",
    "discovery_agent.schemas",
    "discovery_agent.graph.state",
    "discovery_agent.graph.workflow",
    "discovery_agent.ingestion.pipeline",
    "discovery_agent.retrieval.vector_store",
    "discovery_agent.agents.hypothesis_generator",
    "discovery_agent.agents.critic",
    "discovery_agent.sandbox.runner",
    "discovery_agent.sandbox.feedback",
    "discovery_agent.memory.manager",
    "discovery_agent.synthesis.latex",
    "discovery_agent.evaluation.grader",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name: str) -> None:
    importlib.import_module(name)


def test_settings_defaults() -> None:
    from discovery_agent.config import Settings

    s = Settings(_env_file=None, arxiv_categories="cs.LG, cs.AI")
    assert s.arxiv_category_list == ["cs.LG", "cs.AI"]
    assert s.max_debug_iterations > 0


def test_hypothesis_schema_requires_citations() -> None:
    from pydantic import ValidationError

    from discovery_agent.schemas import Hypothesis

    with pytest.raises(ValidationError):
        Hypothesis(statement="x", rationale="y", test_plan="z", expected_outcome="w")
