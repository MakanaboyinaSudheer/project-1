"""Generate and execute experiment code in the isolated sandbox."""

import re
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate

from discovery_agent.config import settings
from discovery_agent.graph.state import ResearchState, trace_event
from discovery_agent.llm import get_llm
from discovery_agent.sandbox.feedback import compute_reward
from discovery_agent.sandbox.runner import run_code
from discovery_agent.schemas import CodeAttempt
from discovery_agent.telemetry.costs import CostTracker

_SAFE_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_CODE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Write one self-contained Python experiment script. It runs in an offline, "
            "resource-limited container with numpy, pandas, scipy, scikit-learn, statsmodels, "
            "matplotlib, and seaborn available. Do not use network access or write outside /output. "
            "Save valid JSON to /output/metrics.json with a named test, p_value (or p_values), "
            "effect_size (or effect_sizes), and sample_size (or sample_sizes). Include finite "
            "numeric values and report methods truthfully. Return only executable Python code.",
        ),
        (
            "human",
            "Hypothesis: {statement}\nRationale: {rationale}\nTest plan: {test_plan}\n"
            "Expected outcome: {expected_outcome}",
        ),
    ]
)


def _extract_code(content: object) -> str:
    if isinstance(content, str):
        text = content.strip()
    elif isinstance(content, list):
        text = "\n".join(
            str(block["text"])
            for block in content
            if isinstance(block, dict) and isinstance(block.get("text"), str)
        ).strip()
    else:
        text = ""

    fenced = re.fullmatch(r"```(?:python|py)?\s*\n?(.*?)\n?```", text, flags=re.DOTALL | re.IGNORECASE)
    return fenced.group(1).strip() if fenced else text


def code_executor_node(state: ResearchState) -> dict:
    """Generate (unless supplied by the debugger), execute, and record one code attempt."""
    hypothesis = state.get("hypothesis")
    if hypothesis is None:
        raise ValueError("Cannot execute an experiment without a hypothesis")

    run_id = state.get("thread_id", "local-run")
    if not isinstance(run_id, str) or not _SAFE_RUN_ID.fullmatch(run_id):
        raise ValueError("thread_id must contain only letters, numbers, dots, underscores, and hyphens")

    code = state.get("code", "").strip()
    if not code:
        tracker = CostTracker(run_id=run_id)
        response = (_CODE_PROMPT | get_llm()).invoke(
            hypothesis.model_dump(),
            config={"callbacks": [tracker]},
        )
        code = _extract_code(response.content)
        if not code:
            raise ValueError("Code generation returned no executable source")

    attempts = state.get("attempts", [])
    iteration = len(attempts) + 1
    attempt_dir = Path(settings.runs_dir) / run_id / f"attempt_{iteration}"
    result = run_code(code, attempt_dir)
    reward = compute_reward(result, iteration - 1)
    attempt = CodeAttempt(iteration=iteration, code=code, result=result, reward=reward)
    return {
        "code": code,
        "attempts": [attempt],
        "trace": [
            trace_event(
                "code_executor",
                f"Attempt {iteration} exited {result.exit_code}; reward {reward:.2f}.",
            )
        ],
    }
