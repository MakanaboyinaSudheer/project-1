"""Code Executor node: writes experiment code and runs it in the sandbox. Owner: Member 3 (Week 1-2)."""

from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate

from discovery_agent.config import settings
from discovery_agent.graph.state import ResearchState, trace_event
from discovery_agent.llm import get_llm
from discovery_agent.sandbox.feedback import compute_reward, update_strategy
from discovery_agent.sandbox.runner import run_code
from discovery_agent.schemas import CodeAttempt

_CODE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Write one self-contained Python experiment script. It must run without network access, "
            "use only the provided scientific hypothesis and test plan, and write valid statistical "
            "metrics to /output/metrics.json. Include a named test, p_value, effect_size and sample_size. "
            "Return only executable Python source code.",
        ),
        (
            "user",
            "Hypothesis: {statement}\nRationale: {rationale}\nTest plan: {test_plan}\n"
            "Expected outcome: {expected_outcome}",
        ),
    ]
)


def code_executor_node(state: ResearchState) -> dict:
    """Generate code for state['hypothesis'].test_plan, run via sandbox.runner, append a CodeAttempt.
    Generated code must write its results to /output/metrics.json."""
    hypothesis = state.get("hypothesis")
    if hypothesis is None:
        raise ValueError("Cannot execute an experiment without a hypothesis")
    code = state.get("code", "").strip()
    if not code:
        response = (_CODE_PROMPT | get_llm()).invoke(hypothesis.model_dump())
        code = response.content if isinstance(response.content, str) else ""
        if code.startswith("```"):
            code = code.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        if not code:
            raise ValueError("Code generation returned no executable source")
    iteration = len(state.get("attempts", [])) + 1
    attempt_dir = Path(settings.runs_dir) / state.get("thread_id", "local-run") / f"attempt_{iteration}"
    result = run_code(code, attempt_dir)
    reward = compute_reward(result, iteration - 1)
    previous_strategy = state.get("debug_strategy")
    if previous_strategy and (result.exit_code != 0 or result.timed_out):
        update_strategy(previous_strategy, reward)
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
