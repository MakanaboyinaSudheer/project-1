"""Self-debugging node: rewrites failed code from stderr + attempt history."""

import re
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate

from discovery_agent.config import settings
from discovery_agent.graph.state import ResearchState, trace_event
from discovery_agent.llm import get_llm
from discovery_agent.sandbox.bandit import select_strategy
from discovery_agent.sandbox.feedback import classify_error
from discovery_agent.telemetry.costs import CostTracker

_SAFE_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_STRATEGY_GUIDANCE = {
    "minimal_patch": "Make the smallest targeted changes needed to fix the error.",
    "rewrite": "Rewrite the script cleanly while preserving the experiment's intent.",
    "simplify": "Simplify the experiment to the smallest valid analysis and output.",
}
_DEBUG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Repair a failed Python experiment. It runs offline in a resource-limited "
            "container with numpy, pandas, scipy, scikit-learn, statsmodels, matplotlib, "
            "and seaborn. Do not access the network or write outside /output. Save valid "
            "JSON to /output/metrics.json with a named test, finite p_value (or p_values), "
            "effect_size (or effect_sizes), and sample_size (or sample_sizes). "
            "Use the requested repair strategy: {strategy_guidance}. "
            "Return only executable Python code.",
        ),
        (
            "human",
            "Hypothesis: {hypothesis}\nError class: {error_class}\n"
            "Previous stderr (truncated):\n{stderr}\n"
            "Previous code:\n```python\n{code}\n```\n"
            "Earlier attempt outcomes:\n{history}",
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
    match = re.fullmatch(r"```(?:python|py)?\s*\n?(.*?)\n?```", text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else text


def debugger_node(state: ResearchState) -> dict:
    """Use the bandit-selected strategy and failed attempt history to produce corrected code."""
    attempts = state.get("attempts", [])
    if not attempts:
        raise ValueError("Cannot debug without a previous code attempt")
    previous = attempts[-1]
    if previous.result.exit_code == 0 and not previous.result.timed_out:
        raise ValueError("Cannot debug a successful code attempt")

    hypothesis = state.get("hypothesis")
    if hypothesis is None:
        raise ValueError("Cannot debug an experiment without a hypothesis")
    run_id = state.get("thread_id", "local-run")
    if not isinstance(run_id, str) or not _SAFE_RUN_ID.fullmatch(run_id):
        raise ValueError("thread_id must contain only letters, numbers, dots, underscores, and hyphens")

    strategy = select_strategy(Path(settings.runs_dir) / "bandit.json")
    history = "\n".join(
        f"Attempt {attempt.iteration}: reward={attempt.reward:.2f}, "
        f"exit_code={attempt.result.exit_code}, timed_out={attempt.result.timed_out}"
        for attempt in attempts[-5:]
    )
    tracker = CostTracker(run_id=run_id)
    error_class = "timeout" if previous.result.timed_out else classify_error(previous.result.stderr)
    response = (_DEBUG_PROMPT | get_llm()).invoke(
        {
            "hypothesis": hypothesis.statement,
            "error_class": error_class,
            "stderr": previous.result.stderr[-8000:],
            "code": previous.code,
            "history": history,
            "strategy_guidance": _STRATEGY_GUIDANCE[strategy],
        },
        config={"callbacks": [tracker]},
    )
    code = _extract_code(response.content)
    if not code:
        raise ValueError("Debugger returned no executable source")

    iteration = state.get("debug_iteration", 0) + 1
    return {
        "code": code,
        "debug_iteration": iteration,
        "debug_strategy": strategy,
        "trace": [trace_event("debugger", f"Selected {strategy} to repair {error_class} failure.")],
    }
