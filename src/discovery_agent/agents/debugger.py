"""Self-debugging node: rewrites failed code from stderr + attempt history. Owner: Member 3 (Week 2)."""

from langchain_core.prompts import ChatPromptTemplate

from discovery_agent.graph.state import ResearchState, trace_event
from discovery_agent.llm import get_llm
from discovery_agent.sandbox.feedback import choose_strategy, classify_error

_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Repair the failed experiment while preserving the hypothesis and test plan. "
            "Use the selected repair strategy. The code runs without network in a constrained Docker "
            "container; all outputs must go under /output and valid metrics must be written to "
            "/output/metrics.json. Return only Python source code.",
        ),
        (
            "user",
            "Strategy: {strategy}\nError class: {error_class}\nHypothesis: {hypothesis}\n"
            "Current code:\n{code}\n\nstderr:\n{stderr}\n\n"
            "stdout:\n{stdout}\nAttempt history:\n{history}",
        ),
    ]
)


def debugger_node(state: ResearchState) -> dict:
    """Use the previous attempts (code, stderr, reward) to produce corrected `code`;
    increment `debug_iteration`."""
    attempts = state.get("attempts", [])
    if not attempts:
        raise ValueError("Cannot debug without a failed code attempt")
    last = attempts[-1]
    if last.result.exit_code == 0 and not last.result.timed_out:
        raise ValueError("Cannot debug a successful code attempt")
    strategy = choose_strategy()
    hypothesis = state.get("hypothesis")
    response = (_PROMPT | get_llm()).invoke(
        {
            "strategy": strategy,
            "error_class": "timeout" if last.result.timed_out else classify_error(last.result.stderr),
            "hypothesis": hypothesis.model_dump_json() if hypothesis else "",
            "code": last.code,
            "stderr": last.result.stderr[-4000:],
            "stdout": last.result.stdout[-2000:],
            "history": "\n".join(
                f"Attempt {attempt.iteration}: reward={attempt.reward:.2f}; "
                f"exit={attempt.result.exit_code}; stderr={attempt.result.stderr[-300:]}"
                for attempt in attempts[-5:]
            ),
        }
    )
    code = response.content if isinstance(response.content, str) else ""
    if code.startswith("```"):
        code = code.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    if not code.strip():
        raise ValueError("Debugger returned no replacement code")
    return {
        "code": code.strip(),
        "debug_iteration": state.get("debug_iteration", 0) + 1,
        "debug_strategy": strategy,
        "trace": [trace_event("debugger", f"Rewrote failed attempt using {strategy} strategy.")],
    }
