"""LangGraph wiring of the agents.

    literature_reviewer -> hypothesis_generator -> code_executor
                                                     |      ^
                                       (error)       v      | (rewrite, < MAX_DEBUG_ITERATIONS)
                                                   debugger-+
                                                     |
                                       (success)     v
                                                  critic -> synthesizer -> END
                         (not significant / invalid) |
                                                     +-> hypothesis_generator (refine)

Owner: Member 2. Node implementations live in discovery_agent.agents.*.
"""

from discovery_agent.graph.state import ResearchState

MAX_REFINEMENTS = 3


def build_graph():
    """Return the compiled StateGraph with a SQLite checkpointer (resumable runs)."""
    raise NotImplementedError("Week 1 - Member 2")


def route_after_execution(state: ResearchState) -> str:
    """'debug' if the last attempt failed and budget remains, 'critic' if it succeeded, else 'fail'."""
    raise NotImplementedError("Week 2 - Member 3")


def route_after_critic(state: ResearchState) -> str:
    """Route significant results to synthesis and retry other verdicts up to the refinement limit."""
    report = state.get("critic_report")
    if report is None:
        return "failed"
    if report.verdict == "significant":
        return "synthesize"
    return "refine" if state.get("refinement_iteration", 0) < MAX_REFINEMENTS else "failed"


def run_discovery(topic: str, thread_id: str | None = None) -> ResearchState:
    raise NotImplementedError("Week 1 - Member 2")
