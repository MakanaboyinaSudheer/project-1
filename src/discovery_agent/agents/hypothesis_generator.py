"""Hypothesis Generator node, grounded in retrieved literature via RAG."""

from langchain_core.prompts import ChatPromptTemplate

from discovery_agent.graph.state import ResearchState, trace_event
from discovery_agent.llm import get_llm
from discovery_agent.schemas import Citation, Hypothesis
from discovery_agent.telemetry.costs import CostTracker

MAX_RETRIES = 3

HYPOTHESIS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert AI researcher. Formulate one novel, computationally testable hypothesis "
            "grounded in the literature. Cite ONLY paper_ids from the evidence list and quote them "
            "verbatim. The test_plan must be runnable offline in Python on simulated or synthetic data. "
            "If a previous hypothesis and critic feedback are given, refine it to address the feedback.",
        ),
        (
            "user",
            "Topic: {topic}\n\nLiterature summary:\n{summary}\n\nEvidence:\n{evidence}\n\n"
            "Previous hypothesis:\n{previous_hypothesis}\n\nCritic feedback:\n{feedback}\n\n"
            "{validation_errors}",
        ),
    ]
)


def _ground(citations: list[Citation], sources: dict[str, list]) -> tuple[list[Citation], list[str]]:
    """Keep only citations of retrieved papers; replace non-verbatim quotes with real source text."""
    grounded, invalid = [], []
    for citation in citations:
        chunks = sources.get(citation.paper_id)
        if not chunks:
            invalid.append(citation.paper_id)
            continue
        quote = citation.quote.strip()
        if not quote or not any(quote in chunk.text for chunk in chunks):
            quote = chunks[0].text[:300]
        title = chunks[0].metadata.get("title") or citation.title
        grounded.append(Citation(paper_id=citation.paper_id, title=title, quote=quote))
    return grounded, invalid


def hypothesis_generator_node(state: ResearchState) -> dict:
    """Produce a Hypothesis whose citations all reference chunks in state['literature']."""
    sources: dict[str, list] = {}
    for chunk in state.get("literature", []):
        sources.setdefault(chunk.paper_id, []).append(chunk)
    if not sources:
        raise ValueError("Cannot generate a hypothesis without retrieved literature")

    previous = state.get("hypothesis")
    report = state.get("critic_report")
    inputs = {
        "topic": state["topic"],
        "summary": state.get("literature_summary", ""),
        "evidence": "\n\n".join(
            f"paper_id={pid} | {chunks[0].metadata.get('title', '')}\n{chunks[0].text[:1500]}"
            for pid, chunks in sources.items()
        ),
        "previous_hypothesis": previous.model_dump_json() if previous else "None",
        "feedback": f"{report.summary}\nIssues: {report.issues}" if report else "None",
        "validation_errors": "",
    }
    tracker = CostTracker(run_id=state.get("thread_id", "local-run"))
    chain = HYPOTHESIS_PROMPT | get_llm().with_structured_output(Hypothesis)

    for _ in range(MAX_RETRIES):
        hypothesis = Hypothesis.model_validate(chain.invoke(inputs, config={"callbacks": [tracker]}))
        citations, invalid = _ground(hypothesis.citations, sources)
        if citations and not invalid:
            hypothesis = hypothesis.model_copy(update={"citations": citations})
            refinement = state.get("refinement_iteration", 0)
            return {
                "hypothesis": hypothesis,
                "hypotheses": [hypothesis],
                "trace": [
                    trace_event(
                        "hypothesis_generator",
                        f"Hypothesis v{refinement + 1} with {len(citations)} verified citations.",
                    )
                ],
            }
        inputs["validation_errors"] = (
            f"Your previous answer cited unknown paper_ids {invalid} or none at all. "
            f"Use only: {list(sources)}."
        )
    raise ValueError(f"Hypothesis generator failed to produce grounded citations in {MAX_RETRIES} tries")


generate_hypothesis = hypothesis_generator_node
