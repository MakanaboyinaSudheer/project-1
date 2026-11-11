"""Literature Reviewer node: retrieve evidence from Milvus and summarise research gaps."""

from discovery_agent.agents.prompts import LITERATURE_REVIEW_PROMPT
from discovery_agent.graph.state import ResearchState, trace_event
from discovery_agent.llm import get_llm
from discovery_agent.retrieval.retriever import retrieve
from discovery_agent.telemetry.costs import CostTracker


def literature_reviewer_node(state: ResearchState) -> dict:
    """Retrieve literature for state['topic']; return `literature` + `literature_summary`."""
    topic = state["topic"]
    literature = retrieve(query=topic, k=10, peer_reviewed_only=state.get("peer_reviewed_only", False))
    if not literature:
        raise RuntimeError(
            f"No literature indexed for {topic!r}. Run `discovery-agent ingest --query ...` first."
        )

    context = "\n\n".join(
        f"[{chunk.paper_id}] {chunk.metadata.get('title', '')}\n{chunk.text}" for chunk in literature
    )
    tracker = CostTracker(run_id=state.get("thread_id", "local-run"))
    response = (LITERATURE_REVIEW_PROMPT | get_llm()).invoke(
        {"topic": topic, "context": context}, config={"callbacks": [tracker]}
    )
    papers = len({chunk.paper_id for chunk in literature})
    return {
        "literature": literature,
        "literature_summary": str(response.content),
        "trace": [
            trace_event("literature_reviewer", f"Retrieved {len(literature)} chunks from {papers} papers.")
        ],
    }


review_literature = literature_reviewer_node
