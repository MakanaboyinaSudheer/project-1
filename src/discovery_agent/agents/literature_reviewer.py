"""Literature Reviewer node: expand the topic into sub-queries, retrieve evidence, summarise gaps."""

from pydantic import BaseModel, Field

from discovery_agent.agents.prompts import LITERATURE_REVIEW_PROMPT, SUB_QUERY_PROMPT
from discovery_agent.graph.state import ResearchState, trace_event
from discovery_agent.llm import get_llm
from discovery_agent.retrieval.retriever import retrieve
from discovery_agent.schemas import RetrievedChunk
from discovery_agent.telemetry.costs import CostTracker

MAX_CHUNKS = 12


class SubQueries(BaseModel):
    queries: list[str] = Field(description="2 to 4 literature search queries")


def literature_reviewer_node(state: ResearchState) -> dict:
    """Retrieve literature for state['topic']; return `literature` + `literature_summary`."""
    topic = state["topic"]
    config = {"callbacks": [CostTracker(run_id=state.get("thread_id", "local-run"))]}
    expanded = (SUB_QUERY_PROMPT | get_llm().with_structured_output(SubQueries)).invoke(
        {"topic": topic}, config=config
    )
    queries = [topic, *[q for q in SubQueries.model_validate(expanded).queries[:4] if q.strip()]]

    seen: dict[str, RetrievedChunk] = {}
    for query in queries:
        for chunk in retrieve(query=query, k=6, peer_reviewed_only=state.get("peer_reviewed_only", False)):
            if chunk.id not in seen or chunk.score > seen[chunk.id].score:
                seen[chunk.id] = chunk
    literature = sorted(seen.values(), key=lambda c: c.score, reverse=True)[:MAX_CHUNKS]
    if not literature:
        raise RuntimeError(
            f"No literature indexed for {topic!r}. Run `discovery-agent ingest --query ...` first."
        )

    context = "\n\n".join(
        f"[{chunk.paper_id}] {chunk.metadata.get('title', '')}\n{chunk.text}" for chunk in literature
    )
    response = (LITERATURE_REVIEW_PROMPT | get_llm()).invoke(
        {"topic": topic, "context": context}, config=config
    )
    papers = len({chunk.paper_id for chunk in literature})
    return {
        "literature": literature,
        "literature_summary": str(response.content),
        "trace": [
            trace_event(
                "literature_reviewer",
                f"{len(queries)} queries retrieved {len(literature)} chunks from {papers} papers.",
            )
        ],
    }


review_literature = literature_reviewer_node
