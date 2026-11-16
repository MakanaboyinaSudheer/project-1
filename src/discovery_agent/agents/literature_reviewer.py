"""Literature Reviewer node: expand the topic, retrieve evidence, and summarize gaps."""

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
    """Retrieve relevant literature and produce a concise review for the research topic."""
    topic = (state.get("research_topic") or state.get("topic") or "").strip()
    if not topic:
        raise ValueError("Cannot review literature without a research topic")

    config = {"callbacks": [CostTracker(run_id=state.get("thread_id", "local-run"))]}
    llm = get_llm()
    expanded = (SUB_QUERY_PROMPT | llm.with_structured_output(SubQueries)).invoke(
        {"topic": topic},
        config=config,
    )
    sub_queries = SubQueries.model_validate(expanded).queries
    queries = list(dict.fromkeys([topic, *(query.strip() for query in sub_queries if query.strip())]))

    general: dict[str, RetrievedChunk] = {}
    peer_reviewed: dict[str, RetrievedChunk] = {}
    peer_reviewed_only = state.get("peer_reviewed_only", False)
    for query in queries:
        searches = (
            [(peer_reviewed, True)]
            if peer_reviewed_only
            else [
                (general, False),
                (peer_reviewed, True),
            ]
        )
        for bucket, peer_only in searches:
            limit = 3 if peer_only else 6
            for chunk in retrieve(query=query, k=limit, peer_reviewed_only=peer_only):
                if chunk.id not in bucket or chunk.score > bucket[chunk.id].score:
                    bucket[chunk.id] = chunk

    def top(bucket: dict[str, RetrievedChunk], limit: int) -> list[RetrievedChunk]:
        return sorted(bucket.values(), key=lambda chunk: chunk.score, reverse=True)[:limit]

    literature = top(peer_reviewed, MAX_CHUNKS // 2)
    literature.extend(chunk for chunk in top(general, MAX_CHUNKS) if chunk.id not in peer_reviewed)
    literature = literature[:MAX_CHUNKS]
    if not literature:
        raise RuntimeError(
            f"No literature indexed for {topic!r}. Run `discovery-agent ingest --query ...` first."
        )

    context = "\n\n".join(
        f"[{chunk.paper_id}] {chunk.metadata.get('title', '')}\n{chunk.text}" for chunk in literature
    )
    response = (LITERATURE_REVIEW_PROMPT | llm).invoke(
        {"topic": topic, "context": context},
        config=config,
    )
    paper_count = len({chunk.paper_id for chunk in literature})
    return {
        "literature": literature,
        "literature_context": literature,
        "literature_summary": str(response.content),
        "trace": [
            trace_event(
                "literature_reviewer",
                f"{len(queries)} queries retrieved {len(literature)} chunks from {paper_count} papers.",
            )
        ],
    }


review_literature = literature_reviewer_node
