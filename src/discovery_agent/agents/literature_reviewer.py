"""Literature Reviewer node. Owner: Member 2 (Week 1)."""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from discovery_agent.graph.state import ResearchState, trace_event
from discovery_agent.llm import get_llm
from discovery_agent.retrieval.retriever import retrieve

_QUERY_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "Generate exactly three concise, distinct literature search queries for the topic."),
        ("user", "Topic: {topic}\nReturn one query per line, without numbering."),
    ]
)
_SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Synthesize only the supplied sources. Identify established findings, disagreements, "
            "and explicit evidence gaps. Cite claims with the supplied paper IDs.",
        ),
        ("user", "Research question: {topic}\n\nSources:\n{sources}"),
    ]
)


def literature_reviewer_node(state: ResearchState) -> dict:
    """Retrieve literature for state['topic']; return `literature` + `literature_summary`."""
    topic = state.get("topic", "").strip()
    if not topic:
        raise ValueError("Research topic cannot be empty")
    model = get_llm()
    expanded = (_QUERY_PROMPT | model | StrOutputParser()).invoke({"topic": topic})
    queries = [topic]
    for line in expanded.splitlines():
        query = line.strip().lstrip("-*0123456789. ").strip()
        if query and query.casefold() not in {item.casefold() for item in queries}:
            queries.append(query)
        if len(queries) == 4:
            break
    if len(queries) < 3:
        raise ValueError("Literature query expansion must produce at least two distinct sub-queries")

    literature = []
    seen_ids = set()
    for query in queries:
        for chunk in retrieve(query, k=4):
            if chunk.id not in seen_ids:
                seen_ids.add(chunk.id)
                literature.append(chunk)
    if not literature:
        raise ValueError(f"No literature was retrieved for topic {topic!r}")
    sources = "\n\n".join(
        f"[{chunk.paper_id}] {chunk.metadata.get('title', chunk.paper_id)}: {chunk.text}"
        for chunk in literature
    )
    summary = (_SUMMARY_PROMPT | model | StrOutputParser()).invoke({"topic": topic, "sources": sources})
    if not summary.strip():
        raise ValueError("Literature reviewer returned an empty summary")
    return {
        "literature": literature,
        "literature_summary": summary.strip(),
        "trace": [
            trace_event(
                "literature_reviewer",
                f"Retrieved {len(literature)} unique chunks from {len(queries)} queries.",
            )
        ],
    }
