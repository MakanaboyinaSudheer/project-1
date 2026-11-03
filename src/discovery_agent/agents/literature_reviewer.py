"""Literature Reviewer node. Owner: Member 2."""

from discovery_agent.agents.prompts import LITERATURE_REVIEW_PROMPT, SUB_QUERY_PROMPT
from discovery_agent.graph.state import ResearchState
from discovery_agent.llm import get_llm
from discovery_agent.retrieval.retriever import retrieve


def literature_reviewer_node(state: ResearchState) -> dict:
    """Retrieve literature for state['topic']; return `literature` + `literature_summary`."""
    topic = state.get("research_topic", "")
    llm = get_llm()

    query_chain = SUB_QUERY_PROMPT | llm
    _sub_queries = query_chain.invoke({"topic": topic})

    retrieved_docs = retrieve(query=topic, k=10, peer_reviewed_only=True)
    context_text = "\n\n".join([str(doc) for doc in retrieved_docs])

    chain = LITERATURE_REVIEW_PROMPT | llm
    response = chain.invoke({"topic": topic, "context": context_text})

    return {
        "literature_context": retrieved_docs,
        "literature_summary": response.content,
    }
