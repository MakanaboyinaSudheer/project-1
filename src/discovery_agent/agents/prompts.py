"""Prompt library for discovery agents."""

from langchain_core.prompts import ChatPromptTemplate

LITERATURE_REVIEW_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert AI research scientist. Synthesize the provided "
            "literature into a concise foundational review. You must identify "
            "key findings and critical research gaps based on the context.",
        ),
        ("user", "Topic: {topic}\n\nLiterature Context:\n{context}"),
    ]
)

SUB_QUERY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Generate 2 to 4 short, specific literature-search queries that together cover the "
            "research topic (methods, populations, outcomes, competing explanations).",
        ),
        ("user", "Topic: {topic}"),
    ]
)
