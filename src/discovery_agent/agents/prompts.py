"""Prompt library for discovery agents."""

from langchain_core.prompts import ChatPromptTemplate

LITERATURE_REVIEW_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert AI research scientist. Synthesize the provided "
            "literature into a concise foundational review. Identify key findings "
            "and critical research gaps based on the context.",
        ),
        ("user", "Topic: {topic}\n\nLiterature Context:\n{context}"),
    ]
)

SUB_QUERY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an AI research assistant. Given a main research topic, "
            "generate 2 to 4 specific sub-queries to search a literature database.",
        ),
        ("user", "Main Topic: {topic}"),
    ]
)
