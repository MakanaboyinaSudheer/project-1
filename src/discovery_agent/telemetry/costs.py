"""Token/cost tracking, written to runs/<thread_id>/costs.jsonl for the dashboard. Owner: Member 3."""

from langchain_core.callbacks import BaseCallbackHandler


class CostTracker(BaseCallbackHandler):
    """LangChain callback: record a CostEvent for every LLM call."""
