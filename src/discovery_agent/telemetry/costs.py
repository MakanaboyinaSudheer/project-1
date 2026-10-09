"""Token and cost tracking for the supervisor dashboard. Owner: Member 4."""

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from discovery_agent.schemas import CostEvent


class CostTracker(BaseCallbackHandler):
    """Collect a CostEvent for every completed LLM call."""

    def __init__(self) -> None:
        self.events: list[CostEvent] = []

    def on_llm_end(self, response: LLMResult, **kwargs: object) -> None:
        """Capture provider-reported usage without depending on provider-specific callbacks."""
        output = response.llm_output or {}
        usage = output.get("token_usage") or output.get("usage") or output
        metadata = kwargs.get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        invocation_params = kwargs.get("invocation_params")
        invocation_params = invocation_params if isinstance(invocation_params, dict) else {}

        self.events.append(
            CostEvent(
                node=str(
                    metadata.get("langgraph_node")
                    or metadata.get("node")
                    or kwargs.get("name")
                    or "llm"
                ),
                model=str(
                    output.get("model_name")
                    or output.get("model")
                    or output.get("model_id")
                    or invocation_params.get("model")
                    or invocation_params.get("model_name")
                    or "unknown"
                ),
                input_tokens=usage.get("prompt_tokens", usage.get("input_tokens", 0)),
                output_tokens=usage.get("completion_tokens", usage.get("output_tokens", 0)),
            )
        )
