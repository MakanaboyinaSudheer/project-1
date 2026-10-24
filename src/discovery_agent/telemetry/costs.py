"""Token and cost tracking for the supervisor dashboard."""

import logging
import math
import re
import threading
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from discovery_agent.config import settings
from discovery_agent.schemas import CostEvent

logger = logging.getLogger(__name__)

# USD per million input and output tokens. Update these estimates when provider
# pricing changes; local Ollama models do not incur provider token fees.
MODEL_PRICING_USD_PER_MILLION_TOKENS: dict[str, tuple[float, float]] = {
    "claude-3-5-sonnet": (3.0, 15.0),
    "claude-3-7-sonnet": (3.0, 15.0),
    "claude-sonnet-4": (3.0, 15.0),
    "claude-sonnet-4-5": (3.0, 15.0),
    "claude-sonnet-5-5": (3.0, 15.0),
    "gpt-4o": (2.5, 10.0),
    "gpt-4o-mini": (0.15, 0.6),
    "llama3.1": (0.0, 0.0),
    "llava": (0.0, 0.0),
}
_PRICING_SCALE = 1_000_000
_SAFE_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")


class CostTracker(BaseCallbackHandler):
    """Collect and persist a ``CostEvent`` for every completed LLM call."""

    def __init__(
        self,
        run_id: str | None = None,
        runs_dir: str | Path | None = None,
        pricing: Mapping[str, tuple[float, float]] | None = None,
    ) -> None:
        self.run_id = str(uuid.uuid4()) if run_id is None else run_id
        if not _SAFE_RUN_ID.fullmatch(self.run_id):
            raise ValueError("run_id must contain only letters, numbers, dots, underscores, and hyphens")

        self.run_dir = Path(runs_dir or settings.runs_dir) / self.run_id
        self.costs_path = self.run_dir / "costs.jsonl"
        configured_pricing = MODEL_PRICING_USD_PER_MILLION_TOKENS if pricing is None else pricing
        self.pricing = {model.casefold(): rates for model, rates in configured_pricing.items()}
        for model, rates in self.pricing.items():
            if len(rates) != 2 or any(
                not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate < 0 for rate in rates
            ):
                raise ValueError(f"pricing for {model!r} must be non-negative (input, output) rates")

        self.events: list[CostEvent] = []
        self._lock = threading.Lock()

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Capture provider-reported usage, estimate cost, and append a JSONL record."""
        output = response.llm_output or {}
        raw_usage = output.get("token_usage") or output.get("usage") or output
        usage = raw_usage if isinstance(raw_usage, Mapping) else {}
        invocation = kwargs.get("invocation_params")
        invocation = invocation if isinstance(invocation, Mapping) else {}
        metadata = kwargs.get("metadata")
        metadata = metadata if isinstance(metadata, Mapping) else {}

        model = str(
            output.get("model_name")
            or output.get("model")
            or output.get("model_id")
            or invocation.get("model")
            or invocation.get("model_name")
            or "unknown"
        )
        input_tokens = int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)
        output_tokens = int(usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0)
        if input_tokens < 0 or output_tokens < 0:
            raise ValueError("LLM token usage cannot be negative")

        pricing_key = model.rsplit(":", maxsplit=1)[-1].casefold()
        rates = self.pricing.get(pricing_key)
        if rates is None:
            rates = next(
                (
                    self.pricing[key]
                    for key in sorted(self.pricing, key=len, reverse=True)
                    if pricing_key.startswith(f"{key}-")
                ),
                None,
            )
        if rates is None:
            logger.warning("No token pricing configured for model %r; recording USD cost as 0", model)
            usd = 0.0
        else:
            usd = round(
                (input_tokens * rates[0] + output_tokens * rates[1]) / _PRICING_SCALE,
                10,
            )

        event = CostEvent(
            node=str(metadata.get("langgraph_node") or metadata.get("node") or kwargs.get("name") or "llm"),
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            usd=usd,
            at=datetime.now(UTC),
        )

        with self._lock:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            with self.costs_path.open("a", encoding="utf-8") as stream:
                stream.write(event.model_dump_json() + "\n")
            self.events.append(event)
