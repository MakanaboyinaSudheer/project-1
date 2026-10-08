"""Provider-agnostic chat model factory.

All agents get their model through `get_llm()` so the provider is switched only via
LLM_MODEL in .env (e.g. "anthropic:claude-sonnet-5-5", "ollama:llama3.1").
"""

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from discovery_agent.config import settings


def get_llm(model: str | None = None, temperature: float | None = None) -> BaseChatModel:
    spec = model or settings.llm_model
    kwargs: dict = {"temperature": settings.llm_temperature if temperature is None else temperature}
    if spec.startswith("ollama:"):
        kwargs["base_url"] = settings.ollama_base_url
    return init_chat_model(spec, **kwargs)


def get_vlm() -> BaseChatModel:
    """Vision-language model used for figure/chart parsing (Week 3)."""
    return get_llm(settings.vlm_model, temperature=0.0)
