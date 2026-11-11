from unittest.mock import patch

from discovery_agent.llm import get_llm


def test_anthropic_models_get_no_temperature_but_others_do():
    with patch("discovery_agent.llm.init_chat_model") as init:
        get_llm("anthropic:claude-sonnet-5-5")
        assert "temperature" not in init.call_args.kwargs
        get_llm("ollama:llama3.1", temperature=0.0)
        assert init.call_args.kwargs["temperature"] == 0.0
