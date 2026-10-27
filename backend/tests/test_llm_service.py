import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError

import pytest

from backend.core.exceptions import AIServiceError
from backend.services.llm_service import LLMService


def settings(**overrides):
    values = {
        "ai_provider": "ollama",
        "ollama_base_url": "http://localhost:11434",
        "ollama_model": "qwen2.5:3b",
        "chat_model": "test-groq-model",
        "groq_api_key": None,
        "request_timeout": 5,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_ollama_provider_calls_local_api_without_constructing_groq():
    response = MagicMock()
    response.__enter__.return_value.read.return_value = json.dumps({
        "message": {"content": '{"answer":"Local answer","citations":["notes.txt"]}'},
        "prompt_eval_count": 12,
        "eval_count": 7,
    }).encode()
    with patch("backend.services.llm_service.urlopen", return_value=response) as local_call, patch(
        "backend.services.llm_service.Groq"
    ) as groq:
        generated = LLMService(settings()).generate("private document prompt")

    request = local_call.call_args.args[0]
    assert request.full_url == "http://localhost:11434/api/chat"
    assert generated.input_tokens == 12
    assert generated.output_tokens == 7
    assert "private document prompt" in request.data.decode()
    groq.assert_not_called()


def test_groq_provider_uses_configured_groq_model():
    result = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content='{"answer":"Hosted","citations":[]}'))],
        usage=SimpleNamespace(prompt_tokens=3, completion_tokens=4),
    )
    client = MagicMock()
    client.chat.completions.create.return_value = result
    with patch("backend.services.llm_service.Groq", return_value=client) as groq:
        generated = LLMService(settings(ai_provider="groq", groq_api_key="test-secret")).generate("prompt")

    groq.assert_called_once()
    assert client.chat.completions.create.call_args.kwargs["model"] == "test-groq-model"
    assert generated.text.startswith('{"answer"')
    assert generated.input_tokens == 3


def test_ollama_connection_failure_is_a_service_error():
    from urllib.error import URLError

    with patch("backend.services.llm_service.urlopen", side_effect=URLError("offline")):
        with pytest.raises(AIServiceError, match="Cannot connect to Ollama"):
            LLMService(settings()).generate("prompt")


def test_ollama_invalid_json_is_a_service_error():
    response = MagicMock()
    response.__enter__.return_value.read.return_value = b"not-json"
    with patch("backend.services.llm_service.urlopen", return_value=response):
        with pytest.raises(AIServiceError, match="invalid response"):
            LLMService(settings()).generate("prompt")


def test_missing_ollama_model_explains_how_to_pull_it():
    error = HTTPError("http://localhost:11434/api/chat", 404, "not found", {}, None)
    with patch("backend.services.llm_service.urlopen", side_effect=error):
        with pytest.raises(AIServiceError, match="ollama pull qwen2.5:3b"):
            LLMService(settings()).generate("prompt")


def test_groq_provider_requires_key():
    with pytest.raises(AIServiceError, match="GROQ_API_KEY"):
        LLMService(settings(ai_provider="groq")).generate("prompt")
