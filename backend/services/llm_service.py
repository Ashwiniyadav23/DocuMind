"""Provider adapter returning a consistent result for Groq and local Ollama."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from groq import APIError, Groq

from ..core.config import Settings
from ..core.exceptions import AIServiceError


@dataclass
class Generation:
    text: str
    input_tokens: int
    output_tokens: int


class LLMService:
    """Select exactly one provider from configuration; never silently fall back."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @property
    def model(self) -> str:
        return self.settings.ollama_model if self.settings.ai_provider == "ollama" else self.settings.chat_model

    def generate(self, prompt: str) -> Generation:
        if self.settings.ai_provider == "ollama":
            return self._generate_ollama(prompt)
        return self._generate_groq(prompt)

    def _generate_groq(self, prompt: str) -> Generation:
        if not self.settings.groq_api_key:
            raise AIServiceError("GROQ_API_KEY is required when AI_PROVIDER=groq.")
        try:
            client = Groq(api_key=self.settings.groq_api_key, timeout=self.settings.request_timeout, max_retries=1)
            completion = client.chat.completions.create(
                model=self.settings.chat_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_completion_tokens=1000,
                response_format={"type": "json_object"},
            )
            text = completion.choices[0].message.content or ""
            usage: Any = completion.usage
            return Generation(text, int(getattr(usage, "prompt_tokens", 0) or 0), int(getattr(usage, "completion_tokens", 0) or 0))
        except APIError as exc:
            raise AIServiceError("Groq could not generate a response. Check provider configuration and API limits.") from exc

    def _generate_ollama(self, prompt: str) -> Generation:
        payload = json.dumps({
            "model": self.settings.ollama_model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.1, "num_predict": 1000},
        }).encode("utf-8")
        request = Request(
            f"{self.settings.ollama_base_url}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.settings.request_timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
            message = body.get("message")
            text = message.get("content") if isinstance(message, dict) else None
            if not isinstance(text, str) or not text.strip():
                raise AIServiceError("Ollama returned an empty or invalid response.")
            return Generation(text, int(body.get("prompt_eval_count", 0) or 0), int(body.get("eval_count", 0) or 0))
        except HTTPError as exc:
            if exc.code == 404:
                detail = f"Ollama model '{self.settings.ollama_model}' is unavailable. Pull it with `ollama pull {self.settings.ollama_model}`."
            else:
                detail = f"Ollama returned HTTP {exc.code}."
            raise AIServiceError(detail) from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise AIServiceError("Cannot connect to Ollama. Start the local Ollama service and check OLLAMA_BASE_URL.") from exc
        except (UnicodeDecodeError, json.JSONDecodeError, AttributeError, TypeError, ValueError) as exc:
            raise AIServiceError("Ollama returned an invalid response.") from exc
