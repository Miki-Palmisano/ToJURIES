"""Gemini (google-genai). Free-text reply."""

from __future__ import annotations

import os

from ..config import ConfigError
from .base import LLMResponse, Provider, ProviderError


class Gemini(Provider):
    """``Gemini(model="...", api_key=None)``; the key defaults to GEMINI_API_KEY / GOOGLE_API_KEY."""

    kind = "gemini"

    def __init__(self, model: str, *, api_key: str | None = None, temperature: float | None = None,
                 max_output_tokens: int | None = 16000, name: str = "gemini") -> None:
        super().__init__(name, model, temperature, max_output_tokens)
        key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise ConfigError("Gemini: missing api_key (argument or GEMINI_API_KEY / GOOGLE_API_KEY)")
        from google import genai
        from google.genai import types

        self._types = types
        self._client = genai.Client(api_key=key)

    def complete(self, user: str) -> LLMResponse:
        kw: dict = {}

        if self.temperature is not None:
            kw["temperature"] = self.temperature

        if self.max_output_tokens:
            kw["max_output_tokens"] = self.max_output_tokens

        resp = self._client.models.generate_content(
            model=self.model, contents=user, config=self._types.GenerateContentConfig(**kw))

        text = getattr(resp, "text", None)
        if not text:
            cands = getattr(resp, "candidates", None) or []
            reason = getattr(cands[0], "finish_reason", None) if cands else None
            raise ProviderError(f"empty Gemini response (finish_reason={reason})")
        um = getattr(resp, "usage_metadata", None)
        usage = {"input_tokens": getattr(um, "prompt_token_count", None),
                 "output_tokens": getattr(um, "candidates_token_count", None)} if um else {}
        return LLMResponse(text=text, usage=usage)
