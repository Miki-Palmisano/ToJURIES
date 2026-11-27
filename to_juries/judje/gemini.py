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
        """
        Create the Gemini judge. The key is read from the argument or, if missing, from the
        environment variables (for example from the ``.env`` file); it is never saved in the results.
        :param model: model identifier (e.g. the one shown in Google AI Studio)
        :param api_key: API key; default: GEMINI_API_KEY, otherwise GOOGLE_API_KEY
        :param temperature: sampling temperature; None means "do not send it"
        :param max_output_tokens: maximum length of the reply; None means "do not send it"
        :param name: name of this judge, used for the result folder and the tables; must be unique
                     among the judges
        :raises ConfigError: if the key is missing
        """
        super().__init__(name, model, temperature, max_output_tokens)
        key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise ConfigError("Gemini: missing api_key (argument or GEMINI_API_KEY / GOOGLE_API_KEY)")
        from google import genai
        from google.genai import types

        self._types = types
        self._client = genai.Client(api_key=key)

    def complete(self, user: str) -> LLMResponse:
        """
        Send the prompt to the model and return the free-text reply (uses tokens).
        :param user: full prompt, sent as the content of the request
        :return: the reply text and the token usage (``input_tokens``, ``output_tokens``)
        :raises ProviderError: if the reply is empty (for example it was blocked or cut); the
                               message includes the ``finish_reason``
        :raises Exception: any other error of the SDK (network, authentication, rate limit) is
                           propagated; the runner retries the call
        """
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
