"""Azure OpenAI (Chat Completions). Free-text reply; the prompt is sent as-is."""

from __future__ import annotations

import os

from ..config import ConfigError
from .base import LLMResponse, Provider, ProviderError

_OPTIONAL = ("temperature", "max_completion_tokens")


class AzureOpenAI(Provider):
    """``AzureOpenAI(deployment="...", endpoint=..., api_version=..., api_key=...)``.

    ``endpoint``, ``api_version`` and ``api_key`` default to AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_VERSION and AZURE_OPENAI_API_KEY. ``deployment`` is the deployment NAME.
    """

    kind = "azure"

    def __init__(self, deployment: str, *, endpoint: str | None = None, api_version: str | None = None,
                 api_key: str | None = None, temperature: float | None = None,
                 max_output_tokens: int | None = 16000, name: str = "azure") -> None:
        super().__init__(name, deployment, temperature, max_output_tokens)

        key = api_key or os.environ.get("AZURE_OPENAI_API_KEY")
        base = endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
        version = api_version or os.environ.get("AZURE_OPENAI_API_VERSION")

        missing = [n for n, v in (("api_key", key), ("endpoint", base)) if not v]
        if missing:
            raise ConfigError(f"Azure OpenAI: missing {', '.join(missing)} (argument or AZURE_OPENAI_* variable)")

        if version:  # API datata: client AzureOpenAI
            from openai import AzureOpenAI as _Client

            self._client = _Client(api_key=key, azure_endpoint=base, api_version=version,
                                   max_retries=0, timeout=900)
        else:  # API v1: nessuna api-version
            from openai import OpenAI

            self._client = OpenAI(api_key=key, base_url=base.rstrip("/") + "/openai/v1/",
                                  max_retries=0, timeout=900)

    def complete(self, user: str) -> LLMResponse:
        kwargs: dict = {"model": self.model, "messages": [{"role": "user", "content": user}]}

        if self.temperature is not None:
            kwargs["temperature"] = self.temperature

        if self.max_output_tokens:
            kwargs["max_completion_tokens"] = self.max_output_tokens

        last: Exception | None = None
        for _ in range(len(_OPTIONAL) + 1):
            try:
                resp = self._client.chat.completions.create(**kwargs)
                break
            except Exception as exc:  # noqa: BLE001 - SDK exception classes vary
                last = exc
                msg = str(exc).lower()
                bad = [p for p in _OPTIONAL if p in msg and p in kwargs]
                if not bad:
                    raise
                kwargs.pop(bad[0])  # parameter not supported by this model: drop and retry
        else:
            raise ProviderError(f"Azure OpenAI call failed: {last}")

        choice = resp.choices[0]
        text = choice.message.content or ""
        if not text:
            raise ProviderError(f"empty Azure OpenAI response (finish_reason={choice.finish_reason})")
        u = getattr(resp, "usage", None)
        usage = {"input_tokens": getattr(u, "prompt_tokens", None),
                 "output_tokens": getattr(u, "completion_tokens", None)} if u else {}
        return LLMResponse(text=text, usage=usage)
