"""A judge = a model with its configuration. Subclass :class:`Provider` to add your own."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ..config import ConfigError  # noqa: F401  (re-exported for providers)


@dataclass
class LLMResponse:
    text: str
    usage: dict[str, Any] = field(default_factory=dict)


class ProviderError(RuntimeError):
    pass


class Provider(ABC):
    kind = "custom"

    def __init__(self, name: str, model: str, temperature: float | None = None,
                 max_output_tokens: int | None = 16000) -> None:
        self.name = name  # unique within an Evaluator; used for the output folders
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

    def describe(self) -> dict[str, Any]:
        """Configuration (never keys); stored in config.json and used for cache invalidation."""
        return {"name": self.name, "provider": self.kind, "model": self.model,
                "temperature": self.temperature, "max_output_tokens": self.max_output_tokens}

    @abstractmethod
    def complete(self, user: str) -> LLMResponse:
        """Send the prompt as a single user message and return the raw text reply."""
