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
        """
        Base constructor of every judge: it stores the common configuration. Subclasses call it
        with ``super().__init__(...)`` and then create their own client.
        :param name: name of the judge, unique within an ``Evaluator``; used for the result folders
                     and the tables
        :param model: model or deployment identifier sent to the API
        :param temperature: sampling temperature; None means "do not send it"
        :param max_output_tokens: maximum length of the reply; None means "do not send it"
        """
        self.name = name  # unique within an Evaluator; used for the output folders
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

    def describe(self) -> dict[str, Any]:
        """
        Configuration of the judge, without keys. It is saved in ``config.json`` and is part of the
        cache key: changing the model or the temperature makes the calls run again.
        :return: dict with ``name``, ``provider`` (the ``kind`` of the class), ``model``,
                 ``temperature`` and ``max_output_tokens``
        """
        return {"name": self.name, "provider": self.kind, "model": self.model,
                "temperature": self.temperature, "max_output_tokens": self.max_output_tokens}

    @abstractmethod
    def complete(self, user: str) -> LLMResponse:
        """
        Send the prompt to the model and return its reply. Every judge must implement this method
        (it is the only one needed to add a new model).
        :param user: full prompt, to be sent as a single user message
        :return: the raw text of the reply and the token usage, if available
        :raises ProviderError: if the reply is empty or the call fails
        """
