"""Validated run options."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class ConfigError(RuntimeError):
    """Invalid or incomplete configuration."""


class RunConfig(BaseModel):
    systems: tuple[str, str] = ("baseline", "candidate")  # (first, second); gain = second - first
    orders: list[Literal["AB", "BA"]] = Field(default_factory=lambda: ["AB", "BA"])
    reps: int = 1
    concurrency: int = 4
    max_retries: int = 3
    bootstrap: int = 5000
    alpha: float = 0.05
    seed: int = 0

    @field_validator("reps", "concurrency", "max_retries")
    @classmethod
    def _positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("must be >= 1")
        return v

    @model_validator(mode="after")
    def _check(self) -> "RunConfig":
        if not self.orders:
            raise ValueError("at least one order is required")
        if self.systems[0] == self.systems[1]:
            raise ValueError("the two systems need different names")
        return self
