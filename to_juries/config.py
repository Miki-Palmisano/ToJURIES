"""Validated run options."""

# Enables modern type annotations (e.g. `tuple[str, str]`) on older Python versions.
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

# =============================================================================
# 1. CONFIGURATION ERROR
# =============================================================================

class ConfigError(RuntimeError):
    """Invalid or incomplete configuration."""
    # Package-specific exception, so callers can catch configuration
    # problems separately from other runtime errors.

# =============================================================================
# 2. RUN OPTIONS
# =============================================================================

class RunConfig(BaseModel):
    # Names of the two systems being compared; the gain is always second - first.
    labels: tuple[str, str] = ("first", "second")

    # Presentation orders to run: "AB" = first label shown as REPORT A,
    # "BA" = swapped. Using both controls the judges' position bias.
    # default_factory creates a new list for each instance (no shared mutable default).
    orders: list[Literal["AB", "BA"]] = Field(default_factory=lambda: ["AB", "BA"])

    # Repetitions of each call (same pair, judge and order), to average out the judges' randomness.
    reps: int = 1

    # Maximum number of calls to the models running at the same time.
    concurrency: int = 4

    # Maximum number of attempts for a failed call.
    max_retries: int = 3

    # Number of bootstrap resamples for the jury's interval.
    bootstrap: int = 5000

    # Significance level of the intervals (0.05 -> 95% intervals).
    alpha: float = 0.05

    # Seed of the random generator (bootstrap), for reproducible results.
    seed: int = 0

    # --- Single-field validation ---------------------------------------------
    # The same validator is applied to the three listed fields.
    # It runs after pydantic has converted the value to int.
    @field_validator("reps", "concurrency", "max_retries")
    @classmethod
    def _positive(cls, v: int) -> int:
        """
        Check that a numeric option is at least 1 (applied to reps, concurrency and max_retries).
        :param v: value of the option being validated
        :return: the same value, if valid
        :raises ValueError: if the value is lower than 1
        """
        if v < 1:
            raise ValueError("must be >= 1")
        return v

    # --- Cross-field validation ----------------------------------------------
    # mode="after": runs on the already-built instance, once every field has passed its own checks.
    @model_validator(mode="after")
    def _check(self) -> "RunConfig":
        """
        Cross-field checks, run after every single option has been validated:
        at least one order must be given and the two labels must be different.
        :return: the validated configuration itself
        :raises ValueError: if orders is empty or the two labels are equal
        """
        # With no orders there would be no calls to run.
        if not self.orders:
            raise ValueError("at least one order is required")

        # Equal labels would make it impossible to tell the two systems apart
        # (pivots and gains in the analysis would break).
        if self.labels[0] == self.labels[1]:
            raise ValueError("the two labels need different names")
        return self
