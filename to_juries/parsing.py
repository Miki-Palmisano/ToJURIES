"""Extract and validate the final JSON object that the evaluation prompt asks for."""

from __future__ import annotations

import json
import re
from typing import Annotated, Literal

from pydantic import BaseModel, Field

Score = Annotated[float, Field(ge=1.0, le=10.0)]
DIMS = ("factual_consistency", "time_series_analysis", "central_argument", "language_quality")


class ParseError(ValueError):
    pass


class ReportScores(BaseModel):
    factual_consistency: Score | None = None
    time_series_analysis: Score | None = None
    central_argument: Score | None = None
    language_quality: Score | None = None
    mean: float | None = None
    overall: float | None = None
    major_contradictions: int | None = None
    temporal_leakage: bool | None = None


class Comparison(BaseModel):
    better_temporal_analysis: str | None = None
    better_overall: str | None = None
    verdict: str = ""


class Verdict(BaseModel):
    report_A: ReportScores
    report_B: ReportScores
    comparison: Comparison | None = None


_FENCE = re.compile(r"```(?:json)?", flags=re.I)


def extract_verdict(text: str) -> Verdict:
    """Parse the LAST JSON object of the reply (the one containing "report_A")."""
    if not text or not text.strip():
        raise ParseError("empty response")
    s = _FENCE.sub("", text)
    key = s.rfind('"report_A"')
    if key == -1:
        raise ParseError('no final JSON object with "report_A" found')
    start = s.rfind("{", 0, key)
    if start == -1:
        raise ParseError("final JSON object is malformed")
    try:
        obj, _ = json.JSONDecoder().raw_decode(s[start:])
    except json.JSONDecodeError as exc:
        raise ParseError(f"invalid JSON: {exc}") from exc
    try:
        return Verdict.model_validate(obj)
    except Exception as exc:  # pydantic.ValidationError
        raise ParseError(f"JSON does not match the expected format: {exc}") from exc


OVERALL_WEIGHTS = {"factual_consistency": 0.30, "time_series_analysis": 0.30,
                   "central_argument": 0.30, "language_quality": 0.10}


def overall_from(scores: ReportScores, weights: dict[str, float] = OVERALL_WEIGHTS) -> float | None:
    num = den = 0.0
    for k, wk in weights.items():
        v = getattr(scores, k)
        if v is not None:
            num += wk * v
            den += wk
    return num / den if den else None


def mean_from(scores: ReportScores) -> float | None:
    vals = [getattr(scores, k) for k in DIMS if getattr(scores, k) is not None]
    return sum(vals) / len(vals) if vals else None


Position = Literal["A", "B"]
