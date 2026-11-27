"""Extract and validate the final JSON object that the evaluation prompt asks for."""

# Enables modern type annotations (e.g. `float | None`, `dict[str, float]`) on older Python versions.
from __future__ import annotations

import json
import re
from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator


# =============================================================================
# 1. TYPES AND CONSTANTS
# =============================================================================

# A score is a float between 1 and 10 (inclusive); pydantic rejects values outside the range.
Score = Annotated[float, Field(ge=1.0, le=10.0)]
# A count of contradictions cannot be negative.
Count = Annotated[int, Field(ge=0)]
# The only accepted answers for "which report is better".
Choice = Literal["A", "B", "TIE"]
# The four dimensions scored by the judges (names as in the prompt's JSON).
DIMS = ("factual_consistency", "time_series_analysis", "central_argument", "language_quality")


class ParseError(ValueError):
    # Raised when the judge's reply cannot be turned into a valid verdict.
    # Subclass of ValueError, so it can also be caught as a generic value error.
    pass


# =============================================================================
# 2. MODELS OF THE FINAL JSON
# =============================================================================

class ReportScores(BaseModel):
    # Scores of the four dimensions; None if the judge left the dimension null
    # (e.g. not applicable to that report).
    factual_consistency: Score | None = None
    time_series_analysis: Score | None = None
    central_argument: Score | None = None
    language_quality: Score | None = None

    # Flags: number of major contradictions and whether the report uses information after the cutoff.
    major_contradictions: Count | None = None   # FIX: >= 0
    temporal_leakage: bool | None = None


# Common spellings of a tie, mapped to "TIE".
_TIE_ALIASES = {"TIE", "TIED", "DRAW", "EQUAL"}


class Comparison(BaseModel):
    # Restricted to "A", "B" or "TIE" (after normalisation), instead of any string.
    better_temporal_analysis: Choice | None = None
    better_overall: Choice | None = None
    # Free-text explanation of the verdict.
    verdict: str = ""

    # Normalise the answer before validation, so "a", " Report B ", "tie" or "Tied"
    # are accepted; anything still not A/B/TIE makes validation fail instead of being
    # silently turned into None later on.
    @field_validator("better_temporal_analysis", "better_overall", mode="before")
    @classmethod
    def _normalise_choice(cls, v):
        """
        Normalise a "which report is better" answer.
        :param v: raw value written by the model
        :return: "A", "B", "TIE", None, or the cleaned value (which then fails validation)
        """
        if not isinstance(v, str):
            return v
        v = v.strip().upper()
        v = re.sub(r"^REPORT[\s_]*", "", v)   # "REPORT A", "Report_B" -> "A", "B"
        if v in ("", "NONE", "NULL", "N/A"):
            return None
        return "TIE" if v in _TIE_ALIASES else v


class Verdict(BaseModel):
    # The two reports are required; the comparison is optional.
    report_A: ReportScores
    report_B: ReportScores
    comparison: Comparison | None = None


# =============================================================================
# 3. EXTRACTING THE VERDICT FROM THE REPLY
# =============================================================================

# Only fences at the start of a line are removed (``` or ```json, any case).
# A JSON string cannot contain a raw newline, so a fence at the start of a line
# is never inside the JSON; backticks inside strings are left untouched.
_FENCE = re.compile(r"^[ \t]*```[a-zA-Z]*", flags=re.M)


def extract_verdict(text: str) -> Verdict:
    """
    Parse the LAST JSON object of the reply, the one containing "report_A", ignoring the free text
    and the markdown fences that come before it, and validate it. The keys of the object can be
    in any order.
    :param text: full reply of the judge
    :return: the validated verdict (scores of the two reports and comparison)
    :raises ParseError: if the reply is empty, the final JSON is missing, malformed or invalid,
                        or its fields do not match the expected format (e.g. a score outside 1-10)
    """
    # Empty or whitespace-only reply.
    if not text or not text.strip():
        raise ParseError("empty response")
    # Remove the fences, so they do not get in the way of the JSON decoding.
    s = _FENCE.sub("", text)
    # Last occurrence of the "report_A" key: the final JSON comes after the reasoning,
    # which may itself quote earlier drafts or examples.
    key = s.rfind('"report_A"')
    if key == -1:
        raise ParseError('no final JSON object with "report_A" found')

    # Instead of assuming the object starts at the last "{" before the key
    # (true only if "report_A" is its first key), try every "{" before the key,
    # from the nearest backwards, and keep the first object that
    #   - decodes as valid JSON,
    #   - spans past the key (so it is the object holding the last "report_A",
    #     not an earlier draft or a nested object such as "comparison"),
    #   - has "report_A" among its top-level keys.
    decoder = json.JSONDecoder()
    first_error: json.JSONDecodeError | None = None
    starts = [m.start() for m in re.finditer(r"\{", s[:key])]
    for start in reversed(starts):
        try:
            # raw_decode reads one JSON value starting at `start` and returns where it ends,
            # ignoring anything after it (e.g. closing remarks by the model).
            obj, end = decoder.raw_decode(s, start)
        except json.JSONDecodeError as exc:
            # Keep the error of the nearest candidate: it is the most informative one.
            if first_error is None:
                first_error = exc
            continue
        if end > key and isinstance(obj, dict) and "report_A" in obj:
            # Validate against the models; any pydantic error becomes a ParseError,
            # so callers only need to catch one exception type.
            try:
                return Verdict.model_validate(obj)
            except Exception as exc:  # pydantic.ValidationError
                raise ParseError(f"JSON does not match the expected format: {exc}") from exc

    # No candidate worked.
    if first_error is not None:
        raise ParseError(f"invalid JSON: {first_error}") from first_error
    raise ParseError("final JSON object is malformed")


# =============================================================================
# 4. MEAN AND OVERALL COMPUTED IN CODE
# =============================================================================

# Weights of the dimensions in OVERALL, as prescribed by the prompt (they sum to 1).
OVERALL_WEIGHTS = {"factual_consistency": 0.30, "time_series_analysis": 0.30,
                   "central_argument": 0.30, "language_quality": 0.10}


def overall_from(
        scores: ReportScores,
        weights: dict[str, float] = OVERALL_WEIGHTS
) -> float | None:
    """
    Compute OVERALL in code as the weighted mean of the four dimensions (0.30, 0.30, 0.30, 0.10).
    Dimensions that are null are excluded and the remaining weights are renormalised, as the prompt
    prescribes. The value written by the model is ignored.
    :param scores: scores of one report
    :param weights: weight of each dimension (default: the ones of the prompt)
    :return: the overall score, or None if all the dimensions are null
    """
    # num = sum of weight * score; den = sum of the weights actually used.
    num = den = 0.0
    for k, wk in weights.items():
        v = getattr(scores, k)
        # Null dimensions are skipped: their weight does not enter the denominator.
        if v is not None:
            num += wk * v
            den += wk
    # Dividing by den renormalises the remaining weights; None if no dimension was scored.
    return num / den if den else None


def mean_from(scores: ReportScores) -> float | None:
    """
    Compute MEAN in code as the plain mean of the available dimensions (null ones are skipped).
    The value written by the model is ignored.
    :param scores: scores of one report
    :return: the mean score, or None if all the dimensions are null
    """
    # Non-null scores only.
    vals = [getattr(scores, k) for k in DIMS if getattr(scores, k) is not None]
    return sum(vals) / len(vals) if vals else None


# Position of a report in a call (REPORT A or REPORT B); used by other modules.
Position = Literal["A", "B"]