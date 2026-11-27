"""Tidy score and flag tables, with each report mapped back to its label."""

# Enables modern type annotations (e.g. `tuple[str, str]`) on older Python versions.
from __future__ import annotations

import pandas as pd

from ..parsing import DIMS, ReportScores, overall_from, mean_from

# =============================================================================
# 1. CONSTANTS
# =============================================================================

# Map from the field names used in the judge's JSON to the short names used
# in the tables (e.g. "time_series_analysis" -> "temporal").
DIM_LABEL = {
    "factual_consistency": "factual",
    "time_series_analysis": "temporal",
    "central_argument": "central",
    "language_quality": "language",
    "mean": "mean",
    "overall": "overall",
}

# =============================================================================
# 2. FROM POSITION (A/B) TO LABEL
# =============================================================================

def system_of(
        position: str,
        order: str,
        systems: tuple[str, str]
) -> str:
    """
    Tell which label (e.g. PRE or POST) was shown in a given position in a given call.
    With order "AB" the first label is REPORT A and the second is REPORT B; with "BA" it is the opposite.
    :param position: "A" or "B", the position shown to the judge
    :param order: "AB" or "BA", the presentation order of the call
    :param systems: the two labels ``(first, second)``
    :return: the label of the report that was in that position
    """
    first, second = systems

    # Order "AB": A = first label, B = second label.
    if order == "AB":
        return first if position == "A" else second

    # Order "BA" (swapped, to control position bias): A = second, B = first.
    return second if position == "A" else first

# =============================================================================
# 3. FROM SAVED CALLS TO TIDY TABLES
# =============================================================================

def to_frames(
        records: list[dict],
        systems: tuple[str, str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Turn the saved calls into two tidy tables, mapping each report back to its label.
    ``mean`` and ``overall`` are computed in code from the four dimensions; the values written by
    the model are ignored.
    :param records: calls loaded with ``load_records`` (``job`` and parsed ``verdict``)
    :param systems: the two labels ``(first, second)``
    :return: a tuple of two DataFrames:
             - scores: one row per report, call and dimension (factual, temporal, central,
               language, mean, overall), with the columns of the job, ``position``, ``system``,
               ``dimension`` and ``score`` (NaN if the dimension is null);
             - flags: one row per report and call, with ``major_contradictions``,
               ``temporal_leakage`` and ``better_overall`` (the winning label, "TIE" or None)
    """
    # srows -> rows of the scores table, frows -> rows of the flags table.
    srows, frows = [], []
    for rec in records:

        # job: call metadata (pair_id, judge, order, repetition, ...)
        # v: the judge's parsed JSON verdict
        job, v = rec["job"], rec["verdict"]

        # --- One block per report shown in the call (positions A and B) ------
        for pos in ("A", "B"):
            # Validate the scores of the report in this position.
            sc = ReportScores.model_validate(v[f"report_{pos}"])

            # Map the position back to the real label (undoing the presentation order).
            system = system_of(pos, job["order"], systems)

            # Columns shared by all rows of this report: job metadata + position + label.
            base = {**job, "position": pos, "system": system}

            # Scores of the four dimensions, read from the validated model.
            values = {k: getattr(sc, k) for k in DIMS}

            # `mean` and `overall` are recomputed in code (the model's own values are ignored),
            # so they are always consistent with the four dimensions.
            values["mean"] = mean_from(sc)
            values["overall"] = overall_from(sc)

            # Long format: one row per dimension, with its short name.
            for k, x in values.items():
                srows.append({**base, "dimension": DIM_LABEL[k], "score": x})

            # One flags row per report (better_overall is added below, once per call).
            frows.append({**base, "major_contradictions": sc.major_contradictions,
                          "temporal_leakage": sc.temporal_leakage})

        # --- Final verdict of the call ---------------------------------------
        # `comparison` may be missing or null: fall back to an empty dict.
        comp = v.get("comparison") or {}
        win = comp.get("better_overall")

        # "A"/"B" -> mapped back to the winning label; "TIE" stays as is;
        # anything else (missing or invalid) -> None.
        winner = system_of(win, job["order"], systems) if win in ("A", "B") else (
            "TIE" if win == "TIE" else None)

        # The verdict belongs to the whole call: write it on the two flags rows
        # just appended for this call (positions A and B).
        for f in frows[-2:]:
            f["better_overall"] = winner
    scores = pd.DataFrame(srows)

    # Null or non-numeric scores become NaN, so the column is numeric.
    scores["score"] = pd.to_numeric(scores["score"], errors="coerce")
    return scores, pd.DataFrame(frows)
