"""Tidy score tables and paired gains (second system minus first)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..parsing import DIMS, ReportScores, overall_from, mean_from

DIM_LABEL = {
    "factual_consistency": "factual",
    "time_series_analysis": "temporal",
    "central_argument": "central",
    "language_quality": "language",
    "mean": "mean",
    "overall": "overall",
}
INDEX = ["pair_id", "judge", "dimension"]


def system_of(position: str, order: str, systems: tuple[str, str]) -> str:
    first, second = systems
    if order == "AB":
        return first if position == "A" else second
    return second if position == "A" else first


def to_frames(records: list[dict], systems: tuple[str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """scores (one row per report x dimension) and flags (contradictions, leakage, verdict)."""
    srows, frows = [], []
    for rec in records:
        job, v = rec["job"], rec["verdict"]
        for pos in ("A", "B"):
            sc = ReportScores.model_validate(v[f"report_{pos}"])
            system = system_of(pos, job["order"], systems)
            base = {**job, "position": pos, "system": system}
            values = {k: getattr(sc, k) for k in DIMS}
            values["mean"] = mean_from(sc)
            values["overall"] = overall_from(sc)

            for k, x in values.items():
                srows.append({**base, "dimension": DIM_LABEL[k], "score": x})
            frows.append({**base, "major_contradictions": sc.major_contradictions,
                          "temporal_leakage": sc.temporal_leakage})

        comp = v.get("comparison") or {}
        win = comp.get("better_overall")
        winner = system_of(win, job["order"], systems) if win in ("A", "B") else (
            "TIE" if win == "TIE" else None)
        for f in frows[-2:]:
            f["better_overall"] = winner
    scores = pd.DataFrame(srows)
    scores["score"] = pd.to_numeric(scores["score"], errors="coerce")
    return scores, pd.DataFrame(frows)


def pair_wide(scores: pd.DataFrame, systems: tuple[str, str]) -> pd.DataFrame:
    """One row per (pair, judge, dimension): score of each system in each order and the gain."""
    first, second = systems
    d = scores.dropna(subset=["score"])
    g = d.groupby(INDEX + ["order", "system"], as_index=False)["score"].mean()
    w = g.pivot_table(index=INDEX, columns=["order", "system"], values="score")
    w.columns = [f"{o}_{s}" for o, s in w.columns]
    w = w.reset_index()
    for o in ("AB", "BA"):
        for s in systems:
            w[f"{o}_{s}"] = w.get(f"{o}_{s}", np.nan)
    w["gain_AB"] = w[f"AB_{second}"] - w[f"AB_{first}"]
    w["gain_BA"] = w[f"BA_{second}"] - w[f"BA_{first}"]
    w["gain"] = w[["gain_AB", "gain_BA"]].mean(axis=1, skipna=True)
    return w
