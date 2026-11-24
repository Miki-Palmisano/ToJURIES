"""Analysis of a run directory -> CSV tables, report.md, summary.json."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import RunConfig
from ..runner import load_records
from .agreement import judge_agreement
from .order_effects import order_effects
from .stats_utils import bootstrap_ci, summarize
from .tables import pair_wide, to_frames

DIMENSIONS = ["overall", "mean", "factual", "temporal", "central", "language"]


def _gain_matrix(wide: pd.DataFrame, dim: str, judges: list[str]) -> pd.DataFrame:
    d = wide[wide["dimension"] == dim]
    return d.pivot_table(index="pair_id", columns="judge", values="gain").reindex(columns=judges)


def _fmt(v, nd=2):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "-"
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def _md(df: pd.DataFrame, cols: list[str] | None = None) -> str:
    if df is None or df.empty:
        return "_(no data)_\n"
    cols = cols or list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    out += ["| " + " | ".join(_fmt(r[c]) for c in cols) + " |" for _, r in df.iterrows()]
    return "\n".join(out) + "\n"


def analyze(out_dir: str | Path, cfg: RunConfig | None = None, judges: list[str] | None = None) -> dict:
    """Analyse a results folder. Options and judge names default to the ones saved in config.json."""
    out = Path(out_dir)
    if cfg is None or judges is None:
        meta = json.loads((out / "config.json").read_text(encoding="utf-8"))
        cfg = cfg or RunConfig.model_validate(meta["options"])
        judges = judges or [j["name"] for j in meta["judges"]]
    records = load_records(out)
    if not records:
        raise RuntimeError(f"no successful records under {out / 'runs'}")
    scores, flags = to_frames(records, cfg.systems)
    judges = [j for j in judges if j in set(scores["judge"])]
    wide = pair_wide(scores, cfg.systems)
    a, nb, seed = cfg.alpha, cfg.bootstrap, cfg.seed

    per_judge, jury = [], []
    for dim in DIMENSIONS:
        m = _gain_matrix(wide, dim, judges)
        if m.empty:
            continue
        for j in judges:
            per_judge.append({"judge": j, "dimension": dim, **summarize(m[j].to_numpy(float), a)})
        pm = m.mean(axis=1, skipna=True).to_numpy(float)
        lo, hi = bootstrap_ci(m.to_numpy(float), None, nb, a, seed)
        jury.append({"dimension": dim, **summarize(pm, a), "ci_low_boot": lo, "ci_high_boot": hi})
    per_judge, jury = pd.DataFrame(per_judge), pd.DataFrame(jury)

    oe = order_effects(wide, cfg.systems, a)
    agree = pd.concat([judge_agreement(_gain_matrix(wide, d, judges), d) for d in ("overall", "temporal")],
                      ignore_index=True)

    f = flags.copy()
    f["temporal_leakage"] = f["temporal_leakage"].astype(float)
    flag_tab = (f.groupby(["judge", "system"])
                .agg(reports=("pair_id", "size"), major_contradictions=("major_contradictions", "sum"),
                     leakage_rate=("temporal_leakage", "mean")).reset_index())
    wins = (flags.drop_duplicates(["pair_id", "judge", "order", "rep"])
            .groupby(["judge", "better_overall"]).size().unstack(fill_value=0).reset_index())

    first, second = cfg.systems
    lines = ["# reportjury - report\n",
             f"- records: **{len(records)}** | judges: {', '.join(judges)} | pairs: {wide['pair_id'].nunique()}",
             f"- gain = `{second}` - `{first}`, averaged over the two presentation orders\n",
             "## Jury (mean of the judges, per pair)\n", _md(jury, ["dimension", "n", "mean", "ci_low", "ci_high",
                                                             "ci_low_boot", "ci_high_boot", "p", "dz", "prob_pos"]),
             "## Per judge\n", _md(per_judge, ["judge", "dimension", "n", "mean", "ci_low", "ci_high", "p", "dz",
                                               "prob_pos", "sign_p"]),
             "## Order effects\n",
             "H = score in position A minus position B (H>0 favours the first position); "
             "S = both reports score higher in one order (run effect).\n",
             _md(oe, ["judge", "dimension", "n_pairs", "H_mean", "H_ci_low", "H_ci_high", "H_p", "S_mean", "S_p",
                      "gain_order1", "gain_order2", "gain_corr_between_orders", "sign_agreement"]),
             "## Agreement between judges (gain per pair)\n", _md(agree),
             "## Verdicts (`better_overall`, counted per call)\n", _md(wins),
             "## Contradictions and temporal leakage reported by the judges\n", _md(flag_tab)]
    (out).mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    scores.to_csv(out / "scores_long.csv", index=False)
    flags.to_csv(out / "flags.csv", index=False)
    wide.to_csv(out / "pair_gains.csv", index=False)
    per_judge.to_csv(out / "per_judge.csv", index=False)
    jury.to_csv(out / "jury.csv", index=False)
    oe.to_csv(out / "order_effects.csv", index=False)
    agree.to_csv(out / "agreement.csv", index=False)
    (out / "summary.json").write_text(
        json.dumps({"records": len(records), "judges": judges,
                    "jury": jury[["dimension", "n", "mean", "ci_low_boot", "ci_high_boot", "p"]].to_dict("records")},
                   indent=1, default=float), encoding="utf-8")
    return {"jury": jury, "per_judge": per_judge, "order_effects": oe, "agreement": agree}
