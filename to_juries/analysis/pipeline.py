"""Analysis of a results folder: mean and overall per judge, the jury's scores and the gain statistics."""

# Enables modern type annotations (e.g. `str | Path`, `list[str]`) on older Python versions too.
from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from ..config import RunConfig
from ..runner import load_run_records
from .stats_utils import bootstrap_ci, summarize
from .tables import to_frames

# Name of the "virtual judge" representing the mean of all real judges.
JURY = "jury"
# Dimensions on which the gain (second - first) is computed.
GAIN_DIMS = ("overall", "mean", "temporal")
# Dimensions on which agreement between judges is measured (subset of the above).
AGREEMENT_DIMS = ("overall", "temporal")

# =============================================================================
# 2. FORMATTING HELPERS FOR THE MARKDOWN REPORT
# =============================================================================

def _fmt(v) -> str:
    """
    Format a value for a markdown table: floats with 2 decimals, missing values as "-".
    :param v: value of a cell
    :return: the text to show
    """
    # Missing values (None or NaN) -> dash, so the table stays readable.
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "-"

    # Floats are rounded to 2 decimals; everything else (ints, strings,
    # categories) is converted to text as is.
    return f"{v:.2f}" if isinstance(v, float) else str(v)


def _md(df: pd.DataFrame) -> str:
    """
    Convert a DataFrame to a markdown table (used for report.md), without extra dependencies.
    :param df: table to convert
    :return: the table as markdown text
    """
    cols = list(df.columns)

    # First line: header "| col1 | col2 | ... |"
    # Second line: separator "|---|---|...|" required by Markdown syntax.
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]

    # One Markdown line per DataFrame row, each cell formatted by _fmt.
    lines += ["| " + " | ".join(_fmt(r[c]) for c in cols) + " |" for _, r in df.iterrows()]
    return "\n".join(lines) + "\n"

# =============================================================================
# 3. AGREEMENT BETWEEN JUDGES
# =============================================================================

def _agreement(
        gains: pd.DataFrame,
        judges: list[str]
) -> pd.DataFrame:
    """
    Measure how much the judges agree, for each pair of judges and for the dimensions in
    ``AGREEMENT_DIMS`` (overall and temporal). A pair of judges is reported only if they share at
    least 3 pairs with a valid gain. The jury is not included.
    :param gains: per-pair gains, with columns ``pair_id``, ``judge``, ``dimension``, ``gain``
    :param judges: names of the judges to compare
    :return: one row per dimension and pair of judges, with ``n`` (common pairs), ``spearman``
             (rank correlation of the gains, NaN if a judge gives constant gains),
             ``sign_agreement`` (share of pairs where both gains have the same sign) and
             ``mean_gain_a`` / ``mean_gain_b`` (mean gain of each judge)
    """
    rows = []
    for dim in AGREEMENT_DIMS:
        # Keep the current dimension and real judges only (the jury is excluded
        # because it is the mean of the judges: agreement with them would be inflated).
        # The pivot yields: rows = pair_id, columns = judge, values = gain.
        w = gains[(gains["dimension"] == dim) & gains["judge"].isin(judges)].pivot(
            index="pair_id", columns="judge", values="gain")

        # All unordered pairs of judges (a, b), without repetition.
        for a, b in itertools.combinations(judges, 2):
            # A judge may have no data for this dimension: skip.
            if a not in w or b not in w:
                continue
            # Keep only the pairs (pair_id) rated by both judges.
            x = w[[a, b]].dropna()

            # With fewer than 3 common pairs the metrics are meaningless.
            if len(x) < 3:
                continue

            # Spearman (rank) correlation: computed only if both judges have
            # non-constant gains, otherwise it is undefined -> NaN.
            rho = stats.spearmanr(x[a], x[b])[0] if x[a].nunique() > 1 and x[b].nunique() > 1 else np.nan
            rows.append({"dimension": dim, "judge_a": a, "judge_b": b, "n": len(x),
                            "spearman": float(rho),

                            # Share of pairs where both gains have the same sign
                            # (i.e. the judges agree on which system is better).
                            "sign_agreement": float((np.sign(x[a]) == np.sign(x[b])).mean()),

                            # Mean gain of each judge over the common pairs.
                            "mean_gain_a": float(x[a].mean()), "mean_gain_b": float(x[b].mean())})

    # Explicit columns: even if `rows` is empty the DataFrame has the right schema.
    return pd.DataFrame(rows, columns=["dimension", "judge_a", "judge_b", "n", "spearman",
                                       "sign_agreement", "mean_gain_a", "mean_gain_b"])

# =============================================================================
# 4. JUDGES' FLAGS
# =============================================================================

def _flags(flags: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Summarise the warnings reported by the judges in the final JSON.
    :param flags: one row per report and call, with ``judge``, ``system``, ``position``,
                  ``major_contradictions``, ``temporal_leakage`` and ``better_overall``
    :return: a tuple of two tables:
             - per judge and label: ``n_reports``, total ``major_contradictions`` and ``leakage_pct``
               (percentage of reports with temporal leakage);
             - per judge: how many times ``better_overall`` indicates each label, TIE or "n/a"
               (counted once per call)
    """
    # Table 1: per judge and system
    #   - n_reports: number of reports rated
    #   - major_contradictions: total major contradictions flagged
    #   - leakage_pct: % of reports with temporal leakage (boolean -> mean * 100)
    summ = (flags.groupby(["judge", "system"])
            .agg(n_reports=("pair_id", "size"), major_contradictions=("major_contradictions", "sum"),
                 leakage_pct=("temporal_leakage", lambda s: 100 * s.astype(bool).mean()))
            .reset_index())

    # Table 2: count of `better_overall` verdicts per judge.
    # Each call produces two rows (position A and B) with the same verdict:
    # filtering on "A" counts each call only once.
    # Missing verdicts are labelled "n/a" so they are not dropped by the groupby.
    votes = (flags[flags["position"] == "A"]  # one row per call (the verdict is the same for A and B)
             .assign(better_overall=lambda d: d["better_overall"].fillna("n/a"))
             .groupby(["judge", "better_overall"]).size().rename("count").reset_index())
    return summ, votes

# =============================================================================
# 5. MAIN FUNCTION
# =============================================================================


def analyze(
        out_dir: str | Path,
        cfg: RunConfig | None = None,
        judges: list[str] | None = None
) -> dict:
    """
    Analyse a results folder (local, no calls to the models, no tokens used). For each pair, judge
    and label it averages the scores over presentation orders and repetitions, adds the jury (plain
    mean of the judges), computes the gain ``second - first`` and its statistics, the agreement
    between judges and the judges' flags. Writes report.md and the CSV tables into the folder.
    :param out_dir: results folder created by ``Evaluator.evaluate``
    :param cfg: run options; if None they are read from ``config.json`` in the folder
    :param judges: judge names; if None they are read from ``config.json``
    :return: dict of DataFrames:
             ``scores`` (per pair, judge or jury, label: mean, overall, temporal),
             ``summary`` (mean and overall averaged over pairs),
             ``gain_stats`` (statistics of the gain per judge and dimension; the jury also has the
             bootstrap interval ``ci_low_boot`` / ``ci_high_boot``),
             ``agreement`` (agreement between judges),
             ``flags`` (contradictions and leakage) and ``votes`` (``better_overall`` counts)
    :raises RuntimeError: if the folder has no successful calls
    """
    # ---------------------------------------------------------------------
    # 5a. Configuration and record loading
    # ---------------------------------------------------------------------
    out = Path(out_dir)

    # If options or judges are not passed explicitly, read them from the
    # config.json saved in the folder at evaluation time.
    if cfg is None or judges is None:
        meta = json.loads((out / "config.json").read_text(encoding="utf-8"))
        cfg = cfg or RunConfig.model_validate(meta["options"])
        judges = judges or [j["name"] for j in meta["judges"]]

    # Load successful calls only.
    records = load_run_records(out)
    if not records:
        raise RuntimeError(f"no successful records under {out / 'runs'}")

    # ---------------------------------------------------------------------
    # 5b. Mean scores per pair, judge and system + jury
    # ---------------------------------------------------------------------
    # `long`: one score per row (pair_id, judge, system, dimension, score, ...)
    # `flags`: the judges' warnings (contradictions, leakage, verdict).
    long, flags = to_frames(records, cfg.labels)

    # Keep only judges actually present in the data (preserving config order).
    judges = [j for j in judges if j in set(long["judge"])]

    # Restrict to the dimensions of interest.
    d = long[long["dimension"].isin(GAIN_DIMS)]

    # Average scores over presentation orders and repetitions; then `unstack`
    # moves each dimension into its own column (overall, mean, temporal).
    # `reindex` ensures all columns exist and follow the GAIN_DIMS order.
    per_judge = (d.groupby(["pair_id", "judge", "system", "dimension"])["score"].mean()
                 .unstack("dimension").reindex(columns=list(GAIN_DIMS)).reset_index())

    # Jury: plain (equal-weight) mean of the judges for each pair and system.
    jury = per_judge.groupby(["pair_id", "system"], as_index=False)[list(GAIN_DIMS)].mean()
    jury["judge"] = JURY

    # The jury is appended as if it were another judge
    # (same columns in the same order as per_judge).
    scores = pd.concat([per_judge, jury[per_judge.columns]], ignore_index=True)

    # ---------------------------------------------------------------------
    # 5c. Gain per pair = second - first
    # ---------------------------------------------------------------------
    first, second = cfg.labels

    # For each dimension: pivot with systems as columns, take second - first,
    # back to long format with a `dimension` column. The three tables are concatenated.
    # Gain > 0 => the second system is rated better.
    gains = pd.concat([
        (scores.pivot(index=["pair_id", "judge"], columns="system", values=dim)
         .pipe(lambda w: (w[second] - w[first]).rename("gain")).reset_index().assign(dimension=dim))
        for dim in GAIN_DIMS], ignore_index=True)

    # ---------------------------------------------------------------------
    # 5d. Gain statistics per judge and dimension
    # ---------------------------------------------------------------------
    rows = []
    for (j, dim), g in gains.groupby(["judge", "dimension"]):
        # The (expensive) bootstrap interval is computed for the jury only,
        # which is the main result; for individual judges it stays NaN.
        lo, hi = bootstrap_ci(g["gain"], cfg.bootstrap, cfg.alpha, cfg.seed) if j == JURY else (np.nan, np.nan)

        # `summarize` returns a dict (n, mean_gain, sd, se, ci_low, ci_high,
        # dz, prob_pos, sign_p) that is unpacked into the row with **.
        rows.append({"judge": j, "dimension": dim, **summarize(g["gain"], cfg.alpha),
                     "ci_low_boot": lo, "ci_high_boot": hi})
    gain_stats = pd.DataFrame(rows)

    # ---------------------------------------------------------------------
    # 5e. Agreement between judges and flags
    # ---------------------------------------------------------------------
    agreement = _agreement(gains, judges)
    flag_summary, votes = _flags(flags)

    # ---------------------------------------------------------------------
    # 5f. Readable, stable ordering of the tables
    # ---------------------------------------------------------------------
    # Ordered categorical types: judges in config order with the jury last,
    # systems in label order (first, second).
    order = pd.CategoricalDtype(judges + [JURY], ordered=True)
    sys_order = pd.CategoricalDtype(list(cfg.labels), ordered=True)
    scores["judge"] = scores["judge"].astype(order)
    scores["system"] = scores["system"].astype(sys_order)
    scores = scores.sort_values(["pair_id", "judge", "system"]).reset_index(drop=True)

    gain_stats["judge"] = gain_stats["judge"].astype(order)

    # Dimensions in descending alphabetical order: temporal, overall, mean.
    gain_stats = gain_stats.sort_values(["judge", "dimension"], ascending=[True, False]).reset_index(drop=True)
    gains["judge"] = gains["judge"].astype(order)
    gains = gains.sort_values(["pair_id", "judge", "dimension"]).reset_index(drop=True)

    # ---------------------------------------------------------------------
    # 5g. Summary: mean over all pairs
    # ---------------------------------------------------------------------
    # `observed=True` avoids rows for category combinations absent from the data.
    summary = (scores.groupby(["judge", "system"], observed=True)
               .agg(n_pairs=("pair_id", "nunique"), mean=("mean", "mean"), overall=("overall", "mean"))
               .reset_index())

    # ---------------------------------------------------------------------
    # 5h. Write the CSV files
    # ---------------------------------------------------------------------
    out.mkdir(parents=True, exist_ok=True)
    long.to_csv(out / "scores_long.csv", index=False)  # raw scores, long format
    scores.to_csv(out / "scores.csv", index=False)  # means per pair/judge/system
    summary.to_csv(out / "summary.csv", index=False)  # means over all pairs
    gains.to_csv(out / "pair_gains.csv", index=False)  # gain per pair
    gain_stats.to_csv(out / "gain_stats.csv", index=False)  # gain statistics
    agreement.to_csv(out / "agreement.csv", index=False)  # agreement between judges
    flags.to_csv(out / "flags.csv", index=False)  # raw flags
    flag_summary.to_csv(out / "flags_summary.csv", index=False)  # summarised flags
    votes.to_csv(out / "votes.csv", index=False)  # better_overall counts

    # ---------------------------------------------------------------------
    # 5i. Write the Markdown report
    # ---------------------------------------------------------------------
    (out / "report.md").write_text(
        # Header with general information about the run
        "# Report\n\n"
        f"- records: {len(records)} | judges: {', '.join(judges)} | labels: {cfg.labels[0]}, {cfg.labels[1]}\n"
        "- scores are averaged over presentation orders and repetitions; "
        f"`{JURY}` is the mean of the judges' scores (equal weights)\n"
        "- `mean` and `overall` are computed in code from the four dimensions\n\n"
        
        # Section 1: averages over pairs
        "## Average over pairs\n\n" + _md(summary)

        # Section 2: gain statistics, with a legend for the columns
        + f"\n## Gain per pair ({second} - {first})\n\n"
        + _md(gain_stats[["judge", "dimension", "n", "mean_gain", "sd", "se", "ci_low", "ci_high",
                          "dz", "prob_pos", "sign_p", "ci_low_boot", "ci_high_boot"]])
        + "\nci = t interval of the mean gain; ci_*_boot = percentile bootstrap over pairs (jury only); "
        "dz = mean/sd; prob_pos = share of pairs with gain > 0; sign_p = sign-test p-value.\n\n"
          
        # Section 3: agreement between judges (or a message if there is not enough data)
        "## Agreement between judges (gain per pair)\n\n"
        + (_md(agreement) if len(agreement) else "not enough judges or pairs (needs 2 judges, 3 pairs)\n")

        # Section 4: flags and votes
        + "\n## Judges' flags\n\n" + _md(flag_summary)
        + "\n`better_overall` votes per judge:\n\n" + _md(votes)

        # Pointers to the detailed CSV files
        + "\nPer-pair scores: `scores.csv`, per-pair gains: `pair_gains.csv`, raw flags: `flags.csv`.\n",
        encoding="utf-8")

    # ---------------------------------------------------------------------
    # 5j. Return value: the main DataFrames, for use in a notebook or other code
    # ---------------------------------------------------------------------
    return {"scores": scores, "summary": summary, "gain_stats": gain_stats,
            "agreement": agreement, "flags": flag_summary, "votes": votes}