"""Small statistical helpers shared by the analyses."""

from __future__ import annotations

import numpy as np
from scipy import stats


def summarize(x, alpha: float = 0.05) -> dict:
    """Summary of a one-sample (e.g. paired-difference) vector."""
    a = np.asarray(x, dtype=float)
    a = a[~np.isnan(a)]
    n = len(a)
    out = {
        "n": n, "mean": np.nan, "sd": np.nan, "se": np.nan, "t": np.nan, "p": np.nan,
        "ci_low": np.nan, "ci_high": np.nan, "dz": np.nan, "prob_pos": np.nan, "sign_p": np.nan,
    }
    if n == 0:
        return out
    mean = float(a.mean())
    out["mean"] = mean
    out["prob_pos"] = float(((a > 0).sum() + 0.5 * (a == 0).sum()) / n)
    nz = int((a != 0).sum())
    if nz > 0:
        out["sign_p"] = float(stats.binomtest(int((a > 0).sum()), nz, 0.5).pvalue)
    if n < 2:
        return out
    sd = float(a.std(ddof=1))
    se = sd / np.sqrt(n)
    out["sd"], out["se"] = sd, float(se)
    out["dz"] = mean / sd if sd > 0 else np.nan
    if se > 0:
        t = mean / se
        tcrit = stats.t.ppf(1 - alpha / 2, n - 1)
        out["t"] = float(t)
        out["p"] = float(2 * stats.t.sf(abs(t), n - 1))
        out["ci_low"], out["ci_high"] = float(mean - tcrit * se), float(mean + tcrit * se)
    else:
        out["p"] = 1.0 if mean == 0 else 0.0
        out["ci_low"] = out["ci_high"] = mean
    return out


def bootstrap_ci(values: np.ndarray, weights: np.ndarray | None, n_boot: int, alpha: float, seed: int):
    """Percentile bootstrap CI of the (row-weighted) mean over resampled rows (pairs).

    ``values`` has shape (n_pairs, n_judges); ``weights`` (n_judges,) are fixed judge weights;
    NaNs are skipped and weights renormalised per pair.
    """
    v = np.asarray(values, dtype=float)
    if v.ndim == 1:
        v = v[:, None]
    w = np.ones(v.shape[1]) if weights is None else np.asarray(weights, dtype=float)

    def pair_means(mat: np.ndarray) -> np.ndarray:
        mask = ~np.isnan(mat)
        num = np.where(mask, mat, 0.0) @ w
        den = mask @ w
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(den > 0, num / den, np.nan)

    pm = pair_means(v)
    pm = pm[~np.isnan(pm)]
    if len(pm) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(pm), size=(n_boot, len(pm)))
    means = pm[idx].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))
