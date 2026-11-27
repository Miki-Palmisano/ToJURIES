"""Summary statistics of a vector of per-pair gains (second system minus first)."""

# Enables modern type annotations (e.g. `tuple[float, float]`) on older Python versions.
from __future__ import annotations

import numpy as np
from scipy import stats

# =============================================================================
# 1. DESCRIPTIVE STATISTICS AND TESTS ON THE GAINS
# =============================================================================

def summarize(
        x,
        alpha: float = 0.05
) -> dict:
    """
    Summary statistics of the per-pair gains (second label minus first label) of one judge
    (or of the jury) on one dimension. Missing values (NaN) are ignored.
    :param x: gains, one per pair
    :param alpha: significance level of the interval (0.05 gives a 95% interval)
    :return: dict with
             ``n`` number of pairs used;
             ``mean_gain`` mean gain;
             ``sd`` and ``se`` standard deviation and standard error of the gains;
             ``ci_low`` and ``ci_high`` t interval of the mean gain;
             ``dz`` effect size, mean / sd;
             ``prob_pos`` share of pairs with gain > 0 (ties count half);
             ``sign_p`` p-value of the two-sided sign test (is the share of positive gains different from 50%?).
             The values that cannot be computed are NaN (e.g. ``sd`` and the interval with fewer than 2 pairs).
    """
    # Convert any input (list, Series, array) to a float array and drop NaNs
    # (e.g. pairs where a judge did not score one of the two systems).
    a = np.asarray(x, dtype=float)
    a = a[~np.isnan(a)]
    n = len(a)

    # Start with every statistic set to NaN: whatever cannot be computed
    # (too few pairs) stays NaN, and the dict always has the same keys.
    out = {"n": n, "mean_gain": np.nan, "sd": np.nan, "se": np.nan, "ci_low": np.nan,
           "ci_high": np.nan, "dz": np.nan, "prob_pos": np.nan, "sign_p": np.nan}

    # No valid pairs: return only n = 0.
    if n == 0:
        return out

    # --- Statistics that need at least 1 pair --------------------------------
    mean = float(a.mean())
    out["mean_gain"] = mean

    # Share of positive gains; ties (gain == 0) count as half a "win",
    # so 0.5 means "no preference" between the two systems.
    out["prob_pos"] = float(((a > 0).sum() + 0.5 * (a == 0).sum()) / n)  # ties count half

    # Sign test: ties carry no information about direction and are excluded.
    # Under the null hypothesis, the number of positive gains among the non-zero
    # ones follows a Binomial(nonzero, 0.5); binomtest is two-sided by default.
    nonzero = int((a != 0).sum())
    if nonzero > 0:
        out["sign_p"] = float(stats.binomtest(int((a > 0).sum()), nonzero, 0.5).pvalue)

    # --- Statistics that need at least 2 pairs -------------------------------
    # With a single pair the sample standard deviation is undefined.
    if n < 2:
        return out

    # Sample standard deviation (ddof=1, i.e. divided by n - 1).
    sd = float(a.std(ddof=1))

    # Standard error of the mean.
    se = sd / np.sqrt(n)

    # Critical value of Student's t with n - 1 degrees of freedom
    # (e.g. alpha = 0.05 -> 97.5th percentile, for a two-sided 95% interval).
    tcrit = stats.t.ppf(1 - alpha / 2, n - 1)

    # t interval: mean ± tcrit * se.
    # dz (Cohen's d for paired data) = mean / sd; undefined if all gains are equal (sd = 0).
    out.update(sd=sd, se=float(se), ci_low=float(mean - tcrit * se), ci_high=float(mean + tcrit * se),
               dz=mean / sd if sd > 0 else np.nan)
    return out

# =============================================================================
# 2. BOOTSTRAP CONFIDENCE INTERVAL
# =============================================================================

def bootstrap_ci(
        x,
        n_boot: int = 5000,
        alpha: float = 0.05,
        seed: int = 0
) -> tuple[float, float]:
    """
    Percentile bootstrap interval of the mean gain: the pairs are resampled with replacement
    ``n_boot`` times, the mean is computed each time, and the interval is given by the
    ``alpha/2`` and ``1 - alpha/2`` percentiles of those means. It does not assume that the gains
    are normally distributed. Missing values (NaN) are ignored.
    :param x: gains, one per pair
    :param n_boot: number of resamples
    :param alpha: significance level (0.05 gives a 95% interval)
    :param seed: seed of the random generator, so the result is reproducible
    :return: ``(lower, upper)`` bounds of the interval; ``(nan, nan)`` with fewer than 2 pairs
    """
    # Same cleaning as in summarize: float array without NaNs.
    a = np.asarray(x, dtype=float)
    a = a[~np.isnan(a)]

    # With fewer than 2 pairs every resample is identical: the interval is meaningless.
    if len(a) < 2:
        return float("nan"), float("nan")

    # Random generator with a fixed seed -> same interval on every run.
    rng = np.random.default_rng(seed)

    # Vectorised resampling: an (n_boot x n) matrix of random indices,
    # each row is one resample with replacement of the n pairs.
    # The mean of each row gives n_boot bootstrap means.
    means = a[rng.integers(0, len(a), size=(n_boot, len(a)))].mean(axis=1)

    # The interval bounds are the alpha/2 and 1 - alpha/2 quantiles of the means
    # (e.g. 2.5th and 97.5th percentiles for a 95% interval).
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)