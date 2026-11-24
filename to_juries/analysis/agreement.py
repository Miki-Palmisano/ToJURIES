"""Between-judge agreement on the per-pair gains (scale-free: ranks and signs)."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats


def judge_agreement(matrix: pd.DataFrame, dimension: str) -> pd.DataFrame:
    """Pairwise Spearman correlation and sign agreement of per-pair gains between judges."""
    rows = []
    for a, b in combinations(list(matrix.columns), 2):
        d = matrix[[a, b]].dropna()
        n = len(d)
        if n < 3:
            continue
        x, y = d[a].to_numpy(float), d[b].to_numpy(float)
        rho = float(stats.spearmanr(x, y).statistic) if x.std() > 0 and y.std() > 0 else np.nan
        rows.append(
            {
                "dimension": dimension, "judge_a": a, "judge_b": b, "n_pairs": n,
                "spearman": rho,
                "sign_agreement": float((np.sign(x) == np.sign(y)).mean()),
                "mean_gain_a": float(x.mean()), "mean_gain_b": float(y.mean()),
            }
        )
    return pd.DataFrame(rows)
