"""Does the presentation order (A/B) affect the judgement?

With PRE = baseline and POST = candidate, and "o1" = baseline shown as A, "o2" = candidate shown as A:

  H (position bias)  = ((PRE_o1 - PRE_o2) + (POST_o2 - POST_o1)) / 2
        score when a report is in position A minus when it is in position B.
        H > 0 favours the first position, H < 0 the second.  The order effect on the gain
        is -2*H and cancels when the two orders are averaged.
  S (common shift)   = ((PRE_o1 + POST_o1) - (PRE_o2 + POST_o2)) / 2
        both reports score higher in one order than in the other: a run/session effect,
        not a favour to A or B.
  Test-retest        correlation and sign agreement of the gain between the two orders.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .stats_utils import summarize


def order_effects(wide: pd.DataFrame, systems: tuple[str, str], alpha: float = 0.05) -> pd.DataFrame:
    base, cand = systems
    need = [f"AB_{base}", f"AB_{cand}", f"BA_{base}", f"BA_{cand}"]
    rows = []
    for (judge, dim), g in wide.groupby(["judge", "dimension"]):
        g = g.dropna(subset=need)
        if len(g) < 2:
            continue
        pre1, post1, pre2, post2 = (g[c].to_numpy(float) for c in need)
        h = ((pre1 - pre2) + (post2 - post1)) / 2
        s = ((pre1 + post1) - (pre2 + post2)) / 2
        g1, g2 = post1 - pre1, post2 - pre2
        corr = float(np.corrcoef(g1, g2)[0, 1]) if g1.std() > 0 and g2.std() > 0 else np.nan
        sh, ss = summarize(h, alpha), summarize(s, alpha)
        rows.append(
            {
                "judge": judge,
                "dimension": dim,
                "n_pairs": len(g),
                "H_mean": sh["mean"], "H_ci_low": sh["ci_low"], "H_ci_high": sh["ci_high"], "H_p": sh["p"],
                "S_mean": ss["mean"], "S_ci_low": ss["ci_low"], "S_ci_high": ss["ci_high"], "S_p": ss["p"],
                "gain_order1": float(g1.mean()), "gain_order2": float(g2.mean()),
                "gain_avg": float((g1.mean() + g2.mean()) / 2),
                "gain_corr_between_orders": corr,
                "sign_agreement": float((np.sign(g1) == np.sign(g2)).mean()),
            }
        )
    return pd.DataFrame(rows)
