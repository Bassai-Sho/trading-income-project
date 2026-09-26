"""
evaluation/costs.py  (PR-002)
=============================
Trading costs in R, the break-even cost, and a spread estimate from bars.

cost_r        round-trip cost of one trade in R: commission per share on both
              sides + `bps_per_side` of price on both sides (spread + slippage),
              divided by the stop distance. Share count cancels out, so this is
              independent of position size.
breakeven_bps the per-side bps cost at which mean net R = 0 (costs are linear
              in bps, so this is exact). THE decision number for PR-002.
abdi_ranaldo  effective spread (fraction of price) from high / low / close bars:
              Abdi & Ranaldo (2017, RFS), "A Simple Estimation of Bid-Ask
              Spreads from Daily Close, High, and Low Prices" — the per-pair
              negative estimates set to 0 ("two-period corrected" version).
              Applied here to 1-minute bars; validated against a known spread
              in tests and on SPY before it is trusted (PR-002).
"""
from __future__ import annotations

import numpy as np

COMMISSION = 0.0035          # $/share/side (the paper's IBKR Pro tiered figure)


def cost_r(entry, exit_, stop_dist, commission: float = COMMISSION, bps_per_side: float = 0.0):
    e, x, d = (np.asarray(v, float) for v in (entry, exit_, stop_dist))
    return (2 * commission + bps_per_side * 1e-4 * (e + x)) / d


def breakeven_bps(gross_r, entry, exit_, stop_dist, commission: float = COMMISSION) -> float:
    g, e, x, d = (np.asarray(v, float) for v in (gross_r, entry, exit_, stop_dist))
    after_comm = (g - 2 * commission / d).mean()
    per_bps = (1e-4 * (e + x) / d).mean()
    return float(after_comm / per_bps) if per_bps > 0 else float("nan")


def abdi_ranaldo(high, low, close) -> float:
    """Effective spread as a fraction of price (e.g. 0.001 = 10 bps full spread)."""
    h, l, c = (np.log(np.asarray(v, float)) for v in (high, low, close))
    if len(c) < 3:
        return float("nan")
    eta = (h + l) / 2
    s2 = 4 * (c[:-1] - eta[:-1]) * (c[:-1] - eta[1:])
    return float(np.sqrt(np.clip(s2, 0, None)).mean())
