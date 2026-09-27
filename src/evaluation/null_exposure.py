"""
evaluation/null_exposure.py  (PR-003 Stage V)
==============================================
A null-alpha test for strategies whose trades are a DIRECTIONAL EXPOSURE that
can flip several times a day (box #3: long/short/flat set at each half-hour
check, no fixed stop-loss) -- the evaluation/null_baseline.py machinery
(bracket trade: one entry, one stop, one exit) does not fit this trade
structure, so it is not reused here; forcing it on would test the wrong
mechanism (see PR-003 Notion log, 27 Sep 2026).

NULL: for each day, was the STRATEGY'S CHOSEN DIRECTION that day better than
a coin flip, given the exact same trade timing, position sizes and costs?
    actual_net(d)  = gross(d) - cost(d)
    mirrored_net(d) = -gross(d) - cost(d)     (opposite call, same cost: a
                       wrong guess still pays the same commission and slippage)
Bootstrap: for `draws` iterations, flip each day independently with p=0.5,
take the mean of the resulting daily series; compare the actual mean's rank.
p = (1 + #null means >= observed) / (1 + draws).

Requires gross and cost split PER DAY (mes_costs / pr003_refine already
compute cost fraction per day from execution fees).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def null_alpha_exposure(gross: pd.Series, cost: pd.Series, draws: int = 1000,
                        alpha: float = 0.05, seed: int = 0) -> dict:
    gross, cost = gross.align(cost, join="inner")
    g, c = gross.to_numpy(), cost.to_numpy()
    observed = float((g - c).mean())
    rng = np.random.default_rng(seed)
    n = len(g)
    signs = rng.choice([-1.0, 1.0], size=(draws, n))
    null_means = (signs * g - c).mean(axis=1)
    p = (1 + int((null_means >= observed).sum())) / (1 + draws)
    return {"n_days": n, "observed": observed, "null_mean": float(null_means.mean()),
           "null_p95": float(np.percentile(null_means, 95)), "p_value": p,
           "passed": p < alpha}
