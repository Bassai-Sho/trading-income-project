"""PR-002 cost tools: cost in R, exact break-even, Abdi-Ranaldo spread estimator."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from evaluation.costs import abdi_ranaldo, breakeven_bps, cost_r


def test_cost_r_matches_hand_calculation():
    # $50 stock, stop $0.20: commission 2*0.0035/0.2 = 0.035R; 5 bps/side = 2*0.0025/0.2 = 0.025*... 
    assert cost_r(50, 50, 0.2) == pytest.approx(0.035)
    assert cost_r(50, 50, 0.2, bps_per_side=5) == pytest.approx(0.035 + 5e-4 * 100 / 0.2)


def test_breakeven_is_exact():
    rng = np.random.default_rng(0)
    g = rng.normal(0.3, 1, 500); e = rng.uniform(20, 200, 500); x = e * (1 + rng.normal(0, .01, 500))
    d = e * 0.004
    b = breakeven_bps(g, e, x, d)
    assert (g - cost_r(e, x, d, bps_per_side=b)).mean() == pytest.approx(0, abs=1e-9)


def _market(spread, n_min=2000, trades_per_min=20, seed=1, px=50.0, vol=0.0003):
    """Mid follows a random walk; trades print at bid or ask at random."""
    rng = np.random.default_rng(seed)
    mid = px * np.exp(np.cumsum(rng.normal(0, vol / np.sqrt(trades_per_min), n_min * trades_per_min)))
    side = rng.choice([-1, 1], len(mid))
    p = mid * (1 + side * spread / 2)
    bars = p.reshape(n_min, trades_per_min)
    return bars.max(1), bars.min(1), bars[:, -1]


@pytest.mark.parametrize("spread", [0.0005, 0.001, 0.003])        # 5, 10, 30 bps
def test_abdi_ranaldo_recovers_a_known_spread(spread):
    h, l, c = _market(spread)
    assert abdi_ranaldo(h, l, c) == pytest.approx(spread, rel=0.35)


def test_abdi_ranaldo_near_zero_without_a_spread():
    h, l, c = _market(0.0)
    assert abdi_ranaldo(h, l, c) < 0.0003
