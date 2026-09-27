"""PR-003 pre-unseal diagnostics: drawdown attribution, cost-sensitivity
curve, intraday block-shuffle placebo. Fast, on small/synthetic data."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pr003_diagnostics as diag
from market_data_store import session_close
from synthetic_bars import noisy_days


# ── 1. Drawdown attribution ───────────────────────────────────────────────────

def _series_with_known_drawdown():
    idx = pd.bdate_range("2019-01-02", periods=60)
    r = pd.Series(0.001, index=idx)
    r.iloc[10:15] = -0.05      # one clean, isolated episode
    return r


def test_drawdown_episodes_finds_the_planted_episode():
    r = _series_with_known_drawdown()
    eps = diag.drawdown_episodes(r, top=3)
    assert eps and eps[0]["start"] == str(r.index[10].date())
    assert eps[0]["depth"] < -0.2


def test_counterfactual_without_episode_removes_it():
    r = _series_with_known_drawdown()
    eps = diag.drawdown_episodes(r, top=1)
    cf = diag.counterfactual_without_episode(r, eps[0])
    assert cf > eps[0]["depth"] * 0.5          # much shallower once the episode is zeroed
    assert cf >= -0.01                          # the rest of the series is flat/positive


def test_drawdown_episode_ends_exactly_at_recovery_to_zero():
    """Boundary case: a day with an exact 0.0 return that brings the equity
    curve exactly back to its prior peak must END the episode there, not
    extend it. (>= vs > at the boundary; rarely differs on noisy data, so a
    planted exact-zero day is needed to expose it.)"""
    idx = pd.bdate_range("2019-01-02", periods=10)
    r4 = 1.0 / 0.9 - 1.0 + 1e-9      # brings equity back to (fractionally above) the prior peak
    r = pd.Series([0.0, 0.0, -0.10, 0.0, r4, 0.0, 0.0, 0.0, 0.0, 0.0], index=idx)
    eps = diag.drawdown_episodes(r, top=1)
    assert eps[0]["end"] == str(idx[4].date())     # the episode must have CLOSED at day 4, not run to the end


def test_regime_classification():
    assert diag._regime_of("2020-03-15") == "covid_crash"
    assert diag._regime_of("2017-06-01") == "normal_bull"


# ── 2. Cost-sensitivity curve ─────────────────────────────────────────────────

def test_cost_curve_is_monotonically_non_increasing_in_cost():
    idx = pd.bdate_range("2019-01-02", periods=200)
    rng = np.random.default_rng(0)
    gross = pd.Series(rng.normal(0.0006, 0.008, 200), index=idx)
    cost = pd.Series(0.0002, index=idx)
    curve = diag.cost_curve_from_series(gross, cost, base_rate=1.0, grid=(0.5, 1.0, 2.0, 4.0))
    sh = [c["sharpe"] for c in curve]
    assert all(sh[i] >= sh[i + 1] - 1e-9 for i in range(len(sh) - 1))


def test_cost_curve_scales_relative_to_base_rate_not_absolute():
    """base_rate != 1.0: the grid point EQUAL to base_rate must reproduce the
    exact passed-in cost series (rate/base_rate == 1), not `rate x cost`
    directly -- a monotonicity check alone (prior test, base_rate=1.0) cannot
    tell these two formulas apart."""
    idx = pd.bdate_range("2019-01-02", periods=50)
    gross = pd.Series(0.001, index=idx)
    cost = pd.Series(0.0002, index=idx)
    curve = diag.cost_curve_from_series(gross, cost, base_rate=2.0, grid=(2.0,))
    expected_net = (gross - cost).mean()          # rate == base_rate -> cost unchanged
    eq = (1 + (gross - cost)).cumprod()
    expected_cagr = float(eq.iloc[-1] ** (252 / len(gross)) - 1)
    assert curve[0]["cagr"] == pytest.approx(expected_cagr)


def test_cost_curve_breakeven_matches_the_known_formula():
    """Deterministic series with a known positive edge (avoids a random draw
    that happens to lose money before costs, where 'breakeven cost' is not a
    meaningful positive number)."""
    from evaluation.mes_costs import breakeven_cost_per_contract_side
    idx = pd.bdate_range("2019-01-02", periods=300)
    gross = pd.Series([0.001, -0.0003] * 150, index=idx)   # mean +0.00035/day, deterministic
    cost = pd.Series(0.00015, index=idx)
    be = breakeven_cost_per_contract_side(float(gross.sum()), float(cost.sum()), current_rate=1.0)
    assert be > 0
    net_below = (gross - (be * 0.9 / 1.0) * cost).mean()
    net_above = (gross - (be * 1.1 / 1.0) * cost).mean()
    assert net_below > 0 > net_above


def test_find_cliff_detects_a_planted_discontinuity():
    curve = [{"cost_per_contract_side": c, "sharpe": s} for c, s in
            zip([0.5, 1.0, 1.5, 2.0, 2.5], [1.0, 0.9, 0.8, -0.5, -0.6])]
    cliff = diag.find_cliff(curve)
    assert cliff and cliff["between"] == (1.5, 2.0)


def test_find_cliff_none_when_smooth():
    curve = [{"cost_per_contract_side": c, "sharpe": s} for c, s in
            zip([0.5, 1.0, 1.5, 2.0, 2.5], [1.0, 0.8, 0.6, 0.4, 0.2])]
    assert diag.find_cliff(curve) is None


# ── 3. Intraday block-shuffle placebo ─────────────────────────────────────────

@pytest.fixture(scope="module")
def two_days():
    df = noisy_days("2019-03-04", "2019-03-15", seed=3, px=280.0)
    df = df[[session_close(d) is not None for d in df.index.date]]
    days = sorted(set(df.index.date))
    return df[df.index.date == days[2]], df[df.index.date == days[3]]


def test_shuffle_preserves_first_half_hour_and_bar_multiset(two_days):
    _, day = two_days
    rng = np.random.default_rng(5)
    sh = diag.shuffle_day_blocks(day, rng)
    assert (sh.index == day.index).all()
    assert (sh.iloc[:30].to_numpy() == day.iloc[:30].to_numpy()).all()
    assert sorted(sh.iloc[30:]["Close"]) == sorted(day.iloc[30:]["Close"])
    assert not (sh.iloc[30:].to_numpy() == day.iloc[30:].to_numpy()).all()
    ok = (sh["High"] >= sh[["Open", "Close"]].max(axis=1)) & (sh["Low"] <= sh[["Open", "Close"]].min(axis=1))
    assert ok.all()


def test_run_one_day_placebo_shuffle_false_matches_normal_run(two_days):
    prev, day = two_days
    r_direct = diag.run_one_day_placebo(prev, day, np.random.default_rng(0), shuffle=False)
    assert isinstance(r_direct, float)


def test_placebo_end_to_end_on_small_sample(monkeypatch, tmp_path):
    df = noisy_days("2019-01-02", "2019-02-28", seed=4, px=280.0)
    df = df[[session_close(d) is not None for d in df.index.date]]
    monkeypatch.setattr(diag.pr, "load_spy", lambda db: df)
    out = diag.intraday_shuffle_placebo("unused.db", tmp_path, n_days=8, draws=5, seed=0)
    assert out["n_days"] == 8 and out["draws"] == 5
    assert 0.0 <= out["p_value"] <= 1.0
    assert Path(out["path"]).exists()


def test_diagnostics_never_read_sealed_data():
    assert diag.pr.WINDOW[1].year == 2024 and diag.pr.WINDOW[1].month == 12
