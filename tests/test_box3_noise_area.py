"""PR-003 box #3 (noise-area intraday momentum) through the real runner + core,
and agreement with the authors' published reference code."""
import sys
from datetime import datetime, time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pr003_replication as pr
from boxes.noise_area_momentum import NoiseAreaMomentumBox, Params, State
from core.backtest_runner import run_box
from core.execution_core import AccountConfig
from core.interfaces import Bar, CoreView
from evaluation.reference_noise_area import reference_backtest
from market_data_store import session_close
from synthetic_bars import noisy_days

NY = "America/New_York"


@pytest.fixture(scope="module")
def data():
    df = noisy_days("2019-01-02", "2019-06-28", seed=5, px=280.0)
    return df[[session_close(d) is not None for d in df.index.date]]


def _run(df, **kw):
    return run_box(NoiseAreaMomentumBox(), Params(**kw), df, "SPY",
                   AccountConfig(cash=100_000.0, leverage=None), fee_fn=pr.fee_fn("fidelity"),
                   keep_reports=True)


def test_trades_only_on_half_hours_at_the_next_open_and_flat_overnight(data):
    res = _run(data)
    fills = [r for r in res.reports if r.status == "FILLED"]
    trades = [r for r in fills if r.tag == "trade"]
    closes = [r for r in fills if r.tag == "close"]
    assert trades and closes
    opens = data["Open"]
    for r in trades:
        ts = pd.Timestamp(r.timestamp)
        assert ts.minute in (0, 30) and time(10, 0) <= ts.time() <= time(15, 30)
        assert r.fill_price == pytest.approx(float(opens.loc[ts]))          # next minute's open
    last_close = data.groupby(data.index.date)["Close"].last()
    for r in closes:                                                      # market-on-close
        assert r.fill_price == pytest.approx(float(last_close.loc[pd.Timestamp(r.timestamp).date()]))
    eq = pd.Series(res.equity_by_day)
    assert len(eq) == len(set(data.index.date))


def test_agrees_with_the_authors_code_after_warm_up(data):
    box = pr.daily_returns(_run(data).equity_by_day)
    ref = reference_backtest(data, {}).dropna()
    j = pd.concat([box.rename("box"), ref.rename("ref")], axis=1).dropna().iloc[20:]
    j = j[(j.box != 0) | (j.ref != 0)]
    assert len(j) > 30 and j.box.corr(j.ref) >= 0.99
    assert (((j.box != 0) ^ (j.ref != 0)).sum()) == 0                   # same trading days


def _bar(ts, o, h, l, c, v=1000.0):
    return Bar("SPY", pd.Timestamp(ts, tz=NY).to_pydatetime(), o, h, l, c, v)


def _view(eq=100_000.0):
    return CoreView(datetime(2019, 3, 12), eq, eq, eq)


def test_dividend_adjusts_the_previous_close_in_the_band_base():
    box, s = NoiseAreaMomentumBox(), State(day=None)
    s, _ = box.on_bar(_bar("2019-03-11 15:59", 100, 100, 100, 100.0), s, Params(), _view())
    p = Params(dividends={"2019-03-12": 1.0})
    s1, _ = box.on_bar(_bar("2019-03-12 09:30", 99.5, 99.6, 99.4, 99.5), s, p, _view())
    s2, _ = box.on_bar(_bar("2019-03-12 09:30", 99.5, 99.6, 99.4, 99.5), s, Params(), _view())
    assert s1.ref_price == 99.5 and s1.lo_base == 99.0          # base = max/min(open, 100 - 1)
    assert s2.ref_price == 100.0 and s2.lo_base == 99.5         # no dividend: base uses 100


def test_vol_target_sizing_and_leverage_cap():
    box = NoiseAreaMomentumBox()
    closes = tuple(100 * (1 + 0.001 * (-1) ** i) for i in range(15))  # ~0.2% daily vol -> capped at 4x
    s = State(day=None, closes=closes, last_close=closes[-1], prev_close=closes[-1])
    s, _ = box.on_bar(_bar("2019-03-12 09:30", 100, 100, 100, 100.0), s, Params(), _view(50_000))
    assert s.shares == round(50_000 * 4 / 100)
    vol = tuple(100 * (1 + 0.02 * (-1) ** i) for i in range(15))       # ~4% vol -> 0.5x
    s = State(day=None, closes=vol, last_close=vol[-1], prev_close=vol[-1])
    s, _ = box.on_bar(_bar("2019-03-12 09:30", 100, 100, 100, 100.0), s, Params(), _view(50_000))
    rets = [vol[i] / vol[i - 1] - 1 for i in range(1, 15)]
    assert s.shares == round(50_000 * min(4, 0.02 / np.std(rets, ddof=1)) / 100)
    s = State(day=None, closes=vol, last_close=vol[-1], prev_close=vol[-1])
    s, _ = box.on_bar(_bar("2019-03-12 09:30", 100, 100, 100, 100.0), s, Params(sizing="full"), _view(50_000))
    assert s.shares == 500


def test_half_day_positions_close_at_the_early_close(data):
    extra = noisy_days("2019-11-01", "2019-11-29", seed=9, px=300.0)
    extra = extra[[session_close(d) is not None for d in extra.index.date]]
    half = extra[extra.index.date == pd.Timestamp("2019-11-29").date()]
    extra = pd.concat([extra[extra.index.date != half.index[0].date()], half.iloc[:210]])
    res = _run(extra)
    late = [r for r in res.reports if r.status == "FILLED" and pd.Timestamp(r.timestamp).date() ==
            pd.Timestamp("2019-11-29").date()]
    assert all(pd.Timestamp(r.timestamp).time() <= time(13, 0) for r in late)


def test_replication_scoring_against_the_paper_table():
    pm = pr.paper_monthly()
    e = pr.evaluate(pm.copy(), pm)                     # identical series -> perfect replication
    assert e["monthly_corr"] == pytest.approx(1.0) and all(e["gates"].values())
    noise = pm * -1                                    # inverted -> must fail
    assert not pr.evaluate(noise, pm)["gates"]["corr"]
