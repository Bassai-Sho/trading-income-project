"""PR-002 box #2 (Zarattini ORB on stocks in play), run through the real
backtest runner and execution core on hand-built sessions."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from boxes.orb_stocks_in_play import OrbStocksInPlayBox, Params, State
from core.backtest_runner import run_box
from core.execution_core import AccountConfig

NY = "America/New_York"


def day(d="2019-03-12", n=390, first5=(100.0, 100.0, 100.0, 100.0), path=None):
    """Flat day at 100 except: the first 5 minutes follow `first5`
    (open, high, low, close of the 5-min candle, spread over the minutes),
    then `path` {minute_index: (o, h, l, c)} overrides individual minutes."""
    idx = pd.date_range(f"{d} 09:30", periods=n, freq="1min", tz=NY)
    o = np.full(n, 100.0); h = o + 0.01; l = o - 0.01; c = o.copy()
    fo, fh, fl, fc = first5
    o[0], c[4] = fo, fc
    h[:5] = fh; l[:5] = fl
    c[:4] = o[1:5] = np.linspace(fo, fc, 5)[:4]
    c[4] = fc
    if n > 5:
        o[5] = h[5] = l[5] = c[5] = fc
        o[5:] = c[5:] = fc; h[5:] = fc + 0.01; l[5:] = fc - 0.01
    for i, (a, b, cc, dd) in (path or {}).items():
        o[i], h[i], l[i], c[i] = a, b, cc, dd
    return pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c, "Volume": 1000.0}, index=idx)


def run(df, atr=2.0, sym="XYZ"):
    d = str(df.index[0].date())
    p = Params(atr14={d: atr})
    res = run_box(OrbStocksInPlayBox(), p, df, sym, AccountConfig(leverage=None))
    return res.state.journal


BULL = (100.0, 101.0, 99.8, 100.8)      # green first candle, high 101.0
BEAR = (100.0, 100.2, 99.0, 99.2)       # red first candle, low 99.0


def test_bullish_candle_goes_long_at_the_high_and_exits_at_the_close():
    df = day(first5=BULL, path={40: (100.9, 101.3, 100.9, 101.2), 389: (101.5, 101.6, 101.4, 101.55)})
    for i in range(41, 389):
        df.iloc[i, :4] = [101.3, 101.35, 101.25, 101.3]
    (t,) = run(df)
    assert (t.direction, t.entry_order_price, t.entry_price) == ("long", 101.0, 101.0)
    assert t.stop_price == pytest.approx(101.0 - 0.2)             # 10% of ATR 2.0 from the fill
    assert (t.exit_reason, t.exit_price) == ("close", 101.55)
    assert t.exit_ts.time().strftime("%H:%M") == "16:00"


def test_bearish_candle_goes_short_only():
    df = day(first5=BEAR, path={30: (99.1, 99.1, 98.8, 98.9)})
    for i in range(31, 390):
        df.iloc[i, :4] = [98.9, 98.95, 98.85, 98.9]
    (t,) = run(df)
    assert (t.direction, t.entry_price, t.exit_reason) == ("short", 99.0, "close")
    assert t.stop_price == pytest.approx(99.2)


def test_doji_first_candle_no_trade():
    assert run(day(first5=(100.0, 101.0, 99.0, 100.0), path={30: (100, 102, 98, 100)})) == ()


def test_wrong_way_break_is_ignored():
    df = day(first5=BULL, path={30: (99.9, 99.9, 98.0, 98.5)})     # breaks the LOW on a green candle
    assert run(df) == ()


def test_gap_through_the_entry_level_fills_at_the_open():
    df = day(first5=BULL, path={20: (101.6, 101.8, 101.5, 101.7)})
    for i in range(21, 390):
        df.iloc[i, :4] = [101.7, 101.75, 101.65, 101.7]
    (t,) = run(df)
    assert t.entry_price == 101.6 and t.stop_price == pytest.approx(101.4)


def test_stop_reached_in_the_entry_minute_is_a_loss():
    df = day(first5=BULL, path={20: (100.9, 101.1, 100.7, 101.05)})   # crosses 101 then 100.8
    (t,) = run(df)
    assert (t.exit_reason, t.exit_price) == ("stop", pytest.approx(100.8))
    assert t.exit_ts == t.entry_fill_ts


def test_half_day_exits_at_the_early_close():
    df = day("2019-11-29", n=210, first5=BULL, path={20: (100.9, 101.1, 100.85, 101.05)})
    for i in range(21, 210):
        df.iloc[i, :4] = [101.05, 101.1, 101.0, 101.08]
    (t,) = run(df)
    assert t.exit_reason == "close" and t.exit_ts.time().strftime("%H:%M") == "13:00"
    assert t.exit_price == pytest.approx(101.08)


def test_no_atr_no_trade_and_untriggered_entry_expires():
    df = day(first5=BULL, path={30: (100.9, 101.2, 100.9, 101.1)})
    assert run(df, atr=0.0) == ()
    assert run(day(first5=BULL)) == ()                              # never reaches 101


def test_one_trade_per_day_even_if_the_level_is_crossed_again():
    df = day(first5=BULL, path={20: (100.9, 101.1, 100.85, 101.0), 21: (101.0, 101.0, 100.7, 100.75),
                                 60: (100.9, 101.4, 100.9, 101.3)})
    assert len(run(df)) == 1


def test_box_state_is_frozen():
    import dataclasses
    assert State.__dataclass_params__.frozen and Params.__dataclass_params__.frozen
    with pytest.raises(dataclasses.FrozenInstanceError):
        State().phase = "x"
