"""P2-122: fill_model='resting' in trading_engine._backtest_orb_full_gate.
Each rule is pinned to an exact price on a hand-built session; the default
('legacy') is proven unchanged."""
import hashlib, json, statistics, sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from trading_engine import _backtest_orb_full_gate, CONFIG
from historical_sim import _to_5min

CFG = {**CONFIG, "ticker": "SPY", "account_balance": 10_000.0, "risk_pct": 0.01}
D = "2019-03-12"


def _session(flat_after=None):
    """Flat ORB at ~100, steady climb from 09:45, optionally flat from `flat_after`."""
    idx = pd.date_range(f"{D} 09:30", f"{D} 15:59", freq="1min", tz="America/New_York")
    t = np.arange(len(idx))
    c = 100 + np.where(t < 15, 0.05 * np.sin(t), 0.05 * (t - 15))
    if flat_after is not None:
        c = np.where(t >= flat_after, c[flat_after], c)
    o = np.r_[c[0], c[:-1]]
    return pd.DataFrame({"Open": o, "High": np.maximum(o, c) + 0.01,
                         "Low": np.minimum(o, c) - 0.01, "Close": c, "Volume": 1000.0}, index=idx)


def _run(df, **kw):
    return _backtest_orb_full_gate(_to_5min(df), CFG, fill_model="resting", df_1m=df, **kw)


def _first(df, **kw):
    r = _run(df, **kw)
    assert r["n"] >= 1, r
    return r["trades"][0]


@pytest.fixture
def base():
    df = _session()
    return df, _first(df)


def _minute(ts):
    return pd.Timestamp(ts).tz_convert("America/New_York")


def test_entry_fills_at_next_minute_open_and_r_is_from_the_fill(base):
    df, t = base
    fill_ts = _minute(t["entry_fill_ts"])
    assert fill_ts == _minute(t["entry_ts"]) + pd.Timedelta(minutes=5)
    df = df.copy()                                    # make the next open differ from the close
    df.loc[fill_ts, ["Open", "High"]] = [t["signal_close"] + 0.07, t["signal_close"] + 0.08]
    t2 = _first(df)
    assert t2["entry_price"] == pytest.approx(t["signal_close"] + 0.07, abs=1e-4)
    risk = t2["entry_price"] - t2["stop_price"]
    assert risk == pytest.approx(t["signal_close"] + 0.07 - t["stop_price"], abs=1e-4)


def test_wick_through_stop_fills_at_stop(base):
    df, t = base
    m = _minute(t["entry_fill_ts"]) + pd.Timedelta(minutes=2)
    df = df.copy(); df.loc[m, "Low"] = t["stop_price"] - 0.01      # wick only; close unchanged
    t2 = _first(df)
    assert (t2["exit_reason"], _minute(t2["exit_ts"])) == ("trailing_stop", m)
    assert t2["exit_price"] == pytest.approx(t["stop_price"], abs=1e-4)
    legacy = _backtest_orb_full_gate(_to_5min(df), CFG)["trades"][0]
    assert legacy["exit_reason"] != "trailing_stop" or _minute(legacy["exit_ts"]) != m


def test_gap_through_stop_fills_at_the_open(base):
    df, t = base
    m = _minute(t["entry_fill_ts"]) + pd.Timedelta(minutes=2)
    df = df.copy(); gap = t["stop_price"] - 0.30
    df.loc[m, ["Open", "Low"]] = [gap, gap - 0.01]
    t2 = _first(df)
    assert t2["exit_reason"] == "trailing_stop" and t2["exit_price"] == pytest.approx(gap, abs=1e-4)


def test_target_wick_fills_at_t1(base):
    df, t = base
    risk = t["entry_price"] - t["stop_price"]
    t1 = t["entry_price"] + risk
    m = _minute(t["entry_fill_ts"]) + pd.Timedelta(minutes=1)
    if _minute(t["exit_ts"]) <= m:
        pytest.skip("base trade already exited")
    df = df.copy(); df.loc[m, "High"] = t1 + 0.01
    t2 = _first(df)
    assert (t2["exit_reason"], _minute(t2["exit_ts"])) == ("target_hit", m)
    assert t2["exit_price"] == pytest.approx(t1, abs=1e-4)


def test_minute_touching_both_is_a_stop(base):
    df, t = base
    risk = t["entry_price"] - t["stop_price"]
    m = _minute(t["entry_fill_ts"]) + pd.Timedelta(minutes=1)
    df = df.copy()
    df.loc[m, ["High", "Low"]] = [t["entry_price"] + risk + 0.05, t["stop_price"] - 0.05]
    t2 = _first(df)
    assert t2["exit_reason"] == "trailing_stop" and t2["exit_price"] == pytest.approx(t["stop_price"], abs=1e-4)


def test_eod_exit_at_open_of_session_end_minute():
    df = _session(flat_after=40)                      # climb, then flat: no stop, no target
    r = _run(df, )
    eod = [t for t in r["trades"] if t["exit_reason"] == "eod"]
    assert eod, [t["exit_reason"] for t in r["trades"]]
    t = eod[-1]
    end = pd.Timestamp(f"{D} 11:00", tz="America/New_York")
    assert _minute(t["exit_ts"]) == end
    assert t["exit_price"] == pytest.approx(df.loc[end, "Open"], abs=1e-4)


def test_no_entry_from_a_signal_at_session_end():
    df = _session()
    r = _run(df)
    assert all(_minute(t["entry_ts"]).time() < CONFIG["session_end"] for t in r["trades"])


def test_entry_that_fills_through_the_stop_is_skipped(base):
    df, t = base
    fill = _minute(t["entry_fill_ts"])
    df = df.copy(); df.loc[fill, ["Open", "Low"]] = [t["stop_price"] - 0.2, t["stop_price"] - 0.3]
    r = _run(df)
    assert r["skipped_entries"] >= 1
    assert all(_minute(x["entry_fill_ts"]) != fill for x in r.get("trades", []))


def test_signal_close_entry_option_for_attribution(base):
    df, _ = base
    t = _first(df, entry_fill="signal_close")
    assert t["entry_price"] == t["signal_close"]


def test_argument_validation():
    df = _session()
    with pytest.raises(ValueError, match="df_1m"):
        _backtest_orb_full_gate(_to_5min(df), CFG, fill_model="resting")
    with pytest.raises(ValueError, match="fill_model"):
        _backtest_orb_full_gate(_to_5min(df), CFG, fill_model="magic")


def test_legacy_default_is_unchanged():
    """Hash of the pre-P2-122 output on fixed synthetic data (all three exit modes)."""
    from test_pr001_runner import _days
    from test_pr001_fill_diagnostic import _reversal_days
    df = pd.concat([_days("2019-03-01", "2019-04-30", seed=2), _reversal_days("2019-05-01", "2019-05-31")])
    out = {}
    for m in ("baseline", "fixed_1_5r", "trail_after_1r"):
        r = _backtest_orb_full_gate(_to_5min(df), CFG, exit_mode=m)
        out[m] = [[t["session_date"], str(t["entry_ts"]), str(t["exit_ts"]), t["exit_reason"],
                   round(t["actual_r"], 10)] for t in r["trades"]]
    digest = hashlib.sha256(json.dumps(out).encode()).hexdigest()
    assert digest == LEGACY_DIGEST


LEGACY_DIGEST = "bed024121908e3821e3b47e973b4182857a930a1076d652fcfde928f7ad615a8"


def test_resting_study_refuses_oos(tmp_path, monkeypatch):
    import resting_fill_is_study as rs
    from market_data_store import MarketDataStore
    monkeypatch.setattr(rs, "IS_WINDOW", ("2023-01-01", "2024-12-31"))
    with pytest.raises(AssertionError, match="in-sample only"):
        rs.run(MarketDataStore(str(tmp_path / "m.db")), ["resting"])
