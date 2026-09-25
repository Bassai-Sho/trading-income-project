"""P2-123 slice 3, gate 1: box-agnostic null-alpha gate (direction + timing
permutation tests, every counterfactual simulated through the real core)."""
import sys
from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluation.null_baseline import TradeIn, _Day, run_null_gate, simulate_bracket
from synthetic_bars import noisy_days

END = time(11, 0)
D = "2019-03-12"


def _flat_day(px=100.0):
    idx = pd.date_range(f"{D} 09:30", f"{D} 15:59", freq="1min", tz="America/New_York")
    return pd.DataFrame({"Open": px, "High": px + 0.01, "Low": px - 0.01, "Close": px,
                         "Volume": 1000.0}, index=idx)


def _ts(hm):
    return pd.Timestamp(f"{D} {hm}", tz="America/New_York")


# ── Exact bracket outcomes (fills come from the real core) ───────────────────

def test_stop_target_gap_and_eod_prices():
    sig = _ts("10:00")                                    # entry at 10:05 open = 100
    df = _flat_day(); df.loc[_ts("10:07"), "Low"] = 98.9  # stop 99.0 (1%) touched
    assert simulate_bracket(_Day("X", df), sig, "long", 0.01, END)[0] == pytest.approx(-1.0)
    df = _flat_day(); df.loc[_ts("10:07"), "High"] = 101.2
    assert simulate_bracket(_Day("X", df), sig, "long", 0.01, END)[0] == pytest.approx(1.0)
    df = _flat_day(); df.loc[_ts("10:07"), ["Open", "Low", "Close"]] = [98.0, 97.9, 98.0]
    df.loc[_ts("10:07"), "High"] = 98.01
    assert simulate_bracket(_Day("X", df), sig, "long", 0.01, END)[0] == pytest.approx(-2.0)
    df = _flat_day(); df.loc[_ts("11:00"), ["Open", "High"]] = [100.5, 100.51]
    assert simulate_bracket(_Day("X", df), sig, "long", 0.01, END)[0] == pytest.approx(0.5)
    df = _flat_day(); df.loc[_ts("10:07"), "Low"] = 98.9
    assert simulate_bracket(_Day("X", df), sig, "short", 0.01, END)[0] == pytest.approx(1.0)  # short target


def test_minute_touching_both_is_a_stop():
    df = _flat_day(); df.loc[_ts("10:07"), ["High", "Low"]] = [101.5, 98.5]
    assert simulate_bracket(_Day("X", df), _ts("10:00"), "long", 0.01, END)[0] == pytest.approx(-1.0)


def test_no_entry_at_or_after_session_end():
    assert simulate_bracket(_Day("X", _flat_day()), _ts("10:55"), "long", 0.01, END) is None


# ── Permutation tests on noisy data ───────────────────────────────────────────

@pytest.fixture(scope="module")
def data():
    return noisy_days("2019-01-02", "2019-06-28", seed=21, px=200.0)


def _log(df, pick, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for d, g in df.groupby(df.index.date):
        c = g["Close"]
        sig = pick(g, rng)
        if sig is None:
            continue
        side = sig[1]
        e = float(g["Open"].iloc[g.index.searchsorted(sig[0] + pd.Timedelta(minutes=5))])
        out.append(TradeIn(str(d), side, sig[0], e, e * (0.997 if side == "long" else 1.003)))
    return out


def _future_move(g, t):
    at = g.index.searchsorted(t + pd.Timedelta(minutes=5))
    end = g.index.searchsorted(pd.Timestamp.combine(t.date(), END).tz_localize(t.tz))
    return float(g["Close"].iloc[end] - g["Open"].iloc[at])


def test_foresight_direction_passes_direction_test(data):
    def pick(g, rng):
        t = g.index[0] + pd.Timedelta(minutes=5 * int(rng.integers(3, 15)))
        return t, "long" if _future_move(g, t) > 0 else "short"
    rep = run_null_gate(_log(data, pick), data, "X", END, draws=300)
    assert rep.direction_p < 0.01 and rep.common_exit_expectancy > rep.direction_null_p95


def test_random_direction_and_time_fails(data):
    def pick(g, rng):
        return (g.index[0] + pd.Timedelta(minutes=5 * int(rng.integers(3, 15))),
                "long" if rng.random() < 0.5 else "short")
    rep = run_null_gate(_log(data, pick, seed=3), data, "X", END, draws=300)
    assert not rep.passed and rep.direction_p > 0.05 and rep.timing_p > 0.05


def test_always_long_on_rising_market_does_not_pass_direction(data):
    """Drift control: shuffling labels keeps the long/short mix, so a box that
    is simply always long gets no credit for the market going up."""
    up = data.copy()
    k = np.arange(len(up)) * 0.0004                       # steady upward drift
    for col in ("Open", "High", "Low", "Close"):
        up[col] = up[col] + k
    def pick(g, rng):
        return g.index[0] + pd.Timedelta(minutes=5 * int(rng.integers(3, 15))), "long"
    rep = run_null_gate(_log(up, pick), up, "X", END, draws=200)
    assert rep.direction_p == 1.0 and not rep.passed


def test_foresight_timing_passes_timing_test(data):
    def pick(g, rng):
        grid = [g.index[0] + pd.Timedelta(minutes=5 * k) for k in range(3, 15)]
        side = "long" if rng.random() < 0.5 else "short"
        best = max(grid, key=lambda t: _future_move(g, t) * (1 if side == "long" else -1))
        return best, side
    rep = run_null_gate(_log(data, pick), data, "X", END, draws=300)
    assert rep.timing_p < 0.01


def test_harness_never_imports_a_box():
    pkg = Path(__file__).resolve().parent.parent / "src" / "evaluation"
    for f in pkg.glob("*.py"):
        code = [l for l in f.read_text().splitlines() if l.strip().startswith(("import", "from"))]
        assert not any("boxes" in l for l in code), f"{f.name} imports a strategy box"
