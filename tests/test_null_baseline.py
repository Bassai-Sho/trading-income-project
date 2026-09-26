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


def test_foresight_direction_passes_gate_and_direction(data):
    def pick(g, rng):
        t = g.index[0] + pd.Timedelta(minutes=5 * int(rng.integers(3, 15)))
        return t, "long" if _future_move(g, t) > 0 else "short"
    rep = run_null_gate(_log(data, pick), data, "X", END, draws=300)
    assert rep.passed and rep.gate_p < 0.01 and rep.direction_p < 0.01


def test_random_entries_fail_the_gate(data):
    def pick(g, rng):
        return (g.index[0] + pd.Timedelta(minutes=5 * int(rng.integers(3, 15))),
                "long" if rng.random() < 0.5 else "short")
    rep = run_null_gate(_log(data, pick, seed=3), data, "X", END, draws=300)
    assert not rep.passed and rep.gate_p > 0.05 and rep.direction_p > 0.05


def test_always_long_on_rising_market_does_not_pass(data):
    """Drift control: shuffling labels keeps the long/short mix, so a box that
    is simply always long gets no credit for the market going up."""
    up = data.copy()
    k = np.arange(len(up)) * 0.0004
    for col in ("Open", "High", "Low", "Close"):
        up[col] = up[col] + k
    def pick(g, rng):
        return g.index[0] + pd.Timedelta(minutes=5 * int(rng.integers(3, 15))), "long"
    rep = run_null_gate(_log(up, pick), up, "X", END, draws=200)
    assert rep.direction_p == 1.0 and not rep.passed


def test_side_kept_timing_null_leaks_the_future(data):
    """Why review round 4's Test B was dropped: a side chosen from what happens
    AFTER the signal, carried to earlier random times, makes the null look
    skilled. The leak shows as side-kept 'before' >> the honest gate null."""
    def pick(g, rng):
        t = g.index[0] + pd.Timedelta(minutes=5 * int(rng.integers(10, 15)))    # late signals
        first = g.index[0] + pd.Timedelta(minutes=15)
        return t, "long" if _future_move(g, first) > 0 else "short"              # side from the morning's move
    rep = run_null_gate(_log(data, pick), data, "X", END, draws=200)
    assert rep.side_kept_timing_before > rep.gate_null_mean + 0.1


@pytest.mark.parametrize("best, expect_low_p", [(True, True), (False, False)])
def test_timing_diagnostic_detects_whipsaw_avoidance(data, best, expect_low_p):
    """Direction-neutral timing: a tight stop makes whipsaws common; picking the
    moments where neither side whipsaws must score p < 0.01, the worst ~1."""
    from evaluation.null_baseline import _Day, simulate_bracket
    F = 0.0008
    rng = np.random.default_rng(0)
    log = []
    for d, g in data.groupby(data.index.date):
        day = _Day("X", g)
        vals = []
        for k in range(3, 15):
            t = g.index[0] + pd.Timedelta(minutes=5 * k)
            a = simulate_bracket(day, t, "long", F, END)
            b = simulate_bracket(day, t, "short", F, END)
            if a is not None and b is not None:
                vals.append((a[0] + b[0], t))
        t = (max if best else min)(vals)[1]
        side = "long" if rng.random() < 0.5 else "short"
        e = float(g["Open"].iloc[g.index.searchsorted(t + pd.Timedelta(minutes=5))])
        log.append(TradeIn(str(d), side, t, e, e * (1 - F) if side == "long" else e * (1 + F)))
    rep = run_null_gate(log, data, "X", END, draws=300, window_start=time(9, 45))
    assert (rep.timing_p < 0.01) if expect_low_p else (rep.timing_p > 0.9)


def test_harness_never_imports_a_box():
    pkg = Path(__file__).resolve().parent.parent / "src" / "evaluation"
    for f in pkg.glob("*.py"):
        code = [l for l in f.read_text().splitlines() if l.strip().startswith(("import", "from"))]
        assert not any("boxes" in l for l in code), f"{f.name} imports a strategy box"


# ── PR-002 common exit: no target, exit at the close; candidate sampling ─────

def test_no_target_exit_at_close_prices():
    sig = _ts("10:00")                                    # entry 10:05 at 100, stop 99 (1%)
    df = _flat_day(); df.loc[_ts("10:07"), "High"] = 101.5  # would hit a 1R target — none now
    df.loc[_ts("15:59"), ["Open", "High", "Low", "Close"]] = [100.2, 100.4, 100.1, 100.3]
    r = simulate_bracket(_Day("X", df), sig, "long", 0.01, time(16, 0), target_r=None, exit="close")
    assert r[0] == pytest.approx(0.3)                     # closed at 100.3, not the 101 target
    df = _flat_day(); df.loc[_ts("10:07"), "Low"] = 98.9
    r = simulate_bracket(_Day("X", df), sig, "long", 0.01, time(16, 0), target_r=None, exit="close")
    assert r[0] == pytest.approx(-1.0)                    # stop still works


def test_exit_at_close_on_a_half_day():
    idx = pd.date_range("2019-11-29 09:30", periods=210, freq="1min", tz="America/New_York")
    df = pd.DataFrame({"Open": 100.0, "High": 100.01, "Low": 99.99, "Close": 100.0, "Volume": 1.0}, index=idx)
    df.iloc[-1, :4] = [100.5, 100.6, 100.4, 100.5]
    r = simulate_bracket(_Day("X", df), pd.Timestamp("2019-11-29 10:00", tz="America/New_York"),
                         "long", 0.01, time(16, 0), target_r=None, exit="close")
    assert r[0] == pytest.approx(0.5)                     # the 12:59 close


def test_candidate_sampling_is_deterministic_and_still_works(data):
    def pick(g, rng):
        t = g.index[0] + pd.Timedelta(minutes=5 * int(rng.integers(3, 15)))
        return t, "long" if _future_move(g, t) > 0 else "short"
    log = _log(data, pick)
    kw = dict(draws=200, target_r=None, exit="close", n_candidates=6, window_start=time(9, 45))
    a = run_null_gate(log, data, "X", time(16, 0), **kw)
    b = run_null_gate(log, data, "X", time(16, 0), **kw)
    assert a == b and a.passed and a.direction_p < 0.01


def test_pooled_multi_symbol_input_matches_single_symbol_result(data):
    """The dict form (many symbols) must give exactly the single-symbol result."""
    def pick(g, rng):
        t = g.index[0] + pd.Timedelta(minutes=5 * int(rng.integers(3, 15)))
        return t, "long" if rng.random() < .5 else "short"
    log = _log(data, pick, seed=5)[:40]
    kw = dict(draws=100, target_r=None, exit="close", n_candidates=4, window_start=time(9, 45))
    single = run_null_gate(log, data, "X", time(16, 0), **kw)
    import dataclasses
    tagged = [dataclasses.replace(t, symbol="X") for t in log]
    by_day = {("X", str(d)): g for d, g in data.groupby(data.index.date)}
    pooled = run_null_gate(tagged, by_day, "", time(16, 0), **kw)
    assert pooled == single
