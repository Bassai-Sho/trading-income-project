"""P2-123 slice 2: box #1 through the execution core must reproduce the legacy
resting-fill backtest trade for trade (entry/exit time and price, stop, exit
reason, net R to 1e-9) — on noisy synthetic data here; on the real 2016-2022
store via src/parity_box1.py against tests/fixtures/golden_orb_resting_trades."""
import dataclasses, sys
from datetime import timedelta
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import parity_box1 as pb
import resting_fill_is_study as rs
from boxes.orb_vwap_legacy import OrbVwapLegacyBox, State
from core.backtest_runner import run_box
from core.execution_core import AccountConfig
from synthetic_bars import noisy_days


def _legacy_rows(df, tkr="SPY"):
    trades, _ = rs.run_resting(df, tkr, "next_open")
    return [{k: (str(t[k]) if k.endswith("_ts") else t[k]) for k in
             ("session_date", "direction", "entry_ts", "entry_fill_ts", "entry_price",
              "stop_price", "exit_ts", "exit_price", "exit_reason", "actual_r")} for t in trades]


def _box_rows(df, tkr="SPY"):
    p = pb.legacy_params()
    res = run_box(OrbVwapLegacyBox(), p, df, tkr, AccountConfig(leverage=None))
    return pb.box_rows(res.state.journal, df, p), res


def _assert_parity(df):
    gold = _legacy_rows(df)
    mine, _ = _box_rows(df)
    rep = pb.compare(gold, mine)
    assert rep["golden_n"] > 20
    assert (rep["diffs"], rep["only_golden"], rep["only_box"]) == ([], [], []), rep["diffs"][:2]
    return gold


@pytest.fixture(scope="module")
def noisy():
    return noisy_days("2019-01-02", "2019-03-29", seed=10, px=280.0)


def test_parity_on_noisy_gappy_data(noisy):
    gold = _assert_parity(noisy)
    reasons = {t["exit_reason"] for t in gold}
    assert reasons == {"eod", "target_hit", "trailing_stop"}
    assert {t["direction"] for t in gold} == {"long", "short"}


def test_parity_when_a_minute_touches_both_stop_and_target(noisy):
    """Force the rare case: right after an entry, one minute spans stop and target."""
    gold = _legacy_rows(noisy)
    t = next(x for x in gold if x["exit_reason"] != "eod"
             and pd.Timestamp(x["exit_ts"]) > pd.Timestamp(x["entry_fill_ts"]) + timedelta(minutes=1))
    m = pd.Timestamp(t["entry_fill_ts"]) + timedelta(minutes=1)
    risk = abs(t["entry_price"] - t["stop_price"])
    df = noisy.copy()
    df.loc[m, "High"] = max(df.loc[m, "High"], t["entry_price"] + 1.2 * risk)
    df.loc[m, "Low"] = min(df.loc[m, "Low"], t["entry_price"] - 1.2 * risk)
    gold2 = _assert_parity(df)
    hit = next(x for x in gold2 if x["entry_fill_ts"] == t["entry_fill_ts"])
    assert hit["exit_reason"] == "trailing_stop" and pd.Timestamp(hit["exit_ts"]) == m


def test_box_is_pure_and_deterministic(noisy):
    one = noisy[noisy.index.date <= pd.Timestamp("2019-01-31").date()]
    a, ra = _box_rows(one)
    b, rb = _box_rows(one)
    assert a == b and ra.state.journal == rb.state.journal
    assert dataclasses.is_dataclass(State) and State.__dataclass_params__.frozen
    with pytest.raises(dataclasses.FrozenInstanceError):
        ra.state.phase = "flat"
    mutable = {k: v for k, v in vars(OrbVwapLegacyBox).items()
               if not k.startswith("__") and isinstance(v, (dict, list, set))}
    assert mutable == {}, f"box holds shared mutable class state: {mutable}"


def test_runner_is_box_agnostic():
    """The runner and core contain no ORB logic."""
    for f in ("core/backtest_runner.py", "core/execution_core.py", "core/interfaces.py"):
        text = (Path(__file__).resolve().parent.parent / "src" / f).read_text().lower()
        assert "orb" not in text.replace("orbit", ""), f
