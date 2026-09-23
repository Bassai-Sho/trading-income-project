"""PR-001 runner: frozen-spec guard, quality filter, quarterly-ADV chunking
equivalence, DSR trial count, and the one-shot Stage 2 lock. Synthetic data, no network."""
import json, sys
from datetime import time as Time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import pr001_instrument_test as pr
from market_data_store import MarketDataStore
from trading_engine import _backtest_orb_full_gate, CONFIG
from historical_sim import _to_5min


def _days(start, end, seed=0, vol=2000.0):
    rng = np.random.default_rng(seed)
    frames = []
    for i, d in enumerate(pd.bdate_range(start, end)):
        idx = pd.date_range(f"{d.date()} 09:30", f"{d.date()} 15:59", freq="1min", tz="America/New_York")
        t = np.arange(len(idx)); sign = 1 if rng.random() < 0.6 else -1
        k = rng.uniform(0.03, 0.09)
        c = 400 + np.where(t < 15, 0.3 * np.sin(t), sign * k * (t - 15) + 0.4 * np.sin(t / 3.0))
        c = c + rng.normal(0, 0.02, len(t)).cumsum()
        o = np.r_[c[0], c[:-1]]
        frames.append(pd.DataFrame({"Open": o, "High": np.maximum(o, c) + 0.03,
                                    "Low": np.minimum(o, c) - 0.03, "Close": c,
                                    "Volume": vol * (1 + i % 5)}, index=idx))
    return pd.concat(frames)


@pytest.fixture
def store(tmp_path):
    s = MarketDataStore(str(tmp_path / "m.db"))
    for j, tkr in enumerate(pr.SPEC["tickers"]):
        s._store_bars(tkr, _days("2019-03-01", "2019-04-30", seed=j), {})
    return s


def test_frozen_config_matches_engine():
    pr.assert_frozen(CONFIG)                                   # current engine == spec
    with pytest.raises(pr.SpecViolation, match="vwap_lookback"):
        pr.assert_frozen({**CONFIG, "vwap_lookback": 4})
    with pytest.raises(pr.SpecViolation, match="session_end"):
        pr.assert_frozen({**CONFIG, "session_end": Time(11, 30)})


def test_quality_filter_excludes_bad_sessions(store):
    with __import__("sqlite3").connect(store.db_path) as c:
        c.execute("UPDATE session_context SET quality_ok=0 WHERE ticker='SPY' AND session_date='2019-03-05'")
    df, q = pr.load_bars(store, "SPY", "2019-03-01", "2019-04-30")
    assert q["excluded"] == 1 and q["quality_ok"] == q["sessions"] - 1
    assert "2019-03-05" not in {str(d) for d in df.index.date}
    assert {"Open", "High", "Low", "Close", "Volume"} <= set(df.columns)


def test_quarterly_chunking_changes_only_costs(store):
    df, _ = pr.load_bars(store, "SPY", "2019-03-01", "2019-04-30")   # spans Q1 and Q2
    chunked = pr.run_trades(df, "SPY")
    whole = _backtest_orb_full_gate(_to_5min(df), {**CONFIG, "ticker": "SPY",
                                    "account_balance": 10_000.0, "risk_pct": 0.01})["trades"]
    key = lambda t: (t["session_date"], t["direction"], t["entry_price"], t["exit_ts"], t["exit_reason"])
    assert len(chunked) >= 10
    assert [key(t) for t in chunked] == [key(t) for t in whole]
    adv = pr.quarterly_adv(df)
    assert set(adv) == {"2019Q1", "2019Q2"}
    for t in chunked:
        q = str(pd.Period(t["session_date"], freq="Q"))
        assert t["adv_used"] == adv[q]


def test_dsr_uses_n_trials_10():
    rng = np.random.default_rng(5)
    rs = list(rng.normal(0.08, 1.0, 300))
    assert pr.stats(rs, 10)["dsr"] < pr.stats(rs, 1)["dsr"]
    assert pr.SPEC["dsr_n_trials"] == 10


def test_stage1_verdict_rules():
    st = {"n": 150, "expectancy": 0.05, "dsr": 0.96}
    rg = {k: {"expectancy": e} for k, e in zip(pr.IS_REGIMES, (0.1, -0.2, 0.05, 0.02))}
    assert pr.stage1_verdict(st, rg)["pass"]
    rg["rate_hike_bear"]["expectancy"] = None                  # regime with no trades
    v = pr.stage1_verdict(st, rg)
    assert not v["pass"] and v["positive_regimes"] == 2
    assert not pr.stage1_verdict({**st, "n": 99}, {k: {"expectancy": 1} for k in pr.IS_REGIMES})["pass"]


def test_stage1_end_to_end_writes_all_four(store, tmp_path, monkeypatch):
    monkeypatch.setitem(pr.SPEC, "is_window", ("2019-03-01", "2019-04-30"))
    r = pr.run_stage1(store, out_dir=tmp_path / "out")
    assert set(r["tickers"]) == {"SPY", "QQQ", "NVDA", "TSLA"}         # all reported
    saved = json.loads(Path(r["path"]).read_text())
    assert saved["spec"]["dsr_n_trials"] == 10 and saved["stage"] == 1
    assert r["stage1_passers"] == []                                    # n < 100 on 2 months


def test_stage2_needs_confirmation_and_runs_once(store, tmp_path, monkeypatch):
    monkeypatch.setitem(pr.SPEC, "is_window", ("2019-03-01", "2019-03-29"))
    monkeypatch.setitem(pr.SPEC, "oos_window", ("2019-04-01", "2019-04-30"))
    out = tmp_path / "out"
    s1 = pr.run_stage1(store, out_dir=out)
    data = json.loads(Path(s1["path"]).read_text())
    data["stage1_passers"] = ["SPY"]                           # force a passer for the test
    Path(s1["path"]).write_text(json.dumps(data))
    with pytest.raises(pr.SpecViolation, match="confirm-one-shot"):
        pr.run_stage2(store, Path(s1["path"]), confirm=False, out_dir=out)
    r = pr.run_stage2(store, Path(s1["path"]), confirm=True, out_dir=out)
    assert r["status"] == "complete" and "SPY" in r["tickers"] and "QQQ" not in r["tickers"]
    with pytest.raises(pr.SpecViolation, match="already been run"):
        pr.run_stage2(store, Path(s1["path"]), confirm=True, out_dir=out)


def test_stage2_with_no_passers_never_loads_oos(store, tmp_path, monkeypatch):
    monkeypatch.setitem(pr.SPEC, "is_window", ("2019-03-01", "2019-03-29"))
    out = tmp_path / "out"
    s1 = pr.run_stage1(store, out_dir=out)
    loaded = []
    monkeypatch.setattr(pr, "load_bars", lambda *a, **k: loaded.append(a) or (pd.DataFrame(), {}))
    r = pr.run_stage2(store, Path(s1["path"]), confirm=True, out_dir=out)
    assert "No Stage-1 passers" in r["note"] and loaded == []
    assert (out / "stage2.json").exists()


def test_stage2_rejects_results_from_a_different_spec(store, tmp_path, monkeypatch):
    monkeypatch.setitem(pr.SPEC, "is_window", ("2019-03-01", "2019-03-29"))
    out = tmp_path / "out"
    s1 = pr.run_stage1(store, out_dir=out)
    data = json.loads(Path(s1["path"]).read_text()); data["spec"]["dsr_n_trials"] = 1
    Path(s1["path"]).write_text(json.dumps(data))
    with pytest.raises(pr.SpecViolation, match="different spec"):
        pr.run_stage2(store, Path(s1["path"]), confirm=True, out_dir=out)
