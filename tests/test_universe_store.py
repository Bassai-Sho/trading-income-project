"""P2-125 universe stage 1: split-safe eligibility, no look-ahead, resumable
download with DEFAULT symbol mapping (follows the company), and removal of the
same company's rows appearing under an old and a new symbol."""
import sqlite3, sys, types
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import universe_store as us
from universe_store import UniverseStore, eligibility_frame


def _split_series(n=40, split_at=20, ratio=4.0, sym="X"):
    """Price 100 with a $1 daily range; 4:1 split at `split_at` (raw /4 after).
    Alpaca-style split-adjusted series: everything on the post-split basis."""
    dates = [str(d.date()) for d in pd.bdate_range("2019-01-02", periods=n)]
    rows = []
    for i, d in enumerate(dates):
        k = 1.0 if i >= split_at else ratio          # raw price multiplier vs post-split basis
        o, h, l, c = 100 / 4 * k, 100.5 / 4 * k, 99.5 / 4 * k, 100 / 4 * k
        v = 4e6 / k
        rows.append({"symbol": sym, "date": d, "open": o, "high": h, "low": l, "close": c,
                     "volume": v, "adj_open": o / k, "adj_high": h / k, "adj_low": l / k,
                     "adj_close": c / k, "adj_volume": v * k})
    return pd.DataFrame(rows)


def test_window_straddling_a_split_is_on_the_days_own_basis():
    e = eligibility_frame(_split_series())
    after = e.iloc[25]                      # window = days 11..24, split at 20
    assert after["atr14"] == pytest.approx(0.25) and after["adv14"] == pytest.approx(4e6)
    before = e.iloc[18]                     # window entirely pre-split
    assert before["atr14"] == pytest.approx(1.0) and before["adv14"] == pytest.approx(1e6)


def test_naive_raw_calculation_would_have_been_wrong():
    g = _split_series()
    tr = (g["high"] - g["low"]).combine(abs(g["high"] - g["close"].shift()), max)
    assert tr.iloc[20] > 50 * 0.25          # the split day's raw 'true range' is enormous


def test_no_look_ahead_today_does_not_affect_today():
    g = _split_series()
    base = eligibility_frame(g).iloc[30]
    g.loc[30, ["high", "volume"]] = [g.loc[30, "high"] * 3, 9e9]
    g.loc[30, ["adj_high", "adj_volume"]] = [g.loc[30, "adj_high"] * 3, 9e9]
    new = eligibility_frame(g).iloc[30]
    assert (new["atr14"], new["adv14"]) == pytest.approx((base["atr14"], base["adv14"]))


def test_needs_a_full_window():
    e = eligibility_frame(_split_series())
    assert e["atr14"].iloc[:14].isna().all() and e["atr14"].iloc[14:].notna().all()


# ── store / download (stubbed Alpaca) ──────────────────────────────────────────

class _Client:
    def __init__(self, fail_first=0, status=None, invalid=()):
        self.calls, self.fail_first, self.status, self.invalid = [], fail_first, status, set(invalid)

    def get_stock_bars(self, req):
        self.calls.append(req)
        bad = [s for s in req.symbol_or_symbols if s in self.invalid]
        if bad:
            from alpaca.common.exceptions import APIError
            raise APIError('{"message":"invalid symbol: %s"}' % bad[0])
        if self.status:
            from alpaca.common.exceptions import APIError

            class _HTTPError(APIError):          # subclass: never mutate APIError itself
                status_code = property(lambda s, c=self.status: c)
            raise _HTTPError('{"message":"forbidden"}')
        if self.fail_first:
            self.fail_first -= 1
            raise ConnectionError("reset")
        frames = []
        for s in req.symbol_or_symbols:
            g = _split_series(sym=s)
            pre = "adj_" if req.adjustment.value == "split" else ""
            idx = pd.MultiIndex.from_arrays(
                [[s] * len(g), pd.to_datetime(g["date"]).dt.tz_localize("America/New_York")
                 .dt.tz_convert("UTC")], names=["symbol", "timestamp"])
            frames.append(pd.DataFrame({k: g[pre + k].to_numpy() for k in
                                        ("open", "high", "low", "close", "volume")}, index=idx))
        return types.SimpleNamespace(df=pd.concat(frames))


def _store(tmp_path, syms=(("AAA", "NASDAQ"), ("BBB", "NYSE"), ("CCC", "OTC"))):
    u = UniverseStore(str(tmp_path / "u.db"))
    with sqlite3.connect(u.db_path) as c:
        c.executemany("INSERT INTO universe_assets VALUES (?,?,?,?,?,?,?)",
                      [(s, s, ex, "active", 1, "us_equity", "") for s, ex in syms])
    return u


def test_otc_excluded_and_default_symbol_mapping_used(tmp_path, monkeypatch):
    monkeypatch.setattr(us._time, "sleep", lambda s: None)
    u = _store(tmp_path)
    assert u.symbols() == ["AAA", "BBB"]
    cl = _Client()
    r = u.download_daily(cl, date(2019, 1, 1), date(2019, 3, 31), batch=10)
    assert r["rows"] == 80 and not r["failed_batches"]
    assert {c.asof for c in cl.calls} == {None} and {c.adjustment.value for c in cl.calls} == {"raw", "split"}
    assert {c.feed.value for c in cl.calls} == {"sip"}


def test_resume_skips_finished_batches_and_retries_failed(tmp_path, monkeypatch):
    monkeypatch.setattr(us._time, "sleep", lambda s: None)
    u = _store(tmp_path, syms=[(f"S{c}", "NYSE") for c in "ABCD"])
    r1 = u.download_daily(_Client(fail_first=1), date(2019, 1, 1), date(2019, 3, 31), batch=2)
    assert len(r1["failed_batches"]) == 1 and r1["rows"] == 80
    cl = _Client()
    r2 = u.download_daily(cl, date(2019, 1, 1), date(2019, 3, 31), batch=2)
    assert r2["rows"] == 80 and len(cl.calls) == 2           # only the failed batch, raw + split


def test_auth_failure_stops_immediately(tmp_path, monkeypatch):
    u = _store(tmp_path)
    with pytest.raises(us.FatalDownloadError):
        u.download_daily(_Client(status=403), date(2019, 1, 1), date(2019, 3, 31))


def test_sealed_window_clamped(tmp_path, monkeypatch):
    monkeypatch.setattr(us._time, "sleep", lambda s: None)
    u = _store(tmp_path)
    cl = _Client()
    u.download_daily(cl, date(2024, 12, 1), date(2025, 6, 30))
    assert all(c.end.date() <= date(2024, 12, 31) for c in cl.calls)


def test_eligibility_thresholds_and_duplicate_report(tmp_path, monkeypatch):
    monkeypatch.setattr(us._time, "sleep", lambda s: None)
    u = _store(tmp_path)
    u.download_daily(_Client(), date(2019, 1, 1), date(2019, 3, 31))
    r = u.compute_eligibility(date(2019, 1, 1), date(2019, 3, 31), min_atr=0.2)
    assert r["rows"] > 0
    with sqlite3.connect(u.db_path) as c:
        assert c.execute("SELECT MIN(open), MIN(atr14) FROM universe_eligibility").fetchone()[0] > 5
    none = u.compute_eligibility(date(2019, 1, 1), date(2019, 3, 31), min_atr=5.0)
    assert none["rows"] == 0
    d = u.duplicate_report(min_days=10)                      # AAA and BBB are identical by construction
    assert list(d[["sym_a", "sym_b"]].iloc[0]) == ["AAA", "BBB"]


def test_dedupe_keeps_active_symbol_and_only_identical_rows(tmp_path, monkeypatch):
    """FB (inactive) and META (active) share identical rows for the same company;
    FB's rows that match are removed, META is untouched, and a non-matching FB
    row survives."""
    monkeypatch.setattr(us._time, "sleep", lambda s: None)
    u = _store(tmp_path, syms=(("META", "NASDAQ"), ("FB", "NASDAQ")))
    with sqlite3.connect(u.db_path) as c:
        c.execute("UPDATE universe_assets SET status='inactive' WHERE symbol='FB'")
    u.download_daily(_Client(), date(2019, 1, 1), date(2019, 3, 31))
    with sqlite3.connect(u.db_path) as c:                  # one FB-only bar that must survive
        c.execute("UPDATE universe_daily SET close=close+1 WHERE symbol='FB' AND date="
                  "(SELECT MIN(date) FROM universe_daily WHERE symbol='FB')")
    r = u.deduplicate()
    assert r["pairs"] == 1 and r["dropped"][0]["kept"] == "META" and r["rows_deleted"] == 39
    with sqlite3.connect(u.db_path) as c:
        n = dict(c.execute("SELECT symbol, COUNT(*) FROM universe_daily GROUP BY symbol").fetchall())
    assert n == {"META": 40, "FB": 1}
    assert u.duplicate_report().empty


def test_dedupe_ignores_a_few_coincidental_identical_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(us._time, "sleep", lambda s: None)
    u = _store(tmp_path, syms=(("AAA", "NYSE"), ("BBB", "NYSE")))
    u.download_daily(_Client(), date(2019, 1, 1), date(2019, 3, 31))
    with sqlite3.connect(u.db_path) as c:                  # make them differ on all but 5 days
        c.execute("UPDATE universe_daily SET volume=volume+1 WHERE symbol='BBB' AND date NOT IN "
                  "(SELECT date FROM universe_daily WHERE symbol='BBB' ORDER BY date LIMIT 5)")
    assert u.deduplicate()["pairs"] == 0


# ── Placeholder symbols, invalid symbols, per-symbol resume (fix of 26 Sep) ──

def test_data_symbol_mapping():
    known = {"AET_DELISTED", "BRAC_DELISTED", "BRAC", "BRK.B", "0029900E0"}
    assert us.data_symbol("AET_DELISTED", known) == "AET"          # ticker free: fetch as AET
    assert us.data_symbol("BRAC_DELISTED", known) is None          # ticker reused: unreachable
    assert us.data_symbol("BRK.B", known) == "BRK.B"
    for placeholder in ("0029900E0", "641ESC017", "B002455", "Y11RGT027"):
        assert us.data_symbol(placeholder, known) is None


def test_delisted_fetched_under_base_ticker_stored_under_asset_symbol(tmp_path, monkeypatch):
    monkeypatch.setattr(us._time, "sleep", lambda s: None)
    u = _store(tmp_path, syms=(("AET_DELISTED", "NYSE"), ("0029900E0", "NYSE"), ("ZZZ", "NYSE")))
    cl = _Client()
    r = u.download_daily(cl, date(2019, 1, 1), date(2019, 3, 31))
    assert r["unreachable"] == 1 and r["delisted_mapped"] == 1
    assert set(cl.calls[0].symbol_or_symbols) == {"AET", "ZZZ"}
    with sqlite3.connect(u.db_path) as c:
        assert {x for (x,) in c.execute("SELECT DISTINCT symbol FROM universe_daily")} == {"AET_DELISTED", "ZZZ"}


def test_one_invalid_symbol_no_longer_sinks_the_batch(tmp_path, monkeypatch):
    monkeypatch.setattr(us._time, "sleep", lambda s: None)
    u = _store(tmp_path, syms=[(f"S{c}", "NYSE") for c in "ABCDE"])
    r = u.download_daily(_Client(invalid={"SC"}), date(2019, 1, 1), date(2019, 3, 31))
    assert r["invalid"] == ["SC"] and not r["failed_batches"] and r["rows"] == 4 * 40


def test_resume_is_per_symbol_and_honours_the_old_per_batch_progress(tmp_path, monkeypatch):
    monkeypatch.setattr(us._time, "sleep", lambda s: None)
    u = _store(tmp_path, syms=[(f"S{c}", "NYSE") for c in "ABCD"])
    u.download_daily(_Client(), date(2019, 1, 1), date(2019, 3, 31))
    cl = _Client()
    assert u.download_daily(cl, date(2019, 1, 1), date(2019, 3, 31))["fetched"] == 0 and not cl.calls
    # simulate the old version: rows + a per-batch progress row, no per-symbol records
    with sqlite3.connect(u.db_path) as c:
        c.execute("DELETE FROM universe_symbol_done")
        c.execute("DELETE FROM universe_daily WHERE symbol IN ('SC','SD')")
        c.execute("INSERT INTO universe_progress VALUES ('daily:old',2,80,'')")
    cl = _Client()
    r = u.download_daily(cl, date(2019, 1, 1), date(2019, 3, 31))
    assert r["fetched"] == 2 and set(cl.calls[0].symbol_or_symbols) == {"SC", "SD"}


def test_fast_duplicate_report_is_quick_and_exact(tmp_path, monkeypatch):
    import time
    monkeypatch.setattr(us._time, "sleep", lambda s: None)
    u = _store(tmp_path, syms=[("S" + a + b, "NYSE") for a in "ABCDEFGHIJKLMNOPQRST" for b in "ABCDEFGHIJKLMNO"])
    u.download_daily(_Client(), date(2019, 1, 1), date(2019, 3, 31), batch=300)
    t0 = time.time()
    d = u.duplicate_report(min_days=10)          # all 300 identical: 44,850 pairs
    assert time.time() - t0 < 10 and len(d) == 300 * 299 // 2 and (d["days"] == 40).all()
