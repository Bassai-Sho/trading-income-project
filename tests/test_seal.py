"""Research seal (28 Sep 2026): seal.py, MarketDataStore guards/roles, seal_migrate.py,
and the runner's redirect to the live store. Every test calls the REAL function on
temporary databases -- none reproduces the logic inline."""
import sqlite3
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import runner
import seal
import seal_migrate as sm
from market_data_store import MarketDataStore


# ── helpers ───────────────────────────────────────────────────────────────────

def _insert_bar(db, ticker, ts_date, minute="09:30", close=100.0):
    ts = f"{ts_date}T{minute}:00-05:00"
    with sqlite3.connect(db) as c:
        c.execute("INSERT OR IGNORE INTO market_bars (ticker, ts, ts_date, ts_time, open, high, low, "
                  "close, volume, vwap, bar_interval, quality_flag) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                  (ticker, ts, ts_date, minute, close, close + 1, close - 1, close, 1000, close, "1m", 0))
        c.execute("INSERT OR IGNORE INTO session_context (ticker, session_date, n_bars) VALUES (?,?,?)",
                  (ticker, ts_date, 1))


def _research_db(tmp_path, dates=("2024-12-30", "2024-12-31", "2025-01-02", "2025-01-03")):
    db = str(tmp_path / "research.db")
    MarketDataStore(db)                                   # default: sealed research store
    for i, d in enumerate(dates):
        _insert_bar(db, "SPY", d, close=100.0 + i)
    return db


@pytest.fixture(autouse=True)
def _no_unlock(monkeypatch):
    monkeypatch.delenv(seal.UNLOCK_ENV, raising=False)


# ── seal.check ────────────────────────────────────────────────────────────────

def test_check_blocks_the_seal_date_and_allows_the_day_before():
    seal.check(date(2024, 12, 31), "x")                   # last unsealed day: allowed
    with pytest.raises(seal.SealedDataError):
        seal.check(date(2025, 1, 1), "x")                 # boundary: first sealed day
    with pytest.raises(seal.SealedDataError):
        seal.check(date(2026, 6, 30), "x")


def test_check_accepts_strings_datetimes_and_timestamps():
    for ok in ("2024-12-31", datetime(2024, 12, 31, 15, 59), pd.Timestamp("2024-12-31 15:59")):
        seal.check(ok, "x")
    for bad in ("2025-01-01", datetime(2025, 1, 1, 9, 30), pd.Timestamp("2025-06-01 10:00")):
        with pytest.raises(seal.SealedDataError):
            seal.check(bad, "x")


def test_unlock_requires_the_exact_phrase(monkeypatch):
    monkeypatch.setenv(seal.UNLOCK_ENV, "yes")
    with pytest.raises(seal.SealedDataError):
        seal.check("2025-06-01", "x")                     # a wrong value does not unlock
    monkeypatch.setenv(seal.UNLOCK_ENV, seal.UNLOCK_PHRASE)
    seal.check("2025-06-01", "x")                         # the exact phrase does


# ── research store guards ─────────────────────────────────────────────────────

def test_research_store_reads_within_the_window_and_refuses_beyond(tmp_path):
    s = MarketDataStore(_research_db(tmp_path))
    assert len(s.get_bars_range("SPY", date(2024, 12, 1), date(2024, 12, 31))) == 2
    with pytest.raises(seal.SealedDataError):
        s.get_bars_range("SPY", date(2024, 12, 1), date(2025, 1, 31))     # range ENDS in the seal
    with pytest.raises(seal.SealedDataError):
        s.get_bars_range("SPY", date(2025, 1, 2), date(2025, 1, 3))
    with pytest.raises(seal.SealedDataError):
        s.get_session_bars("SPY", "2025-01-02")
    with pytest.raises(seal.SealedDataError):
        s.get_session_context("SPY", "2025-01-02")
    with pytest.raises(seal.SealedDataError):
        s.get_date_range("SPY", date(2024, 12, 1), date(2025, 1, 31))
    assert s.get_session_bars("SPY", "2024-12-31").shape[0] == 1          # unsealed session still works


def test_research_store_refuses_to_write_or_update_into_the_seal(tmp_path, monkeypatch):
    s = MarketDataStore(_research_db(tmp_path))
    monkeypatch.setattr(MarketDataStore, "_fetch_from_alpaca",
                        lambda *a, **k: pytest.fail("network reached before the seal guard"))
    with pytest.raises(seal.SealedDataError):
        s.download_and_store("SPY", date(2025, 1, 2), date(2025, 1, 31))


def test_update_refuses_before_any_network_call(tmp_path, monkeypatch):
    """update() has its own guard so it fails BEFORE fetching anything. (Without it, a
    second guard inside download_and_store still raises -- which is exactly why the
    first version of this test could not tell whether update()'s guard existed.)"""
    s = MarketDataStore(_research_db(tmp_path))
    monkeypatch.setattr(MarketDataStore, "_fetch_vix_daily",
                        lambda *a, **k: pytest.fail("update() reached the network before the seal guard"))
    monkeypatch.setattr(MarketDataStore, "_fetch_from_alpaca",
                        lambda *a, **k: pytest.fail("update() reached the network before the seal guard"))
    with pytest.raises(seal.SealedDataError):
        s.update("SPY")


def test_unlocked_research_store_can_read_the_window(tmp_path, monkeypatch):
    s = MarketDataStore(_research_db(tmp_path))
    monkeypatch.setenv(seal.UNLOCK_ENV, seal.UNLOCK_PHRASE)
    assert len(s.get_bars_range("SPY", date(2025, 1, 1), date(2025, 1, 31))) == 2


# ── roles: research vs live ───────────────────────────────────────────────────

def test_research_store_marks_itself_and_cannot_be_reopened_unsealed(tmp_path):
    db = _research_db(tmp_path)
    assert MarketDataStore(db)._get_setting("role") == "research"
    with pytest.raises(seal.SealedDataError):
        MarketDataStore(db, sealed=False)                 # the obvious bypass is closed


def test_new_live_store_is_marked_live_and_can_hold_sealed_dates(tmp_path):
    live = str(tmp_path / "live.db")
    s = MarketDataStore(live, sealed=False)
    assert s._get_setting("role") == "live"
    _insert_bar(live, "SPY", "2026-07-01")
    assert len(s.get_bars_range("SPY", date(2026, 7, 1), date(2026, 7, 31))) == 1
    with pytest.raises(seal.SealedDataError):             # opened sealed (the default), it refuses
        MarketDataStore(live).get_bars_range("SPY", date(2026, 7, 1), date(2026, 7, 31))


def test_unmarked_store_with_data_cannot_be_opened_unsealed(tmp_path):
    """A pre-existing DB (e.g. the current research DB, which has no role yet)
    holding bars must not be quietly reclassified as live."""
    db = str(tmp_path / "old.db")
    MarketDataStore(db)                                   # creates schema (marks research)...
    with sqlite3.connect(db) as c:
        c.execute("DELETE FROM store_settings WHERE key='role'")   # ...simulate a pre-seal DB
    _insert_bar(db, "SPY", "2024-12-31")
    with pytest.raises(seal.SealedDataError):
        MarketDataStore(db, sealed=False)


# ── migration ─────────────────────────────────────────────────────────────────

def test_dry_run_changes_nothing(tmp_path, capsys):
    db, live = _research_db(tmp_path), str(tmp_path / "live.db")
    before = sm.bars_checksum(db)
    assert sm.run(db, live, execute=False, delete=False, confirm=None) == 0
    assert not Path(live).exists()                        # live store not even created
    assert sm.bars_checksum(db) == before
    assert "DRY RUN" in capsys.readouterr().out


def test_execute_copies_and_verifies_without_deleting(tmp_path):
    db, live = _research_db(tmp_path), str(tmp_path / "live.db")
    assert sm.run(db, live, execute=True, delete=False, confirm=None) == 0
    v = sm.verify(db, live)
    assert v["ok"] and v["detail"]["market_bars"] == (2, 2)
    with sqlite3.connect(db) as c:                        # research store untouched
        assert c.execute("SELECT COUNT(*) FROM market_bars WHERE ts_date>='2025-01-01'").fetchone()[0] == 2
    assert MarketDataStore(live, sealed=False)._get_setting("role") == "live"


def test_execute_is_idempotent(tmp_path):
    db, live = _research_db(tmp_path), str(tmp_path / "live.db")
    sm.run(db, live, execute=True, delete=False, confirm=None)
    sm.run(db, live, execute=True, delete=False, confirm=None)
    with sqlite3.connect(live) as c:
        assert c.execute("SELECT COUNT(*) FROM market_bars").fetchone()[0] == 2


def test_delete_requires_execute_and_the_confirm_phrase(tmp_path):
    db, live = _research_db(tmp_path), str(tmp_path / "live.db")
    assert sm.run(db, live, execute=False, delete=True, confirm=sm.CONFIRM_PHRASE) == 2
    assert sm.run(db, live, execute=True, delete=True, confirm="yes") == 2
    with sqlite3.connect(db) as c:
        assert c.execute("SELECT COUNT(*) FROM market_bars WHERE ts_date>='2025-01-01'").fetchone()[0] == 2


def test_full_migration_moves_sealed_rows_backs_up_and_keeps_unsealed(tmp_path):
    db, live = _research_db(tmp_path), str(tmp_path / "live.db")
    assert sm.run(db, live, execute=True, delete=True, confirm=sm.CONFIRM_PHRASE) == 0
    with sqlite3.connect(db) as c:
        assert c.execute("SELECT MAX(ts_date) FROM market_bars").fetchone()[0] == "2024-12-31"
        assert c.execute("SELECT COUNT(*) FROM market_bars").fetchone()[0] == 2     # unsealed rows kept
    with sqlite3.connect(live) as c:
        assert c.execute("SELECT MIN(ts_date) FROM market_bars").fetchone()[0] == "2025-01-02"
    backups = list(tmp_path.glob("research.pre_seal_*.db"))
    assert len(backups) == 1
    with sqlite3.connect(backups[0]) as c:                # the backup is the pre-delete state
        assert c.execute("SELECT COUNT(*) FROM market_bars").fetchone()[0] == 4


def test_delete_stops_and_deletes_nothing_if_the_copy_does_not_verify(tmp_path, monkeypatch):
    db, live = _research_db(tmp_path), str(tmp_path / "live.db")
    real_copy = sm.copy_to_live

    def tampered_copy(r, l):
        out = real_copy(r, l)
        with sqlite3.connect(l) as c:                     # corrupt one copied bar
            c.execute("UPDATE market_bars SET close = close + 1 WHERE ts_date='2025-01-02'")
        return out

    monkeypatch.setattr(sm, "copy_to_live", tampered_copy)
    assert sm.run(db, live, execute=True, delete=True, confirm=sm.CONFIRM_PHRASE) == 1
    with sqlite3.connect(db) as c:
        assert c.execute("SELECT COUNT(*) FROM market_bars").fetchone()[0] == 4   # nothing deleted
    assert not list(tmp_path.glob("research.pre_seal_*.db"))                      # no backup taken yet


def test_the_boundary_day_itself_moves_and_the_day_before_stays(tmp_path):
    """2025-01-01 is a market holiday so real data never has a bar on it -- but the
    boundary must still be exact: a row dated ON the seal date is sealed (copied and
    deleted), the day before is not."""
    db = _research_db(tmp_path, dates=("2024-12-31", "2025-01-01", "2025-01-02"))
    live = str(tmp_path / "live.db")
    assert sm.run(db, live, execute=True, delete=True, confirm=sm.CONFIRM_PHRASE) == 0
    with sqlite3.connect(db) as c:
        assert [r[0] for r in c.execute("SELECT DISTINCT ts_date FROM market_bars ORDER BY 1")] == ["2024-12-31"]
    with sqlite3.connect(live) as c:
        assert [r[0] for r in c.execute("SELECT DISTINCT ts_date FROM market_bars ORDER BY 1")] == \
            ["2025-01-01", "2025-01-02"]


def test_verify_catches_tampering_confined_to_the_boundary_day(tmp_path):
    """A checksum that silently excluded the seal date itself would exclude it from BOTH
    sides and still match -- so tamper with only that one day's bar in the copy."""
    db = _research_db(tmp_path, dates=("2024-12-31", "2025-01-01", "2025-01-02"))
    live = str(tmp_path / "live.db")
    sm.copy_to_live(db, live)
    assert sm.verify(db, live)["ok"]
    with sqlite3.connect(live) as c:
        c.execute("UPDATE market_bars SET close = close + 5 WHERE ts_date='2025-01-01'")
    assert not sm.verify(db, live)["ok"]


def test_checksum_detects_a_single_changed_value(tmp_path):
    db = _research_db(tmp_path)
    before = sm.bars_checksum(db)
    with sqlite3.connect(db) as c:
        c.execute("UPDATE market_bars SET volume = volume + 1 WHERE ts_date='2025-01-03'")
    assert sm.bars_checksum(db) != before


# ── runner ────────────────────────────────────────────────────────────────────

def test_runner_store_update_targets_the_live_store_unsealed(monkeypatch):
    seen = {}

    class _R:
        stdout = ""
        stderr = ""
        returncode = 0

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        return _R()

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    runner.launch_store_update()                          # the real function, default arguments
    code = seen["cmd"][-1]
    assert "live_market_data.db" in code and "sealed=False" in code
    assert "DATA/market_data.db" not in code


def test_pr003_window_ends_before_the_seal():
    """The fixed PR-003 window and the seal must agree: the window's last day is
    the day before SEAL_START."""
    import pr003_replication as pr
    assert pr.WINDOW[1] == date(2024, 12, 31)
    assert (seal.SEAL_START - pr.WINDOW[1]).days == 1
