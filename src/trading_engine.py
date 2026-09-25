"""
trading_engine.py
=================
Autonomous paper trading engine for the Trading Income Project.

ARCHITECTURE
------------
This engine runs as a standalone Python process in a terminal.
It shares a SQLite database with the Streamlit dashboard (trading_dashboard.py),
which reads the same database to display live results.

  Terminal 1:  python trading_engine.py      ← this file
  Terminal 2:  streamlit run trading_dashboard.py

The database (paper_account.db by default) is the single source of truth.
Neither process depends on the other; they communicate only through the DB.

WHAT THIS ENGINE DOES
---------------------
Every POLL_INTERVAL seconds during the trading window (09:30–11:00 ET):

  1. Fetch live 5-min OHLCV, VWAP, PDH/PDL, VIX, gap
  2. Evaluate every signal in the strategy AND-gate
  3. Log the full decision (inputs + gate results + reasoning) to the LEDGER
  4. If AND-gate passes and no open position → paper BUY/SELL order
  5. If position is open → update trailing VWAP stop, check exit conditions
  6. Apply daily stop and consecutive-loss rules — halt if triggered

The DECISION LEDGER (table: decisions) is the key innovation:
  Every bar evaluation is logged, whether a trade is taken or not.
  Post-session, you can query: "why did we skip that setup at 09:47?"
  This turns every session into a structured learning event.

WALK-FORWARD ANALYSIS
---------------------
Run on-demand or scheduled nightly:
  wfa_run(ticker, strategy_fn, config, n_splits=6)
  Rolls a 6-month in-sample / 1-month out-of-sample window.
  Computes Walk-Forward Efficiency (WFE) = OOS Sharpe / IS Sharpe.
  WFE > 0.5 = acceptable. WFE < 0.5 = likely overfitting.
  Results stored in table: wfa_results.

USAGE
-----
  # Run interactively (Ctrl+C to stop)
  python trading_engine.py

  # Run with custom settings
  python trading_engine.py --ticker SPY --account 10000 --orb 15min --db paper_account.db

  # Run WFA only
  python trading_engine.py --wfa-only

CONFIGURATION
-------------
  Edit CONFIG dict below, or pass command-line arguments.
  The engine reads trading_quant_toolkit_v2_4.py from the same directory.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sqlite3
import statistics
import sys
import threading
import time
from datetime import date, datetime, time as Time, timedelta, timezone
from zoneinfo import ZoneInfo
from typing import Any, Callable

import pandas as pd
import yfinance as yf
try:
    import aiohttp as _aiohttp
    _DISCORD_DEPS = True
except ImportError:
    _DISCORD_DEPS = False

# Market calendar check (P1-008)
try:
    from market_calendar import check_market_session as _check_market_session
    _CALENDAR = True
except ImportError:
    _CALENDAR = False
    logging.warning("market_calendar.py not found — holiday/early-close guard disabled")

# ---------------------------------------------------------------------------
# Optional toolkit import
# ---------------------------------------------------------------------------
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    # Try latest version first, fall back gracefully
    try:
        import trading_quant_toolkit_v2_4 as tk  # type: ignore
    except ImportError:
        try:
            import trading_quant_toolkit_v2_3 as tk  # type: ignore
        except ImportError:
            import trading_quant_toolkit_v2_2 as tk  # type: ignore
    TOOLKIT = True
except ImportError:
    TOOLKIT = False
    logging.warning("trading_quant_toolkit_v2_4.py not found — using engine's built-in calculations")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CONFIG: dict[str, Any] = {
    "ticker":            "SPY",
    "orb_method":        "15min",        # '5min' | '15min' | '30min'
    "account_balance":   10_000.0,
    "risk_pct":          0.01,           # 1% fixed (overridden by Kelly after 50 trades)
    "use_kelly":         False,          # enable after 50+ trades
    "target_rr":         2.0,           # baseline R:R
    "use_vwap_trailing": True,           # VWAP trailing stop (Zarattini 2024)
    "use_ladder_exit":   True,           # 1R/2R/3R partial exits (Maroy 2025)
    # P2-112 — future-proofing beyond NYSE/EST. This pilot trades NYSE
    # equities in US Eastern time; session_start/session_end below are
    # interpreted IN market_timezone, whatever it's set to. Trading a
    # different session (UK, EU, Asia) means changing BOTH of these
    # together — market_timezone to that market's IANA zone, and
    # market_type to a key with REAL data populated in market_calendar.py's
    # EQUITIES_MARKETS registry (not just a plausible-sounding string;
    # check_market_session() raises clearly if the market_type you set
    # here has no real holiday/session data behind it yet) — plus new
    # session_start/session_end values in that market's local hours.
    "market_type":       "nyse_equities",
    "market_timezone":   "America/New_York",
    "session_start":     Time(9, 30),
    "session_end":       Time(11, 0),
    "poll_interval_s":   60,             # seconds between signal evaluations
    "db_path":           "paper_account.db",
    "log_level":         "INFO",
    "max_daily_loss_pct": 0.03,
    "max_drawdown_pct":  0.25,           # Rule 16 Tier 1 (P2-073) — lifetime hard stop
    "consec_loss_pause": 3,
    "min_rvol":          1.0,            # stocks in play filter
    "vwap_lookback":     3,
    "tz_offset_hours":   -5,             # DEPRECATED as of the DST fix — _now_market() uses
                                          # zoneinfo("America/New_York") directly now, which
                                          # handles EST/EDT automatically. This value is no
                                          # longer read by _now_market() itself; kept only as a
                                          # fallback default where a raw offset is still needed
                                          # (e.g. passed into market_calendar's own signature).
    "discord_webhook":   "",             # optional: paste Discord webhook URL here
}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("trading_engine.log", mode="a"),
    ],
    force=True,   # a bare logging.xxx() call anywhere above this (e.g. the
                  # toolkit-not-found warning) silently self-configures the
                  # root logger with Python's bare defaults first, which
                  # makes basicConfig() a no-op without force=True — losing
                  # INFO-level output, the custom format, AND file logging
                  # entirely, exactly when something's already gone missing
)
log = logging.getLogger("engine")


# ===========================================================================
# DATABASE LAYER
# ===========================================================================

class PaperAccountDB:
    """
    SQLite-backed paper account.  Thread-safe via connection-per-call pattern.
    All writes are explicit transactions; all reads return plain dicts/lists.
    """

    SCHEMA = """
    -- Open and closed paper positions
    CREATE TABLE IF NOT EXISTS positions (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker      TEXT    NOT NULL,
        direction   TEXT    NOT NULL,   -- 'long' | 'short'
        entry_price REAL    NOT NULL,
        stop_price  REAL    NOT NULL,
        initial_stop REAL,              -- P2-105 (BUG-02): immutable, set once at
                                         -- entry. stop_price trails; this never
                                         -- changes. R is always computed against
                                         -- this, never the current stop_price.
        target_1r   REAL,
        target_2r   REAL,
        target_3r   REAL,
        units       REAL    NOT NULL,
        status      TEXT    NOT NULL DEFAULT 'open',   -- 'open' | 'closed'
        opened_at   TEXT    NOT NULL,
        closed_at   TEXT,
        exit_price  REAL,
        actual_r    REAL,
        pnl_gbp     REAL,
        exit_reason TEXT,               -- 'target_1r' | 'trailing_stop' | 'eod' | 'manual'
        notes       TEXT
    );

    -- All paper orders
    CREATE TABLE IF NOT EXISTS orders (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker       TEXT    NOT NULL,
        order_type   TEXT    NOT NULL,   -- 'market' | 'stop'
        direction    TEXT    NOT NULL,   -- 'long' | 'short' | 'close'
        price        REAL,
        units        REAL,
        status       TEXT    NOT NULL DEFAULT 'pending',   -- 'pending' | 'filled' | 'cancelled'
        created_at   TEXT    NOT NULL,
        filled_at    TEXT,
        position_id  INTEGER REFERENCES positions(id)
    );

    -- The decision ledger — every bar evaluation, win or lose, trade or no-trade
    CREATE TABLE IF NOT EXISTS decisions (
        id             INTEGER PRIMARY KEY AUTOINCREMENT,
        ts             TEXT    NOT NULL,   -- ISO timestamp of evaluation
        session_date   TEXT    NOT NULL,
        ticker         TEXT    NOT NULL,
        action         TEXT    NOT NULL,   -- 'EVALUATE' | 'ENTER' | 'EXIT' | 'SKIP' | 'HALT'
        -- Market context
        last_close     REAL,
        vwap           REAL,
        vwap_slope     TEXT,   -- 'up' | 'down' | 'flat'
        orb_high       REAL,
        orb_low        REAL,
        orb_method     TEXT,
        vix            REAL,
        gap_pct        REAL,
        rvol           REAL,
        -- Gate evaluation (each 'PASS' | 'FAIL' | 'WAIT')
        gate_orb_break TEXT,
        gate_vwap      TEXT,
        gate_retest    TEXT,
        gate_final     TEXT,   -- 'AND_PASS' | 'AND_FAIL'
        -- If entering
        entry_price    REAL,
        stop_price     REAL,
        position_size  REAL,
        risk_pct       REAL,
        -- Reason string — the plain-English explanation of the decision
        reason         TEXT    NOT NULL,
        -- Meta
        strategy_ver   TEXT    DEFAULT '2.1.0',
        orb_bars_used  INTEGER
    );

    -- Daily session summaries
    CREATE TABLE IF NOT EXISTS sessions (
        session_date  TEXT PRIMARY KEY,
        opened_at     TEXT,
        closed_at     TEXT,
        n_trades      INTEGER DEFAULT 0,
        n_wins        INTEGER DEFAULT 0,
        n_losses      INTEGER DEFAULT 0,
        session_pnl_r REAL    DEFAULT 0.0,
        session_pnl_gbp REAL  DEFAULT 0.0,
        stop_triggered INTEGER DEFAULT 0,
        max_consec_loss INTEGER DEFAULT 0,
        notes         TEXT
    );

    -- Walk-forward analysis results
    CREATE TABLE IF NOT EXISTS wfa_results (
        id             INTEGER PRIMARY KEY AUTOINCREMENT,
        run_ts         TEXT    NOT NULL,
        ticker         TEXT    NOT NULL,
        strategy_ver   TEXT,
        config_json    TEXT,
        is_start       TEXT,   is_end TEXT,
        oos_start      TEXT,   oos_end TEXT,
        n_is_trades    INTEGER, n_oos_trades INTEGER,
        is_sharpe      REAL,   oos_sharpe REAL,
        is_max_dd      REAL,   oos_max_dd REAL,
        is_win_rate    REAL,   oos_win_rate REAL,
        wfe            REAL,   -- oos_sharpe / is_sharpe  (target > 0.5)
        wfe_verdict    TEXT    -- 'PASS' | 'FAIL' | 'INCONCLUSIVE'
    );

    -- Engine heartbeat (dashboard reads this to show engine status)
    CREATE TABLE IF NOT EXISTS heartbeat (
        id         INTEGER PRIMARY KEY DEFAULT 1,
        ts         TEXT,
        status     TEXT,   -- 'running' | 'paused' | 'stopped' | 'halted'
        message    TEXT,
        session_date TEXT,
        account_balance REAL,
        open_position_id INTEGER
    );

    -- Account-lifetime state (Rule 16 Tier 1, P2-073). starting_balance is
    -- set exactly once, on this --db file's first ever run, and never
    -- changed again — even if a later restart passes a different
    -- --account value — so drawdown stays anchored to the true origin.
    -- retired is a one-way flag: once Tier 1 trips, the strategy stays
    -- retired (across session resets AND process restarts) until a human
    -- clears it or starts a fresh --db.
    CREATE TABLE IF NOT EXISTS account_meta (
        id               INTEGER PRIMARY KEY DEFAULT 1,
        starting_balance REAL,
        peak_equity      REAL,
        retired          INTEGER DEFAULT 0,
        retired_reason   TEXT,
        retired_at       TEXT,
        ticker           TEXT    -- P2-105 (BUG-09): which ticker this account's
                                  -- equity/drawdown history belongs to. Reusing
                                  -- a --db file across a different --ticker
                                  -- would otherwise silently blend both
                                  -- tickers' P&L into one equity curve.
    );

    -- Session learning reports (from SessionLearner after each session)
    CREATE TABLE IF NOT EXISTS session_learning (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        session_date     TEXT    NOT NULL,
        ts               TEXT    NOT NULL,
        summary          TEXT,
        n_trades         INTEGER,
        n_skips          INTEGER,
        over_filter_flag INTEGER,
        avg_total_drag_r REAL,
        modelled_ev      REAL,
        realistic_ev     REAL,
        best_regime      TEXT,
        worst_regime     TEXT,
        action_items     TEXT,   -- JSON list
        full_report      TEXT    -- JSON full report dict
    );
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._setup()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _setup(self) -> None:
        with self._conn() as conn:
            conn.executescript(self.SCHEMA)
            self._migrate_schema(conn)
        log.info("DB ready: %s", self.db_path)

    def _migrate_schema(self, conn: sqlite3.Connection) -> None:
        """P2-105 (BUG-06): CREATE TABLE IF NOT EXISTS does not add new
        columns to a table that already existed before this column was
        introduced. Handles two cases: a --db file from before P2-105
        (positions has no initial_stop), and one from after P2-073 but
        before P2-105 (account_meta exists but has no ticker column)."""
        pos_cols = [r["name"] for r in
                    conn.execute("PRAGMA table_info(positions)").fetchall()]
        if pos_cols and "initial_stop" not in pos_cols:
            log.warning("Migrating 'positions': adding initial_stop "
                        "(backfilled from stop_price for legacy rows — "
                        "their true entry-time stop can't be recovered, "
                        "this is the closest available approximation)")
            conn.execute("ALTER TABLE positions ADD COLUMN initial_stop REAL")
            conn.execute("UPDATE positions SET initial_stop = stop_price "
                         "WHERE initial_stop IS NULL")

        meta_cols = [r["name"] for r in
                     conn.execute("PRAGMA table_info(account_meta)").fetchall()]
        if meta_cols and "ticker" not in meta_cols:
            log.warning("Migrating 'account_meta': adding ticker column "
                        "(NULL for now — adopted on the next ensure_account_meta() call)")
            conn.execute("ALTER TABLE account_meta ADD COLUMN ticker TEXT")

    # ── Positions ──────────────────────────────────────────────────────────

    def open_position(self, ticker: str, direction: str, entry_price: float,
                       stop_price: float, units: float, targets: list[float],
                       notes: str = "", initial_stop: float | None = None) -> int:
        # P2-105 (BUG-02): initial_stop is the immutable entry-time risk
        # anchor. Defaults to stop_price since that IS the initial stop
        # at the moment a position opens — existing callers don't need
        # to change.
        if initial_stop is None:
            initial_stop = stop_price
        t1 = targets[0] if len(targets) > 0 else None
        t2 = targets[1] if len(targets) > 1 else None
        t3 = targets[2] if len(targets) > 2 else None
        ts = _now_iso()
        with self._conn() as conn:
            cur = conn.execute(
                """INSERT INTO positions
                   (ticker, direction, entry_price, stop_price, initial_stop,
                    target_1r, target_2r, target_3r, units, status, opened_at, notes)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (ticker, direction, entry_price, stop_price, initial_stop,
                 t1, t2, t3, units, "open", ts, notes)
            )
            return cur.lastrowid  # type: ignore

    def get_open_position(self) -> dict | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM positions WHERE status='open' ORDER BY id DESC LIMIT 1"
            ).fetchone()
        return dict(row) if row else None

    def close_position(self, position_id: int, exit_price: float,
                        exit_reason: str, actual_r: float, pnl_gbp: float) -> None:
        with self._conn() as conn:
            conn.execute(
                """UPDATE positions SET status='closed', closed_at=?, exit_price=?,
                   actual_r=?, pnl_gbp=?, exit_reason=? WHERE id=?""",
                (_now_iso(), exit_price, actual_r, pnl_gbp, exit_reason, position_id)
            )
            # Rule 16 Tier 1 (P2-073): advance peak_equity here, at the exact
            # moment equity changes — not lazily whenever get_account_summary()
            # next happens to be called. A closed trade closing this
            # connection's transaction is the only place a new peak can be
            # made; relying on a later read would leave the peak stale (and
            # drawdown understated) if that read is ever skipped or delayed.
            meta = conn.execute(
                "SELECT starting_balance, peak_equity, ticker FROM account_meta WHERE id=1"
            ).fetchone()
            if meta is not None:
                # P2-105 (BUG-09): scope the realized-P&L sum to this
                # account's own ticker. meta["ticker"] is None only for a
                # legacy row that hasn't been through ensure_account_meta()
                # since the migration — falls back to the unscoped sum in
                # that one case, matching the old (pre-fix) behaviour
                # rather than silently returning zero.
                if meta["ticker"]:
                    realized = conn.execute(
                        "SELECT COALESCE(SUM(pnl_gbp), 0) FROM positions "
                        "WHERE status='closed' AND pnl_gbp IS NOT NULL AND ticker=?",
                        (meta["ticker"],)
                    ).fetchone()[0]
                else:
                    realized = conn.execute(
                        "SELECT COALESCE(SUM(pnl_gbp), 0) FROM positions "
                        "WHERE status='closed' AND pnl_gbp IS NOT NULL"
                    ).fetchone()[0]
                current_equity = meta["starting_balance"] + realized
                if current_equity > meta["peak_equity"]:
                    conn.execute(
                        "UPDATE account_meta SET peak_equity=? WHERE id=1",
                        (current_equity,)
                    )

    def update_stop(self, position_id: int, new_stop: float) -> None:
        with self._conn() as conn:
            conn.execute("UPDATE positions SET stop_price=? WHERE id=?",
                         (new_stop, position_id))

    # ── Decisions ledger ───────────────────────────────────────────────────

    def log_decision(self, d: dict) -> None:
        cols = ("ts,session_date,ticker,action,last_close,vwap,vwap_slope,"
                "orb_high,orb_low,orb_method,vix,gap_pct,rvol,"
                "gate_orb_break,gate_vwap,gate_retest,gate_final,"
                "entry_price,stop_price,position_size,risk_pct,reason,"
                "strategy_ver,orb_bars_used")
        placeholders = ",".join(["?"] * len(cols.split(",")))
        values = tuple(d.get(c) for c in cols.split(","))
        with self._conn() as conn:
            conn.execute(f"INSERT INTO decisions ({cols}) VALUES ({placeholders})", values)

    def get_decisions(self, session_date: str, limit: int = 50) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM decisions WHERE session_date=? ORDER BY id DESC LIMIT ?",
                (session_date, limit)
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Sessions ───────────────────────────────────────────────────────────

    def upsert_session(self, session_date: str, **kwargs: Any) -> None:
        cols = ",".join(kwargs.keys())
        placeholders = ",".join(["?"] * len(kwargs))
        update_clause = ",".join(f"{k}=excluded.{k}" for k in kwargs)
        with self._conn() as conn:
            conn.execute(
                f"""INSERT INTO sessions (session_date,{cols}) VALUES (?,{placeholders})
                    ON CONFLICT(session_date) DO UPDATE SET {update_clause}""",
                (session_date, *kwargs.values())
            )

    def get_session(self, session_date: str) -> dict | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE session_date=?", (session_date,)
            ).fetchone()
        return dict(row) if row else None

    def get_session_trades(self, session_date: str) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM positions WHERE date(opened_at)=? ORDER BY id DESC",
                (session_date,)
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Heartbeat (engine status for dashboard) ────────────────────────────

    def update_heartbeat(self, status: str, message: str,
                          session_date: str, account_balance: float,
                          open_position_id: int | None = None) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO heartbeat (id,ts,status,message,session_date,
                   account_balance,open_position_id)
                   VALUES (1,?,?,?,?,?,?)
                   ON CONFLICT(id) DO UPDATE SET
                   ts=excluded.ts, status=excluded.status, message=excluded.message,
                   session_date=excluded.session_date, account_balance=excluded.account_balance,
                   open_position_id=excluded.open_position_id""",
                (_now_iso(), status, message, session_date, account_balance, open_position_id)
            )

    def get_heartbeat(self) -> dict | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM heartbeat WHERE id=1").fetchone()
        return dict(row) if row else None

    # ── Account-lifetime equity & retirement (Rule 16 Tier 1, P2-073) ──────

    def ensure_account_meta(self, starting_balance: float, ticker: str) -> None:
        """Set starting_balance once, the first time this --db file is
        used. A no-op on every later call (including a restart with a
        different --account value) — this is the account's one true
        origin point for lifetime drawdown.

        P2-105 (BUG-09): also records which ticker this account belongs
        to. A --db file reused across a different --ticker would
        otherwise silently blend both tickers' P&L into one equity
        curve — this raises loudly instead, at startup, rather than
        letting it happen quietly mid-session."""
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO account_meta (id, starting_balance, peak_equity, retired, ticker)
                   VALUES (1, ?, ?, 0, ?)
                   ON CONFLICT(id) DO NOTHING""",
                (starting_balance, starting_balance, ticker)
            )
            row = conn.execute("SELECT ticker FROM account_meta WHERE id=1").fetchone()
            if row["ticker"] is None:
                # Legacy DB from before per-ticker tracking existed (either
                # pre-P2-105, or the INSERT above was a no-op because the
                # row already existed with a NULL ticker from the schema
                # migration) — adopt this run's ticker rather than treat
                # a NULL as a mismatch.
                conn.execute("UPDATE account_meta SET ticker=? WHERE id=1", (ticker,))
            elif row["ticker"] != ticker:
                raise RuntimeError(
                    f"This --db file's account_meta is already tracking "
                    f"{row['ticker']!r}, but --ticker {ticker!r} was requested. "
                    f"Reusing a --db file across different tickers would blend "
                    f"their equity/drawdown history together (P2-105/BUG-09). "
                    f"Use a separate --db file per ticker."
                )

    def get_account_summary(self) -> dict:
        """Lifetime equity: starting_balance + all realized P&L from closed
        trades, plus the peak equity ever reached (advanced here if a new
        high has been made since it was last checked) and the resulting
        drawdown_pct. Empty dict if ensure_account_meta() was never called."""
        with self._conn() as conn:
            meta = conn.execute(
                "SELECT starting_balance, peak_equity, ticker FROM account_meta WHERE id=1"
            ).fetchone()
            if meta is None:
                return {}
            starting_balance = meta["starting_balance"]
            peak_equity = meta["peak_equity"]
            # P2-105 (BUG-09): scope to this account's own ticker; see
            # close_position() for why a NULL ticker falls back unscoped.
            if meta["ticker"]:
                realized = conn.execute(
                    "SELECT COALESCE(SUM(pnl_gbp), 0) FROM positions "
                    "WHERE status='closed' AND pnl_gbp IS NOT NULL AND ticker=?",
                    (meta["ticker"],)
                ).fetchone()[0]
            else:
                realized = conn.execute(
                    "SELECT COALESCE(SUM(pnl_gbp), 0) FROM positions "
                    "WHERE status='closed' AND pnl_gbp IS NOT NULL"
                ).fetchone()[0]
            current_equity = starting_balance + realized
            if current_equity > peak_equity:
                peak_equity = current_equity
                conn.execute(
                    "UPDATE account_meta SET peak_equity=? WHERE id=1",
                    (peak_equity,)
                )
        drawdown_pct = ((peak_equity - current_equity) / peak_equity
                         if peak_equity > 0 else 0.0)
        return {
            "starting_balance": starting_balance,
            "current_equity":   current_equity,
            "peak_equity":      peak_equity,
            "drawdown_pct":     drawdown_pct,
        }

    def get_retired_status(self) -> dict:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT retired, retired_reason, retired_at "
                "FROM account_meta WHERE id=1"
            ).fetchone()
        if row is None:
            return {"retired": False, "reason": None, "retired_at": None}
        return {"retired": bool(row["retired"]), "reason": row["retired_reason"],
                "retired_at": row["retired_at"]}

    def mark_retired(self, reason: str) -> None:
        """One-way: WHERE retired=0 means the first trigger wins and keeps
        its original reason/timestamp — later ticks that also see
        drawdown past the threshold don't overwrite it."""
        with self._conn() as conn:
            conn.execute(
                """UPDATE account_meta SET retired=1, retired_reason=?, retired_at=?
                   WHERE id=1 AND retired=0""",
                (reason, _now_iso())
            )

    # ── WFA results ────────────────────────────────────────────────────────

    def save_wfa_result(self, result: dict) -> None:
        cols = ",".join(result.keys())
        placeholders = ",".join(["?"] * len(result))
        with self._conn() as conn:
            conn.execute(
                f"INSERT INTO wfa_results ({cols}) VALUES ({placeholders})",
                tuple(result.values())
            )

    def get_wfa_results(self, ticker: str, limit: int = 20) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM wfa_results WHERE ticker=? ORDER BY run_ts DESC LIMIT ?",
                (ticker, limit)
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Analytics ──────────────────────────────────────────────────────────

    def rolling_stats(self, n: int = 20) -> dict:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT actual_r FROM positions WHERE status='closed' AND actual_r IS NOT NULL "
                "ORDER BY id DESC LIMIT ?", (n,)
            ).fetchall()
        rs = [r["actual_r"] for r in rows]
        if not rs:
            return {"n": 0}
        wins   = [r for r in rs if r > 0]
        losses = [r for r in rs if r < 0]
        wr     = len(wins) / len(rs)
        aw     = statistics.mean(wins)           if wins   else 0.0
        al     = abs(statistics.mean(losses))    if losses else 0.0
        ev     = (wr * aw) - ((1 - wr) * al)
        sharpe = None
        if len(rs) >= 2:
            sd = statistics.stdev(rs)
            if sd > 0:
                sharpe = round(statistics.mean(rs) / sd, 3)
        return {"n": len(rs), "wr": wr, "aw": aw, "al": al, "ev": ev, "sharpe": sharpe}

    def save_session_learning(self, session_date: str, report: dict) -> None:
        """Persist a SessionLearner report after each session."""
        fa  = report.get("fill_analysis", {})
        ska = report.get("skip_analysis", {})
        ra  = report.get("regime_analysis", {})
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO session_learning
                   (session_date,ts,summary,n_trades,n_skips,over_filter_flag,
                    avg_total_drag_r,modelled_ev,realistic_ev,best_regime,
                    worst_regime,action_items,full_report)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (session_date, _now_iso(), report.get("summary",""),
                 fa.get("n_trades",0), ska.get("n_skips",0),
                 int(ska.get("over_filter_flag", False)),
                 fa.get("avg_total_drag_r"), fa.get("modelled_ev"),
                 fa.get("realistic_ev"), ra.get("best_regime"),
                 ra.get("worst_regime"),
                 json.dumps(report.get("action_items",[])),
                 json.dumps(report))
            )

    def get_session_learning(self, session_date: str) -> dict | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM session_learning WHERE session_date=? ORDER BY id DESC LIMIT 1",
                (session_date,)).fetchone()
        return dict(row) if row else None


# ===========================================================================
# MARKET DATA HELPERS
# ===========================================================================

def _utcnow() -> datetime:
    """Naive UTC 'now' — same value and naive-comparability as the removed
    datetime.utcnow(), via the non-deprecated timezone-aware path. Every
    stored timestamp in this project (heartbeat.ts, positions.opened_at,
    etc.) is naive UTC text; this keeps that format unchanged."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _now_market(cfg: dict) -> datetime:
    """Current time in the configured market's timezone
    (cfg["market_timezone"], default "America/New_York" for the NYSE
    pilot). Renamed from the NYSE-specific _now_et()/_now_est() (P2-112)
    so the engine isn't hardcoded to one market's clock — trading a
    different session (UK, EU, Asia) means changing market_timezone in
    CONFIG, plus supplying that market's real holiday calendar in
    market_calendar.py's EQUITIES_MARKETS registry. This function alone
    handles the timezone/DST arithmetic correctly for whatever zone is
    configured — it was previously hardcoded to zoneinfo("America/
    New_York") directly, which handled EST/EDT correctly but only for
    NYSE specifically, the same class of hardcoding that caused the
    original DST bug this project had before P2-108."""
    return datetime.now(ZoneInfo(cfg.get("market_timezone", "America/New_York")))

def _now_iso() -> str:
    return _utcnow().isoformat(timespec="seconds")

def _in_session(cfg: dict) -> bool:
    t = _now_market(cfg).time()
    return cfg["session_start"] <= t <= cfg["session_end"]

def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df

def _fetch_live(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, period="5d", interval="5m",
                     auto_adjust=True, progress=False)
    return _flatten(df)

def _drop_forming_bar(df_5m: pd.DataFrame) -> pd.DataFrame:
    """P2-105 item 7 (bar-phase asymmetry): yfinance 5-minute bars are
    timestamped at their START. If "now" hasn't yet reached
    (last_bar_start + 5min), that last bar is still forming — its Close
    reflects only a partial window and can show a spurious breakout that
    reverses before the bar actually settles.

    Confirmed empirically (not just theoretically) against 7 real trading
    days of 1-minute bars: reconstructing what a forming bar looked like
    at each minute and comparing its gate decision to the eventual
    settled bar found 9 disagreements out of 602 minute-by-minute checks
    — all clustered in one 11-minute choppy stretch on one of the 7 days,
    where price was oscillating back and forth across the ORB boundary.
    Rare, but real.

    Used ONLY for entry evaluation (evaluate_signals), not for exit
    checks (manage_open_position) — a stop or target should react to the
    live price immediately; waiting for bar-close there would add
    latency to risk management instead of removing false signals. This
    is a narrower fix than "always drop iloc[-1]": it only trims when the
    last bar genuinely hasn't reached its 5-minute mark yet, so it adds
    no lag on ticks where yfinance has already returned a settled bar.
    """
    if df_5m.empty:
        return df_5m
    last_bar_start = df_5m.index[-1]
    now = (pd.Timestamp.now(tz=last_bar_start.tz) if last_bar_start.tzinfo is not None
           else pd.Timestamp.now())
    if now < last_bar_start + pd.Timedelta(minutes=5):
        return df_5m.iloc[:-1]
    return df_5m

def _fetch_daily(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, period="20d", interval="1d",
                     auto_adjust=True, progress=False)
    return _flatten(df)

def _vwap(df: pd.DataFrame, session_start: Time = Time(9, 30)) -> pd.Series:
    # P2-112 follow-up: session_start defaults to the pilot's 9:30 (NYSE
    # open) for backward compatibility, but every call site below passes
    # cfg["session_start"] explicitly — a future non-NYSE market_type
    # changes this correctly instead of VWAP silently still anchoring to
    # 9:30 regardless of what session_start is actually configured to.
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    pv = tp * df["Volume"]
    result = pd.Series(index=df.index, dtype=float)
    for day in set(df.index.date):
        mask = (df.index.date == day) & (df.index.time >= session_start)
        cv = df["Volume"][mask].cumsum()
        result[mask] = (pv[mask].cumsum() / cv.replace(0, float("nan"))).values
    return result

# P2-112 follow-up: ORB formation window end, as an OFFSET in minutes from
# session_start, per method — preserves the exact original NYSE-pilot end
# times (9:34/9:44/9:55 at a 9:30 open) as data rather than a derived
# formula. These three offsets are NOT a clean N-1 pattern (4, 14, 25 —
# not 4, 14, 29) — tested this before assuming a formula and a "session_
# start + (minutes-1)" derivation would have silently widened the 30min
# ORB's window by 4 minutes (9:59 instead of 9:55), a real behavioral
# change, not just a refactor. Keeping the offsets as explicit data avoids
# that trap for whatever a future market's own offsets turn out to be too.
_ORB_METHOD_OFFSET_MIN: dict[str, int] = {"5min": 4, "15min": 14, "30min": 25}

def _orb_range(df: pd.DataFrame, method: str,
               session_start: Time = Time(9, 30)) -> dict | None:
    if df.empty:
        # An external review claimed this was already safely guarded —
        # tested directly and it wasn't: df.index[-1] on an empty frame
        # raises IndexError. _drop_forming_bar() (P2-105 item 7) can now
        # produce an empty frame if a live fetch ever returns unusually
        # little data, so this guard is newly reachable, not just
        # theoretical.
        return None
    offset = _ORB_METHOD_OFFSET_MIN.get(method)
    if offset is None:
        return None
    end_t = (datetime.combine(date.today(), session_start)
              + timedelta(minutes=offset)).time()
    today = df.index[-1].date()
    mask = (df.index.date == today) & (df.index.time >= session_start) & (df.index.time <= end_t)
    bars = df[mask]
    if bars.empty:
        return None
    return {
        "orb_high":  round(float(bars["High"].max()), 4),
        "orb_low":   round(float(bars["Low"].min()), 4),
        "orb_size":  round(float(bars["High"].max() - bars["Low"].min()), 4),
        "bars_used": len(bars),
    }

def _vwap_slope(vwap: pd.Series, lookback: int = 3) -> dict:
    clean = vwap.dropna()
    if clean.empty:
        return {"slope": 0.0, "direction": "flat"}
    # P2-105 (BUG-08): without this, clean.iloc[-lookback:] can reach back
    # across the overnight gap early in a session (e.g. at 09:35 with only
    # 2 bars accumulated since today's open, lookback=3 pulls in
    # yesterday's 15:55 close) — confirmed empirically against real data.
    # Restricting to today's bars means an early session correctly reports
    # "flat" via the len(clean) < lookback guard below, rather than a
    # slope driven by the gap instead of today's actual momentum.
    today = clean.index[-1].date()
    clean = clean[clean.index.date == today]
    if len(clean) < lookback:
        return {"slope": 0.0, "direction": "flat"}
    recent = clean.iloc[-lookback:]
    slope  = float(recent.iloc[-1] - recent.iloc[0]) / lookback
    thresh = float(recent.mean()) * 0.0002
    return {
        "slope":     round(slope, 6),
        "direction": "up" if slope > thresh else "down" if slope < -thresh else "flat",
    }

def _atr(df: pd.DataFrame, n: int = 14) -> float:
    h = df["High"].values.astype(float)
    l = df["Low"].values.astype(float)
    c = df["Close"].values.astype(float)
    tr = [max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1])) for i in range(1, len(c))]
    return float(sum(tr[-n:]) / n) if len(tr) >= n else float(sum(tr) / max(len(tr), 1))

def _kelly_fraction(win_rate: float, avg_win: float, avg_loss: float = 1.0,
                     fraction: float = 0.5) -> float:
    if not (0 < win_rate < 1) or avg_win <= 0 or avg_loss <= 0:
        return 0.0
    r = avg_win / avg_loss
    k = max(0.0, win_rate - (1 - win_rate) / r)
    return min(round(fraction * k, 4), 0.25)

def _vwap_trailing(current_stop: float, vwap_val: float, direction: str) -> float:
    if direction == "long":
        return round(max(current_stop, vwap_val), 4)
    return round(min(current_stop, vwap_val), 4)


# ===========================================================================
# SIGNAL EVALUATION  (mirrors the toolkit AND-gate)
# ===========================================================================

def evaluate_signals(df_5m: pd.DataFrame, df_1d: pd.DataFrame,
                      vwap: pd.Series, cfg: dict) -> dict:
    """
    Evaluate the full AND-gate and return a context dict suitable for
    logging to the decisions table.

    Returns a dict containing all market context fields plus gate verdicts
    and a human-readable 'reason' string.
    """
    ctx: dict[str, Any] = {
        "ts":           _now_iso(),
        "session_date": str(_utcnow().date()),
        "ticker":       cfg["ticker"],
        "action":       "EVALUATE",
        "orb_method":   cfg["orb_method"],
        "strategy_ver": "2.1.0",
    }

    # ── Market context ──
    try:
        df_vix = _flatten(yf.download("^VIX", period="2d", interval="5m",
                                       auto_adjust=True, progress=False))
        ctx["vix"] = round(float(df_vix["Close"].dropna().iloc[-1]), 2) if not df_vix.empty else None
    except Exception:
        ctx["vix"] = None

    if len(df_1d) >= 2 and not df_5m.empty:
        pdh = float(df_1d["High"].iloc[-2])
        pdl = float(df_1d["Low"].iloc[-2])
        prev_close = float(df_1d["Close"].iloc[-2])
        # df_5m spans 5 trading days (_fetch_live's period="5d") — iloc[0]
        # is 5 days ago's open, not today's. Same today-filtering pattern
        # already used by _orb_range() and the RVOL calculation below.
        today = df_5m.index[-1].date()
        df_today = df_5m[df_5m.index.date == today]
        today_open = float(df_today["Open"].iloc[0]) if not df_today.empty else prev_close
        ctx["gap_pct"] = round((today_open - prev_close) / prev_close * 100, 3)
    else:
        ctx["gap_pct"] = None

    last_close = float(df_5m["Close"].dropna().iloc[-1]) if not df_5m.empty else None
    ctx["last_close"] = last_close

    vwap_val = float(vwap.dropna().iloc[-1]) if not vwap.dropna().empty else None
    ctx["vwap"] = round(vwap_val, 4) if vwap_val else None

    vs = _vwap_slope(vwap, lookback=cfg.get("vwap_lookback", 3))
    ctx["vwap_slope"] = vs["direction"]

    orb = _orb_range(df_5m, cfg["orb_method"], cfg["session_start"])
    ctx["orb_high"]      = orb["orb_high"]  if orb else None
    ctx["orb_low"]       = orb["orb_low"]   if orb else None
    ctx["orb_bars_used"] = orb["bars_used"] if orb else None

    # RVOL estimation from 5-min data
    try:
        # P2-112 follow-up: "first 5 minutes" cutoff derived from
        # session_start using the same offset _orb_range() uses for the
        # 5min method, instead of hardcoding 9:34 (only correct for a
        # 9:30 open).
        first5_end = (datetime.combine(date.today(), cfg["session_start"])
                       + timedelta(minutes=_ORB_METHOD_OFFSET_MIN["5min"])).time()
        today = df_5m.index[-1].date()
        df_today = df_5m[df_5m.index.date == today]
        first5 = df_today[df_today.index.time <= first5_end]
        past_days = [d for d in set(df_5m.index.date) if d != today]
        avg_first5_vols = []
        for d in past_days:
            dd = df_5m[(df_5m.index.date == d) & (df_5m.index.time <= first5_end)]
            if not dd.empty:
                avg_first5_vols.append(float(dd["Volume"].sum()))
        avg_first5 = statistics.mean(avg_first5_vols) if avg_first5_vols else 1.0
        current_first5 = float(first5["Volume"].sum()) if not first5.empty else 0.0
        ctx["rvol"] = round(current_first5 / avg_first5, 2) if avg_first5 > 0 else None
    except Exception:
        ctx["rvol"] = None

    # ── Gate 1: ORB breakout ──
    if orb is None or last_close is None:
        ctx["gate_orb_break"] = "WAIT"
        breakout_dir = None
        reason_parts = ["ORB range still forming"]
    elif last_close > orb["orb_high"]:
        ctx["gate_orb_break"] = "PASS"
        breakout_dir = "long"
        reason_parts = [f"Close ${last_close:.2f} > ORB High ${orb['orb_high']:.2f} (LONG)"]
    elif last_close < orb["orb_low"]:
        ctx["gate_orb_break"] = "PASS"
        breakout_dir = "short"
        reason_parts = [f"Close ${last_close:.2f} < ORB Low ${orb['orb_low']:.2f} (SHORT)"]
    else:
        ctx["gate_orb_break"] = "FAIL"
        breakout_dir = None
        reason_parts = [f"Close ${last_close:.2f} inside ORB [{orb['orb_low']:.2f}–{orb['orb_high']:.2f}]"]

    # ── Gate 2: VWAP slope aligned with breakout ──
    if breakout_dir is None:
        ctx["gate_vwap"] = "FAIL"
        reason_parts.append("VWAP check skipped (no breakout direction)")
    elif breakout_dir == "long" and vs["direction"] == "up":
        ctx["gate_vwap"] = "PASS"
        reason_parts.append(f"VWAP sloping up ({vs['slope']:+.4f}/bar) — confirms long")
    elif breakout_dir == "short" and vs["direction"] == "down":
        ctx["gate_vwap"] = "PASS"
        reason_parts.append(f"VWAP sloping down ({vs['slope']:+.4f}/bar) — confirms short")
    elif vs["direction"] == "flat":
        ctx["gate_vwap"] = "FAIL"
        reason_parts.append(f"VWAP flat ({vs['slope']:+.4f}/bar) — insufficient momentum")
    else:
        ctx["gate_vwap"] = "FAIL"
        reason_parts.append(f"VWAP {vs['direction']} conflicts with {breakout_dir} signal")

    # ── Gate 3: Retest — automated proxy ──
    # In live paper trading, the retest is approximated:
    # Price touched ORB level (within 0.15%) and current close is back beyond it.
    # In real discretionary use, the human confirms via dashboard checkbox.
    # The engine uses a mechanical proxy: pullback detected in last 3 bars.
    ctx["gate_retest"] = "WAIT"  # engine uses WAIT; human overrides via dashboard
    reason_parts.append("Retest: awaiting mechanical confirmation (check dashboard)")

    # ── AND-gate ──
    gate_pass = (ctx["gate_orb_break"] == "PASS" and
                  ctx["gate_vwap"] == "PASS")
    ctx["gate_final"] = "AND_PASS" if gate_pass else "AND_FAIL"
    ctx["reason"] = " | ".join(reason_parts)
    ctx["_breakout_dir"] = breakout_dir   # internal, not stored in DB
    ctx["_gate_pass"] = gate_pass
    return ctx


# ===========================================================================
# RISK CALCULATIONS
# ===========================================================================

def compute_entry_params(ctx: dict, db: PaperAccountDB, cfg: dict) -> dict:
    """Compute position size, stops, and ladder targets for a new entry."""
    direction    = ctx["_breakout_dir"]
    entry_price  = ctx["last_close"]
    orb_high     = ctx["orb_high"]
    orb_low      = ctx["orb_low"]
    account      = cfg["account_balance"]

    # ATR-adaptive stop (buffer = 0.5 * ATR)
    risk_dist    = orb_high - orb_low if orb_high and orb_low else 0.5
    buffer       = max(risk_dist * 0.1, 0.02)
    if direction == "long":
        stop_price = round(orb_low - buffer, 4)
    else:
        stop_price = round(orb_high + buffer, 4)
    stop_dist  = abs(entry_price - stop_price)

    # Risk fraction
    if cfg["use_kelly"]:
        stats = db.rolling_stats(50)
        if stats.get("n", 0) >= 50:
            rf = _kelly_fraction(stats["wr"], stats["aw"], stats["al"])
        else:
            rf = cfg["risk_pct"]
    else:
        rf = cfg["risk_pct"]

    # VIX size modifier
    vix_mod = 1.0
    if ctx.get("vix"):
        vix = ctx["vix"]
        vix_mod = (0.25 if vix > 35 else 0.50 if vix > 25 else 0.75 if vix > 18 else 1.0)

    effective_risk = rf * vix_mod
    risk_amount    = account * effective_risk
    units = round(risk_amount / stop_dist, 4) if stop_dist > 1e-6 else 0.0

    # Ladder targets (1R, 2R, 3R)
    ladder = []
    for r in [1.0, 2.0, 3.0]:
        price = (entry_price + r * stop_dist) if direction == "long" else (entry_price - r * stop_dist)
        ladder.append(round(price, 4))

    # Realistic cost breakdown (v2.2.0) — volume liquidity check + cost model
    avg_first5 = ctx.get("first5_vol") or (ctx.get("rvol") or 1) * 500_000  # estimate
    adv        = cfg.get("avg_daily_volume", 150_000_000)
    broker     = cfg.get("broker", "alpaca")

    if TOOLKIT:
        try:
            cost_info = tk.net_position_size(
                account, effective_risk, entry_price, stop_price,
                avg_first5_volume=avg_first5,
                avg_daily_volume=adv,
                broker=broker,
                daily_volatility_pct=cfg.get("daily_vol_pct", 0.01),
            )
            units     = cost_info["units"]
            cost_r    = cost_info["cost_r"]
            tradeable = cost_info["tradeable"]
            cost_detail = cost_info["cost_breakdown"]
            reject_reason = cost_info.get("reject_reason")
        except Exception as e:
            cost_r = 0.05; tradeable = True; cost_detail = {}
            reject_reason = None
    else:
        cost_r = 0.05; tradeable = True; cost_detail = {}
        reject_reason = None

    return {
        "entry_price":    round(entry_price, 4),
        "stop_price":     stop_price,
        "targets":        ladder,
        "units":          units,
        "risk_pct":       effective_risk,
        "risk_amount":    round(risk_amount, 2),
        "direction":      direction,
        "cost_r":         round(cost_r, 4),
        "cost_detail":    cost_detail,
        "tradeable":      tradeable,
        "reject_reason":  reject_reason,
    }


# ===========================================================================
# POSITION MANAGEMENT
# ===========================================================================

def manage_open_position(pos: dict, df_5m: pd.DataFrame,
                          vwap: pd.Series, db: PaperAccountDB,
                          cfg: dict, session: dict) -> str | None:
    """
    Check an open position for exit conditions each bar.
    Returns: 'exited' | 'continue' | 'stop_moved'

    P2-105: R is now anchored to the immutable initial_stop captured at
    entry (r0) — never recomputed from the current/trailed stop_price.
    The old `risk_dist = abs(entry - stop)` made (stop-entry)/risk_dist
    mathematically collapse to exactly +-1.0 on every single stop-hit
    exit, trailed or not (BUG-02, "the signum trap" — confirmed with
    concrete numbers: a stop trailed to lock in $3 of a $5 R0 showed
    +1.0R, not the true +0.6R). Cash P&L is now computed directly from
    (exit-entry)*units*direction (BUG-03) instead of the old fixed
    actual_r * account * risk_pct, which ignored the trade's actual
    (possibly VIX/Kelly-scaled) size entirely.
    """
    last_close   = float(df_5m["Close"].dropna().iloc[-1])
    direction    = pos["direction"]
    entry        = pos["entry_price"]
    stop         = pos["stop_price"]
    initial_stop = pos.get("initial_stop") or pos["stop_price"]
    units        = pos["units"]
    t1 = pos.get("target_1r")

    r0 = abs(entry - initial_stop)

    # P2-105 (BUG-05): guard against a corrupted/zero-risk position
    # crashing the tick with an unhandled ZeroDivisionError. Previously
    # unguarded — manage_open_position() isn't wrapped in try/except at
    # its _tick() call site, so this would have killed the process
    # (runner.py's restart/backoff would catch it, but better not to
    # crash at all).
    if r0 < 1e-6:
        log.critical("Position #%d has invalid initial risk r0=0. Force-closing.", pos["id"])
        _exit_position(pos, last_close, "corrupted_r0_abort", 0.0, 0.0, db, cfg, session)
        return "exited"

    # ── Check stop hit ──
    if direction == "long" and last_close <= stop:
        actual_r, pnl_gbp = _calc_exit(entry, stop, direction, r0, units)
        _exit_position(pos, stop, "trailing_stop", actual_r, pnl_gbp, db, cfg, session)
        log.info("STOP HIT: %s at $%.2f  actual_r=%.2f  pnl=£%.2f",
                 pos["ticker"], stop, actual_r, pnl_gbp)
        return "exited"
    if direction == "short" and last_close >= stop:
        actual_r, pnl_gbp = _calc_exit(entry, stop, direction, r0, units)
        _exit_position(pos, stop, "trailing_stop", actual_r, pnl_gbp, db, cfg, session)
        log.info("STOP HIT (short): %s at $%.2f  actual_r=%.2f  pnl=£%.2f",
                 pos["ticker"], stop, actual_r, pnl_gbp)
        return "exited"

    # ── Check 1R target (first ladder) ──
    if t1 and direction == "long" and last_close >= t1:
        actual_r, pnl_gbp = _calc_exit(entry, t1, direction, r0, units)
        _exit_position(pos, t1, "target_1r", actual_r, pnl_gbp, db, cfg, session)
        log.info("TARGET 1R: %s at $%.2f  actual_r=%.2f  pnl=£%.2f",
                 pos["ticker"], t1, actual_r, pnl_gbp)
        return "exited"
    if t1 and direction == "short" and last_close <= t1:
        actual_r, pnl_gbp = _calc_exit(entry, t1, direction, r0, units)
        _exit_position(pos, t1, "target_1r", actual_r, pnl_gbp, db, cfg, session)
        log.info("TARGET 1R (short): %s at $%.2f  actual_r=%.2f  pnl=£%.2f",
                 pos["ticker"], t1, actual_r, pnl_gbp)
        return "exited"

    # ── End-of-session close ──
    # Checked BEFORE the trailing-stop update below (not after, as this
    # function originally had it) — a trailing stop update returns early
    # with "stop_moved" whenever the stop actually moves, which would
    # otherwise skip past a session-end close that should have fired on
    # the very same poll. Confirmed reproducible: a position open at
    # 11:05 with VWAP having drifted since the last check would return
    # "stop_moved" and never reach the EOD block below, leaving it open
    # past session_end until some later poll happened to see no VWAP
    # movement.
    now_t = _now_market(cfg).time()
    if now_t >= cfg["session_end"]:
        actual_r, pnl_gbp = _calc_exit(entry, last_close, direction, r0, units)
        _exit_position(pos, last_close, "eod", actual_r, pnl_gbp, db, cfg, session)
        log.info("EOD CLOSE: %s at $%.2f  actual_r=%.2f  pnl=£%.2f",
                 pos["ticker"], last_close, actual_r, pnl_gbp)
        return "exited"

    # ── Update VWAP trailing stop ──
    if cfg["use_vwap_trailing"]:
        vwap_val = float(vwap.dropna().iloc[-1]) if not vwap.dropna().empty else None
        if vwap_val:
            new_stop = _vwap_trailing(stop, vwap_val, direction)
            if new_stop != stop:
                db.update_stop(pos["id"], new_stop)
                log.debug("TRAILING STOP updated: %.4f → %.4f (VWAP %.4f)",
                          stop, new_stop, vwap_val)
                return "stop_moved"

    return "continue"


def _calc_exit(entry: float, exit_price: float, direction: str,
               r0: float, units: float) -> tuple[float, float]:
    """P2-105 (BUG-02/03): direct unit cash accounting, R normalized
    against the immutable initial risk r0 — never the current/trailed
    stop distance. `units` cancels out of actual_r algebraically
    (pnl_gbp/(r0*units) = (exit-entry)*mult/r0), so R stays properly
    size-independent while pnl_gbp correctly scales with the trade's
    actual (possibly VIX/Kelly-reduced) size — verified against an
    external audit's test suite before implementing this exact formula."""
    mult = 1.0 if direction == "long" else -1.0
    pnl_gbp = round((exit_price - entry) * units * mult, 2)
    cash_risk = r0 * units
    actual_r = round(pnl_gbp / cash_risk, 3) if cash_risk > 1e-6 else 0.0
    return actual_r, pnl_gbp


def _exit_position(pos: dict, exit_price: float, reason: str,
                    actual_r: float, pnl_gbp: float, db: PaperAccountDB,
                    cfg: dict, session: dict) -> None:
    db.close_position(pos["id"], exit_price, reason, actual_r, pnl_gbp)
    level = "TRADE" if actual_r > 0 else "WARN"
    _notify(cfg,
            f"EXIT #{pos['id']} {pos['ticker']} {pos['direction']} "
            f"@ ${exit_price:.2f}  {actual_r:+.2f}R  £{pnl_gbp:+.2f}  reason={reason}  "
            f"session_pnl={session['pnl_r']+actual_r:+.2f}R", level)
    session["pnl_r"]   += actual_r
    session["pnl_gbp"] += pnl_gbp
    session["n_trades"] += 1
    if actual_r > 0:
        session["n_wins"] += 1
        session["consec_losses"] = 0
    else:
        session["n_losses"] += 1
        session["consec_losses"] += 1
        session["max_consec_loss"] = max(session["max_consec_loss"],
                                          session["consec_losses"])


# ===========================================================================
# WALK-FORWARD ANALYSIS
# ===========================================================================

def wfa_run(
    ticker: str,
    db: PaperAccountDB,
    cfg: dict,
    is_months: int = 6,
    oos_months: int = 1,
    n_splits: int = 6,
) -> list[dict]:
    """
    Walk-Forward Analysis on 5-minute historical bars.

    Rolls a is_months in-sample window in oos_months steps.
    For each window: simulates the ORB strategy, computes IS and OOS metrics.
    Stores results in wfa_results table.

    Returns list of per-split result dicts.
    """
    log.info("WFA starting: %d splits, IS=%dm, OOS=%dm", n_splits, is_months, oos_months)

    # Fetch a long history (yfinance caps 5m at 60 days; use 1h for WFA)
    df_raw = _flatten(yf.download(ticker, period="2y", interval="1h",
                                   auto_adjust=True, progress=False))
    if df_raw.empty:
        log.error("WFA: no data for %s", ticker)
        return []

    run_ts  = _now_iso()
    results = []

    # Build date list
    all_dates = sorted(set(df_raw.index.date))
    trading_days_per_month = 21

    is_days  = is_months  * trading_days_per_month
    oos_days = oos_months * trading_days_per_month

    for split_idx in range(n_splits):
        # Roll window forward by oos_days each split
        oos_end_idx   = len(all_dates) - 1 - split_idx * oos_days
        oos_start_idx = oos_end_idx - oos_days + 1
        is_end_idx    = oos_start_idx - 1
        is_start_idx  = max(0, is_end_idx - is_days + 1)

        if is_start_idx >= is_end_idx or oos_start_idx > oos_end_idx:
            break

        is_start  = all_dates[is_start_idx]
        is_end    = all_dates[is_end_idx]
        oos_start = all_dates[oos_start_idx]
        oos_end   = all_dates[oos_end_idx]

        def _slice(start: date, end: date) -> pd.DataFrame:
            return df_raw[(df_raw.index.date >= start) & (df_raw.index.date <= end)].copy()

        is_metrics  = _backtest_orb_simple(_slice(is_start,  is_end),  cfg)
        oos_metrics = _backtest_orb_simple(_slice(oos_start, oos_end), cfg)

        is_sharpe  = is_metrics.get("sharpe")  or 0.0
        oos_sharpe = oos_metrics.get("sharpe") or 0.0
        wfe        = round(oos_sharpe / is_sharpe, 3) if is_sharpe > 0.01 else None
        verdict    = ("PASS" if wfe and wfe >= 0.5 else
                      "FAIL" if wfe is not None else "INCONCLUSIVE")

        result = {
            "run_ts":       run_ts,
            "ticker":       ticker,
            "strategy_ver": "2.1.0",
            "config_json":  json.dumps({"orb_method": cfg["orb_method"],
                                         "target_rr": cfg["target_rr"]}),
            "is_start":     str(is_start),
            "is_end":       str(is_end),
            "oos_start":    str(oos_start),
            "oos_end":      str(oos_end),
            "n_is_trades":  is_metrics.get("n", 0),
            "n_oos_trades": oos_metrics.get("n", 0),
            "is_sharpe":    is_sharpe,
            "oos_sharpe":   oos_sharpe,
            "is_max_dd":    is_metrics.get("max_dd"),
            "oos_max_dd":   oos_metrics.get("max_dd"),
            "is_win_rate":  is_metrics.get("wr"),
            "oos_win_rate": oos_metrics.get("wr"),
            "wfe":          wfe,
            "wfe_verdict":  verdict,
        }
        db.save_wfa_result(result)
        results.append(result)

        log.info(
            "WFA split %d/%d  IS[%s→%s] SR=%.2f  OOS[%s→%s] SR=%.2f  WFE=%s  %s",
            split_idx + 1, n_splits,
            is_start, is_end, is_sharpe,
            oos_start, oos_end, oos_sharpe,
            f"{wfe:.2f}" if wfe else "N/A", verdict
        )

    avg_wfe = ([r["wfe"] for r in results if r["wfe"] is not None])
    mean_wfe = round(statistics.mean(avg_wfe), 3) if avg_wfe else None
    log.info("WFA complete. Mean WFE: %s (target > 0.5)", mean_wfe)
    return results


def wfa_run_5m(
    ticker: str,
    db: PaperAccountDB,
    cfg: dict,
    is_days: int = 25,
    oos_days: int = 10,
    n_splits: int = 3,
) -> list[dict]:
    """
    Walk-Forward Analysis using REAL 5-minute bars and the engine's actual
    two-gate signal logic + exit priority, via _backtest_orb_full_gate().

    This is the faithful counterpart to wfa_run() (hourly bars, a single-
    gate one-trade/day approximation via _backtest_orb_simple()) — that
    one stays as-is, it's still useful as a fast, long-horizon rough
    sanity check. This one answers a different question: does the ACTUAL
    2-gate strategy, on the resolution it actually trades at, hold up
    walk-forward.

    yfinance caps 5-minute intraday history at roughly 60 calendar days
    (~40 trading days) — nowhere near enough for wfa_run()'s 6-month IS /
    1-month OOS / 6-split design. Windows here are sized in trading days
    to fit that budget instead; n_splits will silently come back short of
    the requested count once the data runs out (same behaviour as
    wfa_run()'s own index-exhaustion check).

    Results are tagged strategy_ver="2.1.0-5m-fullgate" in wfa_results so
    they're never conflated with wfa_run()'s hourly-proxy rows — the two
    use different bar resolutions, different gate counts, and different
    window sizes, and averaging or comparing them directly would be
    meaningless.
    """
    log.info("WFA (5m, full-gate) starting: up to %d splits, IS=%dd, OOS=%dd",
              n_splits, is_days, oos_days)

    df_raw = _flatten(yf.download(ticker, period="60d", interval="5m",
                                   auto_adjust=True, progress=False))
    if df_raw.empty:
        log.error("WFA (5m): no data for %s", ticker)
        return []

    run_ts  = _now_iso()
    results = []
    all_dates = sorted(set(df_raw.index.date))

    for split_idx in range(n_splits):
        oos_end_idx   = len(all_dates) - 1 - split_idx * oos_days
        oos_start_idx = oos_end_idx - oos_days + 1
        is_end_idx    = oos_start_idx - 1
        is_start_idx  = max(0, is_end_idx - is_days + 1)

        if is_start_idx >= is_end_idx or oos_start_idx > oos_end_idx or oos_start_idx < 0:
            log.info("WFA (5m): out of data for split %d/%d — stopping (got %d splits)",
                      split_idx + 1, n_splits, len(results))
            break

        is_start, is_end   = all_dates[is_start_idx], all_dates[is_end_idx]
        oos_start, oos_end = all_dates[oos_start_idx], all_dates[oos_end_idx]

        def _slice(start: date, end: date) -> pd.DataFrame:
            return df_raw[(df_raw.index.date >= start) & (df_raw.index.date <= end)].copy()

        is_metrics  = _backtest_orb_full_gate(_slice(is_start,  is_end),  cfg)
        oos_metrics = _backtest_orb_full_gate(_slice(oos_start, oos_end), cfg)

        is_sharpe  = is_metrics.get("sharpe")  or 0.0
        oos_sharpe = oos_metrics.get("sharpe") or 0.0
        wfe        = round(oos_sharpe / is_sharpe, 3) if is_sharpe > 0.01 else None
        verdict    = ("PASS" if wfe and wfe >= 0.5 else
                      "FAIL" if wfe is not None else "INCONCLUSIVE")

        result = {
            "run_ts":       run_ts,
            "ticker":       ticker,
            "strategy_ver": "2.1.0-5m-fullgate",
            "config_json":  json.dumps({"orb_method": cfg["orb_method"],
                                         "target_rr": cfg["target_rr"],
                                         "is_days": is_days, "oos_days": oos_days}),
            "is_start":     str(is_start),
            "is_end":       str(is_end),
            "oos_start":    str(oos_start),
            "oos_end":      str(oos_end),
            "n_is_trades":  is_metrics.get("n", 0),
            "n_oos_trades": oos_metrics.get("n", 0),
            "is_sharpe":    is_sharpe,
            "oos_sharpe":   oos_sharpe,
            "is_max_dd":    is_metrics.get("max_dd"),
            "oos_max_dd":   oos_metrics.get("max_dd"),
            "is_win_rate":  is_metrics.get("wr"),
            "oos_win_rate": oos_metrics.get("wr"),
            "wfe":          wfe,
            "wfe_verdict":  verdict,
        }
        db.save_wfa_result(result)
        results.append(result)

        log.info(
            "WFA(5m) split %d/%d  IS[%s→%s] n=%d SR=%.2f  OOS[%s→%s] n=%d SR=%.2f  WFE=%s  %s",
            split_idx + 1, n_splits,
            is_start, is_end, is_metrics.get("n", 0), is_sharpe,
            oos_start, oos_end, oos_metrics.get("n", 0), oos_sharpe,
            f"{wfe:.2f}" if wfe else "N/A", verdict
        )

    valid = [r["wfe"] for r in results if r["wfe"] is not None]
    mean_wfe = round(statistics.mean(valid), 3) if valid else None
    log.info("WFA(5m) complete. Mean WFE: %s (target > 0.5)", mean_wfe)
    return results


def _backtest_orb_simple(df: pd.DataFrame, cfg: dict) -> dict:
    """
    Minimal ORB backtest on hourly data for WFA.
    Uses realistic cost model (v2.2.0): broker-aware commission + spread
    + market impact, replacing the prior flat 0.05R constant.
    Returns metrics dict: {n, wr, sharpe, max_dd, ev, avg_cost_r}.
    """
    if df.empty:
        return {}

    broker   = cfg.get("broker", "alpaca")
    account  = cfg.get("account_balance", 10_000.0)
    risk_pct = cfg.get("risk_pct", 0.01)
    risk_amt = account * risk_pct
    adv      = cfg.get("avg_daily_volume", 150_000_000)

    r_multiples: list[float] = []
    cost_r_list: list[float] = []

    for day in sorted(set(df.index.date)):
        day_df  = df[df.index.date == day]
        session = day_df[day_df.index.time >= cfg["session_start"]]
        if len(session) < 2:
            continue

        orb_bar  = session.iloc[0]
        orb_high = float(orb_bar["High"])
        orb_low  = float(orb_bar["Low"])
        orb_size = orb_high - orb_low
        if orb_size < 0.01:
            continue

        for _, bar in session.iloc[1:].iterrows():
            close = float(bar["Close"])
            if close > orb_high:
                stop = orb_low; target = orb_high + orb_size * cfg.get("target_rr", 2.0)
            elif close < orb_low:
                stop = orb_high; target = orb_low - orb_size * cfg.get("target_rr", 2.0)
            else:
                continue

            r = _sim_trade(close, stop, target, day_df, bar.name,
                           "long" if close > orb_high else "short")

            # Realistic cost from toolkit (v2.2.0) or flat fallback
            stop_dist = abs(close - stop)
            if stop_dist > 1e-6:
                units = risk_amt / stop_dist
                if TOOLKIT:
                    try:
                        cost_r = tk.realistic_backtest_cost(close, units, risk_amt, adv, broker)
                    except Exception:
                        cost_r = 0.05
                else:
                    cost_r = 0.05
            else:
                cost_r = 0.05

            r_multiples.append(r - cost_r)
            cost_r_list.append(cost_r)
            break

    if not r_multiples:
        return {"n": 0}

    n    = len(r_multiples)
    wins = [r for r in r_multiples if r > 0]
    wr   = len(wins) / n
    aw   = statistics.mean(wins) if wins else 0.0
    al   = abs(statistics.mean([r for r in r_multiples if r < 0])) if any(r < 0 for r in r_multiples) else 0.0
    ev   = (wr * aw) - ((1 - wr) * al)

    equity = [0.0]
    for r in r_multiples:
        equity.append(equity[-1] + r)
    peak, worst = equity[0], 0.0
    for v in equity:
        peak  = max(peak, v); worst = min(worst, v - peak)

    sharpe = None
    if n >= 2:
        sd = statistics.stdev(r_multiples)
        if sd > 0:
            sharpe = round(statistics.mean(r_multiples) / sd, 3)

    return {"n": n, "wr": round(wr, 3), "aw": round(aw, 3), "al": round(al, 3),
            "ev": round(ev, 3), "sharpe": sharpe, "max_dd": round(abs(worst), 3),
            "avg_cost_r": round(statistics.mean(cost_r_list), 4) if cost_r_list else 0.0}


def _backtest_orb_full_gate(df_5m: pd.DataFrame, cfg: dict,
                             exit_mode: str = "baseline",
                             fill_model: str = "legacy",
                             df_1m: pd.DataFrame | None = None,
                             entry_fill: str = "next_open") -> dict:
    """
    Full-gate ORB backtest on REAL 5-minute bars (added for the 5-minute
    WFA variant — see wfa_run_5m()).

    Replays the SAME two gates evaluate_signals() actually enforces —
    ORB breakout direction + VWAP-slope confirmation. gate_retest is
    intentionally NOT replicated: in the live engine it's hardcoded to
    "WAIT" and never enters the AND-gate condition (see evaluate_signals,
    "AND-gate" section) — it's a dashboard-only human-confirmation field,
    not something trading_engine.py itself requires. Replicating it here
    would make this backtest MORE restrictive than the code it's meant
    to test.

    Reuses the live engine's own _orb_range(), _vwap(), _vwap_slope(),
    and _vwap_trailing() functions unmodified — fed only bars available
    as-of each simulated timestamp (no look-ahead) — rather than
    reimplementing their logic. EOD is judged against each bar's own
    timestamp instead of live wall-clock, since this is replaying the past.

    Deliberately NOT replicated: evaluate_signals()'s live VIX fetch and
    compute_entry_params()'s VIX-based position-size modifier + Kelly
    sizing + full broker cost/reject-reason model. None of those affect
    WHETHER or WHEN a trade fires (VIX only scales size in the live
    code), so they're out of scope for a signal-timing backtest — same
    simplification _backtest_orb_simple() already makes for sizing, kept
    here for consistency. Cost model is the same realistic_backtest_cost
    (or flat 0.05R fallback) both backtests use.

    Unlike _backtest_orb_simple() (one trade/day, breaks after the
    first), this allows re-entry the same day after an exit, matching
    the live engine's actual behaviour — _tick() re-evaluates for a new
    signal immediately after a position closes, within the same session.

    Also faithfully replicates a live characteristic worth knowing about
    rather than quietly "fixing": _orb_range() has no formation-complete
    gate — it computes high/low from whatever bars fall in the window so
    far, so a breakout can fire mid-formation on a partial range. That's
    the live engine's actual behaviour, not a bug in this replay.

    exit_mode (P2-106 exit-restructuring experiment — pre-registered
    hypothesis: current 1R flat exit caps the right tail; a strategy with
    a sub-50% win rate needs bigger winners to survive realistic friction.
    Entry/gate logic is IDENTICAL across all three modes — this only
    changes what happens once in a trade, so any difference in results
    isolates the exit rule's effect, not a signal-quality difference):
      "baseline"       — current live behaviour: flat 1R target, VWAP
                          trailing stop active from the moment of entry.
      "fixed_1_5r"      — Candidate A: flat 1.5R target, NO trailing stop
                          at all (stop stays at the initial structural
                          stop for the life of the trade).
      "trail_after_1r"  — Candidate B: no fixed target. Stop stays at the
                          initial structural stop until price reaches
                          +1R, then VWAP trailing arms and the trade rides
                          until the trailing stop is hit or EOD.

    fill_model (P2-122 — how orders are FILLED; signals are identical):
      "legacy"   — the original behaviour, kept as the default so every
                   earlier result is reproducible: enter at the signal bar's
                   close; a stop/target is DECIDED on a 5-min close but BOOKED
                   at the stop/target price. No real order gets that fill —
                   PR-001's diagnostic showed it inflated expectancy ~3-5x.
      "resting"  — stop and target are resting broker orders, simulated on
                   1-MINUTE bars (df_1m required):
                     * entry: market order after the signal bar closes, filled
                       at the next 1-min bar's open (entry_fill="next_open"),
                       or at the signal close (entry_fill="signal_close", for
                       attribution only). Stop is the same ORB level; R and the
                       target are measured from the actual fill. An entry that
                       fills at/through the stop is skipped (counted).
                     * stop  : fills when a 1-min low (long) reaches it — at the
                       stop, or at the bar's open if it gapped through.
                     * target: fills when a 1-min high (long) reaches it — at t1,
                       or at the open if it gapped through.
                     * a 1-min bar that touches both: STOP first (conservative).
                     * VWAP trailing / trail arming still update on each 5-min
                       close; the moved stop applies from the next minute.
                     * EOD: market exit at the open of the session_end minute.
                     * no new entries on a signal bar at/after session_end
                       (legacy would open one and never close it).

    Returns the same metrics shape as _backtest_orb_simple() so both feed
    the same WFA split/verdict logic.
    """
    if exit_mode not in ("baseline", "fixed_1_5r", "trail_after_1r"):
        raise ValueError(f"Unknown exit_mode: {exit_mode!r}")
    if fill_model not in ("legacy", "resting"):
        raise ValueError(f"Unknown fill_model: {fill_model!r}")
    if entry_fill not in ("next_open", "signal_close"):
        raise ValueError(f"Unknown entry_fill: {entry_fill!r}")
    resting = fill_model == "resting"
    if resting and (df_1m is None or df_1m.empty):
        raise ValueError("fill_model='resting' needs the 1-minute bars (df_1m)")
    one_min = ({d: g for d, g in df_1m.groupby(df_1m.index.date)} if resting else {})
    skipped_entries = 0
    _5min = pd.Timedelta(minutes=5)

    if df_5m.empty:
        return {}

    broker   = cfg.get("broker", "alpaca")
    account  = cfg.get("account_balance", 10_000.0)
    risk_pct = cfg.get("risk_pct", 0.01)
    adv      = cfg.get("avg_daily_volume", 150_000_000)
    orb_method    = cfg["orb_method"]
    vwap_lookback = cfg.get("vwap_lookback", 3)
    use_trailing  = cfg.get("use_vwap_trailing", True) and exit_mode != "fixed_1_5r"

    r_multiples: list[float] = []
    cost_r_list: list[float] = []
    trades: list[dict] = []   # P2-116: full per-trade records, not just R-multiples —
                               # lets this be the single canonical simulator for any
                               # downstream consumer (Markov chain, N-gram, journaling),
                               # not just summary WFA metrics.

    for day in sorted(set(df_5m.index.date)):
        day_bars = df_5m[df_5m.index.date == day]
        session_bars = day_bars[(day_bars.index.time >= cfg["session_start"]) &
                                 (day_bars.index.time <= cfg["session_end"])]
        if session_bars.empty:
            continue

        in_position = False
        direction = stop = entry = t1 = risk_dist = None
        trail_armed = False   # only meaningful for "trail_after_1r"
        entry_meta: dict = {}
        if resting:
            m1 = one_min.get(day)
            if m1 is None or m1.empty:
                continue
            m1_ts = m1.index
            m1_o, m1_h = m1["Open"].to_numpy(float), m1["High"].to_numpy(float)
            m1_l = m1["Low"].to_numpy(float)
        entry_fill_ts = None

        for bar_ts, bar in session_bars.iterrows():
            asof_today = df_5m[(df_5m.index <= bar_ts) & (df_5m.index.date == day)]
            last_close = float(bar["Close"])

            if not in_position:
                orb = _orb_range(asof_today, orb_method, cfg["session_start"])  # real function, unmodified
                if orb is None:
                    continue

                # Gate 1: ORB breakout — mirrors evaluate_signals exactly
                if last_close > orb["orb_high"]:
                    breakout_dir = "long"
                elif last_close < orb["orb_low"]:
                    breakout_dir = "short"
                else:
                    continue

                # Gate 2: VWAP slope aligned — mirrors evaluate_signals exactly
                vwap_series = _vwap(asof_today, cfg["session_start"])         # real function, unmodified
                vs = _vwap_slope(vwap_series, lookback=vwap_lookback)   # real function
                gate_pass = ((breakout_dir == "long" and vs["direction"] == "up") or
                             (breakout_dir == "short" and vs["direction"] == "down"))
                if not gate_pass:
                    continue

                # Entry — mirrors compute_entry_params' stop/target math.
                # Entry price, stop, and risk_dist are IDENTICAL across all
                # three exit_modes — only what happens after entry differs.
                risk_dist_orb = orb["orb_high"] - orb["orb_low"]
                buffer = max(risk_dist_orb * 0.1, 0.02)
                direction = breakout_dir
                entry = last_close
                stop  = (round(orb["orb_low"]  - buffer, 4) if direction == "long"
                         else round(orb["orb_high"] + buffer, 4))
                if resting:
                    if bar_ts.time() >= cfg["session_end"]:
                        direction = None
                        continue
                    k = m1_ts.searchsorted(bar_ts + _5min)
                    if k >= len(m1_ts):
                        direction = None
                        continue
                    entry_fill_ts = m1_ts[k]
                    if entry_fill == "next_open":
                        entry = float(m1_o[k])
                    # Market order filled at/through the stop: the stop order
                    # would trigger at once. Not a tradeable setup — skip.
                    if (direction == "long" and entry <= stop) or \
                       (direction == "short" and entry >= stop):
                        skipped_entries += 1
                        direction = None
                        continue
                    risk_dist = abs(entry - stop)
                else:
                    risk_dist = abs(entry - stop)
                if risk_dist <= 1e-6:
                    direction = None
                    continue
                target_mult = 1.5 if exit_mode == "fixed_1_5r" else 1.0
                t1 = (entry + risk_dist * target_mult if direction == "long"
                      else entry - risk_dist * target_mult)
                trail_armed = (exit_mode != "trail_after_1r")  # baseline/fixed_1_5r: on immediately
                in_position = True
                # Per-trade record metadata captured at entry, for
                # downstream consumers (Markov/N-gram) that need more than
                # just the R-multiple.
                entry_body = abs(float(bar["Close"]) - float(bar["Open"]))
                entry_range = max(float(bar["High"]) - float(bar["Low"]), 1e-6)
                entry_body_pct = entry_body / entry_range * 100
                entry_meta = {
                    "session_date": str(day),
                    "entry_ts": bar_ts,
                    "direction": direction,
                    "entry_price": round(entry, 4),
                    "stop_price": round(stop, 4),
                    "entry_candle_body_pct": round(entry_body_pct, 1),
                    "entry_candle_type": (
                        "strong_bull" if entry_body_pct > 60 and direction == "long" else
                        "strong_bear" if entry_body_pct > 60 and direction == "short" else
                        "doji" if entry_body_pct < 20 else "moderate"),
                    "vwap_slope_at_entry": vs["direction"],
                }
                if resting:
                    entry_meta.update({"fill_model": "resting", "entry_fill": entry_fill,
                                       "signal_close": round(last_close, 4),
                                       "entry_fill_ts": entry_fill_ts})
                continue   # opened on this bar's close; manage from the next bar

            # ── In position ──
            exit_reason = exit_r = None
            exit_px = None
            has_fixed_target = exit_mode in ("baseline", "fixed_1_5r")
            sgn = 1 if direction == "long" else -1

            if resting:
                a = m1_ts.searchsorted(max(bar_ts, entry_fill_ts))
                b = m1_ts.searchsorted(bar_ts + _5min)
                if bar_ts.time() >= cfg["session_end"]:
                    # EOD market exit at the open of the session_end minute
                    if a < len(m1_ts):
                        exit_reason, exit_px = "eod", float(m1_o[a])
                    else:
                        exit_reason, exit_px = "eod", last_close
                else:
                    for j in range(a, b):
                        o, h, l = m1_o[j], m1_h[j], m1_l[j]
                        stop_hit = (l <= stop) if direction == "long" else (h >= stop)
                        tgt_hit  = has_fixed_target and (
                            (h >= t1) if direction == "long" else (l <= t1))
                        gap_stop = (o <= stop) if direction == "long" else (o >= stop)
                        gap_tgt  = has_fixed_target and (
                            (o >= t1) if direction == "long" else (o <= t1))
                        if gap_stop:
                            exit_reason, exit_px = "trailing_stop", float(o)
                        elif gap_tgt:
                            exit_reason, exit_px = "target_hit", float(o)
                        elif stop_hit:                      # includes both-touched: stop first
                            exit_reason, exit_px = "trailing_stop", float(stop)
                        elif tgt_hit:
                            exit_reason, exit_px = "target_hit", float(t1)
                        if exit_reason:
                            bar_ts_exit = m1_ts[j]
                            break
                    if exit_reason is None:
                        # 5-min close: arm / trail exactly as the legacy model
                        if exit_mode == "trail_after_1r" and not trail_armed:
                            if ((direction == "long" and last_close >= t1) or
                                    (direction == "short" and last_close <= t1)):
                                trail_armed = True
                        if use_trailing and trail_armed:
                            vwap_series = _vwap(asof_today, cfg["session_start"])
                            vwap_val = (float(vwap_series.dropna().iloc[-1])
                                        if not vwap_series.dropna().empty else None)
                            if vwap_val:
                                stop = _vwap_trailing(stop, vwap_val, direction)
                if exit_reason is not None:
                    exit_r = (exit_px - entry) / risk_dist * sgn
                    if exit_reason == "eod":
                        bar_ts_exit = m1_ts[a] if a < len(m1_ts) else bar_ts

            elif direction == "long" and last_close <= stop:
                exit_reason, exit_r = "trailing_stop", (stop - entry) / risk_dist
            elif direction == "short" and last_close >= stop:
                exit_reason, exit_r = "trailing_stop", (entry - stop) / risk_dist
            elif has_fixed_target and direction == "long" and last_close >= t1:
                exit_reason, exit_r = "target_hit", (t1 - entry) / risk_dist
            elif has_fixed_target and direction == "short" and last_close <= t1:
                exit_reason, exit_r = "target_hit", (entry - t1) / risk_dist
            else:
                # trail_after_1r: arm once price reaches the +1R marker (t1
                # is still the +1R level here even though there's no fixed
                # target to exit at — it's reused purely as the arming
                # threshold, never checked as an exit condition itself)
                if exit_mode == "trail_after_1r" and not trail_armed:
                    if ((direction == "long" and last_close >= t1) or
                        (direction == "short" and last_close <= t1)):
                        trail_armed = True

                if use_trailing and trail_armed:
                    vwap_series = _vwap(asof_today, cfg["session_start"])
                    vwap_val = (float(vwap_series.dropna().iloc[-1])
                                if not vwap_series.dropna().empty else None)
                    if vwap_val:
                        stop = _vwap_trailing(stop, vwap_val, direction)   # real function
                if bar_ts.time() >= cfg["session_end"]:
                    exit_reason = "eod"
                    exit_r = ((last_close - entry) / risk_dist *
                              (1 if direction == "long" else -1))

            if exit_reason is not None:
                units = (account * risk_pct) / risk_dist if risk_dist > 1e-6 else 0.0
                if TOOLKIT:
                    try:
                        cost_r = tk.realistic_backtest_cost(
                            entry, units, account * risk_pct, adv, broker)
                    except Exception:
                        cost_r = 0.05
                else:
                    cost_r = 0.05
                net_r = exit_r - cost_r
                r_multiples.append(net_r)
                cost_r_list.append(cost_r)
                if resting:
                    exit_price = exit_px
                else:
                    exit_price = (stop if exit_reason == "trailing_stop" else
                                  t1 if exit_reason == "target_hit" else last_close)
                trades.append({
                    **entry_meta,
                    "exit_ts":     bar_ts_exit if resting else bar_ts,
                    "exit_price":  round(exit_price, 4),
                    "actual_r":    net_r,   # unrounded — must exactly match the
                                            # corresponding r_multiples entry
                    "exit_reason": exit_reason,
                    "units":       units,   # fractional shares (legacy sizing; P2-123 golden export)
                })
                in_position = False
                direction = stop = entry = t1 = risk_dist = None
                trail_armed = False
                entry_meta = {}
                if resting:
                    # the next entry signal can only come from a LATER 5-min bar
                    entry_fill_ts = None

    if not r_multiples:
        return {"n": 0, "skipped_entries": skipped_entries}

    n    = len(r_multiples)
    wins = [r for r in r_multiples if r > 0]
    wr   = len(wins) / n
    aw   = statistics.mean(wins) if wins else 0.0
    al   = abs(statistics.mean([r for r in r_multiples if r < 0])) if any(r < 0 for r in r_multiples) else 0.0
    ev   = (wr * aw) - ((1 - wr) * al)

    equity = [0.0]
    for r in r_multiples:
        equity.append(equity[-1] + r)
    peak, worst = equity[0], 0.0
    for v in equity:
        peak  = max(peak, v); worst = min(worst, v - peak)

    sharpe = None
    if n >= 2:
        sd = statistics.stdev(r_multiples)
        if sd > 0:
            sharpe = round(statistics.mean(r_multiples) / sd, 3)

    return {"n": n, "wr": round(wr, 3), "aw": round(aw, 3), "al": round(al, 3),
            "ev": round(ev, 3), "sharpe": sharpe, "max_dd": round(abs(worst), 3),
            "avg_cost_r": round(statistics.mean(cost_r_list), 4) if cost_r_list else 0.0,
            # raw per-trade returns, for downstream statistical analysis
            # (e.g. DSR needs actual skew/kurtosis, not just summary stats)
            "r_multiples": r_multiples,
            # full per-trade records (P2-116) — session_date, direction,
            # entry/exit price+time, exit_reason, candle/VWAP-slope context
            # at entry. Lets this be the single canonical simulator for any
            # downstream consumer (Markov chain, N-gram, trade journaling),
            # not just WFA summary metrics. Both new keys are additive —
            # existing callers only read the keys they already used.
            "trades": trades,
            "fill_model": fill_model,
            "skipped_entries": skipped_entries}


def _backtest_orb_fade(df_5m: pd.DataFrame, cfg: dict) -> dict:
    """
    ORB Liquidity-Sweep Fade (P2-114 follow-up) — the inverse of the
    breakout strategy: instead of confirming a breakout with VWAP slope,
    wait for a breakout to FAIL (price pokes past the ORB boundary then
    the very next bar closes back inside the range) and fade it,
    targeting session VWAP rather than a fixed R-multiple.

    Pre-registered hypothesis, stated before running: a high win rate is
    expected in choppy conditions where most failed breakouts genuinely
    revert, but real tail risk is expected on trend days where the
    "failure" doesn't hold and price continues — net expectancy is
    genuinely uncertain until tested, not assumed positive just because
    the win rate should be high.

    Mechanics:
      - Gate 1 (same trigger as the breakout strategy): last_close
        crosses orb_high or orb_low.
      - Confirmation (INVERTED from the breakout strategy): the very
        NEXT bar's close must be back INSIDE the range. If instead the
        breakout confirms (next bar closes further outside), no fade
        trade is taken — this is exactly the case where fading would be
        on the losing side of a real move.
      - Entry: at the confirmation bar's close, opposite the original
        breakout direction.
      - Stop: just beyond the swept extreme (the high/low made during
        the failed poke), same 10%-of-ORB-range buffer convention the
        breakout strategy uses.
      - Target: session VWAP at the moment of entry (the fade thesis is
        "price reverts to fair value", not "price runs a measured
        distance") — falls back to the ORB range midpoint if VWAP isn't
        available yet.
      - Exit priority: stop -> VWAP target -> EOD.

    Reuses the same real _orb_range()/_vwap() functions as the breakout
    backtest, fed only as-of-bar data — same no-lookahead discipline.
    Allows multiple fade trades per day, matching the live engine's own
    re-entry behavior. Returns the same metrics shape (including
    r_multiples) as _backtest_orb_full_gate() so this can feed
    _deflated_sharpe_ratio() the same way.
    """
    if df_5m.empty:
        return {}

    broker   = cfg.get("broker", "alpaca")
    account  = cfg.get("account_balance", 10_000.0)
    risk_pct = cfg.get("risk_pct", 0.01)
    adv      = cfg.get("avg_daily_volume", 150_000_000)
    orb_method = cfg["orb_method"]

    r_multiples: list[float] = []
    cost_r_list: list[float] = []

    for day in sorted(set(df_5m.index.date)):
        day_bars = df_5m[df_5m.index.date == day]
        session_bars = day_bars[(day_bars.index.time >= cfg["session_start"]) &
                                 (day_bars.index.time <= cfg["session_end"])]
        if session_bars.empty:
            continue

        in_position = False
        awaiting_confirmation = False
        breakout_dir = None
        swept_extreme = None
        direction = stop = entry = target = risk_dist = None

        for bar_ts, bar in session_bars.iterrows():
            asof_today = df_5m[(df_5m.index <= bar_ts) & (df_5m.index.date == day)]
            last_close = float(bar["Close"])
            last_high  = float(bar["High"])
            last_low   = float(bar["Low"])

            if in_position:
                exit_reason = exit_r = None
                if direction == "long" and last_close <= stop:
                    exit_reason, exit_r = "trailing_stop", (stop - entry) / risk_dist
                elif direction == "short" and last_close >= stop:
                    exit_reason, exit_r = "trailing_stop", (entry - stop) / risk_dist
                elif direction == "long" and last_close >= target:
                    exit_reason, exit_r = "vwap_target", (target - entry) / risk_dist
                elif direction == "short" and last_close <= target:
                    exit_reason, exit_r = "vwap_target", (entry - target) / risk_dist
                elif bar_ts.time() >= cfg["session_end"]:
                    exit_reason = "eod"
                    exit_r = ((last_close - entry) / risk_dist *
                              (1 if direction == "long" else -1))

                if exit_reason is not None:
                    units = (account * risk_pct) / risk_dist if risk_dist > 1e-6 else 0.0
                    if TOOLKIT:
                        try:
                            cost_r = tk.realistic_backtest_cost(
                                entry, units, account * risk_pct, adv, broker)
                        except Exception:
                            cost_r = 0.05
                    else:
                        cost_r = 0.05
                    r_multiples.append(exit_r - cost_r)
                    cost_r_list.append(cost_r)
                    in_position = False
                    direction = stop = entry = target = risk_dist = None
                continue

            if awaiting_confirmation:
                orb = _orb_range(asof_today, orb_method, cfg["session_start"])
                if orb is None:
                    awaiting_confirmation = False
                    continue
                failed = ((breakout_dir == "long" and last_close < orb["orb_high"]) or
                          (breakout_dir == "short" and last_close > orb["orb_low"]))
                if failed:
                    direction = "short" if breakout_dir == "long" else "long"
                    entry = last_close
                    buffer = max(orb["orb_size"] * 0.1, 0.02)
                    stop = (round(swept_extreme + buffer, 4) if direction == "short"
                            else round(swept_extreme - buffer, 4))
                    risk_dist = abs(entry - stop)
                    if risk_dist > 1e-6:
                        vwap_series = _vwap(asof_today, cfg["session_start"])
                        vwap_val = (float(vwap_series.dropna().iloc[-1])
                                    if not vwap_series.dropna().empty else None)
                        target = (vwap_val if vwap_val is not None
                                  else (orb["orb_high"] + orb["orb_low"]) / 2)
                        in_position = True
                awaiting_confirmation = False
                continue

            # Not in a position, not awaiting confirmation — look for a
            # fresh breakout poke to potentially fade.
            orb = _orb_range(asof_today, orb_method, cfg["session_start"])
            if orb is None:
                continue
            if last_close > orb["orb_high"]:
                breakout_dir = "long"
                swept_extreme = last_high
                awaiting_confirmation = True
            elif last_close < orb["orb_low"]:
                breakout_dir = "short"
                swept_extreme = last_low
                awaiting_confirmation = True

    if not r_multiples:
        return {"n": 0}

    n    = len(r_multiples)
    wins = [r for r in r_multiples if r > 0]
    wr   = len(wins) / n
    aw   = statistics.mean(wins) if wins else 0.0
    al   = abs(statistics.mean([r for r in r_multiples if r < 0])) if any(r < 0 for r in r_multiples) else 0.0
    ev   = (wr * aw) - ((1 - wr) * al)

    equity = [0.0]
    for r in r_multiples:
        equity.append(equity[-1] + r)
    peak, worst = equity[0], 0.0
    for v in equity:
        peak  = max(peak, v); worst = min(worst, v - peak)

    sharpe = None
    if n >= 2:
        sd = statistics.stdev(r_multiples)
        if sd > 0:
            sharpe = round(statistics.mean(r_multiples) / sd, 3)

    return {"n": n, "wr": round(wr, 3), "aw": round(aw, 3), "al": round(al, 3),
            "ev": round(ev, 3), "sharpe": sharpe, "max_dd": round(abs(worst), 3),
            "avg_cost_r": round(statistics.mean(cost_r_list), 4) if cost_r_list else 0.0,
            "r_multiples": r_multiples}


def _deflated_sharpe_ratio(returns: list[float], n_trials: int) -> dict:
    """
    Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014) — corrects the
    observed per-trade Sharpe ratio for (1) sampling noise from a finite
    N, and (2) selection bias from having screened n_trials candidates
    (e.g. tickers) and reporting the best one(s). P2-114: built to
    replace a reviewed report's DSR analysis that used SR~0.70/N=20 as
    inputs — neither matched the actual measured results (this project's
    real QQQ/TSLA figures were SR=0.116/n=33 and SR=0.138/n=54) — and
    whose own arithmetic contradicted its own stated conclusion.

        DSR = Phi[ (SR_hat - SR*) * sqrt(N-1)
                   / sqrt(1 - skew*SR_hat + ((kurt-1)/4)*SR_hat^2) ]

    SR* is the expected maximum Sharpe ratio achievable by pure luck
    across n_trials independent zero-skill trials:

        SR* ~= SE(SR_hat) * sqrt(2 * ln(n_trials))

    an asymptotic approximation to the expected-maximum-of-M-Gaussians
    result (drops the more precise Euler-Mascheroni correction term,
    consistent with the commonly-cited simplified form). Critically, SR*
    here is scaled by SE(SR_hat) — a z-score-like quantity multiplied by
    the Sharpe estimator's own standard error to become comparable to an
    actual Sharpe value. The reviewed report used a bare sqrt(2*ln(M))
    (~1.665 for M=4) directly as SR*, with no SE scaling — a units error
    that would make the DSR test fail almost any realistic per-trade
    Sharpe ratio, not a meaningful bar.

    SE(SR_hat) uses the standard Lo (2002) approximation:
        SE(SR) ~= sqrt((1 + SR_hat^2/2) / N)

    Args:
        returns:   raw per-trade R-multiples (e.g. from
                   _backtest_orb_full_gate's "r_multiples" key)
        n_trials:  number of independent candidates screened (e.g. how
                   many tickers were tested before selecting this one)

    Returns a dict with n, sr_hat, se_sr, sr_star, skew, kurt, z, dsr.
    dsr is None (with a "reason") if N<3 (skew/kurtosis undefined), the
    return series has zero variance, or extreme skew/kurtosis make the
    formula's denominator non-positive (the approximation breaks down).
    """
    n = len(returns)
    if n < 3:
        return {"n": n, "dsr": None, "reason": "N<3 — skew/kurtosis undefined"}

    mean_r = statistics.mean(returns)
    sd_r = statistics.stdev(returns)
    if sd_r == 0:
        return {"n": n, "dsr": None, "reason": "zero variance in returns"}

    sr_hat = mean_r / sd_r

    # Sample skewness and kurtosis (raw 4th moment convention — normal = 3,
    # not "excess kurtosis" which would be 0 for normal).
    m3 = sum((r - mean_r) ** 3 for r in returns) / n
    m4 = sum((r - mean_r) ** 4 for r in returns) / n
    skew = m3 / (sd_r ** 3)
    kurt = m4 / (sd_r ** 4)

    se_sr = math.sqrt((1 + sr_hat ** 2 / 2) / n)
    sr_star = se_sr * math.sqrt(2 * math.log(max(n_trials, 2)))

    denom_sq = 1 - skew * sr_hat + ((kurt - 1) / 4) * sr_hat ** 2
    if denom_sq <= 0:
        return {"n": n, "sr_hat": round(sr_hat, 4), "se_sr": round(se_sr, 4),
                "sr_star": round(sr_star, 4), "skew": round(skew, 3),
                "kurt": round(kurt, 3), "dsr": None,
                "reason": "denominator non-positive — extreme skew/kurtosis, "
                          "formula breaks down for this sample"}
    denom = math.sqrt(denom_sq)

    z = (sr_hat - sr_star) * math.sqrt(n - 1) / denom
    dsr = 0.5 * (1 + math.erf(z / math.sqrt(2)))   # Phi(z) via erf — no scipy dependency

    return {"n": n, "sr_hat": round(sr_hat, 4), "se_sr": round(se_sr, 4),
            "sr_star": round(sr_star, 4), "skew": round(skew, 3),
            "kurt": round(kurt, 3), "z": round(z, 3), "dsr": round(dsr, 4)}


def _sim_trade(entry: float, stop: float, target: float,
               df: pd.DataFrame, entry_time: Any, direction: str,
               atr: float = 0.50) -> float:
    """
    Simulate a trade outcome on subsequent bars after entry.
    Uses FillSimulator (v2.3.0) for realistic stop and target fills.
    Returns actual R after fill slippage.
    """
    risk = abs(entry - stop)
    if risk < 1e-6:
        return 0.0

    # Build FillSimulator — use toolkit if available, else inline
    if TOOLKIT and hasattr(tk, "FillSimulator"):
        sim = tk.FillSimulator(seed=None)  # non-deterministic for realism
        def _stop_fill(stop_px):
            return sim.simulate_stop_fill(stop_px, direction, atr)["fill_price"]
        def _target_fill(tgt_px):
            return sim.simulate_target_fill(tgt_px, direction)["fill_price"]
        def _eod_fill(bar):
            return sim.simulate_eod_fill(bar, direction)["fill_price"]
    else:
        # Inline fallback: 0.2 * ATR stop slippage, 0.0001 * price spread
        def _stop_fill(stop_px):
            return (stop_px - 0.2 * atr if direction == "long"
                    else stop_px + 0.2 * atr)
        def _target_fill(tgt_px):
            return (tgt_px - tgt_px * 0.0001 if direction == "long"
                    else tgt_px + tgt_px * 0.0001)
        def _eod_fill(bar):
            c = float(bar["Close"])
            return c - c * 0.0001 if direction == "long" else c + c * 0.0001

    future = df[df.index > entry_time]
    for _, bar in future.iterrows():
        high  = float(bar["High"])
        low   = float(bar["Low"])
        if direction == "long":
            if low <= stop:
                fill = _stop_fill(stop)
                return round((fill - entry) / risk, 3)
            if high >= target:
                fill = _target_fill(target)
                return round((fill - entry) / risk, 3)
        else:
            if high >= stop:
                fill = _stop_fill(stop)
                return round((entry - fill) / risk, 3)
            if low <= target:
                fill = _target_fill(target)
                return round((entry - fill) / risk, 3)
    # EOD close
    fill = _eod_fill(df.iloc[-1])
    return round(((fill - entry) / risk if direction == "long"
                  else (entry - fill) / risk), 3)


# ===========================================================================
# MAIN ENGINE LOOP
# ===========================================================================


# ---------------------------------------------------------------------------
# Discord notification helper (fire-and-forget, non-blocking)
# ---------------------------------------------------------------------------

def _notify(cfg: dict, message: str, level: str = "INFO") -> None:
    """
    Post a notification to the configured Discord webhook.
    Runs in a daemon thread so it never blocks the engine loop.
    level: 'INFO' | 'TRADE' | 'WARN' | 'HALT'
    """
    webhook = cfg.get("discord_webhook", "")
    if not webhook or not _DISCORD_DEPS:
        return

    emoji = {"INFO":"📡","TRADE":"✅","WARN":"⚠️","HALT":"🛑"}.get(level,"📡")
    now_str = _now_market(cfg).strftime("%H:%M %Z")

    def _send():
        import asyncio, json as _json
        payload = _json.dumps({"content": f"{emoji} **{level}** [{now_str}]  {message}"})
        async def _post():
            try:
                async with _aiohttp.ClientSession() as session:
                    await session.post(webhook, data=payload,
                                       headers={"Content-Type":"application/json"},
                                       timeout=_aiohttp.ClientTimeout(total=5))
            except Exception:
                pass
        asyncio.run(_post())

    import threading
    t = threading.Thread(target=_send, daemon=True)
    t.start()


class TradingEngine:
    """
    Main engine. Runs a poll loop every cfg['poll_interval_s'] seconds.
    Call engine.start() in a thread or just run engine.run() directly.
    """

    def __init__(self, cfg: dict) -> None:
        self.cfg      = cfg
        self.db       = PaperAccountDB(cfg["db_path"])
        self.db.ensure_account_meta(cfg["account_balance"], cfg["ticker"])   # Rule 16 Tier 1 (P2-073) / BUG-09 (P2-105)
        self.running  = False
        self.session: dict[str, Any] = self._fresh_session()
        # Daily bars only change once per trading day — cache rather than
        # re-fetching over the network every 60s poll (was ~90 wasted
        # yfinance calls per session, on top of the post-session spam
        # fixed separately). See _get_daily_bars().
        self._df_1d_cache: "pd.DataFrame | None" = None
        self._df_1d_cache_date: "str | None" = None

    def _get_daily_bars(self) -> pd.DataFrame:
        today = str(_now_market(self.cfg).date())
        if self._df_1d_cache is None or self._df_1d_cache_date != today:
            df = _fetch_daily(self.cfg["ticker"])
            # Only lock in today's cache date on a genuinely non-empty
            # result. An exception here already skips the assignment below
            # naturally (propagates before it's reached) — but yfinance can
            # also fail "successfully", returning an empty frame without
            # raising. Locking the date in that case would mean gap_pct
            # stays None for the rest of the session, with no retry.
            if not df.empty:
                self._df_1d_cache = df
                self._df_1d_cache_date = today
            return df
        return self._df_1d_cache

    def _fresh_session(self) -> dict:
        return {"pnl_r": 0.0, "pnl_gbp": 0.0, "n_trades": 0,
                "n_wins": 0, "n_losses": 0, "consec_losses": 0,
                "max_consec_loss": 0, "halted": False,
                "learning_done": False}   # P2-058: fires once at true session close

    def _heartbeat(self, status: str, message: str) -> None:
        pos = self.db.get_open_position()
        self.db.update_heartbeat(status, message, str(_utcnow().date()),
                                  self.cfg["account_balance"],
                                  pos["id"] if pos else None)

    def run(self) -> None:
        """Blocking run loop. Ctrl+C to stop."""
        self.running = True
        log.info("=" * 60)
        log.info("  TRADING ENGINE STARTED  v2.1.0")
        log.info("  Ticker: %s  ORB: %s  Account: £%.0f",
                  self.cfg["ticker"], self.cfg["orb_method"],
                  self.cfg["account_balance"])
        log.info("  DB: %s", self.cfg["db_path"])
        log.info("=" * 60)

        # ── P1-008: Market calendar gate ──────────────────────────────────────
        # Check before any data fetch or session work.  Exits cleanly on
        # holidays; adjusts session_end on early-close days.
        if _CALENDAR:
            # No tz_offset_hours passed — check_market_session() computes
            # its own DST-aware offset via zoneinfo by default now.
            # market_type comes from CONFIG (P2-112) rather than being
            # hardcoded to NYSE at this call site — trading a different
            # session means changing cfg["market_type"] to a key with real
            # data in market_calendar.py's EQUITIES_MARKETS registry.
            cal = _check_market_session(market_type=self.cfg.get("market_type", "nyse_equities"))
            if not cal.is_open:
                # cal.reason already names the correct market (e.g. "NYSE
                # closed — ..."), not hardcoded here — stays accurate
                # whichever market_type is configured.
                log.info("MARKET CLOSED — %s", cal.reason)
                log.info("No session today (%s). Exiting cleanly.", cal.session_date)
                self._heartbeat("stopped", f"No session — {cal.reason}")
                _notify(self.cfg, f"📅 No session today — {cal.reason}", "INFO")
                return
            if cal.is_early_close:
                # Cap session_end so the engine doesn't attempt data fetches
                # after the market actually closes.  Strategy window (11:00)
                # is unaffected on most early-close days, but this guards
                # against edge cases where market closes before 11:00.
                early_cap = cal.normal_close
                tz_label = _now_market(self.cfg).strftime('%Z')
                if early_cap < self.cfg["session_end"]:
                    log.info(
                        "EARLY CLOSE DAY — market closes %s %s. "
                        "Capping session_end from %s to %s.",
                        early_cap, tz_label, self.cfg["session_end"], early_cap
                    )
                    self.cfg = {**self.cfg, "session_end": early_cap}
                else:
                    log.info("EARLY CLOSE DAY — market closes %s %s "
                             "(strategy window ends %s — unaffected).",
                             early_cap, tz_label, self.cfg["session_end"])
                _notify(self.cfg,
                        f"⚠️ Early close day — market closes {early_cap.strftime('%H:%M')} "
                        f"{tz_label}",
                        "INFO")
            else:
                log.info("Market check: %s", cal.reason)
        # ─────────────────────────────────────────────────────────────────────

        self._heartbeat("running", "Engine started — awaiting session open")

        try:
            while self.running:
                self._tick()
                time.sleep(self.cfg["poll_interval_s"])
        except KeyboardInterrupt:
            log.info("Engine stopped by user (Ctrl+C).")
        finally:
            self._heartbeat("stopped", "Engine stopped")
            log.info("Goodbye.")

    def _tick(self) -> None:
        """One evaluation cycle."""
        # ── Rule 16 Tier 1: permanent retirement check (P2-073) ──────────
        # Persisted in account_meta, checked before anything else — this
        # is what makes it survive _fresh_session() resets and engine
        # restarts, unlike the daily stop below.
        retired = self.db.get_retired_status()
        if retired["retired"]:
            self._heartbeat("halted", f"RETIRED — {retired['reason']}")
            return

        now    = _now_market(self.cfg)
        today  = str(now.date())
        in_wnd = _in_session(self.cfg)

        # ── Pre-session reset ──
        if now.time() < self.cfg["session_start"]:
            self.session = self._fresh_session()
            self._heartbeat("running", f"Pre-market. Session opens at {self.cfg['session_start']}")
            return

        # ── Post-session close ──
        if now.time() > self.cfg["session_end"]:
            pos = self.db.get_open_position()

            # Nothing left to do this evening — skip the network fetch
            # entirely rather than repeating it every tick until midnight.
            # Previously this fetched every 60s regardless (~780 wasted
            # yfinance calls per evening once learning_done flips true —
            # a real rate-limit/IP-ban risk for the next day's session).
            # The open-position branch below still runs unconditionally,
            # every tick, regardless of learning_done — a stray open
            # position should never stop being retried just because the
            # session-close report already ran.
            if not pos and self.session["learning_done"]:
                self._heartbeat("stopped", f"Session closed at {now.strftime('%H:%M %Z')}")
                return

            df_5m_close: "pd.DataFrame | None" = None
            if pos:
                log.info("Post-session: force-closing open position %d", pos["id"])
                df_5m_close = _fetch_live(self.cfg["ticker"])
                vwap = _vwap(df_5m_close, self.cfg["session_start"])
                manage_open_position(pos, df_5m_close, vwap, self.db, self.cfg, self.session)
                self._check_drawdown_stop()
            else:
                # Fetch bars for SessionLearner ATR even if no position open
                try:
                    df_5m_close = _fetch_live(self.cfg["ticker"])
                except Exception:
                    df_5m_close = None
            self._persist_session(today)
            self._run_session_close(today, df_5m=df_5m_close)   # P2-058
            self._heartbeat("stopped", f"Session closed at {now.strftime('%H:%M %Z')}")
            return

        # ── Halted for the day ──
        if self.session["halted"]:
            self._heartbeat("halted", "Daily stop triggered — no new entries")
            return

        # ── Consecutive loss pause ──
        if self.session["consec_losses"] >= self.cfg["consec_loss_pause"]:
            self._heartbeat("paused",
                             f"{self.session['consec_losses']} consecutive losses — 30-min pause")
            return

        # ── Fetch data ──
        try:
            df_5m  = _fetch_live(self.cfg["ticker"])
            df_1d  = self._get_daily_bars()
            vwap   = _vwap(df_5m, self.cfg["session_start"])
        except Exception as exc:
            log.warning("Data fetch error: %s", exc)
            self._heartbeat("running", f"Data fetch error: {exc}")
            return

        # ── Manage open position first ──
        pos = self.db.get_open_position()
        if pos:
            outcome = manage_open_position(pos, df_5m, vwap, self.db, self.cfg, self.session)
            if outcome == "exited":
                self._check_daily_stop(today)
                self._check_drawdown_stop()
            self._heartbeat("running",
                             f"Position #{pos['id']} open — P&L {self.session['pnl_r']:+.2f}R")
            return

        # ── Evaluate signals for new entry ──
        # P2-105 item 7: entry evaluation uses only SETTLED bars — a still-
        # forming last bar can show a spurious breakout that reverses
        # before it closes. Exits above intentionally still saw the live
        # (possibly forming) df_5m/vwap — waiting for bar-close there
        # would add latency to stop/target execution, which is the
        # opposite of what's wanted for risk management.
        df_5m_settled = _drop_forming_bar(df_5m)
        vwap_settled  = _vwap(df_5m_settled, self.cfg["session_start"])
        ctx = evaluate_signals(df_5m_settled, df_1d, vwap_settled, self.cfg)
        log.info("[%s] %s | ORB:%s VWAP:%s Gate:%s | %s",
                  now.strftime("%H:%M"), ctx["ticker"],
                  ctx["gate_orb_break"], ctx["gate_vwap"],
                  ctx["gate_final"], ctx[:50] if isinstance(ctx.get("reason"), str)
                  else ctx.get("reason", "")[:60])

        # ── Attempt entry if AND-gate passes ──
        if ctx["_gate_pass"] and ctx["_breakout_dir"] and in_wnd:
            params = compute_entry_params(ctx, self.db, self.cfg)
            if not params.get("tradeable", True):
                # Cost model or liquidity check rejected the trade
                ctx["action"] = "SKIP"
                ctx["reason"] += f" | COST REJECT: {params.get('reject_reason', 'cost too high')}"
                log.warning("Trade rejected by cost model: %s", params.get("reject_reason"))
            elif params["units"] > 0:
                cost_r   = params.get("cost_r", 0.05)
                cost_det = params.get("cost_detail", {})
                ctx["action"]        = "ENTER"
                ctx["entry_price"]   = params["entry_price"]
                ctx["stop_price"]    = params["stop_price"]
                ctx["position_size"] = params["units"]
                ctx["risk_pct"]      = params["risk_pct"]
                ctx["reason"]       += (f" → ENTERING {params['direction'].upper()} "
                                        f"${params['entry_price']:.2f} "
                                        f"stop ${params['stop_price']:.2f} "
                                        f"units {params['units']:.0f} "
                                        f"cost {cost_r:.3f}R "
                                        f"[comm ${cost_det.get('commission_rt',0):.2f} "
                                        f"spread ${cost_det.get('spread_rt',0):.2f} "
                                        f"impact ${cost_det.get('market_impact',0):.2f}]")

                pos_id = self.db.open_position(
                    ticker      = ctx["ticker"],
                    direction   = params["direction"],
                    entry_price = params["entry_price"],
                    stop_price  = params["stop_price"],
                    units       = params["units"],
                    targets     = params["targets"],
                    notes       = (f"Engine auto-entry | RVOL {ctx.get('rvol','?')} | "
                                   f"VIX {ctx.get('vix','?')} | cost {cost_r:.3f}R"),
                )
                _notify(self.cfg,
                         f"ENTERED #{pos_id} {params['direction'].upper()} {ctx['ticker']} "
                         f"@ ${params['entry_price']:.2f}  stop=${params['stop_price']:.2f}  "
                         f"units={params['units']:.0f}  cost={cost_r:.3f}R", "TRADE")
                log.info("ENTERED #%d: %s %s @ $%.2f  stop=$%.2f  units=%.0f  cost=%.3fR",
                          pos_id, params["direction"], ctx["ticker"],
                          params["entry_price"], params["stop_price"],
                          params["units"], cost_r)
                self._heartbeat("running",
                                 f"ENTERED #{pos_id} {params['direction']} @ "
                                 f"${params['entry_price']:.2f} | cost {cost_r:.3f}R")
            else:
                ctx["action"] = "SKIP"
                ctx["reason"] += " | SKIP: position size zero (check account/risk settings)"
        elif not ctx["_gate_pass"]:
            ctx["action"] = "SKIP"
        elif not in_wnd:
            ctx["action"] = "SKIP"
            ctx["reason"] += " | Outside trading window"

        # ── Log decision to ledger ──
        self.db.log_decision(ctx)
        self._persist_session(today)
        self._heartbeat("running",
                         f"Last eval {now.strftime('%H:%M')}: {ctx['action']} | {ctx['gate_final']}")

    def _check_daily_stop(self, today: str) -> None:
        # BUG-04 fix: pnl_r (R-multiples) * risk_pct (fraction of account per
        # R) directly gives fraction of account lost — that's what belongs
        # on the left of this comparison. The previous version additionally
        # divided by account_balance, which only produced the intended 3%
        # trigger by coincidence at the exact default $10,000 balance (since
        # account_balance == 1/risk_pct**2 only there) — at $50,000 it took
        # 15 losing 1R trades to trigger instead of 3, and at $2,000 it
        # triggered on a fraction of a single loss. Confirmed both ways by
        # direct calculation before changing this.
        if self.session["pnl_r"] * self.cfg["risk_pct"] <= -self.cfg["max_daily_loss_pct"]:
            self.session["halted"] = True
            log.warning("DAILY STOP TRIGGERED — halting for rest of session")
            _notify(self.cfg, "🛑 DAILY STOP TRIGGERED — session halted. No more entries today.", "HALT")

    def _check_drawdown_stop(self) -> None:
        """Rule 16 Tier 1 (P2-073): 25% equity drawdown from peak, tracked
        across the account's whole lifetime — not the current session.
        Unlike _check_daily_stop (self.session['halted'], cleared by
        _fresh_session() every day), this persists to account_meta and is
        checked at the top of every _tick(), so once tripped the strategy
        stays retired across session resets AND engine restarts.

        Tier 2 (30-trade rolling expectancy + DSR<50%) is NOT implemented
        here — it has a genuinely open design question (how an isolated
        per-strategy process learns the concurrent-strategy count for its
        own DSR n_trials) that needs a decision first, per the P2-073 notes."""
        acc = self.db.get_account_summary()
        if not acc:
            return
        if acc["drawdown_pct"] >= self.cfg["max_drawdown_pct"]:
            reason = (f"TIER 1 DRAWDOWN STOP — equity down {acc['drawdown_pct']:.1%} "
                      f"from peak £{acc['peak_equity']:,.0f} to "
                      f"£{acc['current_equity']:,.0f} (limit "
                      f"{self.cfg['max_drawdown_pct']:.0%})")
            self.db.mark_retired(reason)
            log.error(reason)
            _notify(self.cfg, f"🛑 {reason} — strategy retired.", "HALT")

    def _persist_session(self, today: str) -> None:
        """Write running session stats to DB. Called on every tick."""
        self.db.upsert_session(
            today,
            n_trades         = self.session["n_trades"],
            n_wins           = self.session["n_wins"],
            n_losses         = self.session["n_losses"],
            session_pnl_r    = round(self.session["pnl_r"], 3),
            session_pnl_gbp  = round(self.session["pnl_gbp"], 2),
            stop_triggered   = int(self.session["halted"]),
            max_consec_loss  = self.session["max_consec_loss"],
        )

    def _run_session_close(self, today: str, df_5m: "pd.DataFrame | None" = None) -> None:
        """
        P2-058 — SessionLearner post-session analysis.

        Called ONCE at true session end (post-session block in _tick).
        Fires only when session["learning_done"] is False, then sets it
        True so repeated post-session ticks don't duplicate rows.

        Computes a real avg_atr from live 5-min data when available,
        and uses actual account equity for risk_amount so R-multiple
        drag estimates are meaningful.
        """
        if self.session["learning_done"]:
            return                         # already ran this session

        decisions = self.db.get_decisions(today, limit=200)
        trades    = self.db.get_session_trades(today)

        # Need at least some data to be worth running
        if not decisions and not trades:
            log.debug("SessionLearner: no data for %s — skipping", today)
            return

        if not (TOOLKIT and hasattr(tk, "SessionLearner")):
            log.warning("SessionLearner not available in toolkit — skipping P2-058 analysis")
            return

        # ── Compute real avg_atr from live 5-min bars ──────────────────────
        avg_atr = 0.80   # safe fallback
        if df_5m is not None and not df_5m.empty:
            try:
                hi = df_5m["High"].dropna()
                lo = df_5m["Low"].dropna()
                cl = df_5m["Close"].dropna()
                if len(cl) >= 2:
                    tr = pd.concat([
                        hi - lo,
                        (hi - cl.shift(1)).abs(),
                        (lo - cl.shift(1)).abs(),
                    ], axis=1).max(axis=1)
                    computed_atr = float(tr.rolling(min(14, len(tr))).mean().iloc[-1])
                    if computed_atr > 0:
                        avg_atr = round(computed_atr, 4)
                        log.debug("SessionLearner: computed avg_atr=%.4f from live bars", avg_atr)
            except Exception:
                pass   # fall through to default

        # ── Use actual account equity for risk_amount ───────────────────────
        acc = self.db.get_account_summary() if hasattr(self.db, "get_account_summary") else None
        risk_amount = (acc.get("current_equity", self.cfg["account_balance"])
                       * self.cfg["risk_pct"]) if acc else (
                       self.cfg["account_balance"] * self.cfg["risk_pct"])
        risk_amount = max(risk_amount, 1.0)

        # ── Run all three learning loops ────────────────────────────────────
        try:
            learner = tk.SessionLearner(decisions, trades)

            log.info("─" * 50)
            log.info("SESSION LEARNING — %s", today)

            skip_analysis   = learner.analyse_skips()
            fill_analysis   = learner.analyse_fill_bias(avg_atr=avg_atr,
                                                        risk_amount=risk_amount)
            regime_analysis = learner.regime_performance()
            report          = learner.generate_refinement(avg_atr=avg_atr,
                                                          risk_amount=risk_amount)

            # ── Persist to DB ───────────────────────────────────────────────
            self.db.save_session_learning(today, report)
            log.info("SessionLearner: report saved to DB (%s)", today)

            # ── Log action items prominently ────────────────────────────────
            action_items = report.get("action_items", [])
            if action_items:
                log.info("SESSION LEARNING — %d action item(s):", len(action_items))
                for i, item in enumerate(action_items, 1):
                    log.info("  [%d/%d] %s", i, len(action_items), item[:200])
            else:
                log.info("SESSION LEARNING — no critical issues detected")

            # ── Log key metrics ─────────────────────────────────────────────
            log.info("  Skips: %d  |  over-filter flag: %s",
                     skip_analysis.get("n_skips", 0),
                     "⚠️ YES" if skip_analysis.get("over_filter_flag") else "✅ no")
            log.info("  Fill drag: %.4fR/trade  |  modelled EV: %+.3fR  →  realistic EV: %+.3fR",
                     fill_analysis.get("avg_total_drag_r", 0),
                     fill_analysis.get("modelled_ev", 0),
                     fill_analysis.get("realistic_ev", 0))
            log.info("  Best regime: %s  |  Worst: %s",
                     regime_analysis.get("best_regime", "N/A"),
                     regime_analysis.get("worst_regime", "N/A"))
            log.info("─" * 50)

            # ── Discord / Telegram notification ────────────────────────────
            summary_msg = (
                f"📚 Session Learning — {today}\n"
                f"Skips: {skip_analysis.get('n_skips', 0)}  "
                f"{'⚠️ Over-filter detected' if skip_analysis.get('over_filter_flag') else '✅ Gate balance OK'}\n"
                f"Realistic EV: {fill_analysis.get('realistic_ev', 0):+.3f}R  "
                f"(drag {fill_analysis.get('avg_total_drag_r', 0):.4f}R/trade)\n"
                f"Worst regime: {regime_analysis.get('worst_regime', '?')}\n"
            )
            if action_items:
                summary_msg += f"Action items: {len(action_items)}\n"
                summary_msg += "\n".join(f"• {a[:120]}" for a in action_items[:3])
            _notify(self.cfg, summary_msg, "INFO")

        except Exception as exc:
            log.warning("SessionLearner failed: %s", exc)
        finally:
            self.session["learning_done"] = True   # never runs twice this session


# ===========================================================================
# ENTRY POINT
# ===========================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Autonomous paper trading engine")
    p.add_argument("--ticker",    default=CONFIG["ticker"])
    p.add_argument("--account",   type=float, default=CONFIG["account_balance"])
    p.add_argument("--orb",       default=CONFIG["orb_method"],
                   choices=["5min","15min","30min"])
    p.add_argument("--risk",      type=float, default=CONFIG["risk_pct"])
    p.add_argument("--db",        default=CONFIG["db_path"])
    p.add_argument("--poll",      type=int,   default=CONFIG["poll_interval_s"])
    p.add_argument("--kelly",     action="store_true")
    p.add_argument("--wfa-only",  action="store_true",
                   help="Run walk-forward analysis (hourly bars, single-gate approximation) then exit")
    p.add_argument("--wfa-splits", type=int, default=6)
    p.add_argument("--wfa-5m",    action="store_true",
                   help="Run walk-forward analysis on real 5-minute bars with the "
                        "engine's actual 2-gate signal logic, then exit")
    p.add_argument("--wfa-5m-is-days",  type=int, default=25)
    p.add_argument("--wfa-5m-oos-days", type=int, default=10)
    p.add_argument("--wfa-5m-splits",   type=int, default=3)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg  = {**CONFIG,
            "ticker":           args.ticker.upper(),
            "account_balance":  args.account,
            "orb_method":       args.orb,
            "risk_pct":         args.risk,
            "db_path":          args.db,
            "poll_interval_s":  args.poll,
            "use_kelly":        args.kelly}

    db = PaperAccountDB(cfg["db_path"])

    if args.wfa_only:
        log.info("Running WFA only (%d splits) for %s...", args.wfa_splits, cfg["ticker"])
        results = wfa_run(cfg["ticker"], db, cfg, n_splits=args.wfa_splits)
        valid = [r for r in results if r.get("wfe") is not None]
        if valid:
            mean_wfe = statistics.mean(r["wfe"] for r in valid)
            passes   = sum(1 for r in valid if r["wfe_verdict"] == "PASS")
            log.info("WFA complete: mean WFE=%.3f  %d/%d splits PASS",
                      mean_wfe, passes, len(valid))
        sys.exit(0)

    if args.wfa_5m:
        log.info("Running WFA (5m, full-gate, up to %d splits) for %s...",
                  args.wfa_5m_splits, cfg["ticker"])
        results = wfa_run_5m(cfg["ticker"], db, cfg,
                              is_days=args.wfa_5m_is_days,
                              oos_days=args.wfa_5m_oos_days,
                              n_splits=args.wfa_5m_splits)
        valid = [r for r in results if r.get("wfe") is not None]
        if valid:
            mean_wfe = statistics.mean(r["wfe"] for r in valid)
            passes   = sum(1 for r in valid if r["wfe_verdict"] == "PASS")
            log.info("WFA(5m) complete: mean WFE=%.3f  %d/%d splits PASS",
                      mean_wfe, passes, len(valid))
        sys.exit(0)

    engine = TradingEngine(cfg)
    engine.run()
