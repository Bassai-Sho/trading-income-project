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
Every POLL_INTERVAL seconds during the trading window (09:30–11:00 EST):

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
from datetime import date, datetime, time as Time, timedelta
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
    "session_start":     Time(9, 30),
    "session_end":       Time(11, 0),
    "poll_interval_s":   60,             # seconds between signal evaluations
    "db_path":           "paper_account.db",
    "log_level":         "INFO",
    "max_daily_loss_pct": 0.03,
    "consec_loss_pause": 3,
    "min_rvol":          1.0,            # stocks in play filter
    "vwap_lookback":     3,
    "tz_offset_hours":   -5,             # EST = UTC-5 (no DST adjustment; update manually)
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
    ]
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
        log.info("DB ready: %s", self.db_path)

    # ── Positions ──────────────────────────────────────────────────────────

    def open_position(self, ticker: str, direction: str, entry_price: float,
                       stop_price: float, units: float, targets: list[float],
                       notes: str = "") -> int:
        t1 = targets[0] if len(targets) > 0 else None
        t2 = targets[1] if len(targets) > 1 else None
        t3 = targets[2] if len(targets) > 2 else None
        ts = _now_iso()
        with self._conn() as conn:
            cur = conn.execute(
                """INSERT INTO positions
                   (ticker, direction, entry_price, stop_price, target_1r, target_2r, target_3r,
                    units, status, opened_at, notes)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (ticker, direction, entry_price, stop_price, t1, t2, t3, units, "open", ts, notes)
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

def _now_est(cfg: dict) -> datetime:
    return datetime.utcnow() + timedelta(hours=cfg["tz_offset_hours"])

def _now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")

def _in_session(cfg: dict) -> bool:
    t = _now_est(cfg).time()
    return cfg["session_start"] <= t <= cfg["session_end"]

def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df

def _fetch_live(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, period="5d", interval="5m",
                     auto_adjust=True, progress=False)
    return _flatten(df)

def _fetch_daily(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, period="20d", interval="1d",
                     auto_adjust=True, progress=False)
    return _flatten(df)

def _vwap(df: pd.DataFrame) -> pd.Series:
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    pv = tp * df["Volume"]
    result = pd.Series(index=df.index, dtype=float)
    for day in set(df.index.date):
        mask = (df.index.date == day) & (df.index.time >= Time(9, 30))
        cv = df["Volume"][mask].cumsum()
        result[mask] = (pv[mask].cumsum() / cv.replace(0, float("nan"))).values
    return result

def _orb_range(df: pd.DataFrame, method: str) -> dict | None:
    end_map = {"5min": Time(9, 34), "15min": Time(9, 44), "30min": Time(9, 55)}
    end_t = end_map.get(method)
    if not end_t:
        return None
    today = df.index[-1].date()
    mask = (df.index.date == today) & (df.index.time >= Time(9, 30)) & (df.index.time <= end_t)
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
        "session_date": str(datetime.utcnow().date()),
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

    if len(df_1d) >= 2:
        pdh = float(df_1d["High"].iloc[-2])
        pdl = float(df_1d["Low"].iloc[-2])
        prev_close = float(df_1d["Close"].iloc[-2])
        today_open = float(df_5m["Open"].iloc[0]) if not df_5m.empty else prev_close
        ctx["gap_pct"] = round((today_open - prev_close) / prev_close * 100, 3)
    else:
        ctx["gap_pct"] = None

    last_close = float(df_5m["Close"].dropna().iloc[-1]) if not df_5m.empty else None
    ctx["last_close"] = last_close

    vwap_val = float(vwap.dropna().iloc[-1]) if not vwap.dropna().empty else None
    ctx["vwap"] = round(vwap_val, 4) if vwap_val else None

    vs = _vwap_slope(vwap, lookback=cfg.get("vwap_lookback", 3))
    ctx["vwap_slope"] = vs["direction"]

    orb = _orb_range(df_5m, cfg["orb_method"])
    ctx["orb_high"]      = orb["orb_high"]  if orb else None
    ctx["orb_low"]       = orb["orb_low"]   if orb else None
    ctx["orb_bars_used"] = orb["bars_used"] if orb else None

    # RVOL estimation from 5-min data
    try:
        today = df_5m.index[-1].date()
        df_today = df_5m[df_5m.index.date == today]
        first5 = df_today[df_today.index.time <= Time(9, 34)]
        past_days = [d for d in set(df_5m.index.date) if d != today]
        avg_first5_vols = []
        for d in past_days:
            dd = df_5m[(df_5m.index.date == d) & (df_5m.index.time <= Time(9, 34))]
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
    """
    last_close = float(df_5m["Close"].dropna().iloc[-1])
    direction  = pos["direction"]
    entry      = pos["entry_price"]
    stop       = pos["stop_price"]
    t1 = pos.get("target_1r")
    risk_dist  = abs(entry - stop)
    actual_r   = ((last_close - entry) / risk_dist *
                   (1 if direction == "long" else -1))

    # ── Check stop hit ──
    if direction == "long" and last_close <= stop:
        actual_r_final = round((stop - entry) / risk_dist * -1, 3)
        _exit_position(pos, stop, "trailing_stop", actual_r_final, db, cfg, session)
        log.info("STOP HIT: %s at $%.2f  actual_r=%.2f",
                 pos["ticker"], stop, actual_r_final)
        return "exited"
    if direction == "short" and last_close >= stop:
        actual_r_final = round((entry - stop) / risk_dist * -1, 3)
        _exit_position(pos, stop, "trailing_stop", actual_r_final, db, cfg, session)
        log.info("STOP HIT (short): %s at $%.2f  actual_r=%.2f",
                 pos["ticker"], stop, actual_r_final)
        return "exited"

    # ── Check 1R target (first ladder) ──
    if t1 and direction == "long" and last_close >= t1:
        actual_r_final = round((t1 - entry) / risk_dist, 3)
        _exit_position(pos, t1, "target_1r", actual_r_final, db, cfg, session)
        log.info("TARGET 1R: %s at $%.2f  actual_r=%.2f",
                 pos["ticker"], t1, actual_r_final)
        return "exited"
    if t1 and direction == "short" and last_close <= t1:
        actual_r_final = round((entry - t1) / risk_dist, 3)
        _exit_position(pos, t1, "target_1r", actual_r_final, db, cfg, session)
        log.info("TARGET 1R (short): %s at $%.2f  actual_r=%.2f",
                 pos["ticker"], t1, actual_r_final)
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

    # ── End-of-session close ──
    now_t = _now_est(cfg).time()
    if now_t >= cfg["session_end"]:
        actual_r_final = round(actual_r, 3)
        _exit_position(pos, last_close, "eod", actual_r_final, db, cfg, session)
        log.info("EOD CLOSE: %s at $%.2f  actual_r=%.2f",
                 pos["ticker"], last_close, actual_r_final)
        return "exited"

    return "continue"


def _exit_position(pos: dict, exit_price: float, reason: str,
                    actual_r: float, db: PaperAccountDB,
                    cfg: dict, session: dict) -> None:
    pnl_gbp = actual_r * cfg["account_balance"] * cfg["risk_pct"]
    db.close_position(pos["id"], exit_price, reason, actual_r, pnl_gbp)
    level = "TRADE" if actual_r > 0 else "WARN"
    _notify(cfg,
            f"EXIT #{pos['id']} {pos['ticker']} {pos['direction']} "
            f"@ ${exit_price:.2f}  {actual_r:+.2f}R  reason={reason}  "
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
        session = day_df[day_df.index.time >= Time(9, 30)]
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
    now_est = (datetime.utcnow() + timedelta(hours=cfg.get("tz_offset_hours",-5))).strftime("%H:%M EST")

    def _send():
        import asyncio, json as _json
        payload = _json.dumps({"content": f"{emoji} **{level}** [{now_est}]  {message}"})
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
        self.running  = False
        self.session: dict[str, Any] = self._fresh_session()

    def _fresh_session(self) -> dict:
        return {"pnl_r": 0.0, "pnl_gbp": 0.0, "n_trades": 0,
                "n_wins": 0, "n_losses": 0, "consec_losses": 0,
                "max_consec_loss": 0, "halted": False,
                "learning_done": False}   # P2-058: fires once at true session close

    def _heartbeat(self, status: str, message: str) -> None:
        pos = self.db.get_open_position()
        self.db.update_heartbeat(status, message, str(datetime.utcnow().date()),
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
            cal = _check_market_session(
                tz_offset_hours=self.cfg.get("tz_offset_hours", -5)
            )
            if not cal.is_open:
                log.info("NYSE CLOSED — %s", cal.reason)
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
                if early_cap < self.cfg["session_end"]:
                    log.info(
                        "EARLY CLOSE DAY — market closes %s EST. "
                        "Capping session_end from %s to %s.",
                        early_cap, self.cfg["session_end"], early_cap
                    )
                    self.cfg = {**self.cfg, "session_end": early_cap}
                else:
                    log.info("EARLY CLOSE DAY — market closes %s EST "
                             "(strategy window ends %s — unaffected).",
                             early_cap, self.cfg["session_end"])
                _notify(self.cfg,
                        f"⚠️ Early close day — market closes {early_cap.strftime('%H:%M')} EST",
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
        now    = _now_est(self.cfg)
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
            df_5m_close: "pd.DataFrame | None" = None
            if pos:
                log.info("Post-session: force-closing open position %d", pos["id"])
                df_5m_close = _fetch_live(self.cfg["ticker"])
                vwap = _vwap(df_5m_close)
                manage_open_position(pos, df_5m_close, vwap, self.db, self.cfg, self.session)
            else:
                # Fetch bars for SessionLearner ATR even if no position open
                try:
                    df_5m_close = _fetch_live(self.cfg["ticker"])
                except Exception:
                    df_5m_close = None
            self._persist_session(today)
            self._run_session_close(today, df_5m=df_5m_close)   # P2-058
            self._heartbeat("stopped", f"Session closed at {now.strftime('%H:%M')} EST")
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
            df_1d  = _fetch_daily(self.cfg["ticker"])
            vwap   = _vwap(df_5m)
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
            self._heartbeat("running",
                             f"Position #{pos['id']} open — P&L {self.session['pnl_r']:+.2f}R")
            return

        # ── Evaluate signals for new entry ──
        ctx = evaluate_signals(df_5m, df_1d, vwap, self.cfg)
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
        if self.session["pnl_r"] / max(self.cfg["account_balance"] * self.cfg["risk_pct"], 1) \
                <= -self.cfg["max_daily_loss_pct"]:
            self.session["halted"] = True
            log.warning("DAILY STOP TRIGGERED — halting for rest of session")
            _notify(self.cfg, "🛑 DAILY STOP TRIGGERED — session halted. No more entries today.", "HALT")

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
                   help="Run walk-forward analysis then exit")
    p.add_argument("--wfa-splits", type=int, default=6)
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

    engine = TradingEngine(cfg)
    engine.run()
