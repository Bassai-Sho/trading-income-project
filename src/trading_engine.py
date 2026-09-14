"""
trading_engine.py
=================
Autonomous paper trading engine for the Trading Income Project.

ARCHITECTURE
------------
This engine runs as a standalone Python process in a terminal.
It shares a SQLite database with the Streamlit dashboard (trading_dashboard.py),
which reads the same database to display live results.

  Terminal 1:  python src/trading_engine.py
  Terminal 2:  streamlit run src/trading_dashboard.py

The database (DATA/paper_account.db by default) is the single source of truth.
Neither process depends on the other; they communicate only through the DB.
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
from typing import Any, Callable

import pandas as pd
import yfinance as yf

try:
    import aiohttp as _aiohttp
    _DISCORD_DEPS = True
except ImportError:
    _DISCORD_DEPS = False

# ---------------------------------------------------------------------------
# Optional toolkit import
# ---------------------------------------------------------------------------
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import trading_quant_toolkit_v2_4 as tk  # type: ignore
    TOOLKIT = True
except ImportError:
    TOOLKIT = False
    logging.warning("trading_quant_toolkit_v2_4.py not found — using built-in calculations")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CONFIG: dict[str, Any] = {
    "ticker":            "SPY",
    "orb_method":        "15min",        # '5min' | '15min' | '30min'
    "account_balance":   10_000.0,
    "risk_pct":          0.01,           # 1% fixed (overridden by Kelly after 50 trades)
    "use_kelly":         False,          # enable after 50+ trades
    "target_rr":         2.0,            # baseline R:R
    "use_vwap_trailing": True,           # VWAP trailing stop (Zarattini 2024)
    "use_ladder_exit":   True,           # 1R/2R/3R partial exits (Maroy 2025)
    "session_start":     Time(9, 30),
    "session_end":       Time(11, 0),
    "poll_interval_s":   60,             # seconds between signal evaluations
    "db_path":           "DATA/paper_account.db",
    "llm_preset":        os.environ.get("LLM_PRESET", "nuc-pair1"),
    "log_level":         "INFO",
    "max_daily_loss_pct": 0.03,
    "consec_loss_pause": 3,
    "min_rvol":          1.0,
    "vwap_lookback":     3,
    "tz_offset_hours":   -5,             # EST = UTC-5
    "discord_webhook":   os.environ.get("DISCORD_WEBHOOK_URL", ""),
    "telegram_token":    "",
    "telegram_chat_id":  "",
    "run_analyser_eod":  True,           # True = auto-trigger session_analyser.py at EOD
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
# DATABASE LAYER (Self-Healing)
# ===========================================================================

class PaperAccountDB:
    """
    SQLite-backed paper account. Thread-safe via connection-per-call pattern.
    Auto-creates parent directory (DATA/) so uninitialized setups never crash.
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS positions (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker      TEXT    NOT NULL,
        direction   TEXT    NOT NULL,
        entry_price REAL    NOT NULL,
        stop_price  REAL    NOT NULL,
        target_1r   REAL,
        target_2r   REAL,
        target_3r   REAL,
        units       REAL    NOT NULL,
        status      TEXT    NOT NULL DEFAULT 'open',
        opened_at   TEXT    NOT NULL,
        closed_at   TEXT,
        exit_price  REAL,
        actual_r    REAL,
        pnl_gbp     REAL,
        exit_reason TEXT,
        notes       TEXT
    );

    CREATE TABLE IF NOT EXISTS orders (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker       TEXT    NOT NULL,
        order_type   TEXT    NOT NULL,
        direction    TEXT    NOT NULL,
        price        REAL,
        units        REAL,
        status       TEXT    NOT NULL DEFAULT 'pending',
        created_at   TEXT    NOT NULL,
        filled_at    TEXT,
        position_id  INTEGER REFERENCES positions(id)
    );

    CREATE TABLE IF NOT EXISTS decisions (
        id             INTEGER PRIMARY KEY AUTOINCREMENT,
        ts             TEXT    NOT NULL,
        session_date   TEXT    NOT NULL,
        ticker         TEXT    NOT NULL,
        action         TEXT    NOT NULL,
        last_close     REAL,
        vwap           REAL,
        vwap_slope     TEXT,
        orb_high       REAL,
        orb_low        REAL,
        orb_method     TEXT,
        vix            REAL,
        gap_pct        REAL,
        rvol           REAL,
        gate_orb_break TEXT,
        gate_vwap      TEXT,
        gate_retest    TEXT,
        gate_final     TEXT,
        entry_price    REAL,
        stop_price     REAL,
        position_size  REAL,
        risk_pct       REAL,
        reason         TEXT    NOT NULL,
        strategy_ver   TEXT    DEFAULT '2.4.0',
        orb_bars_used  INTEGER
    );

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
        wfe            REAL,
        wfe_verdict    TEXT
    );

    CREATE TABLE IF NOT EXISTS heartbeat (
        id         INTEGER PRIMARY KEY DEFAULT 1,
        ts         TEXT,
        status     TEXT,
        message    TEXT,
        session_date TEXT,
        account_balance REAL,
        open_position_id INTEGER
    );

    CREATE TABLE IF NOT EXISTS accounts (
        id               INTEGER PRIMARY KEY DEFAULT 1,
        starting_balance REAL    NOT NULL,
        current_equity   REAL,
        peak_equity      REAL,
        max_drawdown_pct REAL    DEFAULT 0.0,
        risk_of_ruin_pct REAL,
        updated_at       TEXT
    );

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
        action_items     TEXT,
        full_report      TEXT
    );
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        dirname = os.path.dirname(self.db_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        self._setup()

    def _conn(self) -> sqlite3.Connection:
        dirname = os.path.dirname(self.db_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _setup(self) -> None:
        with self._conn() as conn:
            conn.executescript(self.SCHEMA)
        log.info("DB initialized: %s", self.db_path)

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
            conn.execute("UPDATE positions SET stop_price=? WHERE id=?", (new_stop, position_id))

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
            row = conn.execute("SELECT * FROM sessions WHERE session_date=?", (session_date,)).fetchone()
        return dict(row) if row else None

    def get_session_trades(self, session_date: str) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM positions WHERE date(opened_at)=? ORDER BY id DESC",
                (session_date,)
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Heartbeat ──────────────────────────────────────────────────────────

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

    # ── Analytics & Equity ─────────────────────────────────────────────────

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
        aw     = statistics.mean(wins)        if wins   else 0.0
        al     = abs(statistics.mean(losses)) if losses else 0.0
        ev     = (wr * aw) - ((1 - wr) * al)
        sharpe = None
        if len(rs) >= 2:
            sd = statistics.stdev(rs)
            if sd > 0:
                sharpe = round(statistics.mean(rs) / sd, 3)
        return {"n": len(rs), "wr": wr, "aw": aw, "al": al, "ev": ev, "sharpe": sharpe}

    def init_account(self, starting_balance: float) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO accounts (id, starting_balance, current_equity, peak_equity) "
                "VALUES (1, ?, ?, ?)",
                (starting_balance, starting_balance, starting_balance)
            )

    def get_current_equity(self, starting_balance: float | None = None) -> float:
        try:
            with self._conn() as conn:
                row = conn.execute("SELECT starting_balance FROM accounts WHERE id=1").fetchone()
            sb = row[0] if row else (starting_balance or 10_000.0)
        except Exception:
            sb = starting_balance or 10_000.0

        with self._conn() as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(pnl_gbp), 0.0) FROM positions WHERE status='closed'"
            ).fetchone()
        total_pnl = float(row[0]) if row else 0.0
        return round(sb + total_pnl, 2)

    def update_account_equity(self, current_equity: float, ror_pct: float | None = None) -> None:
        with self._conn() as conn:
            conn.execute(
                """UPDATE accounts SET current_equity=?, updated_at=?,
                   risk_of_ruin_pct=COALESCE(?,risk_of_ruin_pct),
                   peak_equity=MAX(COALESCE(peak_equity,current_equity), current_equity),
                   max_drawdown_pct=CASE WHEN peak_equity>0
                     THEN MIN(COALESCE(max_drawdown_pct,0),
                          (current_equity-MAX(COALESCE(peak_equity,current_equity),current_equity))
                          / MAX(COALESCE(peak_equity,current_equity),current_equity)*100)
                     ELSE 0 END
                   WHERE id=1""",
                (current_equity, _now_iso(), ror_pct)
            )

    def save_session_learning(self, session_date: str, report: dict) -> None:
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


# ===========================================================================
# MARKET DATA HELPERS
# ===========================================================================

def _now_est(cfg: dict) -> datetime:
    return datetime.now(timezone.utc) + timedelta(hours=cfg["tz_offset_hours"])

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def _in_session(cfg: dict) -> bool:
    t = _now_est(cfg).time()
    return cfg["session_start"] <= t <= cfg["session_end"]

def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df

def _fetch_live(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, period="5d", interval="5m", auto_adjust=True, progress=False)
    return _flatten(df)

def _fetch_daily(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, period="20d", interval="1d", auto_adjust=True, progress=False)
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
    if not end_t or df.empty:
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

def _kelly_fraction(win_rate: float, avg_win: float, avg_loss: float = 1.0, fraction: float = 0.5) -> float:
    if not (0 < win_rate < 1) or avg_win <= 0 or avg_loss <= 0:
        return 0.0
    r = avg_win / avg_loss
    k = max(0.0, win_rate - (1 - win_rate) / r)
    return min(round(fraction * k, 4), 0.25)

def _vwap_trailing(current_stop: float, vwap_val: float, direction: str) -> float:
    return round(max(current_stop, vwap_val), 4) if direction == "long" else round(min(current_stop, vwap_val), 4)


# ===========================================================================
# SIGNAL EVALUATION
# ===========================================================================

def evaluate_signals(df_5m: pd.DataFrame, df_1d: pd.DataFrame, vwap: pd.Series, cfg: dict) -> dict:
    ctx: dict[str, Any] = {
        "ts":           _now_iso(),
        "session_date": str(datetime.now(timezone.utc).date()),
        "ticker":       cfg["ticker"],
        "action":       "EVALUATE",
        "orb_method":   cfg["orb_method"],
        "strategy_ver": "2.4.0",
    }

    try:
        df_vix = _flatten(yf.download("^VIX", period="2d", interval="5m", auto_adjust=True, progress=False))
        ctx["vix"] = round(float(df_vix["Close"].dropna().iloc[-1]), 2) if not df_vix.empty else None
    except Exception:
        ctx["vix"] = None

    if len(df_1d) >= 2:
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

    # Gate 1: ORB
    if orb is None or last_close is None:
        ctx["gate_orb_break"] = "WAIT"
        breakout_dir = None
        reason_parts = ["ORB range forming"]
    elif last_close > orb["orb_high"]:
        ctx["gate_orb_break"] = "PASS"
        breakout_dir = "long"
        reason_parts = [f"Close ${last_close:.2f} > ORB High ${orb['orb_high']:.2f}"]
    elif last_close < orb["orb_low"]:
        ctx["gate_orb_break"] = "PASS"
        breakout_dir = "short"
        reason_parts = [f"Close ${last_close:.2f} < ORB Low ${orb['orb_low']:.2f}"]
    else:
        ctx["gate_orb_break"] = "FAIL"
        breakout_dir = None
        reason_parts = [f"Inside ORB [{orb['orb_low']:.2f}–{orb['orb_high']:.2f}]"]

    # Gate 2: VWAP Slope
    if breakout_dir is None:
        ctx["gate_vwap"] = "FAIL"
    elif (breakout_dir == "long" and vs["direction"] == "up") or (breakout_dir == "short" and vs["direction"] == "down"):
        ctx["gate_vwap"] = "PASS"
        reason_parts.append(f"VWAP confirms {breakout_dir}")
    else:
        ctx["gate_vwap"] = "FAIL"
        reason_parts.append(f"VWAP {vs['direction']} conflicts with {breakout_dir}")

    ctx["gate_retest"] = "WAIT"
    gate_pass = (ctx["gate_orb_break"] == "PASS" and ctx["gate_vwap"] == "PASS")
    ctx["gate_final"] = "AND_PASS" if gate_pass else "AND_FAIL"
    ctx["reason"] = " | ".join(reason_parts)
    ctx["_breakout_dir"] = breakout_dir
    ctx["_gate_pass"] = gate_pass
    return ctx


# ===========================================================================
# RISK CALCULATIONS
# ===========================================================================

def compute_entry_params(ctx: dict, db: PaperAccountDB, cfg: dict) -> dict:
    direction    = ctx["_breakout_dir"]
    entry_price  = ctx["last_close"]
    orb_high     = ctx["orb_high"]
    orb_low      = ctx["orb_low"]
    account      = db.get_current_equity(cfg["account_balance"])

    risk_dist = orb_high - orb_low if orb_high and orb_low else 0.5
    buffer    = max(risk_dist * 0.1, 0.02)
    stop_price = round(orb_low - buffer, 4) if direction == "long" else round(orb_high + buffer, 4)
    stop_dist  = abs(entry_price - stop_price)

    rf = cfg["risk_pct"]
    if cfg["use_kelly"]:
        stats = db.rolling_stats(50)
        if stats.get("n", 0) >= 50:
            rf = _kelly_fraction(stats["wr"], stats["aw"], stats["al"])

    vix_mod = 1.0
    if ctx.get("vix"):
        vix = ctx["vix"]
        vix_mod = (0.25 if vix > 35 else 0.50 if vix > 25 else 0.75 if vix > 18 else 1.0)

    effective_risk = rf * vix_mod
    risk_amount    = account * effective_risk
    units = round(risk_amount / stop_dist, 4) if stop_dist > 1e-6 else 0.0

    ladder = []
    for r in [1.0, 2.0, 3.0]:
        price = (entry_price + r * stop_dist) if direction == "long" else (entry_price - r * stop_dist)
        ladder.append(round(price, 4))

    return {
        "entry_price":   round(entry_price, 4),
        "stop_price":    stop_price,
        "targets":       ladder,
        "units":         units,
        "risk_pct":      effective_risk,
        "risk_amount":   round(risk_amount, 2),
        "direction":     direction,
        "cost_r":        0.016,
        "tradeable":     units > 0,
    }


# ===========================================================================
# POSITION MANAGEMENT
# ===========================================================================

def manage_open_position(pos: dict, df_5m: pd.DataFrame, vwap: pd.Series, db: PaperAccountDB, cfg: dict, session: dict) -> str | None:
    last_close = float(df_5m["Close"].dropna().iloc[-1])
    direction  = pos["direction"]
    entry      = pos["entry_price"]
    stop       = pos["stop_price"]
    t1         = pos.get("target_1r")
    risk_dist  = abs(entry - stop)

    actual_r   = ((last_close - entry) / risk_dist * (1 if direction == "long" else -1))

    # Stop Hit
    if (direction == "long" and last_close <= stop) or (direction == "short" and last_close >= stop):
        actual_r_final = round((stop - entry) / risk_dist * (1 if direction == "long" else -1), 3)
        _exit_position(pos, stop, "trailing_stop", actual_r_final, db, cfg, session)
        return "exited"

    # Target Hit
    if t1 and ((direction == "long" and last_close >= t1) or (direction == "short" and last_close <= t1)):
        actual_r_final = round((t1 - entry) / risk_dist * (1 if direction == "long" else -1), 3)
        _exit_position(pos, t1, "target_1r", actual_r_final, db, cfg, session)
        return "exited"

    # VWAP Trail
    if cfg["use_vwap_trailing"]:
        vwap_val = float(vwap.dropna().iloc[-1]) if not vwap.dropna().empty else None
        if vwap_val:
            new_stop = _vwap_trailing(stop, vwap_val, direction)
            if new_stop != stop:
                db.update_stop(pos["id"], new_stop)
                return "stop_moved"

    # EOD Close
    now_t = _now_est(cfg).time()
    if now_t >= cfg["session_end"]:
        _exit_position(pos, last_close, "eod", round(actual_r, 3), db, cfg, session)
        return "exited"

    return "continue"


def _exit_position(pos: dict, exit_price: float, reason: str, actual_r: float, db: PaperAccountDB, cfg: dict, session: dict) -> None:
    current_eq = db.get_current_equity(cfg["account_balance"])
    base_risk  = current_eq * cfg["risk_pct"]
    cost_r     = pos.get("cost_r") or 0.016
    net_actual_r = round(actual_r - cost_r, 4)
    pnl_gbp    = round(net_actual_r * base_risk, 2)

    db.close_position(pos["id"], exit_price, reason, net_actual_r, pnl_gbp)
    level = "TRADE" if actual_r > 0 else "WARN"
    _notify(cfg, f"EXIT #{pos['id']} {pos['ticker']} {pos['direction']} @ ${exit_price:.2f} ({actual_r:+.2f}R) reason={reason}", level)

    session["pnl_r"]   += actual_r
    session["pnl_gbp"] += pnl_gbp
    session["n_trades"] += 1
    if actual_r > 0:
        session["n_wins"] += 1
        session["consec_losses"] = 0
    else:
        session["n_losses"] += 1
        session["consec_losses"] += 1
        session["max_consec_loss"] = max(session["max_consec_loss"], session["consec_losses"])


# ===========================================================================
# NOTIFICATION HELPER
# ===========================================================================

def _notify(cfg: dict, message: str, level: str = "INFO") -> None:
    webhook  = cfg.get("discord_webhook", "")
    if not webhook or not _DISCORD_DEPS:
        return

    emoji = {"INFO":"📡","TRADE":"✅","WARN":"⚠️","HALT":"🛑"}.get(level,"📡")
    now_est = (datetime.now(timezone.utc) + timedelta(hours=cfg.get("tz_offset_hours",-5))).strftime("%H:%M EST")

    def _send():
        import asyncio, json as _json
        payload = _json.dumps({"content": f"{emoji} **{level}** [{now_est}]  {message}"})
        async def _post_discord():
            try:
                async with _aiohttp.ClientSession() as sess:
                    await sess.post(webhook, data=payload, headers={"Content-Type":"application/json"}, timeout=_aiohttp.ClientTimeout(total=5))
            except Exception:
                pass
        asyncio.run(_post_discord())

    threading.Thread(target=_send, daemon=True).start()


# ===========================================================================
# ENGINE CONTROLLER
# ===========================================================================

class TradingEngine:
    def __init__(self, cfg: dict) -> None:
        self.cfg      = cfg
        self.db       = PaperAccountDB(cfg["db_path"])
        self.db.init_account(cfg["account_balance"])
        self.running  = False
        self.session: dict[str, Any] = self._fresh_session()

    def _fresh_session(self) -> dict:
        return {
            "pnl_r": 0.0, "pnl_gbp": 0.0, "n_trades": 0,
            "n_wins": 0, "n_losses": 0, "consec_losses": 0,
            "max_consec_loss": 0, "halted": False,
            "analyser_triggered": False,
        }

    def _heartbeat(self, status: str, message: str) -> None:
        pos = self.db.get_open_position()
        self.db.update_heartbeat(status, message, str(datetime.now(timezone.utc).date()),
                                  self.cfg["account_balance"],
                                  pos["id"] if pos else None)

    def run(self) -> None:
        self.running = True
        log.info("=" * 60)
        log.info("  TRADING ENGINE STARTED  v2.4.0")
        log.info("  Ticker: %s  ORB: %s  DB: %s", self.cfg["ticker"], self.cfg["orb_method"], self.cfg["db_path"])
        log.info("=" * 60)
        self._heartbeat("running", "Engine started — awaiting session open")

        try:
            while self.running:
                self._tick()
                time.sleep(self.cfg["poll_interval_s"])
        except KeyboardInterrupt:
            log.info("Engine stopped by user (Ctrl+C).")
        finally:
            self._heartbeat("stopped", "Engine stopped")

    def _tick(self) -> None:
        now    = _now_est(self.cfg)
        today  = str(now.date())
        in_wnd = _in_session(self.cfg)

        # Pre-session
        if now.time() < self.cfg["session_start"]:
            self.session = self._fresh_session()
            self._heartbeat("running", f"Pre-market. Session opens at {self.cfg['session_start']}")
            return

        # Post-session
        if now.time() > self.cfg["session_end"]:
            pos = self.db.get_open_position()
            if pos:
                df_5m = _fetch_live(self.cfg["ticker"])
                vwap = _vwap(df_5m)
                manage_open_position(pos, df_5m, vwap, self.db, self.cfg, self.session)
            self._persist_session(today)
            self._heartbeat("stopped", f"Session closed at {now.strftime('%H:%M')} EST")
            if self.cfg.get("run_analyser_eod", True):
                self._trigger_analyser(today)
            return

        if self.session["halted"] or self.session["consec_losses"] >= self.cfg["consec_loss_pause"]:
            return

        # Fetch live data
        try:
            df_5m  = _fetch_live(self.cfg["ticker"])
            df_1d  = _fetch_daily(self.cfg["ticker"])
            vwap   = _vwap(df_5m)
        except Exception as exc:
            log.warning("Data fetch error: %s", exc)
            return

        pos = self.db.get_open_position()
        if pos:
            outcome = manage_open_position(pos, df_5m, vwap, self.db, self.cfg, self.session)
            if outcome == "exited":
                self._check_daily_stop(today)
            return

        ctx = evaluate_signals(df_5m, df_1d, vwap, self.cfg)

        if ctx["_gate_pass"] and ctx["_breakout_dir"] and in_wnd:
            params = compute_entry_params(ctx, self.db, self.cfg)
            if params["tradeable"]:
                ctx["action"] = "ENTER"
                pos_id = self.db.open_position(
                    ticker      = ctx["ticker"],
                    direction   = params["direction"],
                    entry_price = params["entry_price"],
                    stop_price  = params["stop_price"],
                    units       = params["units"],
                    targets     = params["targets"],
                    notes       = f"Engine auto-entry | VIX {ctx.get('vix','?')}",
                )
                _notify(self.cfg, f"ENTERED #{pos_id} {params['direction'].upper()} {ctx['ticker']} @ ${params['entry_price']:.2f}", "TRADE")
            else:
                ctx["action"] = "SKIP"
        else:
            ctx["action"] = "SKIP"

        self.db.log_decision(ctx)
        self._persist_session(today)

    def _trigger_analyser(self, session_date: str) -> None:
        """
        Launch session_analyser.py in a background subprocess after EOD.
        Ensures execution occurs strictly once per session and catches all errors.
        """
        if self.session.get("analyser_triggered", False):
            return

        try:
            import subprocess
            analyser = os.path.join(os.path.dirname(os.path.abspath(__file__)), "session_analyser.py")
            if not os.path.exists(analyser):
                log.warning("session_analyser.py not found at %s — skipping post-session analysis", analyser)
                return

            preset = self.cfg.get("llm_preset", "nuc-pair1")
            cmd = [
                sys.executable, analyser,
                "--date", session_date,
                "--db",   self.cfg["db_path"],
                "--preset", preset,
            ]
            if self.cfg.get("discord_webhook"):
                cmd += ["--discord", self.cfg["discord_webhook"]]

            log.info("Launching session analyser: %s", " ".join(cmd))
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.session["analyser_triggered"] = True
            _notify(self.cfg, f"Session complete — launching AI analysis ({preset}) for {session_date}", "INFO")
        except Exception as exc:
            log.error("Failed to launch session analyser subprocess: %s", exc, exc_info=True)

    def _check_daily_stop(self, today: str) -> None:
        if self.session["pnl_r"] / max(self.cfg["account_balance"] * self.cfg["risk_pct"], 1) <= -self.cfg["max_daily_loss_pct"]:
            self.session["halted"] = True
            _notify(self.cfg, "🛑 DAILY STOP TRIGGERED — session halted.", "HALT")

    def _persist_session(self, today: str) -> None:
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
        current_equity = self.db.get_current_equity(self.cfg["account_balance"])
        self.db.update_account_equity(current_equity, None)


# ===========================================================================
# ENTRY POINT
# ===========================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Autonomous paper trading engine")
    p.add_argument("--ticker",    default=CONFIG["ticker"])
    p.add_argument("--account",   type=float, default=CONFIG["account_balance"])
    p.add_argument("--orb",       default=CONFIG["orb_method"], choices=["5min","15min","30min"])
    p.add_argument("--risk",      type=float, default=CONFIG["risk_pct"])
    p.add_argument("--db",        default=CONFIG["db_path"])
    p.add_argument("--preset",    default=CONFIG["llm_preset"])
    p.add_argument("--poll",      type=int,   default=CONFIG["poll_interval_s"])
    return p.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    cfg  = {
        **CONFIG,
        "ticker":           args.ticker.upper(),
        "account_balance":  args.account,
        "orb_method":       args.orb,
        "risk_pct":         args.risk,
        "db_path":          args.db,
        "llm_preset":       args.preset,
        "poll_interval_s":  args.poll,
    }
    engine = TradingEngine(cfg)
    engine.run()
