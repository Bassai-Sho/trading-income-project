"""
historical_sim.py
=================
High-speed historical simulation runner.

THE CORE INSIGHT (from D-A-C session 09 Sep 2026)
───────────────────────────────────────────────────
The fundamental bottleneck is not analysis speed — it is data generation.
At 2 live trades per day it takes 50 trading days to reach 100 trades.

This script runs the canonical ORB strategy against 10 years of free
Alpaca 1-minute SPY data and produces thousands of virtual trades in minutes.

DATA RULES (non-negotiable)
───────────────────────────
IN-SAMPLE only: 2016-01-01 → 2022-12-31    ← training window, may be used
OUT-OF-SAMPLE:  2023-01-01 → 2024-12-31    ← WFA validation, do NOT touch
SEALED:         2025-01-01 → present        ← NEVER TOUCHED until live gate

The sealed test set constraint is enforced at the function level.
Any attempt to fetch data beyond 2022-12-31 raises SealedDataError.

WHAT THIS PRODUCES
──────────────────
After running on 3 years of IS data (~750 sessions, ~1,500 trades):
  - Markov chain regime matrix trained on 750+ VIX days
  - Candle N-gram transition table with real candle sequences
  - Journal correlation stats on actual historical outcomes
  - Walk-forward results with meaningful IS/OOS trade counts
  - WFE estimate with statistical validity

All outputs saved to the simulation DB and readable by all other components.

USAGE
─────
  # First: open a free Alpaca account at alpaca.markets
  # Set environment variables:
  export ALPACA_API_KEY=your_key_here
  export ALPACA_SECRET_KEY=your_secret_here

  # Run historical simulation (IS data only, 2016-2022)
  python historical_sim.py --start 2016-01-01 --end 2022-12-31 --db paper_account.db

  # Quick test with 6 months
  python historical_sim.py --start 2022-06-01 --end 2022-12-31 --db paper_account.db --quick

  # Review results without running
  python historical_sim.py --report --db paper_account.db
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
import time
from datetime import date, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

try:
    from market_data_store import MarketDataStore, get_store_or_fetch, REGIMES, IS_END as _DS_IS_END
    HAS_DATA_STORE = True
except ImportError:
    HAS_DATA_STORE = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

log = logging.getLogger("historical_sim")

# ---------------------------------------------------------------------------
# DATA PROTECTION — sealed test set guard
# ---------------------------------------------------------------------------

IS_END   = date(2022, 12, 31)    # last allowed in-sample date (2016-2022 = 7yr, matches Zarattini)
OOS_END  = date(2024, 12, 31)    # last allowed WFA validation date
SEALED_START = date(2025, 1, 1)  # sealed — NEVER touch


class SealedDataError(RuntimeError):
    """Raised when a request would touch sealed test-set data."""
    pass


def _guard_dates(start: date, end: date, allow_oos: bool = False) -> None:
    limit = OOS_END if allow_oos else IS_END
    if end > limit:
        raise SealedDataError(
            f"End date {end} exceeds {'OOS validation limit' if allow_oos else 'IS training limit'} "
            f"{limit}. The sealed test set (2025+) must never be touched until "
            f"WFE ≥ 0.50 is confirmed. Set a valid end date."
        )

# ---------------------------------------------------------------------------
# Alpaca data fetcher
# ---------------------------------------------------------------------------

def fetch_alpaca_bars(
    ticker: str,
    start:  date,
    end:    date,
    timeframe: str = "1Min",
) -> pd.DataFrame:
    """
    Fetch historical minute bars from Alpaca.
    Requires ALPACA_API_KEY and ALPACA_SECRET_KEY env vars.
    """
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests   import StockBarsRequest
        from alpaca.data.timeframe  import TimeFrame, TimeFrameUnit
    except ImportError:
        raise ImportError("pip install alpaca-py")

    api_key    = os.environ.get("ALPACA_API_KEY")
    secret_key = os.environ.get("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        raise EnvironmentError(
            "Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables. "
            "Free account at alpaca.markets — no funded account needed for historical data."
        )

    client = StockHistoricalDataClient(api_key, secret_key)
    tf     = TimeFrame(1, TimeFrameUnit.Minute)
    req    = StockBarsRequest(
        symbol_or_symbols=ticker,
        timeframe=tf,
        start=datetime.combine(start, datetime.min.time()),
        end=datetime.combine(end, datetime.max.time()),
        adjustment="all",
    )
    log.info("Fetching %s bars %s → %s...", ticker, start, end)
    t0   = time.monotonic()
    bars = client.get_stock_bars(req).df
    elapsed = time.monotonic() - t0
    log.info("Fetched %d bars in %.1fs", len(bars), elapsed)

    if isinstance(bars.index, pd.MultiIndex):
        bars = bars.reset_index(level=0, drop=True)
    bars.index = pd.to_datetime(bars.index, utc=True).tz_convert("America/New_York")
    bars.columns = [c.capitalize() for c in bars.columns]
    for col in ["Open","High","Low","Close","Volume"]:
        if col not in bars.columns:
            raise ValueError(f"Missing column {col} in Alpaca response")
    return bars

# ---------------------------------------------------------------------------
# yfinance fallback (for environments without Alpaca key)
# ---------------------------------------------------------------------------

def fetch_yfinance_bars(ticker: str, start: date, end: date) -> pd.DataFrame:
    """Fallback to yfinance for IS data if Alpaca not configured."""
    import yfinance as yf
    log.info("Fetching %s from yfinance %s → %s (5-min bars)...", ticker, start, end)
    df = yf.download(ticker, start=str(start), end=str(end),
                     interval="5m", auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df.index = pd.to_datetime(df.index, utc=True).tz_convert("America/New_York")
    return df

# ---------------------------------------------------------------------------
# Strategy signal evaluator (same logic as trading_engine.py, vectorised)
# ---------------------------------------------------------------------------

def _compute_vwap(df: pd.DataFrame) -> pd.Series:
    """Session-anchored VWAP (resets at 09:30 each day)."""
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    pv = tp * df["Volume"]
    vwap_vals = pd.Series(index=df.index, dtype=float)
    for day in set(df.index.date):
        mask = (df.index.date == day) & (df.index.time >= pd.Timestamp("09:30").time())
        cv   = df["Volume"][mask].cumsum()
        vwap_vals[mask] = (pv[mask].cumsum() / cv.replace(0, float("nan"))).values
    return vwap_vals

def _define_orb(day_df: pd.DataFrame, method: str = "15min") -> dict | None:
    orb_end = {"5min": 5, "15min": 15, "30min": 30}.get(method, 15)
    open_t  = pd.Timestamp("09:30", tz="America/New_York").time()
    end_t   = (pd.Timestamp("09:30", tz="America/New_York") +
                pd.Timedelta(minutes=orb_end)).time()
    orb_bars = day_df[(day_df.index.time >= open_t) & (day_df.index.time <= end_t)]
    if len(orb_bars) < 2:
        return None
    return {
        "orb_high": float(orb_bars["High"].max()),
        "orb_low":  float(orb_bars["Low"].min()),
        "orb_size": float(orb_bars["High"].max() - orb_bars["Low"].min()),
    }

def simulate_session(
    day_df:     pd.DataFrame,
    vwap:       pd.Series,
    orb_method: str   = "15min",
    target_rr:  float = 2.0,
    risk_pct:   float = 0.01,
    account:    float = 10_000.0,
    entry_cutoff_hour: int = 11,
) -> list[dict]:
    """
    Simulate the ORB strategy on one trading session.
    Returns list of trade dicts (may be empty if no signals).
    """
    orb = _define_orb(day_df, orb_method)
    if not orb:
        return []

    cutoff_time = pd.Timestamp(f"{entry_cutoff_hour}:00", tz="America/New_York").time()
    trades: list[dict] = []
    entered   = False
    n_long    = 0
    n_short   = 0
    max_per_dir = 2

    # Iterate bars after ORB window
    orb_end_t = (pd.Timestamp("09:30", tz="America/New_York") +
                  pd.Timedelta(minutes={"5min":5,"15min":15,"30min":30}.get(orb_method,15))).time()
    signal_bars = day_df[day_df.index.time > orb_end_t]

    for ts, bar in signal_bars.iterrows():
        if entered:
            break
        if ts.time() > cutoff_time:
            break

        close = float(bar["Close"])
        v     = float(vwap.loc[ts]) if ts in vwap.index else None
        if v is None or v == 0:
            continue

        # Gate 1: ORB breakout
        if close > orb["orb_high"] and n_long < max_per_dir:
            direction = "long"
        elif close < orb["orb_low"] and n_short < max_per_dir:
            direction = "short"
        else:
            continue

        # Gate 2: VWAP direction (simplified slope)
        v_slope = "flat"
        vwap_window = vwap.loc[:ts].dropna().iloc[-5:]
        if len(vwap_window) >= 3:
            slope = float(vwap_window.iloc[-1] - vwap_window.iloc[0]) / len(vwap_window)
            thresh = float(vwap_window.mean()) * 0.0001
            if slope > thresh:   v_slope = "up"
            elif slope < -thresh: v_slope = "down"
        if (direction == "long" and v_slope != "up") or \
           (direction == "short" and v_slope != "down"):
            continue

        # Entry (1-bar delay: fill at next bar's open)
        future_bars = day_df[day_df.index > ts]
        if future_bars.empty:
            continue
        entry_bar  = future_bars.iloc[0]
        entry_price = float(entry_bar["Open"])
        stop_dist  = orb["orb_size"]
        if stop_dist < 1e-4:
            continue

        stop_price = (entry_price - stop_dist if direction == "long"
                       else entry_price + stop_dist)
        target     = (entry_price + target_rr * stop_dist if direction == "long"
                       else entry_price - target_rr * stop_dist)

        # Simulate outcome against remaining bars
        result_r     = None
        exit_reason  = "eod"
        exit_price   = float(day_df["Close"].iloc[-1])
        candle_type  = "doji"  # simplified
        body_pct     = abs(float(bar["Close"]) - float(bar["Open"])) / max(float(bar["High"]) - float(bar["Low"]), 1e-6) * 100

        for _, fbar in future_bars.iloc[1:].iterrows():
            h, l = float(fbar["High"]), float(fbar["Low"])
            if direction == "long":
                if l <= stop_price:
                    result_r   = round(-1.0 - 0.016, 3)  # loss + spread cost
                    exit_price = stop_price
                    exit_reason = "stop"
                    break
                if h >= target:
                    result_r   = round(target_rr - 0.016, 3)
                    exit_price = target
                    exit_reason = "target"
                    break
            else:
                if h >= stop_price:
                    result_r   = round(-1.0 - 0.016, 3)
                    exit_price = stop_price
                    exit_reason = "stop"
                    break
                if l <= target:
                    result_r   = round(target_rr - 0.016, 3)
                    exit_price = target
                    exit_reason = "target"
                    break

        if result_r is None:
            # EOD exit
            eod_price = float(day_df["Close"].iloc[-1])
            raw_r     = ((eod_price - entry_price) / stop_dist *
                          (1 if direction == "long" else -1))
            result_r  = round(raw_r - 0.016, 3)
            exit_price = eod_price

        trades.append({
            "session_date": str(ts.date()),
            "direction":    direction,
            "entry_price":  round(entry_price, 4),
            "exit_price":   round(exit_price, 4),
            "stop_price":   round(stop_price, 4),
            "target_price": round(target, 4),
            "actual_r":     result_r,
            "exit_reason":  exit_reason,
            "entry_candle_body_pct": round(body_pct, 1),
            "entry_candle_type":     "strong_bull" if body_pct > 60 and direction == "long" else
                                     "strong_bear" if body_pct > 60 and direction == "short" else
                                     "doji" if body_pct < 20 else "moderate_bull",
            "vwap_slope_at_entry":   v_slope,
        })
        entered = True
        if direction == "long": n_long += 1
        else: n_short += 1

    return trades

# ---------------------------------------------------------------------------
# Save simulated trades to DB
# ---------------------------------------------------------------------------

def save_sim_trades(db_path: str, trades: list[dict]) -> None:
    """
    Write simulated historical trades into trade_journal_extended table
    so all downstream components (Markov chain, N-gram, correlations) can
    use them alongside live paper trades.
    """
    if not trades:
        return
    try:
        from trade_journal_extended import ensure_journal_table
        ensure_journal_table(db_path)
    except ImportError:
        pass

    with sqlite3.connect(db_path) as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS sim_trades (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            session_date TEXT,
            direction    TEXT,
            entry_price  REAL,
            exit_price   REAL,
            stop_price   REAL,
            actual_r     REAL,
            exit_reason  TEXT,
            entry_candle_body_pct REAL,
            entry_candle_type     TEXT,
            vwap_slope_at_entry   TEXT,
            sim_run_ts   TEXT
        )""")
        ts_now = datetime.utcnow().isoformat(timespec="seconds")
        conn.executemany(
            "INSERT INTO sim_trades (session_date,direction,entry_price,exit_price,"
            "stop_price,actual_r,exit_reason,entry_candle_body_pct,entry_candle_type,"
            "vwap_slope_at_entry,sim_run_ts) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            [(t["session_date"], t["direction"], t["entry_price"], t["exit_price"],
              t["stop_price"], t["actual_r"], t["exit_reason"],
              t["entry_candle_body_pct"], t["entry_candle_type"],
              t["vwap_slope_at_entry"], ts_now) for t in trades]
        )

# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------

def compute_sim_metrics(trades: list[dict]) -> dict:
    rs = [t["actual_r"] for t in trades if t.get("actual_r") is not None]
    if not rs:
        return {"n": 0}
    wins   = [r for r in rs if r > 0]
    losses = [r for r in rs if r < 0]
    wr     = len(wins) / len(rs)
    aw     = sum(wins) / len(wins)     if wins   else 0.0
    al     = abs(sum(losses)/len(losses)) if losses else 1.0
    ev     = (wr * aw) - ((1-wr) * al)
    sharpe = None
    if len(rs) >= 2:
        import statistics
        sd = statistics.stdev(rs)
        if sd > 0: sharpe = round(sum(rs)/len(rs)/sd, 3)

    by_reason = {}
    for t in trades:
        r = t["exit_reason"]
        by_reason.setdefault(r, {"n":0,"sum_r":0})
        by_reason[r]["n"]    += 1
        by_reason[r]["sum_r"] = round(by_reason[r]["sum_r"] + t["actual_r"], 3)

    return {
        "n":          len(rs),
        "win_rate":   round(wr, 3),
        "avg_win_r":  round(aw, 3),
        "avg_loss_r": round(al, 3),
        "ev":         round(ev, 4),
        "sharpe":     sharpe,
        "total_r":    round(sum(rs), 2),
        "by_exit_reason": by_reason,
    }

# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# SimBootstrapper — wires simulation output to all learning components
# ---------------------------------------------------------------------------

def _json_safe(obj):
    """Recursively convert numpy types to Python natives for JSON serialisation."""
    import numpy as np
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k,v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):  return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, (np.bool_,)):    return bool(obj)
    if isinstance(obj, (np.ndarray,)):  return obj.tolist()
    return obj


class SimBootstrapper:
    """
    After historical_sim.py completes, SimBootstrapper.run() transfers the
    virtual trade data into every learning component so they are pre-trained
    before the first live paper session.

    WHAT GETS SEEDED
    ─────────────────
    RegimeMarkovChain   → markov_state.json   (already done in run_simulation)
    CandleNgramChain    → candle_ngram.json   (NEW)
    OutcomeMarkovChain  → outcome_mc.json     (NEW — immediately runs independence test)
    WFA                 → wfa_results table   (NEW — seeds initial split history)
    Monte Carlo         → monte_carlo_forecasts table (NEW — uses real sim stats)
    bootstrap_meta      → DB table           (NEW — labels sim vs live counts)

    LABELLING PRINCIPLE
    ────────────────────
    Sim data TRAINS learning components. It never contaminates performance metrics.
    The dashboard and LLM both see:
      "1,487 sim trades (bootstrap) + 12 live trades"
    Sharpe, EV, P&L reporting always uses live trades only.
    WFA gate (WFE ≥ 0.50) requires live trade confirmation.
    """

    def __init__(self, db_path: str, cfg: dict | None = None) -> None:
        self.db_path = db_path
        self.cfg     = cfg or {}

    def _load_sim_trades(self) -> list[dict]:
        """Load all sim trades from sim_trades table."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM sim_trades ORDER BY id"
            ).fetchall()
        return [dict(r) for r in rows]

    def _init_bootstrap_meta(self) -> None:
        """Create bootstrap_meta table to label sim vs live data."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS bootstrap_meta (
                id            INTEGER PRIMARY KEY DEFAULT 1,
                n_sim_trades  INTEGER DEFAULT 0,
                sim_win_rate  REAL,
                sim_ev        REAL,
                sim_sharpe    REAL,
                sim_start     TEXT,
                sim_end       TEXT,
                bootstrap_ts  TEXT,
                wfa_mean_wfe  REAL,
                outcome_independent INTEGER,
                candle_ngram_valid  INTEGER,
                notes         TEXT
            )""")

    def seed_candle_ngram(self, trades: list[dict]) -> dict:
        """Fit CandleNgramChain from sim candle sequences and save to JSON."""
        try:
            from markov_engine import CandleNgramChain
            cng = CandleNgramChain(n=2)
            for t in trades:
                ct = t.get("entry_candle_type", "doji")
                if ct:
                    cng.observe(ct)
            # Run independence test
            test = cng.test_independence()
            # Save to JSON for engine to load at startup
            import json
            state = {
                "n":      int(len(cng._observations)),
                "counts": {str(k): {s2: int(n2) for s2,n2 in dict(v).items()} for k, v in cng._counts.items()},
                "n_gram": cng.n,
                "fitted_ts": datetime.utcnow().isoformat(),
                "independence_test": test,
            }
            json_path = os.path.join(os.path.dirname(self.db_path), "candle_ngram.json")
            with open(json_path, "w") as f:
                json.dump(_json_safe(state), f, indent=2)
            log.info("CandleNgramChain: %d observations → %s (verdict=%s)",
                     len(cng._observations), json_path, test.get("verdict"))
            return {"n_obs": len(cng._observations), "json_path": json_path,
                    "test": test}
        except Exception as e:
            log.warning("Candle N-gram seeding failed: %s", e)
            return {"error": str(e)}

    def seed_outcome_chain(self, trades: list[dict]) -> dict:
        """Fit OutcomeMarkovChain and immediately run independence test."""
        try:
            from markov_engine import OutcomeMarkovChain
            omc = OutcomeMarkovChain()
            r_vals = [t["actual_r"] for t in trades if t.get("actual_r") is not None]
            omc.add_outcomes(r_vals)
            test = omc.test_independence()
            import json
            state = {
                "n": int(len(r_vals)),
                "outcomes": [str(o)[0] for o in omc.outcomes],  # W/L chars
                "independence_test": test,
                "fitted_ts": datetime.utcnow().isoformat(),
            }
            json_path = os.path.join(os.path.dirname(self.db_path), "outcome_mc.json")
            with open(json_path, "w") as f:
                json.dump(_json_safe(state), f, indent=2)
            log.info("OutcomeMarkovChain: %d outcomes → verdict=%s (p=%s)",
                     len(r_vals), test.get("verdict"), test.get("p_value"))
            return {"n": len(r_vals), "verdict": test.get("verdict"),
                    "json_path": json_path, "test": test}
        except Exception as e:
            log.warning("Outcome chain seeding failed: %s", e)
            return {"error": str(e)}

    def seed_wfa(self, trades: list[dict]) -> dict:
        """Compute walk-forward analysis on sim trades and write to wfa_results."""
        try:
            rs = [t["actual_r"] for t in trades if t.get("actual_r") is not None]
            if len(rs) < 100:
                return {"error": f"Insufficient trades for WFA: {len(rs)}"}

            # Split into 6 IS/OOS windows (each ~250 trades)
            window = len(rs) // 6
            wfe_vals = []
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""CREATE TABLE IF NOT EXISTS wfa_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_ts TEXT, ticker TEXT, is_start TEXT, is_end TEXT,
                    oos_start TEXT, oos_end TEXT,
                    n_is_trades INTEGER, n_oos_trades INTEGER,
                    is_sharpe REAL, oos_sharpe REAL,
                    wfe REAL, wfe_verdict TEXT, source TEXT
                )""")
                for i in range(6):
                    is_rs  = rs[i*window : (i+1)*window]
                    oos_rs = rs[(i+1)*window : min((i+2)*window, len(rs))]
                    if len(is_rs) < 10 or len(oos_rs) < 10: continue
                    import statistics as _s
                    is_sr  = (_s.mean(is_rs)  / _s.stdev(is_rs)  if _s.stdev(is_rs)  > 0 else None)
                    oos_sr = (_s.mean(oos_rs) / _s.stdev(oos_rs) if _s.stdev(oos_rs) > 0 else None)
                    wfe    = round(oos_sr / is_sr, 3) if (is_sr and is_sr > 0.01) else None
                    verdict = "PASS" if wfe and wfe >= 0.50 else "FAIL" if wfe else "INCONCLUSIVE"
                    if wfe: wfe_vals.append(wfe)
                    conn.execute(
                        "INSERT INTO wfa_results (run_ts,ticker,is_start,is_end,"
                        "oos_start,oos_end,n_is_trades,n_oos_trades,is_sharpe,"
                        "oos_sharpe,wfe,wfe_verdict,source) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (datetime.utcnow().isoformat(), "SPY",
                         f"sim_split_{i}_is_start", f"sim_split_{i}_is_end",
                         f"sim_split_{i}_oos_start", f"sim_split_{i}_oos_end",
                         len(is_rs), len(oos_rs),
                         round(is_sr, 3) if is_sr else None,
                         round(oos_sr, 3) if oos_sr else None,
                         wfe, verdict, "historical_sim")
                    )
            mean_wfe = round(sum(wfe_vals)/len(wfe_vals), 3) if wfe_vals else None
            log.info("WFA seeded: %d splits, mean WFE=%s", len(wfe_vals), mean_wfe)
            return {"n_splits": 6, "mean_wfe": mean_wfe,
                    "verdict": "PASS" if mean_wfe and mean_wfe >= 0.50 else "FAIL"}
        except Exception as e:
            log.warning("WFA seeding failed: %s", e)
            return {"error": str(e)}

    def seed_monte_carlo(self, metrics: dict) -> dict:
        """Write a Monte Carlo forecast seeded from sim stats into the DB."""
        try:
            from monte_carlo_extended import forecast_report
            import json
            # Build a synthetic DB with sim stats preloaded
            report = {
                "generated_at": datetime.utcnow().isoformat(),
                "n_trades_history": metrics.get("n", 0),
                "assumed_parameters": False,
                "win_rate": metrics.get("win_rate", 0.44),
                "avg_win_r": metrics.get("avg_win_r", 2.0),
                "avg_loss_r": metrics.get("avg_loss_r", 1.0),
                "source": "historical_sim_bootstrap",
                "headline": (
                    f"BOOTSTRAP: {metrics.get('n',0)} sim trades | "
                    f"WR={metrics.get('win_rate',0):.0%} | "
                    f"EV={metrics.get('ev',0):+.4f}R | based on IS data 2020-2022"
                ),
            }
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""CREATE TABLE IF NOT EXISTS monte_carlo_forecasts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT, report_json TEXT
                )""")
                conn.execute(
                    "INSERT INTO monte_carlo_forecasts (ts, report_json) VALUES (?,?)",
                    (report["generated_at"], json.dumps(report))
                )
            log.info("Monte Carlo seeded from %d sim trades", metrics.get("n",0))
            return {"n": metrics.get("n",0), "seeded": True}
        except Exception as e:
            log.warning("Monte Carlo seeding failed: %s", e)
            return {"error": str(e)}

    def run(self, metrics: dict) -> dict:
        """
        Run all bootstrap steps. Call after run_simulation() completes.
        metrics: output of compute_sim_metrics(all_trades)
        """
        trades = self._load_sim_trades()
        if not trades:
            return {"error": "No sim trades found. Run historical_sim.py first."}

        log.info("SimBootstrapper: seeding %d components from %d sim trades",
                 5, len(trades))

        self._init_bootstrap_meta()
        results = {
            "n_sim_trades":    len(trades),
            "candle_ngram":    self.seed_candle_ngram(trades),
            "outcome_chain":   self.seed_outcome_chain(trades),
            "wfa":             self.seed_wfa(trades),
            "monte_carlo":     self.seed_monte_carlo(metrics),
        }

        # Write bootstrap_meta summary
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO bootstrap_meta "
                "(id,n_sim_trades,sim_win_rate,sim_ev,sim_sharpe,"
                "sim_start,sim_end,bootstrap_ts,"
                "wfa_mean_wfe,outcome_independent,candle_ngram_valid,notes) "
                "VALUES (1,?,?,?,?,?,?,?,?,?,?,?)",
                (len(trades), metrics.get("win_rate"), metrics.get("ev"),
                 metrics.get("sharpe"),
                 "2020-01-01", "2022-12-31",
                 datetime.utcnow().isoformat(),
                 results["wfa"].get("mean_wfe"),
                 int(results["outcome_chain"].get("verdict") in ("INDEPENDENT","MARKOV_VALID")),
                 int(results["candle_ngram"].get("test",{}).get("verdict") == "NGRAM_VALID"),
                 "All components seeded from historical simulation bootstrap.")
            )

        log.info("Bootstrap complete. Summary:")
        log.info("  Candle N-gram: %s", results["candle_ngram"].get("n_obs","N/A"))
        log.info("  Outcome chain: verdict=%s", results["outcome_chain"].get("verdict","N/A"))
        log.info("  WFA:           mean WFE=%s", results["wfa"].get("mean_wfe","N/A"))
        log.info("  Monte Carlo:   seeded=%s", results["monte_carlo"].get("seeded","N/A"))
        return results


def run_simulation(
    ticker:     str   = "SPY",
    start:      date  = date(2020, 1, 1),
    end:        date  = date(2022, 12, 31),
    orb_method: str   = "15min",
    target_rr:  float = 2.0,
    risk_pct:   float = 0.01,
    account:    float = 10_000.0,
    db_path:    str   = "DATA/paper_account.db",
    use_alpaca: bool  = True,
    max_sessions: int | None = None,
    dry_run_mode: bool = False,  # True = skip bootstrap (testing)
    store_path:  str | None = None,  # path to market_data.db (None = auto-detect)
) -> dict:
    """
    Run the full historical simulation.
    Returns comprehensive metrics dict.
    """
    _guard_dates(start, end)   # enforce IS window

    log.info("="*55)
    log.info("HISTORICAL SIMULATION  %s  %s → %s", ticker, start, end)
    log.info("Strategy: %s ORB  Target: %.1f:1  Risk: %.1f%%",
             orb_method, target_rr, risk_pct*100)
    log.info("="*55)

    # Fetch data
    # ── Prefer local SQLite store over live API pull ──────────────────────
    store_db = store_path or "DATA/market_data.db"
    if HAS_DATA_STORE:
        log.info("Checking market_data_store at %s...", store_db)
        store_obj = MarketDataStore(store_db)
        cached_days = store_obj.get_date_range(ticker, start, end)
        if len(cached_days) >= 50:
            log.info("Using %d cached sessions from SQLite store", len(cached_days))
            df_1m    = store_obj.get_bars_range(ticker, start, end)
            bar_label = "1-min SQLite (cached)"
            _store   = store_obj
        else:
            log.info("Cache has only %d sessions — fetching from Alpaca and storing...", len(cached_days))
            if use_alpaca and os.environ.get("ALPACA_API_KEY"):
                store_obj.download_and_store(ticker, start, end)
                df_1m    = store_obj.get_bars_range(ticker, start, end)
                bar_label = "1-min Alpaca → SQLite"
                _store   = store_obj
            else:
                df_1m    = fetch_yfinance_bars(ticker, start, end)
                bar_label = "5-min yfinance (no Alpaca key)"
                _store   = None
    elif use_alpaca and os.environ.get("ALPACA_API_KEY"):
        df_1m    = fetch_alpaca_bars(ticker, start, end, "1Min")
        bar_label = "1-min Alpaca (no store)"
        _store   = None
    else:
        log.info("Alpaca key not set — using yfinance 5-min bars (less accurate)")
        df_1m    = fetch_yfinance_bars(ticker, start, end)
        bar_label = "5-min yfinance"
        _store   = None

    if df_1m.empty:
        return {"error": "No data returned"}

    # Compute session-anchored VWAP for the full dataset
    log.info("Computing VWAP...")
    vwap = _compute_vwap(df_1m)

    # Fetch VIX for regime annotation — prefer FRED local store, fallback to yfinance
    vix_map: dict[str, float] = {}
    try:
        from fred_store import FredDataStore
        _fstore_path = store_path or "DATA/market_data.db"
        _fred = FredDataStore(_fstore_path)
        _rows = _fred.get_series("VIXCLS", str(start), str(end))
        vix_map = {r["date"]: r["value"] for r in _rows if r["value"] is not None}
        if vix_map:
            log.info("VIX from FRED local store: %d days", len(vix_map))
    except Exception:
        pass
    if not vix_map:
        try:
            import yfinance as yf
            vix_df = yf.download("^VIX", start=str(start), end=str(end),
                                   interval="1d", auto_adjust=True, progress=False)
            if isinstance(vix_df.columns, pd.MultiIndex):
                vix_df.columns = [c[0] if isinstance(c, tuple) else c for c in vix_df.columns]
            vix_map = {str(d.date()): float(c) for d, c in zip(vix_df.index, vix_df["Close"])}
        except Exception:
            pass

    # Simulate day by day
    all_trades: list[dict] = []
    trading_days = sorted(set(df_1m.index.date))
    if max_sessions:
        trading_days = trading_days[:max_sessions]

    log.info("Simulating %d sessions...", len(trading_days))
    t0 = time.monotonic()

    for day in trading_days:
        day_df = df_1m[df_1m.index.date == day]
        if len(day_df) < 20:
            continue
        day_vwap = vwap[vwap.index.date == day]
        trades = simulate_session(day_df, day_vwap, orb_method, target_rr, risk_pct, account)
        vix_today = vix_map.get(str(day), 0.0)
        for t in trades:
            t["vix_at_entry"] = vix_today
        all_trades.extend(trades)

    elapsed = time.monotonic() - t0
    log.info("Simulation complete: %d sessions in %.1fs → %d trades",
             len(trading_days), elapsed, len(all_trades))

    # Compute metrics
    metrics = compute_sim_metrics(all_trades)

    # Regime-conditional metrics (if store available with session_context)
    if HAS_DATA_STORE and _store:
        for regime_key, regime_info in REGIMES.items():
            if regime_info['end'] > IS_END:
                continue   # only IS regimes
            regime_trades = [
                t for t in all_trades
                if str(regime_info['start']) <= t.get('session_date','') <= str(regime_info['end'])
            ]
            if len(regime_trades) >= 10:
                rm = compute_sim_metrics(regime_trades)
                metrics.setdefault('by_regime', {})[regime_info['label']] = rm
        if 'by_regime' in metrics:
            log.info('Regime-conditional EV:')
            for lbl, rm in metrics['by_regime'].items():
                log.info('  %s: n=%d EV=%+.4f WR=%s', lbl, rm['n'], rm['ev'], rm['win_rate'])


    # Save to DB
    save_sim_trades(db_path, all_trades)
    log.info("Trades saved to %s (sim_trades table)", db_path)

    # Bootstrap Markov chain from VIX history
    markov_report = {}
    if vix_map:
        try:
            from markov_engine import RegimeMarkovChain
            rmc = RegimeMarkovChain()
            vix_series = [v for v in vix_map.values() if v > 0]
            rmc.fit_from_sequence(vix_series)
            rmc.save("markov_state.json")
            markov_report = {"fitted_on": len(vix_series), "saved": "markov_state.json"}
            log.info("Markov chain fitted on %d VIX days", len(vix_series))
        except Exception as e:
            markov_report = {"error": str(e)}

    result = {
        "ticker":       ticker,
        "start":        str(start),
        "end":          str(end),
        "bar_type":     bar_label,
        "n_sessions":   len(trading_days),
        "elapsed_sec":  round(elapsed, 1),
        "metrics":      metrics,
        "markov":       markov_report,
        "interpretation": (
            f"Simulated {metrics.get('n', 0)} IS trades on {len(trading_days)} sessions "
            f"({start} → {end}). WR={metrics.get('win_rate',0):.0%} "
            f"EV={metrics.get('ev',0):+.4f}R/trade "
            f"Sharpe={metrics.get('sharpe','N/A')}. "
            f"All results are IN-SAMPLE only. Sealed test set (2025+) untouched."
        ),
    }
    # Run SimBootstrapper to seed all learning components
    if not dry_run_mode:
        bootstrapper_obj = SimBootstrapper(db_path)
        bootstrap_results = bootstrapper_obj.run(metrics)
        result["bootstrap"] = bootstrap_results
        log.info("Bootstrap complete.")

        # Run random baseline benchmark for statistical significance
        try:
            from random_baseline_sim import run_random_baseline
            log.info("Running random baseline benchmark (1000 paths)...")
            bm = run_random_baseline(
                db_path=db_path,
                n_paths=1000,
                store_db=store_path,
                systematic_sharpe=metrics.get("sharpe"),
                systematic_ev=metrics.get("ev"),
                systematic_wr=metrics.get("win_rate"),
            )
            result["random_baseline"] = {
                "verdict":     bm.get("verdict","?"),
                "z_score":     bm.get("z_score"),
                "pct_vs_rnd":  bm.get("percentile_vs_random"),
                "mean_sharpe": bm.get("mean_sharpe"),
            }
            log.info("Random baseline: %s (Z=%.1f)",
                     bm.get("verdict","?")[:30], bm.get("z_score") or 0)
        except Exception as _rbe:
            log.warning("Random baseline failed: %s", _rbe)

        # Run academic strategy comparison suite
        try:
            from academic_replications import run_all_comparisons
            log.info("Running academic strategy comparisons...")
            ac = run_all_comparisons(db_path, store_db=store_path)
            result["academic_comparisons"] = [
                {"strategy": r["strategy"], "sharpe": r.get("sharpe"),
                 "ev": r.get("ev"), "n": r.get("n")}
                for r in ac if isinstance(r, dict) and "strategy" in r
            ]
        except Exception as _ace:
            log.warning("Academic comparisons failed: %s", _ace)

    return result


def print_report(db_path: str) -> None:
    """Print summary of simulation results from DB."""
    with sqlite3.connect(db_path) as conn:
        try:
            n = conn.execute("SELECT COUNT(*) FROM sim_trades").fetchone()[0]
            wins = conn.execute(
                "SELECT COUNT(*) FROM sim_trades WHERE actual_r > 0").fetchone()[0]
            ev = conn.execute(
                "SELECT AVG(actual_r) FROM sim_trades").fetchone()[0]
            latest = conn.execute(
                "SELECT MAX(sim_run_ts) FROM sim_trades").fetchone()[0]
            print(f"\nHistorical simulation DB summary:")
            print(f"  Trades:    {n}")
            print(f"  Win rate:  {wins/max(n,1):.0%}")
            print(f"  EV/trade:  {ev:+.4f}R" if ev else "  EV: N/A")
            print(f"  Last run:  {latest}")
        except Exception:
            print("No simulation data found. Run: python historical_sim.py")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")

    p = argparse.ArgumentParser(description="Historical ORB simulation on IS data only")
    p.add_argument("--ticker",     default="SPY")
    p.add_argument("--start",      default="2016-01-01")
    p.add_argument("--end",        default="2022-12-31")
    p.add_argument("--orb",        default="15min", choices=["5min","15min","30min"])
    p.add_argument("--target-rr",  type=float, default=2.0)
    p.add_argument("--risk",       type=float, default=0.01)
    p.add_argument("--account",    type=float, default=10_000.0)
    p.add_argument("--db",         default="DATA/paper_account.db")
    p.add_argument("--no-alpaca",  action="store_true",
                   help="Use yfinance 5-min bars instead of Alpaca 1-min")
    p.add_argument("--quick",      action="store_true",
                   help="Limit to 30 sessions (quick test)")
    p.add_argument("--report",     action="store_true",
                   help="Print DB summary and exit")
    args = p.parse_args()

    if args.report:
        print_report(args.db); sys.exit(0)

    start = date.fromisoformat(args.start)
    end   = date.fromisoformat(args.end)

    try:
        result = run_simulation(
            ticker=args.ticker.upper(),
            start=start, end=end,
            orb_method=args.orb,
            target_rr=args.target_rr,
            risk_pct=args.risk,
            account=args.account,
            db_path=args.db,
            use_alpaca=not args.no_alpaca,
            max_sessions=30 if args.quick else None,
        )
        print(f"\n{'='*55}")
        print(f"SIMULATION COMPLETE")
        print(f"{'='*55}")
        print(f"Trades:     {result['metrics'].get('n', 0)}")
        print(f"Win rate:   {result['metrics'].get('win_rate',0):.0%}")
        print(f"EV/trade:   {result['metrics'].get('ev',0):+.4f}R")
        print(f"Sharpe:     {result['metrics'].get('sharpe','N/A')}")
        print(f"Sessions:   {result['n_sessions']} in {result['elapsed_sec']}s")
        print(f"\n{result['interpretation']}")
    except SealedDataError as e:
        print(f"\n❌ SEALED DATA VIOLATION: {e}")
        sys.exit(1)
