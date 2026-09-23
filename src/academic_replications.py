"""
academic_replications.py
========================
Side-by-side simulation of five strategies on the same IS sessions.
Designed to answer the fundamental question: does our signal add value
beyond what the academic literature already validates on SPY?

STRATEGIES
──────────
1. Gao et al. (2018, JFE) — First-30-min → Last-30-min momentum
   The simplest possible academically-validated intraday strategy.
   No ORB, no VWAP. Just: if first 30-min positive → buy at 15:30.
   Published in Journal of Financial Economics (top-2 finance journal).
   Our lowest bar: any strategy we keep must beat this.

2. Zarattini 5-min ORB — SPY proxy of SSRN 4729284
   The original paper trades individual Stocks in Play (RVOL-filtered).
   Here we apply the same 5-min ORB mechanic directly to SPY.
   Honest caveat: without the RVOL filter on individual stocks, expected
   performance is lower than the paper's Sharpe 2.81.

3. Zarattini VWAP Momentum — SPY proxy of SSRN 4824172
   Intraday momentum with VWAP as a trailing stop.
   Paper: 1,985% total return 2007-2024, Sharpe 1.33, annualised 19.6%.
   Implemented here as: enter when price deviates > threshold from VWAP,
   trail stop through VWAP, exit EOD or when VWAP is recrossed.

4. Hybrid 15-min ORB + VWAP (our system)
   Delegates to historical_sim.simulate_session() — the same function
   the main engine uses. This is the control in the experiment.

5. Random bidirectional (null hypothesis)
   Delegates to random_baseline_sim.simulate_random_session().
   50/50 direction, random entry bar, same stops and commission.

USAGE
─────
  python academic_replications.py --db DATA/paper_account.db
  python academic_replications.py --report --db DATA/paper_account.db
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sqlite3
import sys
from datetime import date, datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

log = logging.getLogger("academic_replications")

COMMISSION_R = 0.016   # same as systematic strategy

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS academic_replication_results (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_ts      TEXT,
    strategy    TEXT,   -- strategy name
    citation    TEXT,   -- paper reference
    n_sessions  INTEGER,
    n_trades    INTEGER,
    win_rate    REAL,
    avg_win_r   REAL,
    avg_loss_r  REAL,
    ev          REAL,
    sharpe      REAL,
    total_r     REAL,
    max_dd_r    REAL,
    notes       TEXT
);
"""

# ---------------------------------------------------------------------------
# Strategy 1 — Gao et al. (2018, JFE)
# First-30-min → Last-30-min momentum
# ---------------------------------------------------------------------------

def simulate_gao_session(day_df: pd.DataFrame) -> dict | None:
    """
    Gao, Han, Li, Zhou (2018) — Market Intraday Momentum.
    Journal of Financial Economics.

    Signal: first 30-min return (09:30-10:00) predicts last 30-min return.
    Entry:  15:30 EST (one bar before close)
    Exit:   16:00 EST (market close)
    Direction: long if open-to-10:00 positive, short if negative.

    Note: this produces one trade every session with no position sizing
    complexity. Returns in percentage terms, converted to R using the
    ORB range as the normaliser for comparability across strategies.
    """
    try:
        bars = day_df.copy()
        bars.index = pd.to_datetime(bars.index)

        open_30 = bars.between_time("09:30", "10:00")
        last_30 = bars.between_time("15:30", "15:59")

        if open_30.empty or last_30.empty:
            return None

        open_price  = float(open_30.iloc[0]["Open"])
        close_10    = float(open_30.iloc[-1]["Close"])
        entry_price = float(last_30.iloc[0]["Open"])
        exit_price  = float(bars.iloc[-1]["Close"])

        if open_price <= 0:
            return None

        first_30_ret = (close_10 - open_price) / open_price
        if abs(first_30_ret) < 0.0001:   # no clear signal
            return None

        direction = "long" if first_30_ret > 0 else "short"

        # P&L in % terms
        raw_pct = (exit_price - entry_price) / entry_price
        if direction == "short":
            raw_pct = -raw_pct

        # Normalise to R using ORB range (for comparability)
        orb_bars = bars.between_time("09:30", "09:45")
        orb_range = (float(orb_bars["High"].max()) - float(orb_bars["Low"].min())) if not orb_bars.empty else None
        if orb_range and orb_range > 0:
            actual_r = round(raw_pct * entry_price / orb_range - COMMISSION_R, 4)
        else:
            actual_r = round(raw_pct * 100 - COMMISSION_R, 4)   # fallback: pct-based

        return {
            "direction":   direction,
            "entry_price": round(entry_price, 4),
            "exit_price":  round(exit_price,  4),
            "actual_r":    actual_r,
            "exit_reason": "eod",
            "first_30_ret": round(first_30_ret * 100, 3),
        }
    except Exception as e:
        log.debug("Gao session error: %s", e)
        return None


# ---------------------------------------------------------------------------
# Strategy 2 — Zarattini 5-min ORB (SPY proxy of SSRN 4729284)
# ---------------------------------------------------------------------------

def simulate_zarattini_5min_orb(
    day_df:       pd.DataFrame,
    target_rr:    float = 2.0,
) -> dict | None:
    """
    Zarattini, Barbon, Aziz (2024) — SSRN 4729284.
    5-minute ORB with breakout direction. No VWAP gate, no retest.

    IMPORTANT CAVEAT: the original paper applies this to RVOL-filtered
    individual stocks ('Stocks in Play'). This is a SPY proxy intended
    to test whether the ORB mechanism itself holds on an index ETF without
    the RVOL filter. Lower performance than the paper's Sharpe 2.81 is
    expected and appropriate. It is the ORB mechanism in isolation.
    """
    try:
        bars = day_df.copy()

        # 5-min ORB
        orb_end  = pd.Timestamp("09:35").time()
        orb_bars = bars[bars.index.time <= orb_end]
        if len(orb_bars) < 3:
            return None

        orb_high  = float(orb_bars["High"].max())
        orb_low   = float(orb_bars["Low"].min())
        orb_range = orb_high - orb_low
        if orb_range < 0.01:
            return None

        # Watch for breakout on bars after ORB
        post_orb = bars[bars.index.time > orb_end]
        cutoff   = pd.Timestamp("10:45").time()
        post_orb = post_orb[post_orb.index.time <= cutoff]

        direction    = None
        entry_price  = None
        entry_idx    = None

        for idx, bar in post_orb.iterrows():
            if float(bar["High"]) > orb_high:
                direction   = "long"
                entry_price = orb_high
                entry_idx   = idx
                break
            elif float(bar["Low"]) < orb_low:
                direction   = "short"
                entry_price = orb_low
                entry_idx   = idx
                break

        if direction is None or entry_idx is None:
            return None

        # Simulate from next bar
        stop_price   = entry_price - orb_range if direction == "long" else entry_price + orb_range
        target_price = entry_price + target_rr * orb_range if direction == "long" \
                       else entry_price - target_rr * orb_range

        result_r    = None
        exit_price  = None
        exit_reason = "eod"

        remaining = bars[bars.index > entry_idx]
        for _, bar in remaining.iterrows():
            h, l = float(bar["High"]), float(bar["Low"])
            if direction == "long":
                if l <= stop_price:
                    result_r = round(-1.0 - COMMISSION_R, 3)
                    exit_price = stop_price; exit_reason = "stop"; break
                if h >= target_price:
                    result_r = round(target_rr - COMMISSION_R, 3)
                    exit_price = target_price; exit_reason = "target"; break
            else:
                if h >= stop_price:
                    result_r = round(-1.0 - COMMISSION_R, 3)
                    exit_price = stop_price; exit_reason = "stop"; break
                if l <= target_price:
                    result_r = round(target_rr - COMMISSION_R, 3)
                    exit_price = target_price; exit_reason = "target"; break

        if result_r is None:
            eod = float(bars["Close"].iloc[-1])
            raw = ((eod - entry_price) / orb_range) * (1 if direction == "long" else -1)
            result_r   = round(raw - COMMISSION_R, 3)
            exit_price = eod

        return {
            "direction":   direction,
            "entry_price": round(entry_price, 4),
            "exit_price":  round(exit_price or 0, 4),
            "actual_r":    result_r,
            "exit_reason": exit_reason,
            "orb_range":   round(orb_range, 4),
        }
    except Exception as e:
        log.debug("Zarattini 5-min ORB error: %s", e)
        return None


# ---------------------------------------------------------------------------
# Strategy 3 — Zarattini VWAP Momentum (SPY proxy of SSRN 4824172)
# ---------------------------------------------------------------------------

def _compute_vwap(df: pd.DataFrame) -> pd.Series:
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    cv = df["Volume"].cumsum().replace(0, float("nan"))
    return (tp * df["Volume"]).cumsum() / cv


def simulate_zarattini_vwap_momentum(
    day_df:          pd.DataFrame,
    vwap_threshold:  float = 0.002,   # 0.2% deviation from VWAP to trigger entry
    entry_cutoff:    str   = "10:45",
) -> dict | None:
    """
    Zarattini, Aziz, Barbon (2024) — SSRN 4824172.
    Intraday momentum on SPY with VWAP as a trailing stop.
    Paper result: Sharpe 1.33, 19.6% annualised, 2007-2024.

    Entry: price deviates > vwap_threshold from intraday VWAP (demand/supply imbalance)
    Stop:  VWAP recrossed (trail)
    Exit:  VWAP crossback or EOD

    This is a simplified SPY proxy. The paper uses more sophisticated
    trailing stops and position management; this captures the core mechanism.
    """
    try:
        bars  = day_df.copy()
        vwap  = _compute_vwap(bars)

        cutoff = pd.Timestamp(entry_cutoff).time()
        entry_pool = bars[bars.index.time <= cutoff]

        direction    = None
        entry_price  = None
        entry_idx    = None
        entry_vwap   = None

        for idx, bar in entry_pool.iterrows():
            if idx not in vwap.index:
                continue
            vwap_val = float(vwap[idx])
            price    = float(bar["Close"])
            if vwap_val <= 0:
                continue

            deviation = (price - vwap_val) / vwap_val
            if deviation > vwap_threshold and direction is None:
                direction   = "long"
                entry_price = price
                entry_idx   = idx
                entry_vwap  = vwap_val
                break
            elif deviation < -vwap_threshold and direction is None:
                direction   = "short"
                entry_price = price
                entry_idx   = idx
                entry_vwap  = vwap_val
                break

        if direction is None or entry_price is None:
            return None

        # ORB range for R normalisation
        orb_end  = pd.Timestamp("09:45").time()
        orb_bars = bars[bars.index.time <= orb_end]
        orb_range = (float(orb_bars["High"].max()) - float(orb_bars["Low"].min())) if not orb_bars.empty else None
        if not orb_range or orb_range < 0.01:
            orb_range = entry_price * 0.002  # fallback: 0.2% of price

        # Trail stop: exit when price crosses back through VWAP
        result_r    = None
        exit_price  = None
        exit_reason = "eod"

        remaining = bars[bars.index > entry_idx]
        for idx2, bar in remaining.iterrows():
            if idx2 not in vwap.index:
                continue
            vwap_val = float(vwap[idx2])
            close    = float(bar["Close"])

            # VWAP trail stop hit
            if direction == "long" and close < vwap_val:
                exit_price  = close
                exit_reason = "vwap_cross"
                break
            elif direction == "short" and close > vwap_val:
                exit_price  = close
                exit_reason = "vwap_cross"
                break

        if exit_price is None:
            exit_price = float(bars["Close"].iloc[-1])

        raw_pct = (exit_price - entry_price) / entry_price
        if direction == "short":
            raw_pct = -raw_pct

        result_r = round(raw_pct * entry_price / orb_range - COMMISSION_R, 4)

        return {
            "direction":   direction,
            "entry_price": round(entry_price, 4),
            "exit_price":  round(exit_price,  4),
            "actual_r":    result_r,
            "exit_reason": exit_reason,
            "entry_vwap":  round(entry_vwap or 0, 4),
        }
    except Exception as e:
        log.debug("Zarattini VWAP error: %s", e)
        return None


# ---------------------------------------------------------------------------
# Strategy 4 — Hybrid 15-min ORB + VWAP (our system)
# ---------------------------------------------------------------------------

def simulate_hybrid_session(day_df: pd.DataFrame) -> dict | None:
    """
    Our current system: 15-min ORB + VWAP slope gate + retest mechanic.
    Delegates to historical_sim.simulate_session().
    Requires vwap pre-computation and returns first trade dict (not a list).
    """
    try:
        from historical_sim import simulate_session, _compute_vwap
        vwap   = _compute_vwap(day_df)
        trades = simulate_session(day_df, vwap=vwap, orb_method="15min")
        return trades[0] if trades else None
    except ImportError:
        log.warning("historical_sim not found — hybrid strategy skipped")
        return None
    except Exception as e:
        log.debug("Hybrid session error: %s", e)
        return None


# ---------------------------------------------------------------------------
# Strategy 5 — Random bidirectional (null hypothesis)
# ---------------------------------------------------------------------------

def simulate_random_session_wrapper(day_df: pd.DataFrame, rng: random.Random | None = None) -> dict | None:
    try:
        from random_baseline_sim import simulate_random_session
        return simulate_random_session(day_df, rng=rng)
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _metrics(rs: list[float], strategy: str) -> dict:
    if not rs:
        return {"strategy": strategy, "n": 0}
    wins   = [r for r in rs if r > 0]
    losses = [r for r in rs if r < 0]
    wr     = len(wins) / len(rs)
    aw     = sum(wins)   / len(wins)   if wins   else 0.0
    al     = abs(sum(losses)/len(losses)) if losses else 0.0
    ev     = (wr * aw) - ((1 - wr) * al)
    sd     = float(np.std(rs, ddof=1)) if len(rs) >= 2 else 0.0
    sharpe = round(float(np.mean(rs)) / sd, 4) if sd > 0 else None
    equity = np.cumsum([0.0] + rs)
    peak   = np.maximum.accumulate(equity)
    max_dd = round(float((equity - peak).min()), 4)
    return {
        "strategy":   strategy,
        "n":          len(rs),
        "win_rate":   round(wr, 4),
        "avg_win_r":  round(aw, 4),
        "avg_loss_r": round(al, 4),
        "ev":         round(ev, 4),
        "sharpe":     sharpe,
        "total_r":    round(sum(rs), 3),
        "max_dd_r":   max_dd,
    }


# ---------------------------------------------------------------------------
# Run all comparisons
# ---------------------------------------------------------------------------


def simulate_long_only_orb(
    day_df:    pd.DataFrame,
    orb_mins:  int   = 15,
    target_rr: float = 2.0,
) -> dict | None:
    """
    Long-only 15-min ORB on SPY.

    Independent review recommendation: SPY's structural upward drift and the
    difficulty of intraday short-selling on index ETFs suggest a long-only
    variant may outperform the bidirectional strategy.

    Only takes long entries (ORB breaks above the range). Skips sessions
    where the first break is to the downside.
    """
    try:
        bars     = day_df.copy()
        orb_end  = (pd.Timestamp("09:30") + pd.Timedelta(minutes=orb_mins)).time()
        orb_bars = bars[bars.index.time <= orb_end]
        if len(orb_bars) < 3:
            return None
        orb_high  = float(orb_bars["High"].max())
        orb_low   = float(orb_bars["Low"].min())
        orb_range = orb_high - orb_low
        if orb_range < 0.01:
            return None

        post_orb = bars[bars.index.time > orb_end]
        cutoff   = pd.Timestamp("10:45").time()
        post_orb = post_orb[post_orb.index.time <= cutoff]

        entry_price = entry_idx = None
        for idx, bar in post_orb.iterrows():
            if float(bar["High"]) > orb_high:
                entry_price = orb_high
                entry_idx   = idx
                break
            elif float(bar["Low"]) < orb_low:
                return None   # First break is downside — skip (long-only)

        if entry_price is None:
            return None

        stop_price   = entry_price - orb_range
        target_price = entry_price + target_rr * orb_range
        result_r = exit_price = None
        exit_reason = "eod"

        for _, bar in bars[bars.index > entry_idx].iterrows():
            h, l = float(bar["High"]), float(bar["Low"])
            if l <= stop_price:
                result_r = round(-1.0 - COMMISSION_R, 3)
                exit_price = stop_price; exit_reason = "stop"; break
            if h >= target_price:
                result_r = round(target_rr - COMMISSION_R, 3)
                exit_price = target_price; exit_reason = "target"; break

        if result_r is None:
            eod       = float(bars["Close"].iloc[-1])
            result_r  = round((eod - entry_price) / orb_range - COMMISSION_R, 3)
            exit_price = eod

        return {"direction": "long", "entry_price": round(entry_price, 4),
                "exit_price": round(exit_price or 0, 4),
                "actual_r": result_r, "exit_reason": exit_reason}
    except Exception as e:
        log.debug("Long-only ORB error: %s", e)
        return None


def simulate_orb_no_retest(
    day_df:    pd.DataFrame,
    orb_mins:  int   = 15,
    target_rr: float = 2.0,
) -> dict | None:
    """
    15-min ORB without the retest mechanic — enter immediately on ORB break.

    Compares directly against the hybrid strategy (which includes retest).
    Independent review identified retest as potential adverse selection:
    the strongest trend days do not retest; the filter may be selecting
    for weak, choppy breakouts.

    If this strategy outperforms hybrid_15min_orb_vwap on IS data, the
    retest mechanic is harmful and should be removed.
    """
    try:
        bars     = day_df.copy()
        orb_end  = (pd.Timestamp("09:30") + pd.Timedelta(minutes=orb_mins)).time()
        orb_bars = bars[bars.index.time <= orb_end]
        if len(orb_bars) < 3:
            return None
        orb_high  = float(orb_bars["High"].max())
        orb_low   = float(orb_bars["Low"].min())
        orb_range = orb_high - orb_low
        if orb_range < 0.01:
            return None

        post_orb = bars[bars.index.time > orb_end]
        cutoff   = pd.Timestamp("10:45").time()
        post_orb = post_orb[post_orb.index.time <= cutoff]

        direction = entry_price = entry_idx = None
        for idx, bar in post_orb.iterrows():
            if float(bar["High"]) > orb_high:
                direction = "long"; entry_price = orb_high; entry_idx = idx; break
            elif float(bar["Low"]) < orb_low:
                direction = "short"; entry_price = orb_low; entry_idx = idx; break

        if direction is None:
            return None

        stop_price   = entry_price - orb_range if direction == "long" else entry_price + orb_range
        target_price = entry_price + target_rr * orb_range if direction == "long"                        else entry_price - target_rr * orb_range
        result_r = exit_price = None
        exit_reason = "eod"

        for _, bar in bars[bars.index > entry_idx].iterrows():
            h, l = float(bar["High"]), float(bar["Low"])
            if direction == "long":
                if l <= stop_price:
                    result_r = round(-1.0 - COMMISSION_R, 3)
                    exit_price = stop_price; exit_reason = "stop"; break
                if h >= target_price:
                    result_r = round(target_rr - COMMISSION_R, 3)
                    exit_price = target_price; exit_reason = "target"; break
            else:
                if h >= stop_price:
                    result_r = round(-1.0 - COMMISSION_R, 3)
                    exit_price = stop_price; exit_reason = "stop"; break
                if l <= target_price:
                    result_r = round(target_rr - COMMISSION_R, 3)
                    exit_price = target_price; exit_reason = "target"; break

        if result_r is None:
            eod      = float(bars["Close"].iloc[-1])
            raw      = ((eod - entry_price) / orb_range) * (1 if direction == "long" else -1)
            result_r = round(raw - COMMISSION_R, 3)
            exit_price = eod

        return {"direction": direction, "entry_price": round(entry_price, 4),
                "exit_price": round(exit_price or 0, 4),
                "actual_r": result_r, "exit_reason": exit_reason}
    except Exception as e:
        log.debug("No-retest ORB error: %s", e)
        return None


STRATEGIES = {
    "gao_first_last_30min": {
        "fn": simulate_gao_session,
        "citation": "Gao, Han, Li, Zhou (2018) JFE — Market Intraday Momentum",
        "note": "First-30-min sign predicts last-30-min. Simplest academic benchmark. All SPY data.",
    },
    "zarattini_5min_orb_spy": {
        "fn": simulate_zarattini_5min_orb,
        "citation": "Zarattini, Barbon, Aziz (2024) SSRN 4729284 — ORB Stocks in Play (SPY proxy)",
        "note": "SPY proxy only. Original paper uses individual RVOL-filtered stocks; expected Sharpe lower than 2.81.",
    },
    "zarattini_vwap_momentum": {
        "fn": simulate_zarattini_vwap_momentum,
        "citation": "Zarattini, Aziz, Barbon (2024) SSRN 4824172 — Intraday SPY Momentum (Sharpe 1.33)",
        "note": "VWAP deviation entry + VWAP trail stop. Direct SPY strategy. Closest to published paper.",
    },
    "hybrid_15min_orb_vwap": {
        "fn": simulate_hybrid_session,
        "citation": "Current system — 15-min ORB + VWAP gate + retest (hybrid of 4729284 and 4824172)",
        "note": "Our strategy. Must beat Gao (baseline) and approach VWAP momentum to justify complexity.",
    },
    "random_bidirectional": {
        "fn": lambda df: simulate_random_session_wrapper(df, rng=random.Random(42)),
        "citation": "Null hypothesis — random 50/50 entry, same stops",
        "note": "The floor. Any strategy that does not significantly beat this has no genuine edge.",
    },
    "long_only_15min_orb": {
        "fn": simulate_long_only_orb,
        "citation": "Independent review recommendation — Long-only SPY ORB (structural upward drift hypothesis)",
        "note": "Only takes long entries. If this beats bidirectional, prune the short side immediately.",
    },
    "orb_15min_no_retest": {
        "fn": simulate_orb_no_retest,
        "citation": "Retest adverse selection test — 15-min ORB without hold requirement",
        "note": "If this beats hybrid_15min_orb_vwap, the retest mechanic is harmful and should be removed.",
    },
}


def run_all_comparisons(
    db_path:   str,
    store_db:  str | None = None,
) -> list[dict]:
    """
    Run all five strategies on the same IS sessions.
    Returns list of metrics dicts, one per strategy.
    """
    from random_baseline_sim import _load_sessions
    import time as _time

    with sqlite3.connect(db_path) as c:
        c.executescript(SCHEMA)

    sessions = _load_sessions(db_path, store_db)
    if not sessions:
        return [{"error": "No session bar data. Run historical_sim first."}]

    log.info("Academic replications: %d strategies × %d sessions", len(STRATEGIES), len(sessions))
    ts = datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds")
    all_results: list[dict] = []

    for strat_key, strat_cfg in STRATEGIES.items():
        t0 = _time.monotonic()
        fn = strat_cfg["fn"]
        rs: list[float] = []

        for day_df in sessions:
            try:
                result = fn(day_df)
                if result and result.get("actual_r") is not None:
                    rs.append(float(result["actual_r"]))
            except Exception as e:
                log.debug("%s session error: %s", strat_key, e)

        m = _metrics(rs, strat_key)
        m["citation"]  = strat_cfg["citation"]
        m["note"]      = strat_cfg["note"]
        m["elapsed_s"] = round(_time.monotonic() - t0, 1)
        all_results.append(m)

        # Persist to DB
        with sqlite3.connect(db_path) as c:
            c.execute(
                """INSERT INTO academic_replication_results
                   (run_ts,strategy,citation,n_sessions,n_trades,win_rate,
                    avg_win_r,avg_loss_r,ev,sharpe,total_r,max_dd_r,notes)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (ts, strat_key, strat_cfg["citation"], len(sessions),
                 m.get("n",0), m.get("win_rate"), m.get("avg_win_r"),
                 m.get("avg_loss_r"), m.get("ev"), m.get("sharpe"),
                 m.get("total_r"), m.get("max_dd_r"), strat_cfg["note"])
            )
        log.info("  %-30s  WR=%.0f%%  EV=%.4fR  Sharpe=%s  (%ds)",
                 strat_key, (m.get("win_rate") or 0)*100,
                 m.get("ev") or 0,
                 f"{m['sharpe']:.3f}" if m.get("sharpe") else "N/A",
                 m["elapsed_s"])

    return all_results


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(db_path: str) -> None:
    with sqlite3.connect(db_path) as c:
        c.row_factory = sqlite3.Row
        rows = c.execute(
            """SELECT * FROM academic_replication_results
               WHERE run_ts = (SELECT MAX(run_ts) FROM academic_replication_results)
               ORDER BY COALESCE(sharpe, -99) DESC"""
        ).fetchall()

    if not rows:
        print("No replication results yet. Run: python academic_replications.py")
        return

    print(f"\n{'═'*72}")
    print(f"  ACADEMIC STRATEGY COMPARISON  ({rows[0]['run_ts'][:19]})")
    print(f"{'═'*72}")
    print(f"  {'Strategy':<30} {'Trades':>6} {'WR':>6} {'EV/R':>7} {'Sharpe':>7} {'MaxDD':>7}")
    print(f"  {'─'*30} {'─'*6} {'─'*6} {'─'*7} {'─'*7} {'─'*7}")

    for r in rows:
        sharpe_str = f"{r['sharpe']:+.3f}" if r['sharpe'] else "  N/A"
        print(f"  {r['strategy']:<30} {r['n_trades']:>6,} "
              f"{(r['win_rate'] or 0):>5.0%} "
              f"{(r['ev'] or 0):>+7.4f} "
              f"{sharpe_str:>7} "
              f"{(r['max_dd_r'] or 0):>+7.2f}R")

    print()
    # Key questions
    results = {r['strategy']: dict(r) for r in rows}
    hybrid  = results.get("hybrid_15min_orb_vwap", {})
    gao     = results.get("gao_first_last_30min", {})
    rnd     = results.get("random_bidirectional", {})
    vwap    = results.get("zarattini_vwap_momentum", {})

    if hybrid.get("sharpe") and gao.get("sharpe"):
        beats_gao = hybrid["sharpe"] > gao["sharpe"]
        print(f"  Hybrid vs Gao baseline:  {'✅ BEATS' if beats_gao else '❌ FAILS TO BEAT'} "
              f"({hybrid['sharpe']:+.3f} vs {gao['sharpe']:+.3f})")

    if hybrid.get("sharpe") and vwap.get("sharpe"):
        beats_vwap = hybrid["sharpe"] > vwap["sharpe"]
        print(f"  Hybrid vs VWAP momentum: {'✅ BEATS' if beats_vwap else '⚠  BELOW'} "
              f"({hybrid['sharpe']:+.3f} vs {vwap['sharpe']:+.3f})")

    if hybrid.get("sharpe") and rnd.get("sharpe"):
        z = None
        if rnd.get("sharpe"):
            z = hybrid["sharpe"] - rnd["sharpe"]
        print(f"  Hybrid vs random null:   ✅ +{z:.3f} Sharpe above null" if z else "")

    print(f"\n  Academic citations:")
    for r in rows:
        print(f"    [{r['strategy'][:25]}] {r['citation'][:60]}")
        if r.get("notes"):
            print(f"      Note: {r['notes'][:65]}")
    print(f"{'═'*72}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")
    p = argparse.ArgumentParser(description="Academic strategy replication suite")
    p.add_argument("--db",       default="DATA/paper_account.db")
    p.add_argument("--store-db", default=None,
                   help="Path to market_data.db (1-min bars). Auto-detected if omitted.")
    p.add_argument("--report",   action="store_true")
    args = p.parse_args()

    if args.report:
        print_report(args.db)
    else:
        results = run_all_comparisons(args.db, store_db=args.store_db)
        print_report(args.db)
        print(f"Results saved to {args.db} → academic_replication_results table")
