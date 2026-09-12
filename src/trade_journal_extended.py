"""
trade_journal_extended.py
=========================
Enhanced trade journal schema for the Trading Income Project.

RESEARCH BASIS
──────────────
Built from synthesis of:
  • Dr. Brett Steenbarger (trading psychology) — process vs outcome,
    emotional state before/during/after, frame of mind logging
  • Edgewonk (professional journal software) — custom statistics categories:
    Timeframe, Market Conditions, Price Patterns, Psychology/Feeling
  • TradeZella research — Rule Adherence Score must be > 75% before any
    edge analysis means anything; setup naming is the most critical field
  • Plancana day trading journal research — cortisol from a loss literally
    impairs next decision; emotional state tracking is not optional
  • Market Internals (TradingView community) — TICK, ADD, VOLD for
    breadth confirmation of directional bias

DESIGN PRINCIPLES
──────────────────
1. AUTO-CAPTURE everything the engine can compute (candle anatomy, time
   buckets, daily trend, volume ratio, distance from key levels)
2. MINIMAL HUMAN INPUT for psychology (1–5 scales, single-word enums,
   three one-sentence fields). Target: 90 seconds post-trade.
3. CORRELATION-READY tags — consistent enums, not free text, so the
   SessionLearner and LLM D-A-C can compute actual EV by category
4. PROCESS vs OUTCOME separation (Steenbarger) — grade the PROCESS
   independently of whether the trade made money

USAGE
─────
  from trade_journal_extended import (
      JournalEntry, build_auto_fields, add_journal_columns, get_journal_stats
  )

  # At entry: auto-populate mechanical fields
  auto = build_auto_fields(df_5m, entry_bar, orb_data, daily_df, cfg)
  
  # Post-trade: human fills psychology fields (dashboard form)
  # journal_entry saved to trade_journal_extended table in DB
"""

from __future__ import annotations

import math
import sqlite3
import statistics
from dataclasses import dataclass, field, asdict
from datetime import datetime, time as Time
from typing import Any

# ── Enumerations ──────────────────────────────────────────────────────────────

class EmotionalState:
    CALM     = "calm"        # ideal: focused, neutral
    ANXIOUS  = "anxious"     # fear of loss, hesitation
    FOMO     = "fomo"        # fear of missing out → chasing
    REVENGE  = "revenge"     # trading after a loss to recover
    EXCITED  = "excited"     # overconfident after a win
    BORED    = "bored"       # trading to trade, no clear setup
    NEUTRAL  = "neutral"     # present but not strongly emotional
    ALL      = [CALM, ANXIOUS, FOMO, REVENGE, EXCITED, BORED, NEUTRAL]


class SetupGrade:
    A = "A"    # Perfect: all conditions textbook, high conviction
    B = "B"    # Good: most conditions met, minor imperfection
    C = "C"    # Marginal: took it but something felt off
    ALL = [A, B, C]


class ProcessGrade:
    """
    Grade the PROCESS independently of the trade outcome.
    A perfect-process trade that loses is still a process A.
    A winning revenge trade is still a process F.
    Based on Steenbarger's principle: journal the decision quality,
    not the market's verdict on that decision.
    """
    A = "A"    # Perfect: followed every rule, waited for confirmation
    B = "B"    # Good: minor deviation (e.g. slightly early entry)
    C = "C"    # Poor: broke a rule but had reasons
    F = "F"    # Failed: clear rule violation (revenge, FOMO, no setup)
    ALL = [A, B, C, F]


class MarketSentiment:
    RISK_ON   = "risk_on"    # market trending up, breadth positive
    RISK_OFF  = "risk_off"   # market trending down, risk aversion
    NEUTRAL   = "neutral"    # mixed signals, choppy
    ALL       = [RISK_ON, RISK_OFF, NEUTRAL]


class SessionType:
    GAP_UP   = "gap_up"     # opened above prior close (> +0.5%)
    GAP_DOWN = "gap_down"   # opened below prior close (< -0.5%)
    FLAT     = "flat"       # within ±0.5% of prior close
    ALL      = [GAP_UP, GAP_DOWN, FLAT]


class TimeSlot:
    OPEN_1   = "09:30-09:45"   # first 15 min — highest volatility
    OPEN_2   = "09:45-10:00"   # second 15 min
    MID_1    = "10:00-10:30"   # first half of post-open
    MID_2    = "10:30-11:00"   # second half — last ORB window
    ALL      = [OPEN_1, OPEN_2, MID_1, MID_2]


# ── Extended journal entry dataclass ─────────────────────────────────────────

@dataclass
class JournalEntry:
    """
    Complete journal entry for one trade.

    AUTO fields are populated by build_auto_fields() at entry time.
    HUMAN fields are populated post-trade via the dashboard form (90 seconds).
    """

    # ── Core identifiers (link to positions table) ────────────────────────────
    position_id:    int   = 0
    session_date:   str   = ""
    ticker:         str   = ""
    direction:      str   = ""   # 'long' | 'short'
    strategy_tag:   str   = "hybrid-orb-15min"

    # ── AUTO: Candle anatomy at entry ─────────────────────────────────────────
    # Source: candle_anatomy() Group G of toolkit
    entry_candle_type:       str   = ""   # hammer/doji/strong_bull/strong_bear/inside/engulfing/other
    entry_candle_body_pct:   float = 0.0  # body as % of total candle range
    entry_candle_upper_wick: float = 0.0  # upper wick as % of range
    entry_candle_lower_wick: float = 0.0  # lower wick as % of range
    entry_candle_colour:     str   = ""   # 'green' | 'red' | 'doji'
    entry_candle_quality:    str   = ""   # 'strong' (body>60%) | 'moderate' | 'weak' (<30%)

    # ── AUTO: Volume context ──────────────────────────────────────────────────
    entry_volume_vs_avg:     float = 1.0  # ratio: entry candle vol / 14-bar avg
    rvol_at_entry:           float = 0.0  # instrument RVOL at time of entry

    # ── AUTO: Price context at entry ──────────────────────────────────────────
    entry_vs_vwap:           str   = ""   # 'above' | 'below' | 'at'
    entry_distance_from_orb: float = 0.0  # entry price distance from ORB edge (in $)
    entry_above_pdh:         int   = 0    # 0/1: is entry above Previous Day High?
    entry_below_pdl:         int   = 0    # 0/1: is entry below Previous Day Low?
    orb_size_at_entry:       float = 0.0  # ORB range size in $
    orb_size_atr_ratio:      float = 0.0  # ORB size / ATR — measures range quality

    # ── AUTO: Trend context (multi-timeframe) ─────────────────────────────────
    daily_trend:             str   = ""   # 'up' | 'down' | 'sideways'
    daily_above_sma20:       int   = 0    # 0/1
    daily_above_sma200:      int   = 0    # 0/1
    vwap_slope_at_entry:     str   = ""   # 'up' | 'down' | 'flat'
    vwap_gate_all_pass:      int   = 0    # 0/1: did vwap_slope_gate() fully pass?

    # ── AUTO: Market context ──────────────────────────────────────────────────
    vix_at_entry:            float = 0.0
    vix_regime:              str   = ""   # NORMAL/ELEVATED/HIGH/EXTREME
    gap_pct:                 float = 0.0
    session_type:            str   = ""   # gap_up/gap_down/flat
    time_slot:               str   = ""   # 09:30-09:45 / 09:45-10:00 / etc.
    day_of_week:             int   = 0    # 0=Monday, 4=Friday
    day_name:                str   = ""   # 'Monday' / 'Tuesday' / etc.
    minutes_since_open:      int   = 0    # minutes since 09:30 EST

    # ── AUTO: Market internals proxy ──────────────────────────────────────────
    # True internals (TICK, ADD, VOLD) require dedicated data feeds.
    # These are practical proxies computable from yfinance:
    spy_vs_vwap_at_entry:    str   = ""   # 'above' | 'below' (SPY breadth proxy)
    spy_trend_at_entry:      str   = ""   # 'up' | 'down' | 'flat' (SPY 5-min)
    sector_alignment:        str   = ""   # 'with' | 'against' | 'neutral' (SPY vs ticker)

    # ── AUTO: Gate status ─────────────────────────────────────────────────────
    gate_orb_break:          str   = ""   # PASS / FAIL / WAIT
    gate_vwap:               str   = ""   # PASS / FAIL / WAIT
    gate_retest:             str   = ""   # PASS / FAIL / WAIT (or 'AUTO' if mechanical)

    # ── HUMAN: Psychology (post-trade, 90 seconds) ───────────────────────────
    confidence_pre:          int   = 3    # 1-5: confidence BEFORE entering
    emotional_state:         str   = EmotionalState.NEUTRAL
    fomo_flag:               int   = 0    # 0/1: FOMO influenced decision
    revenge_flag:            int   = 0    # 0/1: revenge trade after prior loss
    hesitation_flag:         int   = 0    # 0/1: hesitated / entered late
    oversize_flag:           int   = 0    # 0/1: took larger size than plan
    
    # ── HUMAN: Quality grades (Edgewonk / Steenbarger framework) ─────────────
    setup_grade:             str   = SetupGrade.B   # A/B/C: quality of setup
    process_grade:           str   = ProcessGrade.B # A/B/C/F: quality of execution PROCESS
    plan_adherence:          int   = 100  # 0-100: % of rules followed
    market_sentiment:        str   = MarketSentiment.NEUTRAL

    # ── HUMAN: Steenbarger three-question reflection ──────────────────────────
    what_went_right:         str   = ""   # one sentence max
    what_went_wrong:         str   = ""   # one sentence max
    lesson_learned:          str   = ""   # one sentence max
    
    # ── HUMAN: Market context notes ──────────────────────────────────────────
    news_context:            str   = ""   # any relevant news/catalyst this session
    
    # ── Metadata ─────────────────────────────────────────────────────────────
    logged_at:               str   = ""
    modified_at:             str   = ""


# ── Auto-field builder ────────────────────────────────────────────────────────

def build_auto_fields(
    df_5m:      "pd.DataFrame",
    entry_bar:  "pd.Series",
    orb_data:   dict,
    daily_df:   "pd.DataFrame",
    cfg:        dict,
    ctx:        dict,
) -> dict:
    """
    Auto-populate all mechanical journal fields from market data.
    Called by the engine at the moment of entry.

    Parameters
    ----------
    df_5m     : 5-minute OHLCV DataFrame (full session)
    entry_bar : the specific bar that triggered entry (bar N+1)
    orb_data  : output of _orb_range() {orb_high, orb_low, orb_size, bars_used}
    daily_df  : daily OHLCV for trend context
    cfg       : engine CONFIG dict
    ctx       : evaluate_signals() output (contains vix, gap_pct, vwap, etc.)

    Returns dict of auto-populated JournalEntry field values.
    """
    try:
        import pandas as pd
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import trading_quant_toolkit_v2_4 as tk
    except ImportError:
        return {}

    fields: dict[str, Any] = {}
    entry_price = float(entry_bar.get("Open", 0))
    entry_time  = entry_bar.name if hasattr(entry_bar, "name") else None

    # ── Candle anatomy ───────────────────────────────────────────────────────
    try:
        anat = tk.candle_anatomy(entry_bar)
        fields["entry_candle_type"]       = anat.get("candle_type", "")
        fields["entry_candle_body_pct"]   = round(anat.get("body_pct", 0), 2)
        fields["entry_candle_upper_wick"] = round(anat.get("upper_wick_pct", 0), 2)
        fields["entry_candle_lower_wick"] = round(anat.get("lower_wick_pct", 0), 2)
        o, c = float(entry_bar.get("Open",0)), float(entry_bar.get("Close",0))
        fields["entry_candle_colour"] = "green" if c > o else ("red" if c < o else "doji")
        bpct = anat.get("body_pct", 0)
        fields["entry_candle_quality"] = (
            "strong" if bpct >= 60 else "moderate" if bpct >= 30 else "weak"
        )
    except Exception:
        pass

    # ── Volume context ───────────────────────────────────────────────────────
    try:
        entry_vol = float(entry_bar.get("Volume", 0))
        recent_bars = df_5m.iloc[-15:]
        avg_vol = float(recent_bars["Volume"].mean())
        fields["entry_volume_vs_avg"] = round(entry_vol / max(avg_vol, 1), 2)
        fields["rvol_at_entry"] = round(ctx.get("rvol", 0) or 0, 2)
    except Exception:
        pass

    # ── Price context ────────────────────────────────────────────────────────
    try:
        vwap_val = ctx.get("vwap", 0) or 0
        fields["entry_vs_vwap"] = (
            "above" if entry_price > vwap_val * 1.0002
            else "below" if entry_price < vwap_val * 0.9998
            else "at"
        )
        if orb_data:
            # Distance from ORB edge in direction of trade
            direction = ctx.get("_breakout_dir", "")
            if direction == "long":
                fields["entry_distance_from_orb"] = round(entry_price - orb_data.get("orb_high", entry_price), 4)
            elif direction == "short":
                fields["entry_distance_from_orb"] = round(orb_data.get("orb_low", entry_price) - entry_price, 4)
            fields["orb_size_at_entry"] = round(orb_data.get("orb_size", 0), 4)
            # ORB size / ATR ratio
            try:
                atr = tk.atr_stop_loss(df_5m, direction, entry_price).get("atr", 1.0)
                fields["orb_size_atr_ratio"] = round(orb_data.get("orb_size", 0) / max(atr, 0.01), 2)
            except Exception:
                pass

        # PDH/PDL
        if len(daily_df) >= 2:
            pdh = float(daily_df["High"].iloc[-2])
            pdl = float(daily_df["Low"].iloc[-2])
            fields["entry_above_pdh"] = int(entry_price > pdh)
            fields["entry_below_pdl"] = int(entry_price < pdl)
    except Exception:
        pass

    # ── Trend context ────────────────────────────────────────────────────────
    try:
        if len(daily_df) >= 50:
            trend = tk.sma_trend_filter(daily_df, fast=20, slow=200)
            fields["daily_trend"]        = trend.get("trend", "")
            fields["daily_above_sma20"]  = int(trend.get("above_fast", False))
            fields["daily_above_sma200"] = int(trend.get("above_slow", False))
    except Exception:
        pass

    fields["vwap_slope_at_entry"] = ctx.get("vwap_slope", "")
    fields["gate_orb_break"]      = ctx.get("gate_orb_break", "")
    fields["gate_vwap"]           = ctx.get("gate_vwap", "")
    fields["gate_retest"]         = ctx.get("gate_retest", "WAIT")

    # ── Market context ───────────────────────────────────────────────────────
    vix = ctx.get("vix", 0) or 0
    fields["vix_at_entry"] = round(vix, 2)
    fields["vix_regime"]   = (
        "EXTREME" if vix > 35 else "HIGH" if vix > 25
        else "ELEVATED" if vix > 18 else "NORMAL"
    )
    gap_pct = ctx.get("gap_pct", 0) or 0
    fields["gap_pct"]     = round(gap_pct, 3)
    fields["session_type"] = (
        SessionType.GAP_UP   if gap_pct > 0.5
        else SessionType.GAP_DOWN if gap_pct < -0.5
        else SessionType.FLAT
    )

    # Time slot
    if entry_time is not None:
        try:
            t = entry_time.time() if hasattr(entry_time, "time") else Time(9, 35)
            mins_since_open = (t.hour - 9) * 60 + t.minute - 30
            fields["minutes_since_open"] = max(0, mins_since_open)
            fields["day_of_week"] = entry_time.weekday() if hasattr(entry_time, "weekday") else 0
            fields["day_name"] = ["Monday","Tuesday","Wednesday","Thursday","Friday"][
                entry_time.weekday() if hasattr(entry_time, "weekday") else 0]
            if   t <= Time(9, 45):  fields["time_slot"] = TimeSlot.OPEN_1
            elif t <= Time(10, 0):  fields["time_slot"] = TimeSlot.OPEN_2
            elif t <= Time(10, 30): fields["time_slot"] = TimeSlot.MID_1
            else:                   fields["time_slot"] = TimeSlot.MID_2
        except Exception:
            pass

    # SPY breadth proxy (only if ticker isn't SPY)
    try:
        ticker = cfg.get("ticker","SPY")
        if not df_5m.empty and vwap_val:
            spy_close = float(df_5m["Close"].dropna().iloc[-1])
            fields["spy_vs_vwap_at_entry"] = "above" if spy_close > vwap_val else "below"
    except Exception:
        pass

    fields["logged_at"] = datetime.utcnow().isoformat(timespec="seconds")
    return fields


# ── Database helpers ──────────────────────────────────────────────────────────

JOURNAL_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS trade_journal_extended (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    position_id             INTEGER,       -- FK to positions table
    session_date            TEXT,
    ticker                  TEXT,
    direction               TEXT,
    strategy_tag            TEXT,
    -- Candle anatomy
    entry_candle_type       TEXT,
    entry_candle_body_pct   REAL,
    entry_candle_upper_wick REAL,
    entry_candle_lower_wick REAL,
    entry_candle_colour     TEXT,
    entry_candle_quality    TEXT,
    -- Volume
    entry_volume_vs_avg     REAL,
    rvol_at_entry           REAL,
    -- Price context
    entry_vs_vwap           TEXT,
    entry_distance_from_orb REAL,
    entry_above_pdh         INTEGER,
    entry_below_pdl         INTEGER,
    orb_size_at_entry       REAL,
    orb_size_atr_ratio      REAL,
    -- Trend context
    daily_trend             TEXT,
    daily_above_sma20       INTEGER,
    daily_above_sma200      INTEGER,
    vwap_slope_at_entry     TEXT,
    vwap_gate_all_pass      INTEGER,
    gate_orb_break          TEXT,
    gate_vwap               TEXT,
    gate_retest             TEXT,
    -- Market context
    vix_at_entry            REAL,
    vix_regime              TEXT,
    gap_pct                 REAL,
    session_type            TEXT,
    time_slot               TEXT,
    day_of_week             INTEGER,
    day_name                TEXT,
    minutes_since_open      INTEGER,
    -- Internals proxy
    spy_vs_vwap_at_entry    TEXT,
    spy_trend_at_entry      TEXT,
    sector_alignment        TEXT,
    -- Psychology (human input)
    confidence_pre          INTEGER,
    emotional_state         TEXT,
    fomo_flag               INTEGER,
    revenge_flag            INTEGER,
    hesitation_flag         INTEGER,
    oversize_flag           INTEGER,
    setup_grade             TEXT,
    process_grade           TEXT,
    plan_adherence          INTEGER,
    market_sentiment        TEXT,
    -- Reflection
    what_went_right         TEXT,
    what_went_wrong         TEXT,
    lesson_learned          TEXT,
    news_context            TEXT,
    -- Meta
    logged_at               TEXT,
    modified_at             TEXT
);
"""


def ensure_journal_table(db_path: str) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.executescript(JOURNAL_TABLE_SQL)


def save_journal_entry(db_path: str, entry: dict) -> int:
    """Insert a journal entry dict. Returns the new row id."""
    ensure_journal_table(db_path)
    cols   = ", ".join(entry.keys())
    pholds = ", ".join(["?"] * len(entry))
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute(
            f"INSERT INTO trade_journal_extended ({cols}) VALUES ({pholds})",
            tuple(entry.values())
        )
        return cur.lastrowid


def update_psychology_fields(db_path: str, position_id: int, fields: dict) -> None:
    """
    Update psychology + reflection fields after a trade closes.
    Called from the dashboard post-trade input form.
    """
    fields["modified_at"] = datetime.utcnow().isoformat(timespec="seconds")
    set_clause = ", ".join(f"{k} = ?" for k in fields)
    values     = list(fields.values()) + [position_id]
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            f"UPDATE trade_journal_extended SET {set_clause} WHERE position_id = ?",
            values
        )


def get_journal_entries(db_path: str, n: int = 50) -> list[dict]:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM trade_journal_extended ORDER BY id DESC LIMIT ?", (n,)
        ).fetchall()
    return [dict(r) for r in rows]


# ── Statistical correlation analysis (used by SessionLearner + LLM tool) ─────

def get_journal_stats(db_path: str, min_samples: int = 5) -> dict:
    """
    Compute EV and win-rate correlations across all journal dimensions.
    Only reports buckets with min_samples or more trades.
    Returns a structured dict for LLM consumption.
    """
    ensure_journal_table(db_path)
    entries = get_journal_entries(db_path, n=500)

    # Merge with positions to get actual_r
    pos_map: dict[int, float] = {}
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            for row in conn.execute(
                "SELECT id, actual_r FROM positions WHERE status='closed' AND actual_r IS NOT NULL"
            ).fetchall():
                pos_map[row["id"]] = row["actual_r"]
    except Exception:
        pass

    rows = []
    for e in entries:
        pid = e.get("position_id")
        r   = pos_map.get(pid)
        if r is not None:
            rows.append({**e, "actual_r": r})

    if not rows:
        return {"n": 0, "message": "No closed trades with journal entries yet"}

    def _stats_for(group_key: str) -> dict:
        buckets: dict[str, list[float]] = {}
        for row in rows:
            val = str(row.get(group_key, "") or "unknown")
            buckets.setdefault(val, []).append(row["actual_r"])
        result = {}
        for val, rs in buckets.items():
            if len(rs) < min_samples:
                continue
            wins = [r for r in rs if r > 0]
            wr   = len(wins) / len(rs)
            ev   = sum(rs) / len(rs)
            result[val] = {"n": len(rs), "win_rate": round(wr, 3), "ev": round(ev, 4)}
        return result

    correlations = {
        "by_candle_type":    _stats_for("entry_candle_type"),
        "by_candle_quality": _stats_for("entry_candle_quality"),
        "by_candle_colour":  _stats_for("entry_candle_colour"),
        "by_time_slot":      _stats_for("time_slot"),
        "by_day_of_week":    _stats_for("day_name"),
        "by_session_type":   _stats_for("session_type"),
        "by_vix_regime":     _stats_for("vix_regime"),
        "by_daily_trend":    _stats_for("daily_trend"),
        "by_setup_grade":    _stats_for("setup_grade"),
        "by_process_grade":  _stats_for("process_grade"),
        "by_emotional_state": _stats_for("emotional_state"),
        "by_market_sentiment": _stats_for("market_sentiment"),
        "fomo_vs_not":       _stats_for("fomo_flag"),
        "revenge_vs_not":    _stats_for("revenge_flag"),
        "by_entry_vs_vwap":  _stats_for("entry_vs_vwap"),
        "body_pct_correlation": _body_pct_corr(rows),
    }

    # Highlight top and bottom performers
    insights = []
    for cat, data in correlations.items():
        if not data or not isinstance(data, dict):
            continue
        valid = {k: v for k, v in data.items() if isinstance(v, dict)}
        if len(valid) < 2:
            continue
        best  = max(valid, key=lambda k: valid[k]["ev"])
        worst = min(valid, key=lambda k: valid[k]["ev"])
        ev_diff = valid[best]["ev"] - valid[worst]["ev"]
        if ev_diff > 0.15:   # only report meaningful differences
            insights.append({
                "dimension":  cat,
                "best":  {"label": best,  "ev": valid[best]["ev"],  "n": valid[best]["n"]},
                "worst": {"label": worst, "ev": valid[worst]["ev"], "n": valid[worst]["n"]},
                "ev_gap": round(ev_diff, 4),
                "action": f"Focus on '{best}' setups (+{ev_diff:.3f}R vs '{worst}')",
            })

    # Sort insights by impact
    insights.sort(key=lambda x: x["ev_gap"], reverse=True)

    # Psychology flags
    fomo_ev    = correlations["fomo_vs_not"].get("1", {}).get("ev")
    revenge_ev = correlations["revenge_vs_not"].get("1", {}).get("ev")

    return {
        "n":                 len(rows),
        "correlations":      correlations,
        "top_insights":      insights[:8],
        "psychology_alerts": {
            "fomo_ev":      fomo_ev,
            "revenge_ev":   revenge_ev,
            "fomo_warning":    fomo_ev is not None and fomo_ev < -0.1,
            "revenge_warning": revenge_ev is not None and revenge_ev < -0.2,
        },
    }


def _body_pct_corr(rows: list[dict]) -> dict:
    """Bucket candle body_pct into ranges and show EV per bucket."""
    buckets: dict[str, list[float]] = {
        "0-30% (weak)":    [],
        "30-60% (moderate)": [],
        "60-100% (strong)": [],
    }
    for row in rows:
        bp = row.get("entry_candle_body_pct", 0) or 0
        r  = row["actual_r"]
        if bp < 30:
            buckets["0-30% (weak)"].append(r)
        elif bp < 60:
            buckets["30-60% (moderate)"].append(r)
        else:
            buckets["60-100% (strong)"].append(r)
    result = {}
    for label, rs in buckets.items():
        if len(rs) >= 3:
            result[label] = {"n": len(rs), "ev": round(sum(rs)/len(rs), 4)}
    return result


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile, os

    tmp = tempfile.mktemp(suffix=".db")
    ensure_journal_table(tmp)

    # Test save and retrieve
    entry = {
        "position_id": 1, "session_date": "2026-09-08",
        "ticker": "SPY", "direction": "long",
        "strategy_tag": "hybrid-orb-15min",
        "entry_candle_type": "strong_bull", "entry_candle_body_pct": 72.0,
        "entry_candle_quality": "strong", "entry_candle_colour": "green",
        "time_slot": "09:30-09:45", "day_of_week": 1, "day_name": "Tuesday",
        "vix_at_entry": 14.5, "vix_regime": "NORMAL",
        "session_type": "gap_up", "gap_pct": 0.8,
        "daily_trend": "up", "setup_grade": "A", "process_grade": "A",
        "confidence_pre": 4, "emotional_state": "calm",
        "fomo_flag": 0, "revenge_flag": 0,
        "plan_adherence": 100,
        "what_went_right": "Waited for clean retest of ORB level.",
        "what_went_wrong": "Exited first ladder target slightly early.",
        "lesson_learned": "Trust the VWAP trailing stop — let it run.",
        "logged_at": datetime.utcnow().isoformat(),
    }
    row_id = save_journal_entry(tmp, entry)
    assert row_id > 0

    entries = get_journal_entries(tmp)
    assert len(entries) == 1 and entries[0]["entry_candle_quality"] == "strong"
    print(f"Journal entry saved and retrieved: row_id={row_id}")

    # Test psychology update
    update_psychology_fields(tmp, 1, {
        "emotional_state": "fomo",
        "fomo_flag": 1,
        "lesson_learned": "Updated lesson."
    })
    entries2 = get_journal_entries(tmp)
    assert entries2[0]["fomo_flag"] == 1

    print("Psychology update OK")
    print()
    print("EmotionalState values:", EmotionalState.ALL)
    print("SetupGrade values:", SetupGrade.ALL)
    print("ProcessGrade values:", ProcessGrade.ALL)
    print("TimeSlot values:", TimeSlot.ALL)

    os.unlink(tmp)
    print("\n✅ trade_journal_extended.py verified")
