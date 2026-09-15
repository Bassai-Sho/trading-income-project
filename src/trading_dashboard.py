"""
trading_dashboard.py
====================
Standalone Streamlit desktop dashboard for the Trading Income Project.

Theme: Fully adaptive to Streamlit Light and Dark modes.
Run:   streamlit run src/trading_dashboard.py
"""

from __future__ import annotations

import os
import re
import sys
import time
import math
import statistics
from datetime import datetime, time as Time, date, timedelta, timezone
from typing import Any

import streamlit as st
import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# Optional toolkit import
# ---------------------------------------------------------------------------
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import trading_quant_toolkit_v2_4 as tk  # type: ignore
    TOOLKIT_AVAILABLE = True
except ImportError:
    TOOLKIT_AVAILABLE = False

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ORB Trading Cockpit",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS — Adaptive Light / Dark Theme Styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* Base Typography */
html, body, [class*="css"] { 
    font-family: 'JetBrains Mono', 'Fira Code', 'Consolas', monospace; 
}

/* Header Bar */
.cockpit-header {
    background: var(--secondary-background-color);
    border: 1px solid rgba(128, 128, 128, 0.2);
    border-radius: 6px;
    padding: 0.75rem 1.25rem;
    margin-bottom: 1.25rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 0.5rem;
}
.header-left {
    display: flex;
    align-items: center;
    gap: 0.75rem;
}
.header-title {
    font-size: 1.05rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    color: var(--text-color);
}
.header-badge {
    background: rgba(59, 130, 246, 0.15);
    color: #3b82f6;
    border: 1px solid rgba(59, 130, 246, 0.3);
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 600;
}
.header-right {
    display: flex;
    align-items: center;
    gap: 1rem;
}
.status-pill {
    padding: 3px 10px;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.05em;
}
.pill-open { background: rgba(34, 197, 94, 0.15); color: #22c55e; border: 1px solid rgba(34, 197, 94, 0.3); }
.pill-pre  { background: rgba(245, 158, 11, 0.15); color: #f59e0b; border: 1px solid rgba(245, 158, 11, 0.3); }
.pill-post { background: rgba(128, 128, 128, 0.15); color: #888888; border: 1px solid rgba(128, 128, 128, 0.3); }

/* Section Headers */
h2 {
    font-size: 0.85rem !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
    color: rgba(128, 128, 128, 0.85) !important;
    margin: 1.4rem 0 0.6rem !important;
    border-bottom: 1px solid rgba(128, 128, 128, 0.15);
    padding-bottom: 0.3rem;
}

/* Signal Cards */
.sig-card {
    background: var(--secondary-background-color);
    border: 1px solid rgba(128, 128, 128, 0.18);
    border-radius: 5px;
    padding: 0.8rem 1rem;
    margin: 0.25rem 0;
}
.sig-go   { border-left: 4px solid #22c55e; }
.sig-no   { border-left: 4px solid #ef4444; }
.sig-wait { border-left: 4px solid #f59e0b; }

/* Status Badges */
.badge-green { display:inline-block; background:rgba(34, 197, 94, 0.15); color:#22c55e; border:1px solid rgba(34, 197, 94, 0.3); padding:2px 7px; border-radius:3px; font-size:0.72rem; font-weight:600; }
.badge-red   { display:inline-block; background:rgba(239, 68, 68, 0.15); color:#ef4444; border:1px solid rgba(239, 68, 68, 0.3); padding:2px 7px; border-radius:3px; font-size:0.72rem; font-weight:600; }
.badge-amber { display:inline-block; background:rgba(245, 158, 11, 0.15); color:#f59e0b; border:1px solid rgba(245, 158, 11, 0.3); padding:2px 7px; border-radius:3px; font-size:0.72rem; font-weight:600; }
.badge-grey  { display:inline-block; background:rgba(128, 128, 128, 0.15); color:var(--text-color); border:1px solid rgba(128, 128, 128, 0.25); padding:2px 7px; border-radius:3px; font-size:0.72rem; }

/* Metrics Containers */
[data-testid="metric-container"] {
    background: var(--secondary-background-color);
    border: 1px solid rgba(128, 128, 128, 0.18);
    border-radius: 5px;
    padding: 0.6rem 0.85rem;
}
[data-testid="metric-container"] label {
    font-size: 0.7rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
}

/* Master Gate Banner */
.gate-banner {
    margin-top: 0.75rem;
    padding: 0.75rem 1rem;
    border-radius: 5px;
    text-align: center;
    font-size: 0.95rem;
    font-weight: 700;
    letter-spacing: 0.12em;
}
.gate-open {
    background: rgba(34, 197, 94, 0.15);
    border: 1px solid #22c55e;
    color: #22c55e;
}
.gate-closed {
    background: rgba(239, 68, 68, 0.15);
    border: 1px solid #ef4444;
    color: #ef4444;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------
def _init_state():
    defaults = {
        "account":          10_000.0,
        "ticker":           "SPY",
        "orb_method":       "15min",
        "risk_mode":        "Fixed 1%",
        "journal":          [],
        "session_pnl":      0.0,
        "consec_losses":    0,
        "stop_session":     False,
        "pause":            False,
        "trade_count":      0,
        "psych_confirmed":  False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
@st.cache_data(ttl=60)
def _fetch(ticker: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval,
                     auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df

def _vwap_series(df: pd.DataFrame) -> pd.Series:
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    pv = tp * df["Volume"]
    vwap = pd.Series(index=df.index, dtype=float)
    for day in set(df.index.date):
        mask = (df.index.date == day) & (df.index.time >= Time(9, 30))
        cv = df["Volume"][mask].cumsum()
        vwap[mask] = (pv[mask].cumsum() / cv.replace(0, float("nan"))).values
    return vwap

def _orb_range(df: pd.DataFrame, method: str) -> dict:
    end_map = {"5min": Time(9, 34), "15min": Time(9, 44), "30min": Time(9, 55)}
    end_t = end_map.get(method, Time(9, 44))
    if df.empty:
        return {}
    today = df.index[-1].date()
    mask = (df.index.date == today) & (df.index.time >= Time(9, 30)) & (df.index.time <= end_t)
    bars = df[mask]
    if bars.empty:
        return {}
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
    slope = float(recent.iloc[-1] - recent.iloc[0]) / lookback
    thresh = float(recent.mean()) * 0.0002
    direction = "up" if slope > thresh else "down" if slope < -thresh else "flat"
    return {"slope": round(slope, 5), "direction": direction}

def _kelly(win_rate: float, avg_win: float, avg_loss: float = 1.0) -> float:
    if win_rate <= 0 or win_rate >= 1 or avg_win <= 0 or avg_loss <= 0:
        return 0.0
    r = avg_win / avg_loss
    k = win_rate - (1.0 - win_rate) / r
    return max(0.0, min(round(k * 0.5, 4), 0.25))

def _rolling_stats(journal: list[dict]) -> dict:
    if not journal:
        return {}
    n = len(journal)
    rs = [t["actual_r"] for t in journal]
    wins = [r for r in rs if r > 0]
    losses = [r for r in rs if r < 0]
    wr = len(wins) / n
    aw = statistics.mean(wins) if wins else 0.0
    al = abs(statistics.mean(losses)) if losses else 0.0
    ev = (wr * aw) - ((1 - wr) * al)
    sharpe = None
    if n >= 2:
        sd = statistics.stdev(rs)
        if sd > 0:
            sharpe = round(statistics.mean(rs) / sd, 3)
    return {"n": n, "wr": wr, "aw": aw, "al": al, "ev": ev, "sharpe": sharpe}

def _badge(text: str, colour: str) -> str:
    return f'<span class="badge-{colour}">{text}</span>'

def _sig_card(label: str, status: str, detail: str) -> str:
    css = {"GO": "go", "NO": "no", "WAIT": "wait"}.get(status, "wait")
    icon = {"GO": "✓", "NO": "✕", "WAIT": "⊙"}.get(status, "—")
    badge_col = {"GO": "green", "NO": "red", "WAIT": "amber"}.get(status, "grey")
    return (
        f'<div class="sig-card sig-{css}">'
        f'<b style="font-size:0.75rem;letter-spacing:0.08em">{label}</b>&nbsp;&nbsp;'
        f'{_badge(f"{icon} {status}", badge_col)}'
        f'<br><span style="opacity:0.75;font-size:0.75rem">{detail}</span></div>'
    )

# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⬡ SESSION SETUP")
    st.session_state["account"] = st.number_input(
        "Account (£)", value=st.session_state["account"],
        min_value=100.0, step=500.0, format="%.0f")
    st.session_state["ticker"] = st.text_input(
        "Primary ticker", value=st.session_state["ticker"]).upper()
    st.session_state["orb_method"] = st.selectbox(
        "ORB window", ["5min", "15min", "30min"],
        index=["5min","15min","30min"].index(st.session_state["orb_method"]))
    st.session_state["risk_mode"] = st.selectbox(
        "Position sizing", ["Fixed 1%", "Fixed 0.5% (commodity)", "Half-Kelly"],
        index=["Fixed 1%","Fixed 0.5% (commodity)","Half-Kelly"].index(
            st.session_state["risk_mode"]))

    st.markdown("---")
    st.markdown("## ☰ SESSION CONTROLS")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("↺ Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    with col2:
        if st.button("⏹ Reset", use_container_width=True):
            st.session_state["session_pnl"]   = 0.0
            st.session_state["consec_losses"]  = 0
            st.session_state["stop_session"]   = False
            st.session_state["pause"]          = False
            st.session_state["trade_count"]    = 0
            st.rerun()

    st.markdown("---")
    st.markdown("## ✓ DISCIPLINE CHECKLIST")
    st.session_state["psych_confirmed"] = st.checkbox(
        "Pre-market script read",
        value=st.session_state["psych_confirmed"])
    st.caption("PDH/PDL marked  ·  VIX assessed  ·  Gap identified")

# ---------------------------------------------------------------------------
# TOP COCKPIT HEADER BAR
# ---------------------------------------------------------------------------
now_est = datetime.now(timezone.utc) - timedelta(hours=5)
session_ok = Time(9, 30) <= now_est.time() <= Time(11, 0)
is_pre = now_est.time() < Time(9, 30)

status_label = "MARKET OPEN (ORB ACTIVE)" if session_ok else ("PRE-MARKET (OPENS 09:30 EST)" if is_pre else "POST-ORB WINDOW")
status_css   = "pill-open" if session_ok else ("pill-pre" if is_pre else "pill-post")

st.markdown(f"""
<div class="cockpit-header">
    <div class="header-left">
        <span class="header-title">📈 ORB TRADING COCKPIT</span>
        <span class="header-badge">{st.session_state["ticker"]} · {st.session_state["orb_method"].upper()}</span>
    </div>
    <div class="header-right">
        <span class="status-pill {status_css}">{status_label}</span>
        <span style="font-size:0.75rem;font-weight:600;opacity:0.8;">{now_est.strftime("%H:%M:%S EST")}</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Data Ingestion ──
ticker = st.session_state["ticker"]
df_5m  = _fetch(ticker, "2d", "5m")
df_1d  = _fetch(ticker, "20d", "1d")

# ── 1. PRE-MARKET INTELLIGENCE ──────────────────────────────────────────────
st.markdown("## PRE-MARKET INTELLIGENCE")
col_vix, col_pdh, col_pdl, col_gap, col_regime = st.columns(5)

try:
    df_vix = _fetch("^VIX", "2d", "5m")
    vix_val = float(df_vix["Close"].dropna().iloc[-1])
    regime = ("EXTREME" if vix_val > 35 else "HIGH" if vix_val > 25
              else "ELEVATED" if vix_val > 18 else "NORMAL")
    size_mod = (0.25 if vix_val > 35 else 0.50 if vix_val > 25
                else 0.75 if vix_val > 18 else 1.00)
    col_vix.metric("VIX", f"{vix_val:.1f}", f"{regime}")
    col_regime.metric("Size Modifier", f"{size_mod:.0%}", "Risk budget scale")
except Exception:
    col_vix.metric("VIX", "—")
    col_regime.metric("Size Modifier", "100%")
    vix_val = 20.0
    size_mod = 1.0

if len(df_1d) >= 2:
    pdh = float(df_1d["High"].iloc[-2])
    pdl = float(df_1d["Low"].iloc[-2])
    prev_close = float(df_1d["Close"].iloc[-2])
    col_pdh.metric("PDH (Prev High)", f"${pdh:.2f}")
    col_pdl.metric("PDL (Prev Low)",  f"${pdl:.2f}")
    if not df_5m.empty:
        today_open = float(df_5m["Open"].iloc[0])
        gap_pct = (today_open - prev_close) / prev_close * 100
        gap_dir = "↑" if gap_pct > 0.1 else "↓" if gap_pct < -0.1 else "→"
        col_gap.metric("Overnight Gap", f"{gap_pct:+.2f}%", gap_dir)
else:
    col_pdh.metric("PDH", "—")
    col_pdl.metric("PDL", "—")
    col_gap.metric("Gap", "—")

# ── 2. ORB & VWAP SETUP ─────────────────────────────────────────────────────
st.markdown("## ORB & VWAP SETUP")

orb = _orb_range(df_5m, st.session_state["orb_method"]) if not df_5m.empty else {}
vwap = _vwap_series(df_5m) if not df_5m.empty else pd.Series()
vs = _vwap_slope(vwap, lookback=3)

col_orb1, col_orb2, col_orb3, col_vwap, col_vs = st.columns(5)
if orb:
    col_orb1.metric("ORB High", f"${orb['orb_high']:.2f}")
    col_orb2.metric("ORB Low",  f"${orb['orb_low']:.2f}")
    col_orb3.metric("Range Size", f"${orb['orb_size']:.2f}", f"{orb['bars_used']} bars")
else:
    col_orb1.metric("ORB High", "Forming…")
    col_orb2.metric("ORB Low",  "Forming…")
    col_orb3.metric("Range Size", "—")

if not df_5m.empty:
    last_close = float(df_5m["Close"].dropna().iloc[-1])
    last_vwap  = float(vwap.dropna().iloc[-1]) if not vwap.dropna().empty else 0.0
    col_vwap.metric("VWAP", f"${last_vwap:.2f}")
    col_vs.metric("VWAP Slope", vs["direction"].upper(), f"{vs['slope']:+.4f}/bar")

    # Breakout check
    if orb:
        if last_close > orb["orb_high"]:
            breakout_signal = "long"
        elif last_close < orb["orb_low"]:
            breakout_signal = "short"
        else:
            breakout_signal = None
    else:
        breakout_signal = None
else:
    last_close = 100.0
    breakout_signal = None

# ── 3. SIGNAL AND-GATE ──────────────────────────────────────────────────────
st.markdown("## SIGNAL AND-GATE")

if df_5m.empty or not orb:
    st.info("Waiting for market open and ORB range formation (09:30 EST)…")
else:
    if breakout_signal == "long":
        g1_status, g1_detail = "GO", f"Close ${last_close:.2f} > ORB High ${orb['orb_high']:.2f}"
    elif breakout_signal == "short":
        g1_status, g1_detail = "GO", f"Close ${last_close:.2f} < ORB Low ${orb['orb_low']:.2f}"
    else:
        g1_status, g1_detail = "WAIT", f"Price inside range [${orb['orb_low']:.2f} – ${orb['orb_high']:.2f}]"

    if breakout_signal == "long" and vs["direction"] == "up":
        g2_status, g2_detail = "GO", "VWAP sloping UP ↑ — confirms long breakout"
    elif breakout_signal == "short" and vs["direction"] == "down":
        g2_status, g2_detail = "GO", "VWAP sloping DOWN ↓ — confirms short breakout"
    elif vs["direction"] == "flat":
        g2_status, g2_detail = "NO", "VWAP flat — insufficient institutional momentum"
    else:
        g2_status, g2_detail = "NO", f"VWAP {vs['direction']} conflicts with {breakout_signal or 'no'} breakout"

    gc1, gc2, gc3 = st.columns(3)
    with gc1:
        st.markdown(_sig_card("① ORB BREAKOUT", g1_status, g1_detail), unsafe_allow_html=True)
    with gc2:
        st.markdown(_sig_card("② VWAP SLOPE", g2_status, g2_detail), unsafe_allow_html=True)
    with gc3:
        retest_confirmed = st.checkbox("Retest confirmed", value=False)
        g3_status = "GO" if retest_confirmed else "WAIT"
        g3_detail = "Level held & closed away" if retest_confirmed else "Waiting for level touch & confirmation"
        st.markdown(_sig_card("③ RETEST MECHANIC", g3_status, g3_detail), unsafe_allow_html=True)

    and_gate = (g1_status == "GO" and g2_status == "GO" and g3_status == "GO")
    gate_cls = "gate-open" if and_gate else "gate-closed"
    gate_txt = "✓  AND-GATE OPEN — EXECUTION PERMITTED" if and_gate else "✕  AND-GATE CLOSED — NO ENTRY"
    st.markdown(f'<div class="gate-banner {gate_cls}">{gate_txt}</div>', unsafe_allow_html=True)

# ── 4. POSITION CALCULATOR ──────────────────────────────────────────────────
st.markdown("## POSITION CALCULATOR")

pc1, pc2 = st.columns(2)
with pc1:
    entry_price = st.number_input("Planned Entry ($)", value=float(last_close), step=0.05, format="%.2f")
    stop_price  = st.number_input("Stop Price ($)", value=round(entry_price * 0.995, 2), step=0.05, format="%.2f")
    target_rr   = st.slider("Target R:R Ratio", min_value=1.0, max_value=4.0, value=2.0, step=0.5)

with pc2:
    account = st.session_state["account"]
    risk_mode = st.session_state["risk_mode"]
    risk_dist = abs(entry_price - stop_price)

    if risk_mode == "Fixed 1%":
        risk_frac = 0.01
    elif risk_mode == "Fixed 0.5% (commodity)":
        risk_frac = 0.005
    else:
        stats = _rolling_stats(st.session_state["journal"])
        risk_frac = _kelly(stats.get("wr", 0.45), stats.get("aw", 2.0), stats.get("al", 1.0)) if stats.get("n", 0) >= 50 else 0.01

    effective_risk = risk_frac * size_mod
    risk_amount    = account * effective_risk
    units          = risk_amount / risk_dist if risk_dist > 1e-6 else 0.0

    is_long      = stop_price < entry_price
    reward_dist  = risk_dist * target_rr
    target_price = round((entry_price + reward_dist) if is_long else (entry_price - reward_dist), 2)
    be_price     = round((entry_price + reward_dist * 0.5) if is_long else (entry_price - reward_dist * 0.5), 2)

    col_u, col_r, col_t, col_be = st.columns(4)
    col_u.metric("Position Units", f"{units:.0f} shs")
    col_r.metric("Capital at Risk", f"£{risk_amount:.0f}", f"{effective_risk:.2%} eff.")
    col_t.metric("Target Price", f"${target_price:.2f}", f"{target_rr:.1f}R reward")
    col_be.metric("Breakeven Trigger", f"${be_price:.2f}", "Move stop here")

    # Ladder targets
    ladders = []
    for r_lvl in [1.0, 2.0, 3.0]:
        p = round((entry_price + r_lvl * risk_dist) if is_long else (entry_price - r_lvl * risk_dist), 2)
        ladders.append({"Target": f"{r_lvl:.0f}R", "Price": f"${p:.2f}", "Scale Out": "33%"})

    st.caption("Partial Scale-Out Ladder (VWAP + Ladder Rule):")
    st.dataframe(pd.DataFrame(ladders), hide_index=True, use_container_width=True)

# ── 5. SESSION RISK MONITOR ─────────────────────────────────────────────────
st.markdown("## SESSION RISK STATE")

rs1, rs2, rs3, rs4 = st.columns(4)
session_pnl_pct = (st.session_state["session_pnl"] / max(account, 1)) * 100
rs1.metric("Session P&L", f"£{st.session_state['session_pnl']:.0f}", f"{session_pnl_pct:+.2f}%")
rs2.metric("Consecutive Losses", str(st.session_state["consec_losses"]), "Pause if 3" if st.session_state["consec_losses"] < 3 else "⏸ PAUSE NOW")
rs3.metric("Trades Taken", str(st.session_state["trade_count"]))
rs4.metric("Daily Max Loss", f"£{account * 0.03:.0f}", "3% hard limit")

if st.session_state["stop_session"]:
    st.error("🛑 SESSION HALTED — Daily loss limit reached. Stand down.")
elif st.session_state["pause"]:
    st.warning("⏸ MANDATORY 30-MIN PAUSE — 3 consecutive losses.")
elif not st.session_state["psych_confirmed"]:
    st.info("ℹ️ Confirm the pre-session checklist in the sidebar to authorize entries.")
else:
    st.success("✅ Operational risk within parameters.")

# ── 6. TRADE JOURNAL ────────────────────────────────────────────────────────
st.markdown("## TRADE JOURNAL")

with st.expander("📝 Log a Completed Trade", expanded=False):
    lc1, lc2, lc3 = st.columns(3)
    log_ticker  = lc1.text_input("Ticker", value=ticker, key="log_ticker").upper()
    log_dir     = lc1.selectbox("Direction", ["long", "short"], key="log_dir")
    log_entry   = lc2.number_input("Fill Entry ($)", value=entry_price, key="log_entry")
    log_stop    = lc2.number_input("Fill Stop ($)", value=stop_price, key="log_stop")
    log_exit    = lc3.number_input("Fill Exit ($)", value=entry_price, key="log_exit")
    log_outcome = lc3.selectbox("Outcome", ["win", "loss", "breakeven"], key="log_outcome")
    log_notes   = st.text_input("Session Notes", key="log_notes")

    if st.button("Commit Trade to Journal", use_container_width=True):
        risk_d = abs(log_entry - log_stop)
        actual_r = round((log_exit - log_entry) / risk_d * (1 if log_dir == "long" else -1), 3) if risk_d > 1e-6 else 0.0
        trade = {
            "date":      str(date.today()),
            "ticker":    log_ticker,
            "direction": log_dir,
            "entry":     log_entry,
            "stop":      log_stop,
            "exit":      log_exit,
            "actual_r":  actual_r,
            "outcome":   log_outcome,
            "notes":     log_notes,
        }
        st.session_state["journal"].append(trade)
        st.session_state["session_pnl"] += actual_r * risk_amount
        st.session_state["trade_count"] += 1
        if actual_r < 0:
            st.session_state["consec_losses"] += 1
        else:
            st.session_state["consec_losses"] = 0
        if st.session_state["session_pnl"] / account <= -0.03:
            st.session_state["stop_session"] = True
        st.session_state["pause"] = (st.session_state["consec_losses"] >= 3 and not st.session_state["stop_session"])
        st.success(f"Logged: {actual_r:+.2f}R")
        st.rerun()

# Journal Stats & History Table
if st.session_state["journal"]:
    stats = _rolling_stats(st.session_state["journal"])
    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("Win Rate", f"{stats.get('wr', 0):.1%}", "Target > 40%")
    sc2.metric("EV per Trade", f"{stats.get('ev', 0):+.3f}R", "Positive EV required")
    sc3.metric("Sharpe Ratio", str(stats.get("sharpe", "—")))
    sc4.metric("Logged Trades", str(stats.get("n", 0)), f"{max(0, 100 - stats.get('n', 0))} to milestone")

    df_j = pd.DataFrame(st.session_state["journal"])[["date", "ticker", "direction", "entry", "exit", "actual_r", "outcome"]]

    # Pandas 2.2+ safe Styler map
    styler = df_j.style
    map_func = getattr(styler, "map", getattr(styler, "applymap", None))
    if map_func:
        styler = map_func(
            lambda v: "color:#22c55e;font-weight:600;" if v == "win" else ("color:#ef4444;font-weight:600;" if v == "loss" else ""),
            subset=["outcome"]
        )
    st.dataframe(styler, use_container_width=True, hide_index=True)
else:
    st.caption("No trades logged in current session.")
