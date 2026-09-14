"""
trading_dashboard.py
====================
Standalone Streamlit desktop dashboard for the Trading Income Project.

ARCHITECTURE PRINCIPLE: This app complements the trading_quant_toolkit but
is NOT dependent on it. It imports toolkit functions where convenient but
contains its own fallbacks and can run standalone. The toolkit is the
computational truth layer; this dashboard is the real-time cockpit.

Sections
--------
SIDEBAR     -- Session controls: account balance, ticker, risk mode
PRE-MARKET  -- Stocks in Play scanner + VIX regime + PDH/PDL
ORB SETUP   -- Live ORB range (5/15/30min), VWAP slope, breakout signal
SIGNAL GATE -- AND-gate visual status (ORB / VWAP / Retest)
POSITION    -- Size calculator (fixed%, Kelly) + stop/target/trailing
RISK STATE  -- Session P&L, consecutive losses, daily stop check
JOURNAL     -- Log trades inline, rolling stats table
SETTINGS    -- Universe watchlist, refresh interval

Run:  streamlit run trading_dashboard.py
"""

from __future__ import annotations
import re

import os
import sys
import time
import math
import statistics
import tempfile
from datetime import datetime, time as Time, date, timedelta
from typing import Any

import streamlit as st
import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# Optional toolkit import — graceful fallback if path differs
# ---------------------------------------------------------------------------
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import trading_quant_toolkit_v2_4 as tk  # type: ignore
    TOOLKIT_AVAILABLE = True
except ImportError:
    TOOLKIT_AVAILABLE = False

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ORB Trading Cockpit",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS — trading terminal aesthetic
# Dark base, phosphor-green signals, amber warnings, crisp monospace data
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* Base */
[data-testid="stAppViewContainer"] { background: #0d0d0f; }
[data-testid="stSidebar"] { background: #111116; border-right: 1px solid #222230; }
section[data-testid="stSidebarContent"] { padding: 1rem; }

/* Typography */
html, body, [class*="css"] { font-family: 'JetBrains Mono', 'Fira Code', 'Consolas', monospace; }
h1 { font-size: 1.1rem; letter-spacing: 0.12em; color: #c8ccd4; font-weight: 500;
     text-transform: uppercase; border-bottom: 1px solid #1e1e2e; padding-bottom: 0.4rem; }
h2 { font-size: 0.8rem; letter-spacing: 0.18em; color: #6b7280;
     text-transform: uppercase; margin: 1.2rem 0 0.5rem; }
h3 { font-size: 0.75rem; color: #9ca3af; letter-spacing: 0.1em; }
p, label, div { color: #c8ccd4; font-size: 0.82rem; }

/* Signal cards */
.sig-card { background: #111116; border: 1px solid #1e1e2e; border-radius: 4px;
             padding: 0.75rem 1rem; margin: 0.25rem 0; }
.sig-go    { border-left: 3px solid #22c55e; }
.sig-no    { border-left: 3px solid #ef4444; }
.sig-wait  { border-left: 3px solid #f59e0b; }

/* Metric overrides */
[data-testid="metric-container"] { background: #111116; border: 1px solid #1e1e2e;
    border-radius: 4px; padding: 0.6rem 0.8rem; }
[data-testid="metric-container"] label { color: #6b7280 !important; font-size: 0.7rem;
    letter-spacing: 0.12em; text-transform: uppercase; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #e2e8f0 !important;
    font-size: 1.3rem; font-weight: 600; }
[data-testid="metric-container"] [data-testid="stMetricDelta"] { font-size: 0.75rem; }

/* Status badge */
.badge-green  { display:inline-block; background:#14532d; color:#4ade80;
                padding:2px 8px; border-radius:3px; font-size:0.72rem; font-weight:600; }
.badge-red    { display:inline-block; background:#450a0a; color:#f87171;
                padding:2px 8px; border-radius:3px; font-size:0.72rem; font-weight:600; }
.badge-amber  { display:inline-block; background:#451a03; color:#fbbf24;
                padding:2px 8px; border-radius:3px; font-size:0.72rem; font-weight:600; }
.badge-grey   { display:inline-block; background:#1e1e2e; color:#6b7280;
                padding:2px 8px; border-radius:3px; font-size:0.72rem; }

/* Table */
.dataframe { background:#111116 !important; color:#c8ccd4 !important; font-size:0.75rem; }
.dataframe th { background:#1e1e2e !important; color:#9ca3af !important; }

/* Input widgets */
.stSelectbox > div > div, .stNumberInput > div > div > input,
.stTextInput > div > div > input { background:#1a1a24 !important; color:#e2e8f0 !important;
    border: 1px solid #2a2a3e !important; font-family: inherit; font-size: 0.8rem; }
.stButton > button { background:#1a2744; color:#93c5fd; border:1px solid #1e3a5f;
    border-radius:3px; font-size:0.78rem; letter-spacing:0.08em; padding:0.4rem 0.9rem; }
.stButton > button:hover { background:#1e3a5f; }

/* Divider */
hr { border-color: #1e1e2e; margin: 0.8rem 0; }

/* Session banner */
.session-banner { background:#1a1a24; border:1px solid #2a2a3e; border-radius:4px;
    padding:0.5rem 1rem; margin-bottom:0.8rem; display:flex; justify-content:space-between;
    align-items:center; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state initialisation
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
        "last_refresh":     0.0,
        "cached_data":      {},
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
    return max(0.0, min(round(k * 0.5, 4), 0.25))   # half-Kelly, capped 25%


def _rolling_stats(journal: list[dict]) -> dict:
    if not journal:
        return {}
    n = len(journal)
    rs = [t["actual_r"] for t in journal]
    wins = [r for r in rs if r > 0]
    losses = [r for r in rs if r < 0]
    wr = len(wins) / n
    aw = statistics.mean(wins)   if wins   else 0.0
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
    return (f'<div class="sig-card sig-{css}">'
            f'<b style="color:#9ca3af;font-size:0.72rem;letter-spacing:.1em">{label}</b>&nbsp;&nbsp;'
            f'{_badge(f"{icon} {status}", badge_col)}'
            f'<br><span style="color:#6b7280;font-size:0.75rem">{detail}</span></div>')


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⬡ SESSION SETUP")
    st.session_state["account"] = st.number_input(
        "Account (£)", value=st.session_state["account"],
        min_value=100.0, step=100.0, format="%.0f")
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
        if st.button("↺ Refresh"):
            st.cache_data.clear()
            st.rerun()
    with col2:
        if st.button("⏹ Reset Session"):
            st.session_state["session_pnl"]   = 0.0
            st.session_state["consec_losses"]  = 0
            st.session_state["stop_session"]   = False
            st.session_state["pause"]          = False
            st.session_state["trade_count"]    = 0
            st.rerun()

    st.markdown("---")
    st.markdown("## ✓ PRE-SESSION CHECKLIST")
    st.session_state["psych_confirmed"] = st.checkbox(
        "Discipline script read",
        value=st.session_state["psych_confirmed"])
    st.caption("PDH/PDL marked  ·  VIX checked  ·  Gap assessed")

    st.markdown("---")
    st.markdown(
        '<span style="color:#374151;font-size:0.65rem">'
        'v2.1.0 · ORB Trading Cockpit · Not financial advice</span>',
        unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# MAIN LAYOUT
# ---------------------------------------------------------------------------

# Session banner
now_est = datetime.utcnow() - timedelta(hours=5)   # approx EST
session_ok = Time(9, 30) <= now_est.time() <= Time(11, 0)
market_str = "MARKET OPEN — ORB WINDOW ACTIVE" if session_ok else (
    "PRE-MARKET" if now_est.time() < Time(9, 30) else "POST ORB WINDOW")
banner_col = "#14532d" if session_ok else "#1a1a24"
st.markdown(
    f'<div class="session-banner" style="border-color:{banner_col}">'
    f'<span style="color:#9ca3af;font-size:0.72rem;letter-spacing:.15em">'
    f'{market_str}</span>'
    f'<span style="color:#4b5563;font-size:0.7rem">'
    f'{now_est.strftime("%H:%M:%S EST")}</span></div>',
    unsafe_allow_html=True)

# ── Fetch data ──
ticker = st.session_state["ticker"]
df_5m = _fetch(ticker, "2d", "5m")
df_1d = _fetch(ticker, "20d", "1d")

# ── SECTION 1: PRE-MARKET INTELLIGENCE ──────────────────────────────────────
st.markdown("## PRE-MARKET")

col_vix, col_pdh, col_pdl, col_gap, col_regime = st.columns(5)

try:
    df_vix = _fetch("^VIX", "2d", "5m")
    vix_val = float(df_vix["Close"].dropna().iloc[-1])
    regime = ("EXTREME" if vix_val > 35 else "HIGH" if vix_val > 25
              else "ELEVATED" if vix_val > 18 else "NORMAL")
    size_mod = (0.25 if vix_val > 35 else 0.50 if vix_val > 25
                else 0.75 if vix_val > 18 else 1.00)
    vix_col = "inverse" if vix_val > 25 else "normal"
    col_vix.metric("VIX", f"{vix_val:.1f}", f"{regime}")
    col_regime.metric("Size modifier", f"{size_mod:.0%}", "of planned size")
except Exception:
    col_vix.metric("VIX", "—")
    regime = "UNKNOWN"; size_mod = 1.0

if len(df_1d) >= 2:
    pdh = float(df_1d["High"].iloc[-2])
    pdl = float(df_1d["Low"].iloc[-2])
    prev_close = float(df_1d["Close"].iloc[-2])
    col_pdh.metric("PDH", f"${pdh:.2f}")
    col_pdl.metric("PDL", f"${pdl:.2f}")
    if not df_5m.empty:
        today_open = float(df_5m["Open"].iloc[0])
        gap_pct = (today_open - prev_close) / prev_close * 100
        gap_dir = "↑" if gap_pct > 0.1 else "↓" if gap_pct < -0.1 else "→"
        gap_col = "#22c55e" if gap_pct > 0.1 else "#ef4444" if gap_pct < -0.1 else "#f59e0b"
        col_gap.metric("Gap", f"{gap_pct:+.2f}%", gap_dir)
else:
    col_pdh.metric("PDH", "—"); col_pdl.metric("PDL", "—"); col_gap.metric("Gap", "—")

# ── SECTION 2: ORB + VWAP ───────────────────────────────────────────────────
st.markdown("## ORB SETUP")

if not df_5m.empty:
    vwap = _vwap_series(df_5m)
    orb  = _orb_range(df_5m, st.session_state["orb_method"])
    vs   = _vwap_slope(vwap, lookback=3)

    col_orb1, col_orb2, col_orb3, col_vwap, col_vs = st.columns(5)
    if orb:
        col_orb1.metric("ORB High", f"${orb['orb_high']:.2f}")
        col_orb2.metric("ORB Low",  f"${orb['orb_low']:.2f}")
        col_orb3.metric("Range",    f"${orb['orb_size']:.2f}",
                        f"{orb['bars_used']} bars")
    else:
        col_orb1.metric("ORB High", "Building…")
        col_orb2.metric("ORB Low",  "Building…")
        col_orb3.metric("Range",    "—")

    last_close = float(df_5m["Close"].dropna().iloc[-1])
    last_vwap  = float(vwap.dropna().iloc[-1]) if not vwap.dropna().empty else 0.0
    col_vwap.metric("VWAP", f"${last_vwap:.2f}")
    slope_col = "#22c55e" if vs["direction"] == "up" else "#ef4444" if vs["direction"] == "down" else "#f59e0b"
    col_vs.metric("VWAP slope", vs["direction"].upper(),
                  f"{vs['slope']:+.4f}/bar")

    # Breakout signal
    if orb:
        if last_close > orb["orb_high"]:
            breakout_signal = "long"
        elif last_close < orb["orb_low"]:
            breakout_signal = "short"
        else:
            breakout_signal = None
    else:
        breakout_signal = None

# ── SECTION 3: AND-GATE SIGNAL STATUS ───────────────────────────────────────
st.markdown("## SIGNAL GATE")

if df_5m.empty or not orb:
    st.info("Waiting for data…")
else:
    # Gate 1: ORB breakout
    if breakout_signal == "long":
        g1_status, g1_detail = "GO",   f"Close ${last_close:.2f} > ORB High ${orb['orb_high']:.2f}"
    elif breakout_signal == "short":
        g1_status, g1_detail = "GO",   f"Close ${last_close:.2f} < ORB Low ${orb['orb_low']:.2f}"
    else:
        g1_status, g1_detail = "WAIT", f"Close ${last_close:.2f} inside range [{orb['orb_low']:.2f}–{orb['orb_high']:.2f}]"

    # Gate 2: VWAP slope aligned with breakout direction
    if breakout_signal == "long"  and vs["direction"] == "up":
        g2_status, g2_detail = "GO",   "VWAP sloping up ↑ — confirms long"
    elif breakout_signal == "short" and vs["direction"] == "down":
        g2_status, g2_detail = "GO",   "VWAP sloping down ↓ — confirms short"
    elif vs["direction"] == "flat":
        g2_status, g2_detail = "NO",   f"VWAP flat — insufficient institutional momentum"
    else:
        g2_status, g2_detail = "NO",   f"VWAP {vs['direction']} conflicts with {breakout_signal or 'no'} signal"

    # Gate 3: Retest (user confirmation)
    g3_status = "WAIT"
    g3_detail = "Awaiting manual retest confirmation"

    and_gate = (g1_status == "GO" and g2_status == "GO" and g3_status == "GO")

    gc1, gc2, gc3 = st.columns(3)
    with gc1:
        st.markdown(_sig_card("① ORB BREAKOUT", g1_status, g1_detail), unsafe_allow_html=True)
    with gc2:
        st.markdown(_sig_card("② VWAP SLOPE", g2_status, g2_detail), unsafe_allow_html=True)
    with gc3:
        retest_confirmed = st.checkbox("③ Retest confirmed", value=False)
        g3_status = "GO" if retest_confirmed else "WAIT"
        g3_detail = "Level held, confirmation candle closed" if retest_confirmed else "Waiting for retest + candle close"
        st.markdown(_sig_card("③ RETEST", g3_status, g3_detail), unsafe_allow_html=True)

    and_gate = (g1_status == "GO" and g2_status == "GO" and g3_status == "GO")
    gate_html = (
        '<div style="margin-top:0.6rem;padding:0.6rem 1rem;'
        f'background:{"#14532d" if and_gate else "#450a0a"};'
        'border-radius:4px;text-align:center">'
        f'<b style="font-size:0.95rem;letter-spacing:.15em;color:{"#4ade80" if and_gate else "#f87171"}">'
        f'{"✓  AND-GATE OPEN — ENTRY PERMITTED" if and_gate else "✕  AND-GATE CLOSED — NO ENTRY"}'
        '</b></div>'
    )
    st.markdown(gate_html, unsafe_allow_html=True)

    in_orb_window = (Time(9, 30) <= now_est.time() <= Time(11, 0))
    if not in_orb_window:
        st.warning("⏰ Outside ORB window (09:30–11:00 EST). No new entries.")

# ── SECTION 4: POSITION CALCULATOR ──────────────────────────────────────────
st.markdown("## POSITION CALCULATOR")

pc1, pc2 = st.columns([1, 1])
with pc1:
    entry_price = st.number_input("Entry price ($)", value=float(last_close) if not df_5m.empty else 100.0, step=0.01, format="%.2f")
    stop_price  = st.number_input("Stop price ($)",  value=round(entry_price * 0.995, 2), step=0.01, format="%.2f")
    target_rr   = st.slider("Target R:R", min_value=1.0, max_value=5.0, value=2.0, step=0.5)

with pc2:
    account = st.session_state["account"]
    risk_mode = st.session_state["risk_mode"]
    risk_dist = abs(entry_price - stop_price)

    # Risk fraction
    if risk_mode == "Fixed 1%":
        risk_frac = 0.01
    elif risk_mode == "Fixed 0.5% (commodity)":
        risk_frac = 0.005
    else:
        # Half-Kelly from journal
        stats = _rolling_stats(st.session_state["journal"])
        if stats.get("n", 0) >= 50:
            risk_frac = _kelly(stats["wr"], stats["aw"], stats["al"])
        else:
            risk_frac = 0.01
            st.caption(f"⚠ Kelly needs 50+ trades ({stats.get('n',0)} logged). Using 1%.")

    # Apply VIX modifier
    effective_risk = risk_frac * size_mod
    risk_amount    = account * effective_risk
    units = risk_amount / risk_dist if risk_dist > 1e-6 else 0.0

    # Target & breakeven
    is_long = stop_price < entry_price
    reward_dist = risk_dist * target_rr
    target_price = round((entry_price + reward_dist) if is_long else (entry_price - reward_dist), 2)
    be_price     = round((entry_price + reward_dist * 0.5) if is_long else (entry_price - reward_dist * 0.5), 2)

    # Ladder targets
    ladders = []
    for r in [1.0, 2.0, 3.0]:
        p = round((entry_price + r * risk_dist) if is_long else (entry_price - r * risk_dist), 2)
        ladders.append({"R": f"{r:.0f}R", "Price": f"${p:.2f}", "Exit": "33%"})

    col_u, col_r, col_t, col_be = st.columns(4)
    col_u.metric("Units",           f"{units:.0f}")
    col_r.metric("£ at risk",       f"£{risk_amount:.0f}",  f"{effective_risk:.1%} eff.")
    col_t.metric("Target",          f"${target_price:.2f}", f"2:1 → {target_rr:.1f}R")
    col_be.metric("Breakeven move", f"${be_price:.2f}",     "move stop here")

    st.markdown("**Ladder exits (VWAP + Ladder):**")
    st.dataframe(pd.DataFrame(ladders), hide_index=True, use_container_width=True)

# ── SECTION 5: RISK STATE ────────────────────────────────────────────────────
st.markdown("## SESSION RISK")

rs1, rs2, rs3, rs4 = st.columns(4)
session_pnl_pct = st.session_state["session_pnl"] / account * 100
rs1.metric("Session P&L", f"£{st.session_state['session_pnl']:.0f}",
           f"{session_pnl_pct:+.1f}%")
rs2.metric("Consec. losses", str(st.session_state["consec_losses"]),
           "pause if 3" if st.session_state["consec_losses"] < 3 else "⏸ PAUSE NOW")
rs3.metric("Trades today", str(st.session_state["trade_count"]))
rs4.metric("Daily limit", "£300 (3%)", f"£{account * 0.03:.0f} max loss")

if st.session_state["stop_session"]:
    st.error("🛑 SESSION STOPPED — daily loss limit hit. Close the platform.")
elif st.session_state["pause"]:
    st.warning("⏸ PAUSE 30 MINUTES — 3 consecutive losses. No new entries.")
elif not st.session_state["psych_confirmed"]:
    st.warning("⚠ Read the pre-session discipline script before trading.")
else:
    st.success("✅ Trading permitted.")

# ── SECTION 6: TRADE JOURNAL ─────────────────────────────────────────────────
st.markdown("## TRADE JOURNAL")

with st.expander("Log a trade", expanded=False):
    lc1, lc2, lc3 = st.columns(3)
    log_ticker  = lc1.text_input("Ticker", value=ticker, key="log_ticker").upper()
    log_dir     = lc1.selectbox("Direction", ["long", "short"], key="log_dir")
    log_entry   = lc2.number_input("Entry ($)", value=entry_price, key="log_entry")
    log_stop    = lc2.number_input("Stop ($)",  value=stop_price,  key="log_stop")
    log_exit    = lc3.number_input("Exit ($)",  value=entry_price, key="log_exit")
    log_outcome = lc3.selectbox("Outcome", ["win", "loss", "breakeven"], key="log_outcome")
    log_notes   = st.text_input("Notes", key="log_notes")

    if st.button("Log trade"):
        risk_d = abs(log_entry - log_stop)
        if risk_d > 1e-6:
            actual_r = round((log_exit - log_entry) / risk_d *
                              (1 if log_dir == "long" else -1), 3)
        else:
            actual_r = 0.0
        trade = {
            "date":     str(date.today()),
            "ticker":   log_ticker,
            "direction": log_dir,
            "entry":    log_entry,
            "stop":     log_stop,
            "exit":     log_exit,
            "actual_r": actual_r,
            "outcome":  log_outcome,
            "notes":    log_notes,
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
        st.session_state["pause"] = (st.session_state["consec_losses"] >= 3
                                      and not st.session_state["stop_session"])
        st.success(f"Logged: {actual_r:+.2f}R")
        st.rerun()

# Rolling stats
if st.session_state["journal"]:
    stats = _rolling_stats(st.session_state["journal"])
    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("Win rate",     f"{stats.get('wr', 0):.1%}",
               "▲" if stats.get("wr", 0) > 0.34 else "▼ below breakeven")
    sc2.metric("EV/trade",     f"{stats.get('ev', 0):+.3f}R",
               "positive" if stats.get("ev", 0) > 0 else "negative")
    sc3.metric("Sharpe",       str(stats.get("sharpe", "—")))
    sc4.metric("Trades logged", str(stats.get("n", 0)),
               f"{max(0, 100 - stats.get('n',0))} to milestone")

    df_j = pd.DataFrame(st.session_state["journal"])
    df_j = df_j[["date", "ticker", "direction", "entry", "exit", "actual_r", "outcome"]]
    styler = df_j.style
    map_func = getattr(styler, "map", getattr(styler, "applymap", None))
    if map_func:
        styler = map_func(
            lambda v: "color:#4ade80" if v == "win" else "color:#f87171" if v == "loss" else "",
            subset=["outcome"]
        )
    st.dataframe(styler, use_container_width=True, hide_index=True)
else:
    st.caption("No trades logged this session.")
