"""
morning_brief.py
================
Pre-market morning routine CLI for the Trading Income Project.
Run before 09:25 EST every trading day. Completes in < 5 minutes.

Usage:
    python src/morning_brief.py            # Default: SPY
    python src/morning_brief.py --ticker QQQ
    python src/morning_brief.py --orb 30min

Prints a structured session brief covering:
  1. VIX regime + position size modifier
  1b. Macro economic calendar (FOMC, OPEX, Early Close)
  2. Previous Day High/Low (PDH/PDL)
  3. Overnight gap size, direction, and quality
  4. SMA trend filter (daily chart)
  5. Session parameters: ORB window, max attempts, risk %
  6. Pre-session psychological discipline script prompt
  7. Confirmed D-A-C patterns & institutional sentiment
  8. Go / No-go recommendation
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone, time as Time

# ── Try toolkit import ──────────────────────────────────────────────────────
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import trading_quant_toolkit_v2_4 as tk
    TOOLKIT = True
except ImportError:
    TOOLKIT = False

try:
    import yfinance as yf
    import pandas as pd
    DATA_DEPS = True
except ImportError:
    DATA_DEPS = False

# ── Terminal colours ────────────────────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"


def _col(text: str, colour: str) -> str:
    return f"{colour}{text}{RESET}"

def _bar(label: str, value: str, colour: str = "") -> None:
    print(f"  {DIM}{label:<24}{RESET} {_col(value, colour) if colour else value}")

def _section(title: str) -> None:
    print(f"\n{BOLD}{CYAN}{'─'*54}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'─'*54}{RESET}")

def _fetch(ticker: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval,
                     auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df


def run_brief(ticker: str = "SPY", orb_method: str = "15min",
              account: float = 10_000.0, risk_pct: float = 0.01) -> None:

    now_est = datetime.now(timezone.utc) - timedelta(hours=5)
    go_signals   = 0
    total_checks = 0

    # Safe fallback default for vix_data
    vix_data = {"vix": 20.0, "regime": "NORMAL", "size_modifier": 1.0}

    # ── HEADER ────────────────────────────────────────────────────────────────
    print(f"\n{BOLD}{'='*54}{RESET}")
    print(f"{BOLD}  ⬡ TRADING MORNING BRIEF  {now_est.strftime('%a %d %b %Y  %H:%M EST')}{RESET}")
    print(f"{BOLD}  Ticker: {ticker}   ORB: {orb_method}   Account: £{account:,.0f}{RESET}")
    print(f"{BOLD}{'='*54}{RESET}")

    if not DATA_DEPS:
        print(f"{RED}  Missing dependencies: pip install yfinance pandas{RESET}")
        return

    # ── 1. VIX REGIME ─────────────────────────────────────────────────────────
    _section("1. VIX REGIME")
    try:
        from fred_store import get_macro_context as _fred_macro
        _db_path = os.environ.get("MARKET_DATA_DB", "DATA/market_data.db")
        _macro   = _fred_macro(_db_path)
        vix      = _macro.get("vix")
        _source  = _macro.get("source", "yfinance")

        if vix is None:
            raise ValueError("VIX unavailable from FRED or yfinance")

        if vix > 35:   regime, mod = "EXTREME",  0.25
        elif vix > 25: regime, mod = "HIGH",      0.50
        elif vix > 18: regime, mod = "ELEVATED",  0.75
        else:          regime, mod = "NORMAL",    1.00

        vix_colour = GREEN if regime == "NORMAL" else YELLOW if regime == "ELEVATED" else RED
        _bar("VIX", f"{vix:.1f}  [{_source}]", vix_colour)
        _bar("Regime", regime, vix_colour)
        _bar("Position size modifier", f"{mod:.0%} of planned", vix_colour)
        _bar("Effective risk", f"{risk_pct * mod:.2%} of account", vix_colour)
        vix_data = {"vix": vix, "regime": regime, "size_modifier": mod}

        # Macro yield & credit spreads if FRED is populated
        if _macro.get("yield_curve") is not None:
            _yc = _macro["yield_curve"]
            _yc_note = "  INVERTED ⚠️" if _yc < 0 else ""
            _bar("Yield curve (10Y-2Y)", f"{_yc:+.3f}%{_yc_note}", RED if _yc < 0 else DIM)
        if _macro.get("hy_spread") is not None:
            _hy = _macro["hy_spread"]
            _hy_note = "  RISK-OFF" if _hy > 500 else ""
            _bar("HY spread", f"{_hy:.2f}%{_hy_note}", RED if _hy > 500 else YELLOW if _hy > 350 else DIM)

        if regime in ("NORMAL", "ELEVATED"):
            go_signals += 1
        total_checks += 1
    except Exception as e:
        _bar("VIX", f"ERROR: {e}", RED)

    # ── 1b. MACRO CALENDAR ───────────────────────────────────────────────────
    _section("1b. MACRO CALENDAR")
    try:
        import pandas_market_calendars as mcal
        _nyse  = mcal.get_calendar("NYSE")
        _today = now_est.date()
        _sched = _nyse.schedule(start_date=str(_today), end_date=str(_today), tz="America/New_York")
        _is_opex = (
            _today.weekday() == 4
            and _today.month in (3, 6, 9, 12)
            and 15 <= _today.day <= 21
        )
        if _is_opex:
            _bar("TODAY", "QUARTERLY OPEX — triple witching", YELLOW)
            _bar("Note", "Elevated volume — ORB range may be wider than normal", YELLOW)
        if not _sched.empty:
            _close = _sched.iloc[0]["market_close"].tz_convert("America/New_York")
            if _close.hour < 16:
                _bar("TODAY", f"EARLY CLOSE at {_close.strftime('%H:%M')} EST", YELLOW)
    except Exception:
        pass

    try:
        today_str = now_est.strftime("%Y-%m-%d")
        FOMC_2026 = {"2026-01-28","2026-03-18","2026-05-06","2026-06-17",
                     "2026-07-29","2026-09-16","2026-11-04","2026-12-16"}
        if today_str in FOMC_2026:
            _bar("TODAY", "FOMC RATE DECISION (High Impact)", RED)
            _bar("Size override", "50% position size recommended", YELLOW)
        else:
            _bar("Macro catalysts", "No major scheduled Fed shocks today", GREEN)
        go_signals += 1
        total_checks += 1
    except Exception as e:
        _bar("Macro calendar", f"ERROR: {e}", RED)

    # ── 2. PRICE LEVELS (Consolidated: Local Store → yfinance Fallback) ──────
    _section("2. PRICE LEVELS (PDH / PDL)")
    pdh_val, pdl_val, prev_date_str = None, None, ""
    try:
        # Check local MarketDataStore first
        from market_data_store import MarketDataStore
        _mds_path = os.environ.get("MARKET_DATA_DB", "DATA/market_data.db")
        if os.path.exists(_mds_path):
            _mds = MarketDataStore(_mds_path)
            _yesterday = (now_est - timedelta(days=1)).strftime("%Y-%m-%d")
            _ctx = _mds.get_session_context(ticker, _yesterday)
            if _ctx and _ctx.get("pdh") and _ctx.get("pdl"):
                pdh_val = _ctx["pdh"]
                pdl_val = _ctx["pdl"]
                prev_date_str = f"Store: {_yesterday}"

        # Fallback to yfinance if not in local store
        if pdh_val is None or pdl_val is None:
            df1d = _fetch(ticker, "5d", "1d")
            if len(df1d) >= 2:
                pdh_val = float(df1d["High"].iloc[-2])
                pdl_val = float(df1d["Low"].iloc[-2])
                prev_date_str = str(df1d.index[-2].date())

        if pdh_val is not None and pdl_val is not None:
            _bar("Date", prev_date_str, DIM)
            _bar("PDH (prev high)", f"${pdh_val:.2f}", CYAN)
            _bar("PDL (prev low)",  f"${pdl_val:.2f}", CYAN)
            _bar("Range", f"${pdh_val - pdl_val:.2f}", DIM)
            print(f"\n  {YELLOW}▶ Mark PDH ${pdh_val:.2f} and PDL ${pdl_val:.2f} on chart now.{RESET}")
            go_signals += 1
        else:
            _bar("PDH/PDL", "Insufficient price data to compute levels", RED)
        total_checks += 1
    except Exception as e:
        _bar("PDH/PDL", f"ERROR: {e}", RED)

    # ── 3. OVERNIGHT GAP ──────────────────────────────────────────────────────
    _section("3. OVERNIGHT GAP")
    try:
        gap_data = tk.fetch_premarket_gap(ticker) if TOOLKIT else None
        if gap_data is None:
            df_pm = yf.download(ticker, period='1d', interval='5m', auto_adjust=True, progress=False)
            if isinstance(df_pm.columns, pd.MultiIndex):
                df_pm.columns = [c[0] if isinstance(c, tuple) else c for c in df_pm.columns]
            df_1d = _fetch(ticker, "5d", "1d")
            prev_close = float(df_1d["Close"].iloc[-2])
            curr_open  = float(df_pm["Open"].iloc[0])
            gap_pct = round((curr_open - prev_close) / prev_close * 100, 3)
            direction = "up" if gap_pct > 0.1 else "down" if gap_pct < -0.1 else "flat"
            gap_data = {"gap_pct": gap_pct, "gap_direction": direction,
                        "prev_close": prev_close, "current_open": curr_open}

        gap_pct   = gap_data["gap_pct"]
        direction = gap_data["gap_direction"]
        gap_colour = GREEN if abs(gap_pct) >= 0.3 else YELLOW if abs(gap_pct) >= 0.1 else DIM
        _bar("Previous close", f"${gap_data['prev_close']:.2f}")
        _bar("Current open",   f"${gap_data['current_open']:.2f}")
        _bar("Gap",            f"{gap_pct:+.2f}%  {direction.upper()}", gap_colour)

        # Gap quality assessment
        if abs(gap_pct) >= 2.0:
            quality, q_col = "A-GRADE (Shocking catalyst)", GREEN
        elif abs(gap_pct) >= 0.5:
            quality, q_col = "B-GRADE (Actionable move)", YELLOW
        elif abs(gap_pct) >= 0.1:
            quality, q_col = "C-GRADE (Minor noise)", DIM
        else:
            quality, q_col = "FLAT (No catalyst)", RED
        _bar("Gap quality", quality, q_col)

        # Window suggestion
        if abs(gap_pct) >= 0.5 and vix_data.get("regime", "") in ("NORMAL", "ELEVATED"):
            rec, rec_col = "15-min ORB (Clean gap, normal volatility)", GREEN
        elif abs(gap_pct) < 0.1 or vix_data.get("vix", 20) > 20:
            rec, rec_col = "30-min ORB ⚠️ (Ambiguous gap or elevated VIX)", YELLOW
        else:
            rec, rec_col = "15-min ORB (Default baseline)", ""
        _bar("Recommended window", rec, rec_col)

        if abs(gap_pct) >= 0.1:
            go_signals += 1
        total_checks += 1
    except Exception as e:
        _bar("Gap", f"ERROR: {e}", RED)

    # ── 4. TREND FILTER (DAILY) ───────────────────────────────────────────────
    _section("4. TREND FILTER (DAILY SMA 20 / 200)")
    try:
        df_daily = _fetch(ticker, "1y", "1d")
        if len(df_daily) >= 200:
            if TOOLKIT:
                trend_data = tk.sma_trend_filter(df_daily, fast=20, slow=200)
            else:
                p = float(df_daily["Close"].iloc[-1])
                s20 = float(df_daily["Close"].rolling(20).mean().iloc[-1])
                s200 = float(df_daily["Close"].rolling(200).mean().iloc[-1])
                af = p > s20
                as_ = p > s200
                t_dir = "up" if (af and as_) else "down" if (not af and not as_) else "mixed"
                trend_data = {"price": p, "sma_fast": s20, "sma_slow": s200, "trend": t_dir,
                              "above_fast": af, "above_slow": as_}

            trend_colour = GREEN if trend_data["trend"] == "up" else RED if trend_data["trend"] == "down" else YELLOW
            _bar("Price",    f"${trend_data['price']:.2f}")
            _bar("SMA(20)",  f"${trend_data['sma_fast']:.2f}")
            _bar("SMA(200)", f"${trend_data['sma_slow']:.2f}")
            _bar("Trend",    trend_data["trend"].upper(), trend_colour)
            _bar("Above 20 SMA",  "YES" if trend_data["above_fast"] else "NO", GREEN if trend_data["above_fast"] else RED)
            _bar("Above 200 SMA", "YES" if trend_data["above_slow"] else "NO", GREEN if trend_data["above_slow"] else RED)

            if trend_data["trend"] in ("up", "mixed"):
                go_signals += 1
            total_checks += 1
        else:
            _bar("Daily trend", f"Insufficient history ({len(df_daily)} bars, need 200)", YELLOW)
    except Exception as e:
        _bar("Trend filter", f"ERROR: {e}", RED)

    # ── 5. SESSION PARAMETERS ─────────────────────────────────────────────────
    _section("5. SESSION SIZING & RISK PARAMETERS")
    effective_risk = risk_pct * vix_data.get("size_modifier", 1.0)
    risk_amount    = account * effective_risk
    window_str     = "09:30-09:44" if orb_method == "15min" else "09:30-09:55" if orb_method == "30min" else "09:30-09:34"

    _bar("ORB window",      f"{window_str} EST")
    _bar("Trading window",  "09:30-11:00 EST (Strict cutoff)")
    _bar("Max attempts",    "2 per direction (long / short)")
    _bar("Risk per trade",  f"{effective_risk:.2%} → £{risk_amount:.0f}")
    _bar("Daily stop",      f"3% → £{account * 0.03:.0f}")
    _bar("Consecutive loss", "3 losses = mandatory 30-min pause")

    # ── 6. CONFIRMED PATTERNS & SENTIMENT ─────────────────────────────────────
    _section("6. RECURRING D-A-C PATTERNS & SENTIMENT")
    try:
        from findings_store import get_confirmed_patterns
        _p_db = os.environ.get("PAPER_ACCOUNT_DB", "DATA/paper_account.db")
        _patterns = get_confirmed_patterns(_p_db)
        if _patterns:
            for _p in _patterns[:3]:
                _bar(f"[{_p['area'].upper()}] {_p['observation'][:40]}",
                     f"Seen {_p['occurrence_count']}x — {(_p.get('implication') or '')[:35]}", YELLOW)
        else:
            _bar("Recurring patterns", "None confirmed yet (requires 3+ recurring findings)", DIM)
    except Exception:
        _bar("Recurring patterns", "findings_store not active", DIM)

    try:
        from sentiment_store import get_sentiment_context as _get_sentiment
        _s_db = os.environ.get("MARKET_DATA_DB", "DATA/market_data.db")
        _sctx = _get_sentiment(_s_db)
        if _sctx.get("equity_pc") is not None:
            _eq = _sctx["equity_pc"]
            _bar(f"CBOE Equity P/C", f"{_eq:.2f} ({_sctx.get('equity_sentiment','Neutral')})",
                 RED if _eq > 0.85 else YELLOW if _eq > 0.70 else GREEN)
        if _sctx.get("lev_net") is not None:
            _bar("CFTC Lev Money Net", f"{_sctx['lev_net']:+,} ({_sctx.get('lev_sentiment','')})", DIM)
    except Exception:
        pass

    # ── 7. PRE-SESSION DISCIPLINE ─────────────────────────────────────────────
    _section("7. PRE-SESSION DISCIPLINE CHECK")
    disciplines = [
        "I am a risk manager first, a trader second.",
        "My job today is to execute the process correctly; P&L will follow.",
        "A losing trade executed according to rules is a successful process.",
        "I will not enter during candle formation — only on closed bars.",
        "If my daily stop is hit, I close the platform immediately.",
    ]
    for d in disciplines:
        print(f"  {DIM}•{RESET} {d}")

    # ── 8. GO / NO-GO ─────────────────────────────────────────────────────────
    _section("8. GO / NO-GO READINESS ASSESSMENT")
    score_pct = (go_signals / max(total_checks, 1)) * 100
    if score_pct >= 75:
        verdict = "✅ GO — Market conditions favourable for breakout execution"
        v_col   = GREEN
    elif score_pct >= 50:
        verdict = "⚠️  CAUTIOUS GO — Reduce position sizing, demand strict entry confirmation"
        v_col   = YELLOW
    else:
        verdict = "🛑 NO-GO — High risk or ambiguous conditions. Stand down today."
        v_col   = RED

    print(f"\n  {BOLD}{_col(verdict, v_col)}{RESET}")
    print(f"  {DIM}Core checks passed: {go_signals}/{total_checks} ({score_pct:.0f}%){RESET}\n")
    print(f"{BOLD}{'='*54}{RESET}\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Pre-market morning brief")
    p.add_argument("--ticker",  default="SPY")
    p.add_argument("--orb",     default="15min", choices=["5min","15min","30min"])
    p.add_argument("--account", type=float, default=10_000.0)
    p.add_argument("--risk",    type=float, default=0.01)
    args = p.parse_args()

    run_brief(ticker=args.ticker.upper(), orb_method=args.orb,
              account=args.account, risk_pct=args.risk)
