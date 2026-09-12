"""
morning_brief.py
================
Pre-market morning routine CLI for the Trading Income Project.
Run before 09:25 EST every trading day. Completes in < 5 minutes.

Usage:
    python morning_brief.py            # Default: SPY
    python morning_brief.py --ticker QQQ
    python morning_brief.py --orb 30min

Prints a structured session brief covering:
  1. VIX regime + position size modifier
  2. Previous Day High/Low (PDH/PDL)
  3. Overnight gap size, direction, and quality
  4. SMA trend filter (daily chart)
  5. Session parameters: ORB window, max attempts, risk %
  6. Pre-session psychological discipline script prompt
  7. Go / No-go recommendation
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta, time as Time

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
    print(f"  {DIM}{label:<22}{RESET} {_col(value, colour) if colour else value}")

def _section(title: str) -> None:
    print(f"\n{BOLD}{CYAN}{'─'*52}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'─'*52}{RESET}")

def _fetch(ticker: str, period: str, interval: str):
    df = yf.download(ticker, period=period, interval=interval,
                     auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c,tuple) else c for c in df.columns]
    return df


def run_brief(ticker: str = "SPY", orb_method: str = "15min",
              account: float = 10_000.0, risk_pct: float = 0.01) -> None:

    now_est = datetime.utcnow() - timedelta(hours=5)
    go_signals   = 0
    total_checks = 0

    # ── HEADER ────────────────────────────────────────────────────────────────
    print(f"\n{BOLD}{'='*52}{RESET}")
    print(f"{BOLD}  ⬡ TRADING MORNING BRIEF  {now_est.strftime('%a %d %b %Y  %H:%M EST')}{RESET}")
    print(f"{BOLD}  Ticker: {ticker}   ORB: {orb_method}   Account: £{account:,.0f}{RESET}")
    print(f"{BOLD}{'='*52}{RESET}")

    if not DATA_DEPS:
        print(f"{RED}  Missing: pip install yfinance pandas{RESET}")
        return

    # ── 1. VIX REGIME ─────────────────────────────────────────────────────────
    _section("1. VIX REGIME")
    try:
        # Prefer FRED local store (authoritative CBOE source, no network call)
        # Falls back to yfinance automatically if FRED not yet downloaded
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
        # Bundle for use in later sections (Fixes NameError on vix_data)
        vix_data = {"vix": vix, "regime": regime, "size_modifier": mod}

        # Additional macro context if FRED data is available
        if _macro.get("yield_curve") is not None:
            _yc = _macro["yield_curve"]
            _yc_note = "  INVERTED" if _yc < 0 else ""
            _bar("Yield curve (10Y-2Y)", f"{_yc:+.3f}%{_yc_note}",
                 RED if _yc < 0 else DIM)
        if _macro.get("hy_spread") is not None:
            _hy = _macro["hy_spread"]
            _hy_note = "  RISK-OFF" if _hy > 500 else ""
            _bar("HY spread", f"{_hy:.2f}%{_hy_note}",
                 RED if _hy > 500 else YELLOW if _hy > 350 else DIM)

        if regime in ("NORMAL", "ELEVATED"):
            go_signals += 1
        total_checks += 1
    except Exception as e:
        _bar("VIX", f"ERROR: {e}", RED)

    # ── 1b. MACRO CALENDAR ───────────────────────────────────────────────
    _section("1b. MACRO CALENDAR")
    # OPEX and early-close detection via pandas_market_calendars
    try:
        import pandas_market_calendars as mcal
        _nyse  = mcal.get_calendar("NYSE")
        _today = (datetime.utcnow() - timedelta(hours=5)).date()
        _sched = _nyse.schedule(start_date=str(_today), end_date=str(_today),
                                tz="America/New_York")
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
                _close_str = _close.strftime("%H:%M")
                _bar("TODAY", f"EARLY CLOSE at {_close_str} EST", YELLOW)
    except Exception:
        pass
    macro_day = False
    try:
        today_str = (datetime.utcnow() - timedelta(hours=5)).strftime("%Y-%m-%d")
        FOMC_2026 = {"2026-01-28","2026-03-18","2026-05-06","2026-06-17",
                     "2026-07-29","2026-09-16","2026-11-04","2026-12-16"}
        macro_day = today_str in FOMC_2026
        if macro_day:
            _bar("TODAY", "FOMC / HIGH-IMPACT EVENT", RED)
            _bar("Size override", "50% of planned position size", YELLOW)
            _bar("VWAP gate", "Tighten to range_factor=0.15", YELLOW)
        else:
            _bar("Macro events", "None detected today", GREEN)
        go_signals += 1
        total_checks += 1
    except Exception as e:
        _bar("Macro calendar", f"ERROR: {e}", RED)

    # ── 2. PDH/PDL (market_data_store → yfinance fallback) ───────────────────────
    _section("2. PRICE LEVELS")
    try:
        from market_data_store import MarketDataStore
        _mds_path = os.environ.get("MARKET_DATA_DB", "DATA/market_data.db")
        _mds = MarketDataStore(_mds_path)
        _yesterday = (datetime.utcnow() - timedelta(hours=5) - timedelta(days=1)).strftime("%Y-%m-%d")
        _ctx = _mds.get_session_context(ticker, _yesterday)
        if _ctx and _ctx.get("orb_15_high"):
            pdh = _ctx["pdh"] or 0
            pdl = _ctx["pdl"] or 0
            gap = _ctx.get("gap_pct", 0) or 0
            _bar("PDH (prev day high)", f"${pdh:.2f}", DIM)
            _bar("PDL (prev day low)",  f"${pdl:.2f}", DIM)
            _gap_col = RED if abs(gap) > 1.5 else YELLOW if abs(gap) > 0.5 else GREEN
            _bar("Gap vs prev close",   f"{gap:+.2f}%", _gap_col)
            go_signals += 1
        else:
            _bar("PDH/PDL", "market_data_store not populated — run data bootstrap", YELLOW)
        total_checks += 1
    except Exception as _e:
        _bar("PDH/PDL", f"store error: {_e}", YELLOW)

    # ── 2. PDH/PDL ────────────────────────────────────────────────────────────
    try:
        pdh_data = tk.fetch_pdh_pdl(ticker) if TOOLKIT else None
        if pdh_data is None:
            df1d = _fetch(ticker, "5d", "1d")
            pdh = float(df1d["High"].iloc[-2])
            pdl = float(df1d["Low"].iloc[-2])
            pdh_data = {"pdh": pdh, "pdl": pdl, "prev_date": str(df1d.index[-2].date()),
                        "range": round(pdh - pdl, 4)}
        _bar("Date",  pdh_data["prev_date"])
        _bar("PDH",   f"${pdh_data['pdh']:.2f}")
        _bar("PDL",   f"${pdh_data['pdl']:.2f}")
        _bar("Range", f"${pdh_data['range']:.2f}")
        print(f"\n  {YELLOW}▶ Mark PDH ${pdh_data['pdh']:.2f} and PDL ${pdh_data['pdl']:.2f} on chart now.{RESET}")
        go_signals += 1; total_checks += 1
    except Exception as e:
        _bar("PDH/PDL", f"ERROR: {e}", RED)

    # ── 3. OVERNIGHT GAP ──────────────────────────────────────────────────────
    _section("3. OVERNIGHT GAP")
    try:
        gap_data = tk.fetch_premarket_gap(ticker) if TOOLKIT else None
        if gap_data is None:
            df_pm = yf.download(ticker, period='1d', interval='5m', auto_adjust=True, progress=False)
            if isinstance(df_pm.columns, pd.MultiIndex):
                df_pm.columns = [c[0] if isinstance(c,tuple) else c for c in df_pm.columns]
            df_1d = _fetch(ticker, "5d", "1d")
            prev_close = float(df_1d["Close"].iloc[-2])
            curr_open  = float(df_pm["Open"].iloc[0])
            gap_pct = round((curr_open - prev_close) / prev_close * 100, 3)
            direction = "up" if gap_pct > 0.1 else "down" if gap_pct < -0.1 else "flat"
            gap_data = {"gap_pct": gap_pct, "gap_direction": direction,
                        "prev_close": prev_close, "current_open": curr_open}

        gap_pct = gap_data["gap_pct"]
        direction = gap_data["gap_direction"]
        gap_colour = GREEN if abs(gap_pct) >= 0.3 else YELLOW if abs(gap_pct) >= 0.1 else DIM
        _bar("Previous close", f"${gap_data['prev_close']:.2f}")
        _bar("Current open",   f"${gap_data['current_open']:.2f}")
        _bar("Gap",            f"{gap_pct:+.2f}%  {direction.upper()}", gap_colour)

        # Gap quality assessment
        if abs(gap_pct) >= 2.0:
            quality, q_col = "A-GRADE (shocking)", GREEN
        elif abs(gap_pct) >= 0.5:
            quality, q_col = "B-GRADE (meaningful)", YELLOW
        elif abs(gap_pct) >= 0.1:
            quality, q_col = "C-GRADE (minor)", DIM
        else:
            quality, q_col = "FLAT (no catalyst)", RED
        _bar("Gap quality", quality, q_col)

        # ORB window recommendation
        if abs(gap_pct) >= 0.5 and vix_data.get("regime","") in ("NORMAL","ELEVATED"):
            rec = "15-min ORB ✅ (clean gap, normal vol)"
            rec_col = GREEN
        elif abs(gap_pct) < 0.1 or vix_data.get("vix",20) > 20:
            rec = "30-min ORB ⚠️  (ambiguous gap or elevated VIX)"
            rec_col = YELLOW
        else:
            rec = f"15-min ORB (default)"
            rec_col = ""
        _bar("Recommended ORB", rec, rec_col)

        if abs(gap_pct) >= 0.1:
            go_signals += 1
        total_checks += 1
    except Exception as e:
        _bar("Gap", f"ERROR: {e}", RED)

    # ── 4. TREND FILTER (DAILY) ───────────────────────────────────────────────
    _section("4. TREND FILTER (DAILY)")
    try:
        df_daily = _fetch(ticker, "1y", "1d")
        if len(df_daily) >= 200 and TOOLKIT:
            trend_data = tk.sma_trend_filter(df_daily, fast=20, slow=200)
            trend_colour = GREEN if trend_data["trend"] == "up" else RED if trend_data["trend"] == "down" else YELLOW
            _bar("Price",    f"${trend_data['price']:.2f}")
            _bar("SMA(20)",  f"${trend_data['sma_fast']:.2f}")
            _bar("SMA(200)", f"${trend_data['sma_slow']:.2f}")
            _bar("Trend",    trend_data["trend"].upper(), trend_colour)
            _bar("Above 20 SMA", "YES" if trend_data["above_fast"] else "NO",
                 GREEN if trend_data["above_fast"] else RED)
            _bar("Above 200 SMA", "YES" if trend_data["above_slow"] else "NO",
                 GREEN if trend_data["above_slow"] else RED)
            if trend_data["trend"] in ("up", "mixed"):
                go_signals += 1
            total_checks += 1
        else:
            _bar("Daily trend", "Insufficient data (need 200 days)")
    except Exception as e:
        _bar("Trend filter", f"ERROR: {e}", RED)

    # ── 5. SESSION PARAMETERS ─────────────────────────────────────────────────
    _section("5. TODAY'S SESSION PARAMETERS")
    effective_risk = risk_pct * vix_data.get("size_modifier", 1.0)
    risk_amount    = account * effective_risk
    _bar("ORB window",     f"09:30-{'09:44' if orb_method=='15min' else '09:55' if orb_method=='30min' else '09:34'} EST")
    _bar("Trading window", "09:30-11:00 EST")
    _bar("Max attempts",   "2 per direction (long / short)")
    _bar("Risk per trade", f"{effective_risk:.2%} → £{risk_amount:.0f}")
    _bar("Daily stop",     f"3% → £{account*0.03:.0f}")
    _bar("3-loss pause",   "30-minute mandatory pause")

    # ── 6. PSYCHOLOGICAL DISCIPLINE ───────────────────────────────────────────
    _section("6. PRE-SESSION DISCIPLINE")
    script_items = [
        "I am a risk manager first, a trader second.",
        "My job today: execute process correctly.",
        "A losing trade executed perfectly = success.",
        "A winning trade taken recklessly = failure.",
        "I will only trade my one defined setup.",
        "Boredom is not a valid reason to enter.",
        "I will not enter during candle formation.",
        "If I hit my daily stop, I close the platform.",
        "The market will be here tomorrow. My capital must be too.",
    ]
    for item in script_items:
        print(f"  {DIM}•{RESET} {item}")

    print(f"\n  {BOLD}{YELLOW}▶ Have you read the full discipline script?{RESET}")
    print(f"  {DIM}(P1-005: print it and place it beside your monitor){RESET}")

    # ── Confirmed D-A-C patterns ──────────────────────────────────────────────
    _section("CONFIRMED PATTERNS (D-A-C)")
    try:
        from findings_store import get_confirmed_patterns
        import os as _os
        _db = _os.environ.get("PAPER_ACCOUNT_DB", "DATA/paper_account.db")
        _patterns = get_confirmed_patterns(_db)
        if _patterns:
            for _p in _patterns[:4]:
                _bar(
                    f"[{_p['area'].upper()}] {_p['observation'][:50]}",
                    f"seen {_p['occurrence_count']}x — {(_p.get('implication') or '')[:40]}",
                    YELLOW,
                )
        else:
            _bar("No confirmed patterns yet", "need 3+ recurring D-A-C findings", DIM)
    except Exception:
        _bar("Confirmed patterns", "findings_store unavailable", DIM)

    # ── Market sentiment (CBOE P/C + CFTC COT) ─────────────────────────────────
    _section("MARKET SENTIMENT")
    try:
        from sentiment_store import get_sentiment_context as _get_sentiment
        _db_path = os.environ.get("MARKET_DATA_DB", "DATA/market_data.db")
        _sctx    = _get_sentiment(_db_path)
        if _sctx.get("equity_pc") is not None:
            _eq = _sctx["equity_pc"]
            _eq_col = RED if _eq > 0.85 else YELLOW if _eq > 0.70 else DIM
            _bar(f"CBOE equity P/C ({_sctx.get('cboe_date','')[:10]})",
                 f"{_eq:.2f}  {_sctx.get('equity_sentiment','')}",
                 _eq_col)
        if _sctx.get("lev_net") is not None:
            _lev = _sctx["lev_net"]
            _lev_col = RED if abs(_lev) > 200_000 else YELLOW
            _bar(f"CFTC lev funds ({_sctx.get('cot_date','')[:10]})",
                 f"{_lev:+,} net  ({_sctx.get('lev_pct_long')}% long)"
                 f"  {_sctx.get('lev_sentiment','')}",
                 _lev_col)
        if _sctx.get("asset_net") is not None:
            _bar("Asset mgr net", f"{_sctx['asset_net']:+,}", DIM)
        if not _sctx:
            _bar("Sentiment", "Run: python src/sentiment_store.py --download", DIM)
    except Exception as _se:
        _bar("Sentiment", f"sentiment_store unavailable: {_se}", DIM)

    # ── 7. GO / NO-GO ─────────────────────────────────────────────────────────
    _section("7. GO / NO-GO ASSESSMENT")
    score_pct = go_signals / max(total_checks, 1) * 100
    if score_pct >= 75:
        verdict = "✅ GO — conditions favourable for trading"
        v_col   = GREEN
    elif score_pct >= 50:
        verdict = "⚠️  CAUTIOUS GO — reduce position size, be selective"
        v_col   = YELLOW
    else:
        verdict = "🛑 NO-GO — unfavourable conditions. Consider sitting out."
        v_col   = RED
    print(f"\n  {BOLD}{_col(verdict, v_col)}{RESET}")
    print(f"  {DIM}Checks passed: {go_signals}/{total_checks}{RESET}")
    print(f"\n{BOLD}{'='*52}{RESET}\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Pre-market morning brief")
    p.add_argument("--ticker",  default="SPY")
    p.add_argument("--orb",     default="15min", choices=["5min","15min","30min"])
    p.add_argument("--account", type=float, default=10_000.0)
    p.add_argument("--risk",    type=float, default=0.01)
    args = p.parse_args()
    run_brief(ticker=args.ticker.upper(), orb_method=args.orb,
              account=args.account, risk_pct=args.risk)
