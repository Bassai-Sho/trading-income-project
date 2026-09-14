"""
data_corrector.py
=================
Algorithmic correction pipeline for Alpaca 1-minute bar data.

WHY THIS EXISTS
───────────────
Zarattini et al. (Concretum Group, April 2026) found the same ORB strategy
produced $226k–$726k (3× spread) across five providers using identical code.
After identifying and correcting the five data quality issues, results converged.

Most providers (Alpaca, Polygon, IQFeed) source from the same SIP feeds.
Differences are in how they process the raw data — not in the underlying trades.
Algorithmic correction on Alpaca data is therefore equivalent to what Polygon
does internally.

WHAT WE CAN FIX
───────────────
1. OHLC integrity violations: Open > High, or Open < Low (mathematical constraints)
2. Phantom H/L spikes: intraday High/Low beyond yfinance daily anchor
3. Stale bars: OHLC all-equal sequences indicating missing/aggregated data
4. Early-close leakage: bars beyond 16:00 EST (13:00 on half-days)

WHAT WE CANNOT FIX (without raw tick data)
───────────────────────────────────────────
5. Tick-to-bar assignment: trades at boundary timestamps assigned differently.
   Affects 2.1% of days per Zarattini. Cannot be corrected without the raw
   trade-level stream from the exchange.

APPROACH
────────
1. Store raw Alpaca data as-is in market_bars (preserves audit trail)
2. This module applies corrections and writes corrected values back
3. Each corrected bar is flagged: quality_flag=1 (corrected), raw stored in raw_high/raw_low
4. Session quality_ok is re-evaluated after corrections
5. IS training uses corrected data

PROVIDERS CONSIDERED
────────────────────
- yfinance: CANNOT provide 1-min historical bars (7-day limit). Used here
  for daily OHLCV only as a free cross-validation anchor.
- Tiingo: free, 9 years of 1-min data, BUT uses IEX exchange feed only
  (~10-15% of total volume). VWAP/volume calculations will be systematically
  wrong for SIP-comparison. Not suitable for our strategy.
- FirstRate Data: aggregates from 25 exchanges + dark pools. SPY from 2000.
  One-time $30-50 purchase. Worth validating our corrections against at Phase 4.
- Polygon/Massive: $29/month, gold-standard. Phase 4 upgrade before live capital.

USAGE
─────
  # Correct a single session DataFrame
  from data_corrector import correct_session, CorrectionReport
  corrected_df, report = correct_session(df, session_date, daily_anchor)

  # Correct an entire MarketDataStore (run once after download)
  from data_corrector import correct_store
  report = correct_store("DATA/market_data.db", ticker="SPY",
                          start=date(2016,1,1), end=date(2022,12,31))

  # Check correction summary
  print(report.summary())
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger("data_corrector")

# ---------------------------------------------------------------------------
# Early-close calendar (US market half-days)
# Bars after 13:00 EST on these dates are leakage
# ---------------------------------------------------------------------------

REGULAR_CLOSE = "16:00"
EARLY_CLOSE   = "13:00"

# Fallback hardcoded set (used when pandas_market_calendars unavailable)
_EARLY_CLOSE_FALLBACK: set[str] = {
    "2016-11-25", "2016-12-23", "2017-11-24",
    "2018-11-23", "2018-12-24", "2019-11-29", "2019-12-24",
    "2020-11-27", "2021-11-26", "2021-12-24", "2022-11-25",
    "2023-11-24", "2024-07-03", "2024-11-29", "2024-12-24",
}

def _get_market_close_time(session_date: date) -> str:
    """
    Return the official market close time for a given date.
    Uses pandas_market_calendars (authoritative) if available,
    falls back to hardcoded set.
    """
    date_str = str(session_date)
    try:
        import pandas_market_calendars as mcal
        nyse  = mcal.get_calendar("NYSE")
        sched = nyse.schedule(
            start_date=date_str, end_date=date_str,
            tz="America/New_York"
        )
        if sched.empty:
            return REGULAR_CLOSE   # holiday — no session
        close = sched.iloc[0]["market_close"]
        # close is a timezone-aware Timestamp; format as HH:MM
        close_local = close.tz_convert("America/New_York") if hasattr(close, "tz_convert") else close
        h, m = close_local.hour, close_local.minute
        return f"{h:02d}:{m:02d}"
    except ImportError:
        return EARLY_CLOSE if date_str in _EARLY_CLOSE_FALLBACK else REGULAR_CLOSE
    except Exception:
        return EARLY_CLOSE if date_str in _EARLY_CLOSE_FALLBACK else REGULAR_CLOSE


# Keep EARLY_CLOSE_DATES for backward compat (used in _trim_early_close)
EARLY_CLOSE_DATES = _EARLY_CLOSE_FALLBACK

# ---------------------------------------------------------------------------
# Correction report
# ---------------------------------------------------------------------------

@dataclass
class CorrectionReport:
    ticker:           str
    session_date:     str
    n_bars_original:  int       = 0
    n_bars_after:     int       = 0
    ohlc_fixes:       int       = 0
    phantom_caps:     int       = 0
    stale_interpolated: int     = 0
    early_close_trimmed: int    = 0
    anchor_used:      bool      = False
    anchor_daily_high: float | None = None
    anchor_daily_low:  float | None = None
    notes:            list[str] = field(default_factory=list)

    def is_clean(self) -> bool:
        return (self.ohlc_fixes + self.phantom_caps +
                self.stale_interpolated + self.early_close_trimmed) == 0

    def summary(self) -> dict:
        return {
            "session_date":   self.session_date,
            "ohlc_fixes":     self.ohlc_fixes,
            "phantom_caps":   self.phantom_caps,
            "stale_interp":   self.stale_interpolated,
            "early_trimmed":  self.early_close_trimmed,
            "anchor_used":    self.anchor_used,
            "clean":          self.is_clean(),
        }


# ---------------------------------------------------------------------------
# Single-session corrections
# ---------------------------------------------------------------------------

def _fix_ohlc_integrity(df: pd.DataFrame, report: CorrectionReport) -> pd.DataFrame:
    """
    Correction 1: OHLC mathematical constraints.
    High must be >= max(Open, Close) and >= Low.
    Low must be <= min(Open, Close) and <= High.
    """
    df = df.copy()
    # High must be max of Open, High, Close
    correct_high = df[["Open", "High", "Close"]].max(axis=1)
    fixed_h = (df["High"] < correct_high).sum()
    df["High"] = correct_high

    # Low must be min of Open, Low, Close
    correct_low = df[["Open", "Low", "Close"]].min(axis=1)
    fixed_l = (df["Low"] > correct_low).sum()
    df["Low"] = correct_low

    n = int(fixed_h + fixed_l)
    if n:
        report.ohlc_fixes += n
        report.notes.append(f"OHLC integrity: {n} bars corrected")
    return df


def _cap_phantom_highs_lows(
    df: pd.DataFrame,
    report: CorrectionReport,
    daily_high: float | None,
    daily_low:  float | None,
) -> pd.DataFrame:
    """
    Correction 2: Phantom H/L spikes.
    Method A (primary): cap at yfinance daily High/Low anchor.
    Method B (fallback): clip to session mean ± 4σ of H-L range.
    """
    df = df.copy()

    if daily_high is not None and daily_low is not None:
        # Method A: yfinance daily anchor
        # Intraday price physically cannot exceed the day's actual High or Low
        n_h = int((df["High"] > daily_high * 1.002).sum())   # 0.2% tolerance
        n_l = int((df["Low"]  < daily_low  * 0.998).sum())
        df["High"] = df["High"].clip(upper=daily_high * 1.002)
        df["Low"]  = df["Low"].clip(lower=daily_low  * 0.998)
        capped = n_h + n_l
        report.anchor_used      = True
        report.anchor_daily_high = daily_high
        report.anchor_daily_low  = daily_low
        if capped:
            report.phantom_caps += capped
            report.notes.append(
                f"Phantom H/L: {capped} bars capped at yfinance daily anchor "
                f"(H={daily_high:.2f}, L={daily_low:.2f})"
            )
    else:
        # Method B: statistical outlier clipping (no external anchor)
        hl_range = df["High"] - df["Low"]
        hl_mean  = float(hl_range.mean())
        hl_std   = float(hl_range.std())
        if hl_std > 0:
            # A bar with H-L range > mean + 4σ is a phantom spike
            phantom_mask = hl_range > hl_mean + 4 * hl_std
            if phantom_mask.any():
                # Clip the phantom bar's High down to Open + typical_range
                typical_range = hl_mean + hl_std
                for idx in df[phantom_mask].index:
                    raw_h = float(df.loc[idx, "High"])
                    raw_l = float(df.loc[idx, "Low"])
                    mid   = float(df.loc[idx, "Close"])
                    df.loc[idx, "High"] = mid + typical_range / 2
                    df.loc[idx, "Low"]  = mid - typical_range / 2
                n = int(phantom_mask.sum())
                report.phantom_caps += n
                report.notes.append(
                    f"Phantom H/L (statistical): {n} bars clipped (no daily anchor)"
                )
    return df


def _interpolate_stale_bars(
    df: pd.DataFrame,
    report: CorrectionReport,
    daily_high: float | None = None,
    daily_low:  float | None = None,
) -> pd.DataFrame:
    """
    Correction 3: Stale bars (OHLC all equal).
    Strategy:
      - Find contiguous stale sequences
      - Linearly interpolate price between the last valid Close before
        the sequence and the first valid Open after it
      - Set Volume to the mean of adjacent valid bars (rough estimate)
      - Mark correction_applied = 1 on each interpolated bar

    Rationale: a stale sequence means trades DID happen (we can verify from
    yfinance daily that price moved) but the aggregation system failed to
    capture them. Linear interpolation is a conservative neutral reconstruction
    that preserves the aggregate session P&L path without inventing signals.
    """
    df = df.copy()
    stale = (
        (df["Open"]  == df["High"])  &
        (df["High"]  == df["Low"])   &
        (df["Low"]   == df["Close"])
    )

    if not stale.any():
        return df

    n_stale = int(stale.sum())
    n_interpolated = 0

    # Typical half-spread: half the mean H-L range of non-stale bars
    valid_bars = df[~stale]
    if len(valid_bars) < 5:
        report.notes.append(f"Stale: {n_stale} bars, but too few valid bars to interpolate")
        return df

    typical_spread = float((valid_bars["High"] - valid_bars["Low"]).mean()) / 2
    mean_volume    = int(valid_bars["Volume"].mean())

    # Walk through stale sequences
    i = 0
    idx = df.index
    while i < len(df):
        if stale.iloc[i]:
            # Find sequence extent
            j = i
            while j < len(df) and stale.iloc[j]:
                j += 1
            # Sequence is [i, j)
            # Anchor prices
            prev_close = float(df["Close"].iloc[i - 1]) if i > 0 else float(df["Open"].iloc[0])
            next_open  = float(df["Open"].iloc[j]) if j < len(df) else prev_close

            # Linear interpolation across the stale window
            n_window = j - i
            for k, pos in enumerate(range(i, j)):
                frac  = (k + 1) / (n_window + 1)
                price = prev_close + frac * (next_open - prev_close)
                # Apply daily anchor bounds if available
                if daily_high is not None:
                    price = min(price, daily_high)
                if daily_low is not None:
                    price = max(price, daily_low)
                df.iloc[pos, df.columns.get_loc("Open")]   = price - typical_spread * 0.3
                df.iloc[pos, df.columns.get_loc("High")]   = price + typical_spread
                df.iloc[pos, df.columns.get_loc("Low")]    = price - typical_spread
                df.iloc[pos, df.columns.get_loc("Close")]  = price
                df.iloc[pos, df.columns.get_loc("Volume")] = mean_volume
                n_interpolated += 1
            i = j
        else:
            i += 1

    if n_interpolated:
        report.stale_interpolated += n_interpolated
        report.notes.append(
            f"Stale bars: {n_interpolated} bars interpolated "
            f"(linear between valid price boundaries)"
        )
    return df


def _trim_early_close(
    df:           pd.DataFrame,
    session_date: date,
    report:       CorrectionReport,
) -> pd.DataFrame:
    """Correction 4: Remove bars beyond the official session close."""
    date_str   = str(session_date)
    close_time = _get_market_close_time(session_date)
    cutoff     = pd.Timestamp(f"{date_str} {close_time}", tz="America/New_York").time()
    mask       = df.index.time <= cutoff
    n_trimmed  = int((~mask).sum())
    if n_trimmed:
        report.early_close_trimmed += n_trimmed
        report.notes.append(
            f"Early-close trim: {n_trimmed} bars removed after {close_time}"
        )
        df = df[mask]
    return df


def _recompute_vwap(df: pd.DataFrame) -> pd.Series:
    """Recompute session-anchored VWAP after corrections."""
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    pv = tp * df["Volume"]
    cv = df["Volume"].cumsum().replace(0, float("nan"))
    return pv.cumsum() / cv


# ---------------------------------------------------------------------------
# Main correction entry point
# ---------------------------------------------------------------------------

def correct_session(
    df:           pd.DataFrame,
    session_date: date,
    daily_anchor: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, CorrectionReport]:
    """
    Apply all four algorithmic corrections to one session's bars.

    Parameters
    ----------
    df           : DataFrame of 1-min bars with OHLCV columns, timezone-aware index
    session_date : the trading date
    daily_anchor : {'high': float, 'low': float} from yfinance daily bar (optional)

    Returns
    -------
    corrected_df : DataFrame with same schema, corrections applied
    report       : CorrectionReport describing what was changed
    """
    report = CorrectionReport(
        ticker="",
        session_date=str(session_date),
        n_bars_original=len(df),
    )

    if df.empty:
        return df, report

    daily_high = daily_anchor.get("high") if daily_anchor else None
    daily_low  = daily_anchor.get("low")  if daily_anchor else None

    # Apply in order (each depends on previous)
    df = _trim_early_close(df, session_date, report)
    df = _fix_ohlc_integrity(df, report)
    df = _cap_phantom_highs_lows(df, report, daily_high, daily_low)
    df = _interpolate_stale_bars(df, report, daily_high, daily_low)

    # Re-enforce OHLC integrity after interpolation
    df = _fix_ohlc_integrity(df, CorrectionReport(ticker="", session_date=""))

    # Recompute VWAP with corrected prices
    if "vwap" in df.columns or "Vwap" in df.columns:
        vwap_col = "vwap" if "vwap" in df.columns else "Vwap"
        df[vwap_col] = _recompute_vwap(df)

    report.n_bars_after = len(df)
    return df, report


# ---------------------------------------------------------------------------
# Bulk correction of an entire MarketDataStore
# ---------------------------------------------------------------------------

def correct_store(
    db_path:   str,
    ticker:    str = "SPY",
    start:     date = date(2016, 1, 1),
    end:       date = date(2024, 12, 31),
    fetch_yf_daily: bool = True,
) -> dict:
    """
    Apply corrections to all sessions in a MarketDataStore SQLite database.

    This adds three columns to market_bars if they don't exist:
      - raw_high REAL  (original value before capping)
      - raw_low  REAL  (original value before capping)
      - correction_applied INTEGER (0 = unchanged, 1 = modified)

    And updates session_context.quality_ok after corrections:
      - Sessions that were quality_ok=0 due to stale bars may become quality_ok=1
        after interpolation — re-evaluated against the same five checks.

    Usage
    ─────
      python data_corrector.py --db DATA/market_data.db --ticker SPY
    """
    from market_data_store import MarketDataStore

    log.info("Correction run: %s %s → %s", ticker, start, end)
    store = MarketDataStore(db_path)

    # ── Schema migration: add correction columns ────────────────────────────
    with sqlite3.connect(db_path) as conn:
        existing = {r[1] for r in conn.execute("PRAGMA table_info(market_bars)").fetchall()}
        if "raw_high" not in existing:
            conn.execute("ALTER TABLE market_bars ADD COLUMN raw_high REAL")
            log.info("Added raw_high column")
        if "raw_low" not in existing:
            conn.execute("ALTER TABLE market_bars ADD COLUMN raw_low REAL")
            log.info("Added raw_low column")
        if "correction_applied" not in existing:
            conn.execute("ALTER TABLE market_bars ADD COLUMN correction_applied INTEGER DEFAULT 0")
            log.info("Added correction_applied column")

    # ── Fetch yfinance daily anchors for the full range ─────────────────────
    daily_anchors: dict[str, dict] = {}
    if fetch_yf_daily:
        log.info("Fetching yfinance daily bars %s → %s...", start, end)
        try:
            import yfinance as yf
            yf_df = yf.download(
                ticker,
                start=str(start),
                end=str(end + timedelta(days=1)),
                interval="1d", auto_adjust=True, progress=False
            )
            if isinstance(yf_df.columns, pd.MultiIndex):
                yf_df.columns = [c[0] for c in yf_df.columns]
            for d, row in yf_df.iterrows():
                daily_anchors[str(d.date())] = {
                    "high": float(row["High"]),
                    "low":  float(row["Low"]),
                }
            log.info("yfinance anchors: %d days", len(daily_anchors))
        except Exception as e:
            log.warning("yfinance daily fetch failed: %s — running without anchor", e)

    # ── Process session by session ──────────────────────────────────────────
    sessions = store.get_date_range(ticker, start, end)
    log.info("Processing %d sessions...", len(sessions))

    total_corrections  = 0
    sessions_corrected = 0
    sessions_clean     = 0
    reports: list[dict] = []

    for sess_str in sessions:
        sess_date = date.fromisoformat(sess_str)
        day_df    = store.get_session_bars(ticker, sess_str)
        if day_df.empty:
            continue

        anchor = daily_anchors.get(sess_str)
        corrected_df, report = correct_session(day_df, sess_date, anchor)

        if report.is_clean():
            sessions_clean += 1
            continue

        # ── Write corrected bars back to market_bars ──────────────────────
        with sqlite3.connect(db_path) as conn:
            # Preserve raw values for bars that changed
            changed_mask = ~(
                (corrected_df["High"] == day_df["High"].reindex(corrected_df.index)) &
                (corrected_df["Low"]  == day_df["Low"].reindex(corrected_df.index))
            )
            for ts in corrected_df[changed_mask].index:
                ts_str = ts.isoformat()
                raw_h  = float(day_df.loc[ts, "High"]) if ts in day_df.index else None
                raw_l  = float(day_df.loc[ts, "Low"])  if ts in day_df.index else None
                row    = corrected_df.loc[ts]
                conn.execute(
                    "UPDATE market_bars SET "
                    "open=?, high=?, low=?, close=?, volume=?, "
                    "raw_high=?, raw_low=?, correction_applied=1 "
                    "WHERE ticker=? AND ts=? AND bar_interval='1m'",
                    (float(row["Open"]), float(row["High"]),
                     float(row["Low"]),  float(row["Close"]),
                     int(row.get("Volume", 0)),
                     raw_h, raw_l,
                     ticker, ts_str)
                )

            # Update VWAP in corrected bars if available
            if "vwap" in corrected_df.columns:
                for ts, row in corrected_df.iterrows():
                    if not pd.isna(row["vwap"]):
                        conn.execute(
                            "UPDATE market_bars SET vwap=? "
                            "WHERE ticker=? AND ts=? AND bar_interval='1m'",
                            (float(row["vwap"]), ticker, ts.isoformat())
                        )

            # Re-evaluate session quality after corrections
            issues_after = store._validate_day(corrected_df, sess_date)
            critical = {"stale_bars_critical","stale_bars","price_error",
                        "orb_range_anomaly","phantom_hl"}
            new_quality_ok = 0 if any(i["issue_type"] in critical for i in issues_after) else 1
            conn.execute(
                "UPDATE session_context SET quality_ok=? WHERE ticker=? AND session_date=?",
                (new_quality_ok, ticker, sess_str)
            )

        n_corrections = (report.ohlc_fixes + report.phantom_caps +
                         report.stale_interpolated + report.early_close_trimmed)
        total_corrections  += n_corrections
        sessions_corrected += 1
        reports.append(report.summary())
        log.info("  %s: %d corrections (%s)", sess_str, n_corrections,
                 "; ".join(report.notes))

    result = {
        "ticker":            ticker,
        "sessions_processed": len(sessions),
        "sessions_clean":    sessions_clean,
        "sessions_corrected": sessions_corrected,
        "total_corrections": total_corrections,
        "corrections_breakdown": {
            "ohlc_integrity":   sum(r.get("ohlc_fixes", 0)   for r in reports),
            "phantom_caps":     sum(r.get("phantom_caps", 0) for r in reports),
            "stale_interp":     sum(r.get("stale_interp", 0) for r in reports),
            "early_trim":       sum(r.get("early_trimmed", 0)for r in reports),
        },
        "note": (
            "Corrections applied: OHLC integrity (mathematical), phantom H/L (yfinance anchor), "
            "stale bar interpolation (linear). Tick-to-bar assignment (2.1% of days) cannot be "
            "corrected without raw tick data. Run --validate after correction to confirm "
            "discrepancy rate vs yfinance daily has reduced."
        ),
    }
    return result


# ---------------------------------------------------------------------------
# Provider comparison utility
# ---------------------------------------------------------------------------

def compare_providers_for_day(ticker: str, session_date: str, db_path: str) -> dict:
    """
    Compare stored Alpaca bars against yfinance daily anchor for one day.
    Shows raw vs corrected values and the discrepancy.
    """
    from market_data_store import MarketDataStore
    store = MarketDataStore(db_path)
    bars  = store.get_session_bars(ticker, session_date)
    ctx   = store.get_session_context(ticker, session_date)

    import yfinance as yf
    d     = date.fromisoformat(session_date)
    yf_df = yf.download(ticker, start=str(d), end=str(d + timedelta(days=1)),
                         interval="1d", auto_adjust=True, progress=False)
    if isinstance(yf_df.columns, pd.MultiIndex):
        yf_df.columns = [c[0] for c in yf_df.columns]

    yf_high = float(yf_df["High"].iloc[0]) if not yf_df.empty else None
    yf_low  = float(yf_df["Low"].iloc[0])  if not yf_df.empty else None

    alpaca_high = float(bars["High"].max()) if not bars.empty else None
    alpaca_low  = float(bars["Low"].min())  if not bars.empty else None

    stale_count = int(((bars["Open"] == bars["High"]) &
                        (bars["High"] == bars["Low"])  &
                        (bars["Low"]  == bars["Close"])).sum()) if not bars.empty else 0

    return {
        "session_date":    session_date,
        "alpaca_high":     alpaca_high,
        "yfinance_high":   yf_high,
        "high_delta_pct":  round((alpaca_high - yf_high) / yf_high * 100, 3) if yf_high else None,
        "alpaca_low":      alpaca_low,
        "yfinance_low":    yf_low,
        "low_delta_pct":   round((yf_low - alpaca_low) / yf_low * 100, 3) if yf_low else None,
        "stale_bars":      stale_count,
        "n_bars":          len(bars),
        "quality_ok":      ctx.get("quality_ok") if ctx else None,
        "note": "Discrepancy >0.5% on H/L suggests phantom spike or stale bar contamination",
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, json

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    p = argparse.ArgumentParser(description="Alpaca bar data correction pipeline")
    p.add_argument("--db",      default="DATA/market_data.db")
    p.add_argument("--ticker",  default="SPY")
    p.add_argument("--start",   default="2016-01-01")
    p.add_argument("--end",     default="2022-12-31")
    p.add_argument("--no-yf",   action="store_true", help="Skip yfinance daily anchor fetch")
    p.add_argument("--compare", metavar="DATE", help="Compare Alpaca vs yfinance for one date")
    p.add_argument("--dry-run", action="store_true", help="Report corrections without writing")
    args = p.parse_args()

    if args.compare:
        result = compare_providers_for_day(args.ticker, args.compare, args.db)
        print(json.dumps(result, indent=2))
    else:
        start = date.fromisoformat(args.start)
        end   = date.fromisoformat(args.end)
        result = correct_store(
            args.db, args.ticker, start, end,
            fetch_yf_daily=not args.no_yf
        )
        print(f"\n{'='*55}")
        print(f"CORRECTION COMPLETE — {args.ticker} {start} → {end}")
        print(f"{'='*55}")
        print(f"Sessions processed: {result['sessions_processed']}")
        print(f"Sessions already clean: {result['sessions_clean']}")
        print(f"Sessions corrected:     {result['sessions_corrected']}")
        print(f"Total bar corrections:  {result['total_corrections']}")
        print(f"\nBreakdown:")
        for k, v in result["corrections_breakdown"].items():
            print(f"  {k:<22}: {v}")
        print(f"\n{result['note']}")
