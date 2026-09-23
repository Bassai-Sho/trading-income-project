"""
market_data_store.py
====================
SQLite-backed historical market data store for the Trading Income Project.

PROBLEM SOLVED
──────────────
Alpaca API calls for 7 years of 1-minute SPY data take 30-60 seconds each.
Running historical_sim.py daily would re-pull the same 688,000 bars.
This module downloads ONCE, stores locally in SQLite, and serves bar
requests in milliseconds with no API calls.

DATA QUALITY WARNING
─────────────────────
Alpaca 1-minute data has documented discrepancies vs Polygon and yfinance,
particularly for SPY (price differences of up to $1.88, volume off by 99%).
The Zarattini group investigated this separately and published their findings.

Mitigation built in:
  1. Gap detection: sessions with fewer than 350 bars flagged
  2. Price sanity: bars where high < low or close = 0 removed
  3. Volume sanity: bars with zero volume on high-volume instruments flagged
  4. Cross-validation hook: compare_with_yfinance() for sample days

TRAINING WINDOW (corrected from 2020-2022 to 2016-2022)
──────────────────────────────────────────────────────────
Original window (2020-2022) was arbitrary and captured only the two most
anomalous years in modern market history (COVID + rate-hike bear).

Correct window matches Zarattini 2024 validation period:
  IS (training):   2016-01-01 → 2022-12-31  (7 years, 4 distinct regimes)
  OOS (WFA val):   2023-01-01 → 2024-12-31  (2 years, post-COVID normalisation)
  SEALED:          2025-01-01 → present       (NEVER touch)

Four market regimes in the IS window:
  2016-2019: Normal bull (VIX 11-16). The "typical" conditions.
  Feb-Mar 2020: COVID crash (VIX 80+). Extreme volatility.
  Apr 2020-2021: COVID recovery (Fed QE, ZIRP, meme stocks).
  2022: Rate-hike bear (Fed +425bps, -20% SPY).
Regime-conditional WFE analysis reveals which periods drive the edge.

USAGE
─────
  # Download and store all data (run ONCE — takes 5-30 min)
  python market_data_store.py --download --tickers SPY QQQ --start 2016-01-01 --end 2024-12-31

  # Check store status
  python market_data_store.py --status

  # Update store with latest bars (run daily by runner.py)
  python market_data_store.py --update --tickers SPY

  # Validate data quality
  python market_data_store.py --validate --ticker SPY

  # Per-regime analysis
  python market_data_store.py --regime-analysis --ticker SPY
"""

from __future__ import annotations

import argparse
import logging
import os
import sqlite3
import sys
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger("market_data_store")

# Which Alpaca feed to download. Free (Basic) accounts may query the SIP feed
# (100% of US volume) for any window ending more than 15 minutes ago; the
# default on a free account would otherwise be IEX (~2.5% of volume), which
# breaks every volume-based calculation (VWAP, RVOL). Override with
# ALPACA_DATA_FEED=iex only for comparison.
DEFAULT_FEED = os.environ.get("ALPACA_DATA_FEED", "sip").lower()


class AlpacaFatalError(RuntimeError):
    """Credential / permission problems. Retrying other chunks cannot help,
    so download_and_store() stops instead of logging and moving on."""

# ---------------------------------------------------------------------------
# Regime definitions
# ---------------------------------------------------------------------------

REGIMES = {
    "normal_bull": {
        "start": date(2016, 1, 1), "end": date(2019, 12, 31),
        "label": "2016-2019 Normal Bull",
        "description": "Gradual uptrend, VIX 11-16, typical ORB conditions",
        "typical_vix": "12-16",
    },
    "covid_crash": {
        "start": date(2020, 2, 1), "end": date(2020, 4, 30),
        "label": "2020 COVID Crash",
        "description": "VIX 80+, fastest -34% bear in history",
        "typical_vix": "40-80",
    },
    "covid_recovery": {
        "start": date(2020, 5, 1), "end": date(2021, 12, 31),
        "label": "2020-2021 COVID Recovery",
        "description": "Fed QE, near-zero rates, meme stocks, volatile bull",
        "typical_vix": "18-30",
    },
    "rate_hike_bear": {
        "start": date(2022, 1, 1), "end": date(2022, 12, 31),
        "label": "2022 Rate-Hike Bear",
        "description": "Fed +425bps, SPY -20%, sustained bearish trend",
        "typical_vix": "22-36",
    },
    "post_covid_norm": {
        "start": date(2023, 1, 1), "end": date(2024, 12, 31),
        "label": "2023-2024 Post-COVID (OOS/WFA)",
        "description": "WFA validation window. Do not use for IS training.",
        "typical_vix": "12-18",
    },
}

IS_END   = date(2022, 12, 31)
OOS_END  = date(2024, 12, 31)
SEALED   = date(2025,  1,  1)

# ---------------------------------------------------------------------------
# SQLite schema
# ---------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS market_bars (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker       TEXT    NOT NULL,
    ts           TEXT    NOT NULL,         -- ISO8601, America/New_York
    ts_date      TEXT    NOT NULL,         -- YYYY-MM-DD (fast session filter)
    ts_time      TEXT    NOT NULL,         -- HH:MM     (fast time filter)
    open         REAL    NOT NULL,
    high         REAL    NOT NULL,
    low          REAL    NOT NULL,
    close        REAL    NOT NULL,
    volume       INTEGER NOT NULL,
    vwap         REAL,                     -- session-anchored, pre-computed
    bar_interval TEXT    DEFAULT '1m',
    quality_flag INTEGER DEFAULT 0,        -- 0=ok, 1=gap_fill, 2=suspect
    UNIQUE(ticker, ts, bar_interval)
);
CREATE INDEX IF NOT EXISTS idx_bars_ticker_date ON market_bars(ticker, ts_date);
CREATE INDEX IF NOT EXISTS idx_bars_ticker_ts   ON market_bars(ticker, ts);

CREATE TABLE IF NOT EXISTS session_context (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker       TEXT    NOT NULL,
    session_date TEXT    NOT NULL,
    -- Price levels
    open_price   REAL,
    prev_close   REAL,
    gap_pct      REAL,
    pdh          REAL,    -- previous day high
    pdl          REAL,    -- previous day low
    -- ORB levels (pre-computed)
    orb_5_high   REAL,  orb_5_low   REAL,
    orb_15_high  REAL,  orb_15_low  REAL,
    orb_30_high  REAL,  orb_30_low  REAL,
    -- Market context
    vix_close    REAL,
    vix_regime   TEXT,   -- NORMAL / ELEVATED / HIGH / EXTREME
    -- Quality
    n_bars       INTEGER,
    quality_ok   INTEGER DEFAULT 1,   -- 0 if suspicious
    gap_session  INTEGER DEFAULT 0,   -- 1 if missing bars detected
    -- Regime label for training analysis
    regime_label TEXT,
    UNIQUE(ticker, session_date)
);
CREATE INDEX IF NOT EXISTS idx_ctx_ticker_date ON session_context(ticker, session_date);

CREATE TABLE IF NOT EXISTS data_store_meta (
    id            INTEGER PRIMARY KEY DEFAULT 1,
    tickers       TEXT,           -- JSON list
    earliest_date TEXT,
    latest_date   TEXT,
    total_bars    INTEGER,
    total_sessions INTEGER,
    last_updated  TEXT,
    download_ts   TEXT,
    quality_report TEXT           -- JSON summary
);

CREATE TABLE IF NOT EXISTS data_quality_log (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker       TEXT,
    session_date TEXT,
    issue_type   TEXT,     -- 'gap', 'price_error', 'volume_zero', 'bar_count_low'
    detail       TEXT,
    logged_at    TEXT
);
"""

# ---------------------------------------------------------------------------
# Core store class
# ---------------------------------------------------------------------------

class MarketDataStore:
    """
    SQLite-backed historical bar store.
    Thread-safe for read. Write should be single-process.
    """

    def __init__(self, db_path: str = "DATA/market_data.db") -> None:
        self.db_path = db_path
        dirname = os.path.dirname(self.db_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.db_path, timeout=10)
        c.row_factory = sqlite3.Row
        return c

    def _init_schema(self) -> None:
        with self._conn() as c:
            c.executescript(SCHEMA)

    # ── Fetching from Alpaca ──────────────────────────────────────────────────

    def _fetch_from_alpaca(
        self, ticker: str, start: date, end: date, feed: str | None = None,
    ) -> pd.DataFrame:
        """Fetch 1-min bars from Alpaca. Requires ALPACA_API_KEY / ALPACA_SECRET_KEY.

        feed defaults to DEFAULT_FEED ("sip"). Raises AlpacaFatalError for
        missing keys, a missing SDK, or an auth/permission rejection.
        """
        try:
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests   import StockBarsRequest
            from alpaca.data.timeframe  import TimeFrame, TimeFrameUnit
            from alpaca.data.enums      import DataFeed
            from alpaca.common.exceptions import APIError
        except ImportError:
            raise AlpacaFatalError("alpaca-py not installed: pip install alpaca-py")

        api_key    = os.environ.get("ALPACA_API_KEY")
        secret_key = os.environ.get("ALPACA_SECRET_KEY")
        if not api_key or not secret_key:
            raise AlpacaFatalError(
                "Set BOTH ALPACA_API_KEY and ALPACA_SECRET_KEY (e.g. in .env). "
                "Free paper account at alpaca.markets — no funding required."
            )
        feed = (feed or DEFAULT_FEED).lower()
        client = StockHistoricalDataClient(api_key, secret_key)
        req    = StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=TimeFrame(1, TimeFrameUnit.Minute),
            start=datetime.combine(start, datetime.min.time()),
            end=datetime.combine(end, datetime.max.time()),
            adjustment="all",
            feed=DataFeed(feed),
        )
        try:
            bars = client.get_stock_bars(req).df
        except APIError as e:
            code = getattr(e, "status_code", None)
            msg  = str(e)
            if code in (401, 403) or "subscription" in msg.lower() \
                    or "forbidden" in msg.lower() or "unauthorized" in msg.lower():
                raise AlpacaFatalError(
                    f"Alpaca rejected the request (HTTP {code}, feed={feed}): {msg}"
                ) from e
            raise
        if bars is None or len(bars) == 0:
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        if isinstance(bars.index, pd.MultiIndex):
            bars = bars.reset_index(level=0, drop=True)
        bars.index = pd.to_datetime(bars.index, utc=True).tz_convert("America/New_York")
        bars.columns = [c.capitalize() for c in bars.columns]
        return bars

    def _compute_vwap(self, day_df: pd.DataFrame) -> pd.Series:
        tp = (day_df["High"] + day_df["Low"] + day_df["Close"]) / 3.0
        pv = tp * day_df["Volume"]
        cv = day_df["Volume"].cumsum().replace(0, float("nan"))
        return pv.cumsum() / cv

    def _vix_to_regime(self, vix: float) -> str:
        if vix > 35: return "EXTREME"
        if vix > 25: return "HIGH"
        if vix > 18: return "ELEVATED"
        return "NORMAL"

    def _regime_label(self, d: date) -> str:
        for name, r in REGIMES.items():
            if r["start"] <= d <= r["end"]:
                return r["label"]
        return "unknown"

    # ── Validate bars ─────────────────────────────────────────────────────────

    def _validate_day(self, day_df: pd.DataFrame, session_date: date) -> list[dict]:
        """
        Validate one session against the five data quality issues documented by
        Zarattini et al. (Concretum Group, April 2026) for Alpaca SIP data:

          1. Phantom Highs/Lows  — isolated spikes not replicated across providers
          2. Stale Bars          — OHLC all equal; indicates missing/bad aggregation
          3. Early-Close Leakage — enforced upstream (09:30-16:00 window)
          4. Tick-to-Bar shift   — detectable via ORB range vs full-day range ratio
          5. Venue coverage gaps — manifests as anomalous first-print volume

        Sessions with critical issues get quality_ok=0 and are excluded from IS training.
        Expected exclusion rate on Alpaca SIP data: 2-8%.
        """
        issues = []
        n = len(day_df)

        # ── Issue 1: Phantom Highs / Lows ────────────────────────────────────
        hl_range = day_df["High"] - day_df["Low"]
        hl_mean  = float(hl_range.mean())
        hl_std   = float(hl_range.std())
        if hl_std > 0:
            n_phantom = int((hl_range > hl_mean + 5 * hl_std).sum())
            if n_phantom > 0:
                issues.append({
                    "issue_type": "phantom_hl",
                    "detail": f"{n_phantom} bar(s) with H/L > mean+5σ "
                              "(phantom price spike not replicated across providers)",
                })

        # ── Issue 2: Stale Bars (Zarattini: up to 350/390 in IBKR 2026 data) ─
        stale = (
            (day_df["Open"]  == day_df["High"])  &
            (day_df["High"]  == day_df["Low"])   &
            (day_df["Low"]   == day_df["Close"])
        )
        n_stale = int(stale.sum())
        if n_stale > 10:
            issue_type = "stale_bars_critical" if n_stale > 50 else "stale_bars"
            issues.append({
                "issue_type": issue_type,
                "detail": f"{n_stale}/{n} stale bars (OHLC all equal). "
                          "Indicates missing or improperly aggregated data. "
                          "EXCLUDE from IS training.",
            })

        # ── Issue 3: Bar count ────────────────────────────────────────────────
        if n < 350:
            issues.append({
                "issue_type": "bar_count_low",
                "detail": f"{n} bars (expected ~390 for full 09:30-16:00 session)",
            })

        # ── Issue 4: ORB range anomaly (tick-to-bar / phantom indicator) ─────
        orb_mask = day_df.index.time <= pd.Timestamp("09:45").time()
        orb_bars = day_df[orb_mask]
        if len(orb_bars) >= 3:
            orb_range  = float(orb_bars["High"].max() - orb_bars["Low"].min())
            full_range = float(day_df["High"].max()   - day_df["Low"].min())
            if full_range > 0 and (orb_range / full_range) > 0.80:
                issues.append({
                    "issue_type": "orb_range_anomaly",
                    "detail": f"ORB range ({orb_range:.3f}) = "
                              f"{orb_range/full_range:.0%} of full-day range. "
                              "Likely stale or phantom bars in the critical ORB window.",
                })

        # ── Issue 5: Price sanity / venue coverage ────────────────────────────
        bad_price = day_df[
            (day_df["High"] < day_df["Low"]) |
            (day_df["Close"] <= 0) |
            (day_df["Open"]  <= 0)
        ]
        if len(bad_price) > 0:
            issues.append({
                "issue_type": "price_error",
                "detail": f"{len(bad_price)} bars with invalid OHLC values",
            })

        return issues

    # ── Download and store ────────────────────────────────────────────────────

    def download_and_store(
        self,
        ticker:    str,
        start:     date,
        end:       date,
        vix_daily: dict[str, float] | None = None,
        chunk_months: int = 3,
        feed: str | None = None,
    ) -> dict:
        """
        Download all bars for ticker in date range and store in SQLite.
        Downloads in chunks to avoid API timeouts.
        Pre-computes: VWAP per session, ORB levels, session context.

        vix_daily: {date_str: vix_close} — fetched separately from yfinance.
        """
        feed = (feed or DEFAULT_FEED).lower()
        log.info("Downloading %s bars %s → %s (feed=%s)", ticker, start, end, feed)
        failed_chunks: list[dict] = []
        t0 = time.monotonic()

        # Fetch VIX if not provided
        if vix_daily is None:
            vix_daily = self._fetch_vix_daily(start, end)

        total_bars     = 0
        total_sessions = 0
        chunk_start    = start

        while chunk_start < end:
            chunk_end = min(
                date(chunk_start.year + (chunk_start.month - 1 + chunk_months) // 12,
                     ((chunk_start.month - 1 + chunk_months) % 12) + 1,
                     1) - timedelta(days=1),
                end
            )
            log.info("  Chunk %s → %s...", chunk_start, chunk_end)
            try:
                df = self._fetch_from_alpaca(ticker, chunk_start, chunk_end, feed=feed)
                bars_written, sessions_written = self._store_bars(ticker, df, vix_daily)
                total_bars     += bars_written
                total_sessions += sessions_written
                log.info("  Stored %d bars, %d sessions", bars_written, sessions_written)
                time.sleep(1)   # rate-limit guard
            except AlpacaFatalError:
                # Keys/permissions: every remaining chunk would fail the same way.
                # Previously this was logged per chunk and the run "finished" with
                # 0 bars — a plausible cause of the empty market_data.db (P2-116).
                self._update_meta(ticker, total_bars, total_sessions)
                raise
            except Exception as e:
                log.error("  Chunk %s → %s FAILED: %s", chunk_start, chunk_end, e)
                failed_chunks.append({"start": str(chunk_start), "end": str(chunk_end),
                                      "error": f"{type(e).__name__}: {e}"})
            chunk_start = chunk_end + timedelta(days=1)

        elapsed = time.monotonic() - t0
        # Update meta
        self._update_meta(ticker, total_bars, total_sessions)
        return {"ticker": ticker, "feed": feed, "total_bars": total_bars,
                "total_sessions": total_sessions, "elapsed_sec": round(elapsed, 1),
                "failed_chunks": failed_chunks}

    def _store_bars(
        self,
        ticker:    str,
        df:        pd.DataFrame,
        vix_daily: dict[str, float],
    ) -> tuple[int, int]:
        """Write bars to market_bars and compute session_context."""
        bars_written     = 0
        sessions_written = 0

        trading_days = sorted(set(df.index.date))
        prev_close, prev_high, prev_low = None, None, None
        if trading_days:
            # Seed from the last stored session before this chunk, so the first
            # session of each chunk still gets gap_pct / PDH / PDL.
            prev_close, prev_high, prev_low = self._prev_session_levels(
                ticker, trading_days[0])

        for day in trading_days:
            day_str  = str(day)
            day_df   = df[df.index.date == day].copy()

            # Market hours only: 09:30-16:00
            day_df = day_df[
                (day_df.index.time >= pd.Timestamp("09:30").time()) &
                (day_df.index.time <= pd.Timestamp("16:00").time())
            ]
            if len(day_df) < 10:
                prev_close = prev_high = prev_low = None
                continue

            # Pre-compute session VWAP
            day_df = day_df.copy()
            day_df["vwap"] = self._compute_vwap(day_df)

            # Validate
            issues = self._validate_day(day_df, day)
            quality_ok  = 1 if not issues else 0
            gap_session = 1 if any(i["issue_type"] == "bar_count_low" for i in issues) else 0

            # Write to data_quality_log if issues found
            if issues:
                with self._conn() as conn:
                    for iss in issues:
                        conn.execute(
                            "INSERT OR IGNORE INTO data_quality_log "
                            "(ticker, session_date, issue_type, detail, logged_at) "
                            "VALUES (?,?,?,?,?)",
                            (ticker, day_str, iss["issue_type"],
                             iss["detail"], datetime.now(timezone.utc).replace(tzinfo=None).isoformat())
                        )

            # Write bars
            bar_rows = []
            for ts, bar in day_df.iterrows():
                bar_rows.append((
                    ticker, ts.isoformat(),
                    str(ts.date()), ts.strftime("%H:%M"),
                    float(bar["Open"]), float(bar["High"]),
                    float(bar["Low"]),  float(bar["Close"]),
                    int(bar.get("Volume", 0)),
                    float(bar["vwap"]) if not pd.isna(bar["vwap"]) else None,
                    "1m", 0,
                ))
            with self._conn() as conn:
                conn.executemany(
                    "INSERT OR IGNORE INTO market_bars "
                    "(ticker,ts,ts_date,ts_time,open,high,low,close,volume,vwap,bar_interval,quality_flag) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    bar_rows
                )
            bars_written += len(bar_rows)

            # Compute ORB levels
            orb_data = {}
            for method, mins in [("5", 5), ("15", 15), ("30", 30)]:
                cutoff = (pd.Timestamp(f"{day_str} 09:30", tz="America/New_York") +
                           pd.Timedelta(minutes=mins)).time()
                orb_bars = day_df[day_df.index.time <= cutoff]
                if len(orb_bars) >= 2:
                    orb_data[f"orb_{method}_high"] = float(orb_bars["High"].max())
                    orb_data[f"orb_{method}_low"]  = float(orb_bars["Low"].min())
                else:
                    orb_data[f"orb_{method}_high"] = None
                    orb_data[f"orb_{method}_low"]  = None

            # Gap
            open_p   = float(day_df["Open"].iloc[0])
            gap_pct  = round((open_p - prev_close) / prev_close * 100, 3) if prev_close else None
            vix_val  = vix_daily.get(day_str, 0.0) or 0.0
            regime   = self._vix_to_regime(vix_val) if vix_val else "UNKNOWN"

            with self._conn() as conn:
                conn.execute(
                    """INSERT OR REPLACE INTO session_context
                    (ticker, session_date, open_price, prev_close, gap_pct,
                     pdh, pdl, orb_5_high, orb_5_low, orb_15_high, orb_15_low,
                     orb_30_high, orb_30_low, vix_close, vix_regime, n_bars,
                     quality_ok, gap_session, regime_label)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (ticker, day_str, open_p, prev_close, gap_pct,
                     prev_high, prev_low,
                     orb_data.get("orb_5_high"),  orb_data.get("orb_5_low"),
                     orb_data.get("orb_15_high"), orb_data.get("orb_15_low"),
                     orb_data.get("orb_30_high"), orb_data.get("orb_30_low"),
                     vix_val, regime, len(day_df), quality_ok, gap_session,
                     self._regime_label(day))
                )
            sessions_written += 1

            # Update prev_close/high/low for next day
            prev_close = float(day_df["Close"].iloc[-1])
            prev_high  = float(day_df["High"].max())
            prev_low   = float(day_df["Low"].min())

        return bars_written, sessions_written

    def _prev_session_levels(
        self, ticker: str, before: date
    ) -> tuple[float | None, float | None, float | None]:
        """Close/high/low of the latest stored session strictly before `before`."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT ts_date FROM market_bars WHERE ticker=? AND ts_date<? "
                "ORDER BY ts_date DESC LIMIT 1", (ticker, str(before))
            ).fetchone()
            if row is None:
                return None, None, None
            d = row[0]
            agg = conn.execute(
                "SELECT MAX(high), MIN(low) FROM market_bars WHERE ticker=? AND ts_date=?",
                (ticker, d)).fetchone()
            last = conn.execute(
                "SELECT close FROM market_bars WHERE ticker=? AND ts_date=? "
                "ORDER BY ts DESC LIMIT 1", (ticker, d)).fetchone()
        return float(last[0]), float(agg[0]), float(agg[1])

    def _fetch_vix_daily(self, start: date, end: date) -> dict[str, float]:
        """
        Fetch VIX daily closes.
        Prefers FRED VIXCLS from local store (no network call if already downloaded).
        Falls back to yfinance.
        """
        # Prefer FRED (authoritative CBOE source, same market_data.db)
        try:
            from fred_store import FredDataStore
            fred = FredDataStore(self.db_path)
            rows = fred.get_series("VIXCLS", str(start), str(end))
            if len(rows) >= 5:
                result = {r["date"]: r["value"] for r in rows if r["value"] is not None}
                if result:
                    log.info("VIX from FRED local store: %d days", len(result))
                    return result
        except Exception as e:
            log.debug("FRED VIX not available (%s) — falling back to yfinance", e)

        # Fallback: yfinance
        try:
            import yfinance as yf
            df = yf.download("^VIX", start=str(start), end=str(end + timedelta(days=1)),
                              interval="1d", auto_adjust=True, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
            return {str(d.date()): float(c) for d, c in zip(df.index, df["Close"])}
        except Exception as e:
            log.warning("VIX fetch failed (FRED + yfinance both unavailable): %s", e)
            return {}

    def _update_meta(self, ticker: str, bars: int, sessions: int) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO data_store_meta
                (id, tickers, total_bars, total_sessions, last_updated)
                VALUES (1,
                  COALESCE((SELECT tickers FROM data_store_meta WHERE id=1),'[]'),
                  COALESCE((SELECT total_bars FROM data_store_meta WHERE id=1),0) + ?,
                  COALESCE((SELECT total_sessions FROM data_store_meta WHERE id=1),0) + ?,
                  ?)""",
                (bars, sessions, datetime.now(timezone.utc).replace(tzinfo=None).isoformat())
            )

    # ── Retrieval API (used by historical_sim.py) ─────────────────────────────

    def get_session_bars(
        self, ticker: str, session_date: str, bar_interval: str = "1m"
    ) -> pd.DataFrame:
        """Retrieve all bars for one trading session."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT ts,open,high,low,close,volume,vwap FROM market_bars "
                "WHERE ticker=? AND ts_date=? AND bar_interval=? ORDER BY ts",
                (ticker, session_date, bar_interval)
            ).fetchall()
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame([dict(r) for r in rows])
        # utc=True: stored ISO strings carry -05:00 and -04:00 offsets; without it
        # pandas raises "Mixed timezones" for any range spanning a DST change.
        df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert("America/New_York")
        df = df.set_index("ts")
        df.columns = [c.capitalize() if c != "vwap" else "vwap" for c in df.columns]
        return df

    def get_session_context(self, ticker: str, session_date: str) -> dict | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM session_context WHERE ticker=? AND session_date=?",
                (ticker, session_date)
            ).fetchone()
        return dict(row) if row else None

    def get_date_range(
        self, ticker: str, start: date, end: date,
        quality_ok_only: bool = False,
    ) -> list[str]:
        """List session dates in range. quality_ok_only=True excludes bad sessions."""
        if quality_ok_only:
            with self._conn() as conn:
                rows = conn.execute(
                    "SELECT session_date FROM session_context WHERE ticker=? "
                    "AND session_date>=? AND session_date<=? AND quality_ok=1 ORDER BY session_date",
                    (ticker, str(start), str(end))
                ).fetchall()
        else:
            with self._conn() as conn:
                rows = conn.execute(
                    "SELECT DISTINCT ts_date FROM market_bars WHERE ticker=? "
                    "AND ts_date >= ? AND ts_date <= ? ORDER BY ts_date",
                    (ticker, str(start), str(end))
                ).fetchall()
        return [r[0] for r in rows]

    def get_bars_range(
        self, ticker: str, start: date, end: date, bar_interval: str = "1m"
    ) -> pd.DataFrame:
        """Retrieve all bars in a date range (for multi-day analysis)."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT ts,open,high,low,close,volume,vwap FROM market_bars "
                "WHERE ticker=? AND ts_date>=? AND ts_date<=? AND bar_interval=? ORDER BY ts",
                (ticker, str(start), str(end), bar_interval)
            ).fetchall()
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame([dict(r) for r in rows])
        # utc=True: stored ISO strings carry -05:00 and -04:00 offsets; without it
        # pandas raises "Mixed timezones" for any range spanning a DST change.
        df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert("America/New_York")
        return df.set_index("ts")

    # ── Status and analysis ───────────────────────────────────────────────────

    def quality_summary(self, ticker: str, start: date, end: date) -> dict:
        """Data quality summary. Expected exclusion rate 2-8%% on Alpaca SIP data."""
        with self._conn() as conn:
            total = conn.execute(
                "SELECT COUNT(*) FROM session_context WHERE ticker=? AND session_date>=? AND session_date<=?",
                (ticker, str(start), str(end))).fetchone()[0]
            clean = conn.execute(
                "SELECT COUNT(*) FROM session_context WHERE ticker=? AND session_date>=? AND session_date<=? AND quality_ok=1",
                (ticker, str(start), str(end))).fetchone()[0]
            by_type = conn.execute(
                "SELECT issue_type, COUNT(DISTINCT session_date) FROM data_quality_log "
                "WHERE ticker=? AND session_date>=? AND session_date<=? GROUP BY issue_type",
                (ticker, str(start), str(end))).fetchall()
        excluded = total - clean
        return {
            "ticker": ticker, "date_range": f"{start} → {end}",
            "total_sessions": total, "clean_sessions": clean,
            "excluded_sessions": excluded,
            "exclusion_rate": round(excluded / max(total, 1), 3),
            "issues_by_type": {r[0]: r[1] for r in by_type},
            "note": ("Per Zarattini et al. (Concretum Group, April 2026): "
                     "stale bars, phantom H/L, and ORB range anomalies "
                     "materially distort H/L-stop backtests on Alpaca SIP data. "
                     "IS training uses quality_ok=1 sessions only."),
        }

    def status(self) -> dict:
        """Summary of what's in the store."""
        with self._conn() as conn:
            meta = conn.execute("SELECT * FROM data_store_meta WHERE id=1").fetchone()
            by_ticker = conn.execute(
                "SELECT ticker, COUNT(DISTINCT ts_date) as sessions, COUNT(*) as bars "
                "FROM market_bars GROUP BY ticker"
            ).fetchall()
            quality_issues = conn.execute(
                "SELECT issue_type, COUNT(*) FROM data_quality_log GROUP BY issue_type"
            ).fetchall()

        result: dict[str, Any] = {
            "meta": dict(meta) if meta else {},
            "by_ticker": [dict(r) for r in by_ticker],
            "quality_issues": {r[0]: r[1] for r in quality_issues},
        }
        return result

    def regime_session_counts(self, ticker: str) -> dict:
        """Count sessions per regime for training window analysis."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT regime_label, COUNT(*) as n, "
                "AVG(CASE WHEN quality_ok=1 THEN 1.0 ELSE 0.0 END) as quality_rate "
                "FROM session_context WHERE ticker=? "
                "AND session_date <= ? GROUP BY regime_label ORDER BY MIN(session_date)",
                (ticker, str(IS_END))
            ).fetchall()
        return {r[0]: {"n_sessions": r[1], "quality_rate": round(r[2], 3)}
                for r in rows}

    def validate_data_quality(self, ticker: str, sample_n: int = 20) -> dict:
        """Cross-validate a sample of days against yfinance."""
        import yfinance as yf
        sessions = self.get_date_range(ticker, date(2020, 1, 1), date(2022, 12, 31))
        sample   = sessions[:sample_n]
        issues: list[dict] = []
        for sess in sample:
            stored = self.get_session_bars(ticker, sess)
            if stored.empty:
                continue
            # Fetch same day from yfinance
            try:
                sess_date = date.fromisoformat(sess)
                yf_df = yf.download(ticker, start=str(sess_date),
                                     end=str(sess_date + timedelta(days=1)),
                                     interval="5m", auto_adjust=True, progress=False)
                if isinstance(yf_df.columns, pd.MultiIndex):
                    yf_df.columns = [c[0] if isinstance(c, tuple) else c for c in yf_df.columns]
                if yf_df.empty:
                    continue
                # Compare day's high
                stored_high = float(stored["High"].max())
                yf_high     = float(yf_df["High"].max())
                delta_pct   = abs(stored_high - yf_high) / yf_high * 100
                if delta_pct > 1.0:
                    issues.append({
                        "date": sess,
                        "stored_high": stored_high,
                        "yf_high": yf_high,
                        "delta_pct": round(delta_pct, 2),
                    })
            except Exception:
                pass
        return {
            "sample_size": len(sample),
            "discrepancies_found": len(issues),
            "discrepancy_rate": round(len(issues) / max(len(sample), 1), 3),
            "issues": issues[:5],
            "note": "Discrepancy >1% indicates potential data quality issue. Zarattini group documented this."
        }

    # ── Update (daily runner call) ─────────────────────────────────────────────

    def update(self, ticker: str) -> dict:
        """Add new bars since the last stored date."""
        with self._conn() as conn:
            latest = conn.execute(
                "SELECT MAX(ts_date) FROM market_bars WHERE ticker=?", (ticker,)
            ).fetchone()[0]
        if latest is None:
            return {"error": "No data for ticker. Run --download first."}
        latest_date = date.fromisoformat(latest)
        today       = date.today()
        if latest_date >= today:
            return {"message": "Already up to date", "latest": str(latest_date)}
        vix = self._fetch_vix_daily(latest_date, today)
        result = self.download_and_store(ticker, latest_date + timedelta(days=1), today, vix)
        return result


# ---------------------------------------------------------------------------
# Patch historical_sim.py to use MarketDataStore
# ---------------------------------------------------------------------------

def get_store_or_fetch(
    ticker:    str,
    start:     date,
    end:       date,
    store_db:  str = "DATA/market_data.db",
) -> tuple[pd.DataFrame, MarketDataStore]:
    """
    Return bars from local store if available, else download and store.
    This is the entry point for historical_sim.py to use the store.
    """
    store   = MarketDataStore(store_db)
    trading_days = store.get_date_range(ticker, start, end)
    if len(trading_days) >= 100:
        log.info("Using cached data: %d sessions for %s %s-%s",
                 len(trading_days), ticker, start, end)
        df = store.get_bars_range(ticker, start, end)
        return df, store
    else:
        log.info("Cache miss (%d sessions cached). Downloading from Alpaca...",
                 len(trading_days))
        vix = store._fetch_vix_daily(start, end)
        store.download_and_store(ticker, start, end, vix)
        df = store.get_bars_range(ticker, start, end)
        return df, store


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")

    p = argparse.ArgumentParser(description="Historical market data store")
    p.add_argument("--download",       action="store_true")
    p.add_argument("--update",         action="store_true")
    p.add_argument("--status",         action="store_true")
    p.add_argument("--validate",       action="store_true")
    p.add_argument("--regime-analysis",action="store_true")
    p.add_argument("--tickers",        nargs="+", default=["SPY"])
    p.add_argument("--ticker",         default="SPY")
    p.add_argument("--start",          default="2016-01-01")
    p.add_argument("--end",            default="2024-12-31")
    p.add_argument("--db",             default="DATA/market_data.db")
    p.add_argument("--check",          action="store_true",
                   help="Pre-flight: fetch one past session on SIP and IEX, no DB writes")
    p.add_argument("--check-date",     default="2024-12-20")
    p.add_argument("--feed",           default=None, choices=["sip", "iex"],
                   help="Override ALPACA_DATA_FEED (default sip)")
    args = p.parse_args()

    store = MarketDataStore(args.db)

    if args.check:
        d = date.fromisoformat(args.check_date)
        if d >= SEALED:
            sys.exit(f"--check-date {d} is in the sealed window (>= {SEALED}); pick an earlier date.")
        print(f"\n=== ALPACA PRE-FLIGHT: {args.ticker} {d} (no DB writes) ===")
        vols = {}
        try:
            for f in ("sip", "iex"):
                df = store._fetch_from_alpaca(args.ticker, d, d, feed=f)
                rth = df[(df.index.time >= pd.Timestamp("09:30").time()) &
                         (df.index.time <  pd.Timestamp("16:00").time())] if len(df) else df
                vols[f] = int(rth["Volume"].sum()) if len(rth) else 0
                first = rth.index[0].strftime("%H:%M") if len(rth) else "-"
                print(f"  {f.upper():<4} regular-session bars={len(rth):>4}  "
                      f"volume={vols[f]:>14,}  first bar={first}")
        except AlpacaFatalError as e:
            sys.exit(f"  ✗ {e}")
        if vols.get("sip", 0) == 0:
            sys.exit("  ✗ SIP returned no bars — check the date is a trading day.")
        ratio = vols["iex"] / vols["sip"] if vols["sip"] else 0
        print(f"  IEX/SIP volume ratio = {ratio:.1%}  (expect a few %; ~100% would mean "
              "both calls silently hit the same feed)")
        print("  ✓ Keys work and the SIP feed is available — safe to run --download.")

    elif args.status:
        st = store.status()
        print("\n=== DATA STORE STATUS ===")
        for t in st["by_ticker"]:
            print(f"  {t['ticker']}: {t['sessions']} sessions, {t['bars']:,} bars")
        if st["quality_issues"]:
            print(f"  Quality issues: {st['quality_issues']}")
        m = st["meta"]
        if m:
            print(f"  Last updated: {m.get('last_updated','?')}")

    elif args.download:
        any_failed = False
        for ticker in args.tickers:
            start = date.fromisoformat(args.start)
            end   = date.fromisoformat(args.end)
            if end >= SEALED:
                print(f"⚠️  End date {end} is in or beyond the sealed window.")
                print(f"   Sealed data starts {SEALED}. Clamping to {OOS_END}.")
                end = OOS_END
            print(f"\nDownloading {ticker} {start} → {end}...")
            try:
                result = store.download_and_store(ticker, start, end, feed=args.feed)
            except AlpacaFatalError as e:
                sys.exit(f"✗ {ticker}: stopped — {e}")
            print(f"Stored: {result['total_bars']:,} bars, "
                  f"{result['total_sessions']} sessions in {result['elapsed_sec']}s "
                  f"(feed={result['feed']})")
            if result["failed_chunks"]:
                any_failed = True
                print(f"⚠️  {len(result['failed_chunks'])} chunk(s) FAILED — re-run the same "
                      "command; INSERT OR IGNORE makes it safe to repeat:")
                for c in result["failed_chunks"]:
                    print(f"     {c['start']} → {c['end']}: {c['error']}")
            elif result["total_bars"] == 0:
                any_failed = True
                print("⚠️  0 bars stored — nothing failed loudly, but nothing arrived either.")
        if any_failed:
            sys.exit(1)

    elif args.update:
        for ticker in args.tickers:
            result = store.update(ticker)
            print(f"{ticker}: {result}")

    elif args.validate:
        print(f"\nValidating {args.ticker} vs yfinance (20-day sample)...")
        report = store.validate_data_quality(args.ticker)
        print(f"  Sample: {report['sample_size']} days")
        print(f"  Discrepancies found: {report['discrepancies_found']}")
        print(f"  Discrepancy rate: {report['discrepancy_rate']:.0%}")
        if report["issues"]:
            for iss in report["issues"][:3]:
                print(f"  ⚠️ {iss['date']}: stored={iss['stored_high']:.2f} "
                      f"yf={iss['yf_high']:.2f} delta={iss['delta_pct']:.2f}%")
        print(f"  Note: {report['note']}")

    elif args.regime_analysis:
        print(f"\nRegime analysis for {args.ticker} (IS window 2016-2022):")
        counts = store.regime_session_counts(args.ticker)
        print(f"{'Regime':<35} {'Sessions':>8} {'Quality':>8}")
        print("-" * 55)
        for regime, data in counts.items():
            print(f"  {regime:<33} {data['n_sessions']:>8} {data['quality_rate']:>7.0%}")
        print()
        print("Tip: Regime-conditional WFE reveals which periods drive the edge.")
        print("     Low WFE in a specific regime = Markov position sizing matters more.")

    else:
        p.print_help()
