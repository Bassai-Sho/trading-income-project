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

# Price adjustment for downloaded bars. "raw" is correct for an intraday
# strategy: splits never happen inside a session, and ORB/VWAP/stops only need
# prices consistent within the day. Split-adjusted ("all"/"split") history
# divides old prices (NVDA 2016 by 40, TSLA by 15); Alpaca then rounds them,
# which manufactured ~95% of NVDA/TSLA's "stale bar" flags (verified: NVDA
# 2020-07-08 had 66 stale bars adjusted vs 0 raw) and made the engine's
# absolute $0.02 stop-buffer floor ~40x too large on old sessions.
DEFAULT_ADJUSTMENT = os.environ.get("ALPACA_ADJUSTMENT", "raw").lower()

# Quality issues that exclude a session from training (quality_ok=0). Anything
# else is logged for information only.
CRITICAL_ISSUES = {"phantom_spike", "stale_bars_critical", "price_error", "bar_count_low"}

# Isolated-spike threshold: how far a bar's wick may extend beyond BOTH
# neighbours and its own body, in multiples of the session's median bar range.
# Calibrated 23 Sep 2026 on real 1-min bars (SPY/QQQ/NVDA/TSLA, 32 sessions):
# genuine wicks never exceeded 3.5x; recorded bad prints are typically 10-20x.
SPIKE_MULT      = 8.0
SPIKE_MIN_FRAC  = 0.0005   # ...and at least 0.05% of price (ignores dead-quiet days)
# Overnight open/prev-close ratio outside this band = split or similar event:
# prior-day levels are not comparable, so gap/PDH/PDL are left NULL.
DISCONTINUITY_BAND = (0.65, 1.6)


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
# Exchange calendar: each session's real close
# ---------------------------------------------------------------------------

_CLOSE_CACHE: dict[date, Any] = {}
_CAL_WARNED = False

def session_close(d: date):
    """NYSE close time (America/New_York) for session d; None if d is not a
    session. Uses exchange_calendars (already a project dependency, P1-008).
    Returns the normal 16:00 if the library is unavailable — then early closes
    are not trimmed (logged once)."""
    global _CAL_WARNED
    if not _CLOSE_CACHE:
        try:
            import exchange_calendars as xcals
            sched = xcals.get_calendar("XNYS", start="2010-01-01").schedule
            closes = sched["close"].dt.tz_convert("America/New_York")
            _CLOSE_CACHE.update({ts.date(): c.time() for ts, c in closes.items()})
        except Exception as e:                                   # pragma: no cover
            if not _CAL_WARNED:
                log.warning("exchange_calendars unavailable (%s): early closes not trimmed", e)
                _CAL_WARNED = True
            return pd.Timestamp("16:00").time()
    if d > max(_CLOSE_CACHE):                                    # beyond calendar horizon
        return pd.Timestamp("16:00").time()
    return _CLOSE_CACHE.get(d)


def expected_bars(close_t) -> int:
    """1-min bars from 09:30 up to (not including) the close."""
    return (close_t.hour * 60 + close_t.minute) - (9 * 60 + 30)


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
    issue_type   TEXT,     -- see _validate_day(); CRITICAL_ISSUES exclude a session
    detail       TEXT,
    logged_at    TEXT
);

CREATE TABLE IF NOT EXISTS store_settings (
    key   TEXT PRIMARY KEY,   -- e.g. 'adjustment:SPY'
    value TEXT
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
            # Migration: data_quality_log had no uniqueness rule, so every
            # re-run duplicated its rows. Keep the first copy, then enforce.
            c.execute("""DELETE FROM data_quality_log WHERE id NOT IN (
                           SELECT MIN(id) FROM data_quality_log
                           GROUP BY ticker, session_date, issue_type)""")
            c.execute("""CREATE UNIQUE INDEX IF NOT EXISTS idx_dql_unique
                         ON data_quality_log(ticker, session_date, issue_type)""")

    # ── Settings / bookkeeping ────────────────────────────────────────────────

    def _get_setting(self, key: str) -> str | None:
        with self._conn() as c:
            r = c.execute("SELECT value FROM store_settings WHERE key=?", (key,)).fetchone()
        return r[0] if r else None

    def _set_setting(self, key: str, value: str) -> None:
        with self._conn() as c:
            c.execute("INSERT OR REPLACE INTO store_settings(key, value) VALUES (?,?)",
                      (key, value))

    def stored_adjustment(self, ticker: str) -> str | None:
        """Adjustment of the bars stored for ticker. None = no bars stored.
        Bars stored before this setting existed were downloaded as 'all'."""
        with self._conn() as c:
            has = c.execute("SELECT 1 FROM market_bars WHERE ticker=? LIMIT 1",
                            (ticker,)).fetchone()
        if not has:
            return None
        return self._get_setting(f"adjustment:{ticker}") or "all"

    def wipe(self, ticker: str) -> dict:
        """Delete every stored bar, session and quality row for ticker.
        FRED / sentiment tables in the same file are untouched."""
        with self._conn() as c:
            n = {t: c.execute(f"DELETE FROM {t} WHERE ticker=?", (ticker,)).rowcount
                 for t in ("market_bars", "session_context", "data_quality_log")}
            c.execute("DELETE FROM store_settings WHERE key=?", (f"adjustment:{ticker}",))
        self._refresh_meta()
        return n

    def _record_issues(self, conn: sqlite3.Connection, ticker: str,
                       day_str: str, issues: list[dict]) -> None:
        """Replace this session's quality rows (so a re-run never duplicates
        and a changed check never leaves stale flags behind)."""
        conn.execute("DELETE FROM data_quality_log WHERE ticker=? AND session_date=?",
                     (ticker, day_str))
        now = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()
        conn.executemany(
            "INSERT OR REPLACE INTO data_quality_log "
            "(ticker, session_date, issue_type, detail, logged_at) VALUES (?,?,?,?,?)",
            [(ticker, day_str, i["issue_type"], i["detail"], now) for i in issues])

    # ── Fetching from Alpaca ──────────────────────────────────────────────────

    def _fetch_from_alpaca(
        self, ticker: str, start: date, end: date, feed: str | None = None,
        adjustment: str | None = None,
    ) -> pd.DataFrame:
        """Fetch 1-min bars from Alpaca. Requires ALPACA_API_KEY / ALPACA_SECRET_KEY.

        feed defaults to DEFAULT_FEED ("sip"). Raises AlpacaFatalError for
        missing keys, a missing SDK, or an auth/permission rejection.
        """
        try:
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests   import StockBarsRequest
            from alpaca.data.timeframe  import TimeFrame, TimeFrameUnit
            from alpaca.data.enums      import DataFeed, Adjustment
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
        adjustment = (adjustment or DEFAULT_ADJUSTMENT).lower()
        client = StockHistoricalDataClient(api_key, secret_key)
        req    = StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=TimeFrame(1, TimeFrameUnit.Minute),
            start=datetime.combine(start, datetime.min.time()),
            end=datetime.combine(end, datetime.max.time()),
            adjustment=Adjustment(adjustment),
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
        Validate one session. Returns issue dicts; types in CRITICAL_ISSUES make
        the session quality_ok=0 (excluded from IS training), others are logged
        for information. Checks follow the data problems documented by
        Zarattini et al. (Concretum Group, April 2026) for Alpaca SIP data:

          phantom_spike      (critical) a bar whose wick sticks out beyond BOTH
                             neighbours and its own body by > SPIKE_MULT x the
                             session's median bar range — an isolated print that
                             immediately reverses. The first and last bar
                             (opening/closing auctions) are not tested: they are
                             legitimately wide. REPLACES the old 'phantom_hl'
                             test (any bar range > mean+5 sigma), which fired on
                             67-87% of sessions because the opening bar is
                             routinely that wide.
          stale_bars_critical (critical) > 50 bars with O=H=L=C
          stale_bars         (info) 11-50 such bars
          bar_count_low      (critical) fewer than 90% of the bars the exchange
                             calendar expects for that session
          early_close        (info) calendar close before 16:00 (bars after the
                             close are trimmed before validation)
          orb_range_anomaly  (info) 09:30-09:45 range > 80% of the day's range —
                             happens legitimately on quiet trend-less days
          price_error        (critical) high < low or non-positive prices
        """
        issues: list[dict] = []
        n = len(day_df)
        if n == 0:
            return [{"issue_type": "bar_count_low", "detail": "0 bars"}]
        h = day_df["High"].to_numpy(float);  l = day_df["Low"].to_numpy(float)
        o = day_df["Open"].to_numpy(float);  c = day_df["Close"].to_numpy(float)

        # Isolated spikes
        med_range = float(np.median(h - l))
        if n >= 3 and med_range > 0:
            limit = max(SPIKE_MULT * med_range, SPIKE_MIN_FRAC * float(np.median(c)))
            up = h[1:-1] - np.maximum.reduce([o[1:-1], c[1:-1], h[:-2], h[2:]])
            dn = np.minimum.reduce([o[1:-1], c[1:-1], l[:-2], l[2:]]) - l[1:-1]
            worst = np.maximum(up, dn)
            hits = np.where(worst > limit)[0] + 1
            if len(hits):
                times = ", ".join(day_df.index[i].strftime("%H:%M") for i in hits[:5])
                issues.append({
                    "issue_type": "phantom_spike",
                    "detail": f"{len(hits)} isolated spike(s) at {times} "
                              f"(max {worst.max()/med_range:.1f}x median bar range; "
                              f"limit {SPIKE_MULT:g}x)",
                })

        # Stale bars
        n_stale = int(((o == h) & (h == l) & (l == c)).sum())
        if n_stale > 10:
            issues.append({
                "issue_type": "stale_bars_critical" if n_stale > 50 else "stale_bars",
                "detail": f"{n_stale}/{n} bars with O=H=L=C",
            })

        # Bar count / early close (calendar-aware)
        close_t = session_close(session_date) or pd.Timestamp("16:00").time()
        expect  = expected_bars(close_t)
        if expect < 390:
            issues.append({"issue_type": "early_close",
                           "detail": f"close {close_t.strftime('%H:%M')}; {n}/{expect} bars"})
        if n < 0.9 * expect:
            issues.append({"issue_type": "bar_count_low",
                           "detail": f"{n} bars (expected {expect})"})

        # ORB range vs day range (informational)
        orb = day_df[day_df.index.time < pd.Timestamp("09:45").time()]
        if len(orb) >= 3:
            orb_range  = float(orb["High"].max() - orb["Low"].min())
            full_range = float(h.max() - l.min())
            if full_range > 0 and orb_range / full_range > 0.80:
                issues.append({"issue_type": "orb_range_anomaly",
                               "detail": f"ORB range = {orb_range/full_range:.0%} of day range"})

        # Price sanity
        n_bad = int(((h < l) | (c <= 0) | (o <= 0)).sum())
        if n_bad:
            issues.append({"issue_type": "price_error",
                           "detail": f"{n_bad} bars with invalid OHLC"})
        return issues

    @staticmethod
    def _is_quality_ok(issues: list[dict]) -> int:
        return 0 if any(i["issue_type"] in CRITICAL_ISSUES for i in issues) else 1

    @staticmethod
    def _discontinuity(open_p: float, prev_close: float | None) -> dict | None:
        """Split-like overnight jump: prior-day levels are not comparable."""
        if not prev_close:
            return None
        ratio = open_p / prev_close
        lo, hi = DISCONTINUITY_BAND
        if lo < ratio < hi:
            return None
        return {"issue_type": "overnight_discontinuity",
                "detail": f"open/prev_close = {ratio:.3f} (split or similar); "
                          "gap/PDH/PDL left NULL"}

    def download_and_store(
        self,
        ticker:    str,
        start:     date,
        end:       date,
        vix_daily: dict[str, float] | None = None,
        chunk_months: int = 3,
        feed: str | None = None,
        adjustment: str | None = None,
    ) -> dict:
        """
        Download all bars for ticker in date range and store in SQLite.
        Downloads in chunks to avoid API timeouts.
        Pre-computes: VWAP per session, ORB levels, session context.

        vix_daily: {date_str: vix_close} — fetched separately from yfinance.
        """
        feed = (feed or DEFAULT_FEED).lower()
        adjustment = (adjustment or DEFAULT_ADJUSTMENT).lower()
        stored = self.stored_adjustment(ticker)
        if stored is not None and stored != adjustment:
            # Mixing adjusted and raw prices in one series would corrupt
            # prev-close/gap levels and every multi-day calculation.
            raise AlpacaFatalError(
                f"{ticker} is already stored with adjustment='{stored}'; refusing to "
                f"add '{adjustment}' bars. Run --wipe --tickers {ticker} first.")
        self._set_setting(f"adjustment:{ticker}", adjustment)
        log.info("Downloading %s bars %s → %s (feed=%s, adjustment=%s)",
                 ticker, start, end, feed, adjustment)
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
                df = self._fetch_from_alpaca(ticker, chunk_start, chunk_end,
                                             feed=feed, adjustment=adjustment)
                bars_written, sessions_written = self._store_bars(ticker, df, vix_daily)
                total_bars     += bars_written
                total_sessions += sessions_written
                log.info("  Stored %d bars, %d sessions", bars_written, sessions_written)
                time.sleep(1)   # rate-limit guard
            except AlpacaFatalError:
                # Keys/permissions: every remaining chunk would fail the same way.
                # Previously this was logged per chunk and the run "finished" with
                # 0 bars — a plausible cause of the empty market_data.db (P2-116).
                self._refresh_meta()
                raise
            except Exception as e:
                log.error("  Chunk %s → %s FAILED: %s", chunk_start, chunk_end, e)
                failed_chunks.append({"start": str(chunk_start), "end": str(chunk_end),
                                      "error": f"{type(e).__name__}: {e}"})
            chunk_start = chunk_end + timedelta(days=1)

        elapsed = time.monotonic() - t0
        self._refresh_meta()
        return {"ticker": ticker, "feed": feed, "adjustment": adjustment, "total_bars": total_bars,
                "total_sessions": total_sessions, "elapsed_sec": round(elapsed, 1),
                "failed_chunks": failed_chunks}

    def _store_bars(
        self,
        ticker:    str,
        df:        pd.DataFrame,
        vix_daily: dict[str, float],
        contiguous: bool = True,
    ) -> tuple[int, int]:
        """Write bars to market_bars and compute session_context.

        contiguous=False: the sessions are NOT consecutive for this ticker (e.g.
        the scattered stock-days picked by universe_select), so no prior-day
        levels are carried: prev_close / gap_pct / PDH / PDL stay NULL."""
        bars_written     = 0
        sessions_written = 0

        trading_days = sorted(set(df.index.date))
        prev_close, prev_high, prev_low = None, None, None
        if trading_days and contiguous:
            # Seed from the last stored session before this chunk, so the first
            # session of each chunk still gets gap_pct / PDH / PDL.
            prev_close, prev_high, prev_low = self._prev_session_levels(
                ticker, trading_days[0])

        for day in trading_days:
            day_str  = str(day)
            day_df   = df[df.index.date == day].copy()
            if not contiguous:
                prev_close = prev_high = prev_low = None

            # Regular session only: bars starting 09:30-15:59 (390 bars). A bar
            # labelled 16:00 covers 16:00-16:01, i.e. after-hours trading.
            day_df = day_df[
                (day_df.index.time >= pd.Timestamp("09:30").time()) &
                (day_df.index.time <  pd.Timestamp("16:00").time())
            ]
            close_t = session_close(day)
            if close_t is None:
                log.warning("  %s %s is not an NYSE session — %d bars skipped",
                            ticker, day_str, len(day_df))
                continue
            day_df = day_df[day_df.index.time < close_t]   # drops post-close prints on half-days
            if len(day_df) < 10:
                prev_close = prev_high = prev_low = None
                continue

            # Pre-compute session VWAP
            day_df = day_df.copy()
            day_df["vwap"] = self._compute_vwap(day_df)

            # Validate
            issues  = self._validate_day(day_df, day)
            open_p  = float(day_df["Open"].iloc[0])
            disc    = self._discontinuity(open_p, prev_close)
            if disc:
                issues.append(disc)
                prev_close = prev_high = prev_low = None
            quality_ok  = self._is_quality_ok(issues)
            gap_session = 1 if any(i["issue_type"] == "bar_count_low" for i in issues) else 0
            with self._conn() as conn:
                self._record_issues(conn, ticker, day_str, issues)

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
                cur = conn.executemany(
                    "INSERT OR IGNORE INTO market_bars "
                    "(ticker,ts,ts_date,ts_time,open,high,low,close,volume,vwap,bar_interval,quality_flag) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    bar_rows
                )
                bars_written += max(cur.rowcount, 0)   # rows actually inserted, not attempted

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

    def _refresh_meta(self) -> None:
        """Recompute data_store_meta from the tables (it used to ADD each run's
        attempted counts, so re-runs inflated it)."""
        import json
        with self._conn() as c:
            tickers = [r[0] for r in c.execute(
                "SELECT DISTINCT ticker FROM session_context ORDER BY ticker")]
            rng = c.execute("SELECT MIN(session_date), MAX(session_date), COUNT(*) "
                            "FROM session_context").fetchone()
            nbars = c.execute("SELECT COUNT(*) FROM market_bars").fetchone()[0]
            c.execute(
                """INSERT OR REPLACE INTO data_store_meta
                   (id, tickers, earliest_date, latest_date, total_bars, total_sessions,
                    last_updated, download_ts, quality_report)
                   VALUES (1, ?, ?, ?, ?, ?, ?,
                     (SELECT download_ts FROM data_store_meta WHERE id=1),
                     (SELECT quality_report FROM data_store_meta WHERE id=1))""",
                (json.dumps(tickers), rng[0], rng[1], nbars, rng[2],
                 datetime.now(timezone.utc).replace(tzinfo=None).isoformat()))

    def revalidate(self, ticker: str) -> dict:
        """Re-run quality checks on stored bars (no network). Also removes bars
        stored after each session's real close (half-day after-hours prints),
        and recomputes n_bars and prior-day levels (prev_close/PDH/PDL/gap_pct)
        from the trimmed sessions. Use after a check or threshold changes."""
        with self._conn() as c:
            days = [r[0] for r in c.execute(
                "SELECT DISTINCT ts_date FROM market_bars WHERE ticker=? ORDER BY ts_date",
                (ticker,))]
        prev = None                     # (close, high, low) of previous kept session
        n_ok = trimmed = 0
        for d in days:
            dd = date.fromisoformat(d)
            close_t = session_close(dd) or pd.Timestamp("16:00").time()
            with self._conn() as c:
                trimmed += c.execute(
                    "DELETE FROM market_bars WHERE ticker=? AND ts_date=? AND ts_time>=?",
                    (ticker, d, close_t.strftime("%H:%M"))).rowcount
            day_df = self.get_session_bars(ticker, d)
            if day_df.empty:
                continue
            issues = self._validate_day(day_df, dd)
            open_p = float(day_df["Open"].iloc[0])
            disc = self._discontinuity(open_p, prev[0] if prev else None)
            if disc:
                issues.append(disc)
            p_close, p_high, p_low = (None, None, None) if (disc or not prev) else prev
            gap = round((open_p - p_close) / p_close * 100, 3) if p_close else None
            ok  = self._is_quality_ok(issues)
            gs  = 1 if any(i["issue_type"] == "bar_count_low" for i in issues) else 0
            with self._conn() as c:
                self._record_issues(c, ticker, d, issues)
                c.execute("UPDATE session_context SET quality_ok=?, gap_session=?, n_bars=?, "
                          "prev_close=?, gap_pct=?, pdh=?, pdl=? "
                          "WHERE ticker=? AND session_date=?",
                          (ok, gs, len(day_df), p_close, gap, p_high, p_low, ticker, d))
            n_ok += ok
            prev = (float(day_df["Close"].iloc[-1]), float(day_df["High"].max()),
                    float(day_df["Low"].min()))
        self._refresh_meta()
        return {"ticker": ticker, "sessions": len(days), "quality_ok": n_ok,
                "bars_trimmed": trimmed}

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
            ok_rows = conn.execute(
                "SELECT ticker, COUNT(*), SUM(quality_ok) FROM session_context GROUP BY ticker"
            ).fetchall()

        result: dict[str, Any] = {
            "meta": dict(meta) if meta else {},
            "by_ticker": [dict(r) for r in by_ticker],
            "quality_issues": {r[0]: r[1] for r in quality_issues},
            "quality_ok": {r[0]: (r[2] or 0, r[1]) for r in ok_rows},
        }
        for t in result["by_ticker"]:
            t["adjustment"] = self.stored_adjustment(t["ticker"])
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
    p.add_argument("--adjustment",     default=None, choices=["raw", "split", "all"],
                   help="Override ALPACA_ADJUSTMENT (default raw — right for intraday)")
    p.add_argument("--wipe",           action="store_true",
                   help="Delete stored bars/sessions/quality rows for --tickers "
                        "(FRED/sentiment tables untouched)")
    p.add_argument("--revalidate",     action="store_true",
                   help="Re-run quality checks on stored bars for --tickers (no network)")
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

    elif args.wipe:
        for ticker in args.tickers:
            n = store.wipe(ticker)
            print(f"{ticker}: deleted {n['market_bars']:,} bars, "
                  f"{n['session_context']:,} sessions, {n['data_quality_log']:,} quality rows")
        print("Tip: run VACUUM to return the freed disk space (optional).")

    elif args.revalidate:
        for ticker in args.tickers:
            r = store.revalidate(ticker)
            n = r["sessions"]
            print(f"{ticker}: quality_ok {r['quality_ok']}/{n}"
                  + (f" ({r['quality_ok']/n:.0%})" if n else "")
                  + f"; {r['bars_trimmed']:,} post-close bars removed")

    elif args.status:
        st = store.status()
        print("\n=== DATA STORE STATUS ===")
        for t in st["by_ticker"]:
            ok, n = st["quality_ok"].get(t["ticker"], (0, 0))
            print(f"  {t['ticker']}: {t['sessions']} sessions, {t['bars']:,} bars, "
                  f"adjustment={t['adjustment']}, quality_ok {ok}/{n}"
                  + (f" ({ok/n:.0%})" if n else ""))
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
                result = store.download_and_store(ticker, start, end, feed=args.feed,
                                                  adjustment=args.adjustment)
            except AlpacaFatalError as e:
                sys.exit(f"✗ {ticker}: stopped — {e}")
            print(f"Stored: {result['total_bars']:,} bars, "
                  f"{result['total_sessions']} sessions in {result['elapsed_sec']}s "
                  f"(feed={result['feed']}, adjustment={result['adjustment']})")
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
