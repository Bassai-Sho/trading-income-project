"""
fred_store.py
=============
FRED macro data store — same download-once, query-forever pattern as market_data_store.py.

WHY
───
The system's current VIX source is yfinance — an unofficial scraper that can
fail at 09:00 right when the morning brief needs it. FRED's VIXCLS is the
authoritative CBOE source. Storing it locally means the morning brief never
makes a network call for VIX.

Same reasoning as Alpaca: download once, store in SQLite, update daily.
Seven series, ~51,000 rows total, under 4MB.

SERIES STORED
─────────────
VIXCLS          VIX close              Daily   1990+  Primary regime indicator
DGS2            2Y Treasury yield      Daily   1976+  Short-rate expectation
DGS10           10Y Treasury yield     Daily   1962+  Term premium / long rate
T10YIE          10Y Breakeven infl.    Daily   2003+  Inflation expectations
BAMLH0A0HYM2    HY OA spread           Daily   1997+  Risk-on / risk-off
FEDFUNDS        Fed Funds Rate         Monthly 1954+  Rate environment
CPIAUCSL        CPI All Items          Monthly 1947+  Inflation reality

INTEGRATION
───────────
  morning_brief.py — get_macro_context() replaces yfinance VIX fetch
  data_corrector.py — FRED VIX replaces yfinance daily anchor where available
  session_context   — vix_close populated from FRED (more authoritative)
  runner.py         — daily 16:30 update job

USAGE
─────
  # Free API key at: https://fred.stlouisfed.org/docs/api/api_key.html
  export FRED_API_KEY=your_key_here

  # Download all series (once, ~30 seconds)
  python fred_store.py --download --db DATA/market_data.db

  # Daily update (called by runner.py at 16:30)
  python fred_store.py --update --db DATA/market_data.db

  # Show what's stored
  python fred_store.py --status --db DATA/market_data.db

  # Query macro context for a date
  python fred_store.py --date 2022-06-13 --db DATA/market_data.db
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any

import requests

log = logging.getLogger("fred_store")

FRED_BASE = "https://api.stlouisfed.org/fred"
FRED_RATE_LIMIT_DELAY = 0.6   # 120 req/min → 0.5s min; use 0.6s for safety

# ---------------------------------------------------------------------------
# Series definitions
# ---------------------------------------------------------------------------

SERIES: dict[str, dict] = {
    "VIXCLS": {
        "title":     "CBOE Volatility Index (VIX)",
        "frequency": "D",
        "units":     "Index",
        "use":       "regime",
        "note":      "Primary regime indicator. Replaces yfinance ^VIX.",
    },
    "DGS2": {
        "title":     "2-Year Treasury Constant Maturity Rate",
        "frequency": "D",
        "units":     "Percent",
        "use":       "rates",
        "note":      "Short-rate expectation. Part of yield curve spread.",
    },
    "DGS10": {
        "title":     "10-Year Treasury Constant Maturity Rate",
        "frequency": "D",
        "units":     "Percent",
        "use":       "rates",
        "note":      "Long rate. Yield curve = DGS10 - DGS2.",
    },
    "T10YIE": {
        "title":     "10-Year Breakeven Inflation Rate",
        "frequency": "D",
        "units":     "Percent",
        "use":       "inflation",
        "note":      "Market's inflation expectation. High = stagflation risk.",
    },
    "BAMLH0A0HYM2": {
        "title":     "ICE BofA US High Yield Index OAS",
        "frequency": "D",
        "units":     "Percent",
        "use":       "credit",
        "note":      "HY spread. >500bp = risk-off / stress. <300bp = risk-on.",
    },
    "FEDFUNDS": {
        "title":     "Federal Funds Effective Rate",
        "frequency": "M",
        "units":     "Percent",
        "use":       "rates",
        "note":      "Current rate environment. Rising = headwind for equities.",
    },
    "CPIAUCSL": {
        "title":     "Consumer Price Index for All Urban Consumers",
        "frequency": "M",
        "units":     "Index 1982-84=100",
        "use":       "inflation",
        "note":      "YoY % change signals inflation regime.",
    },
}

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS fred_series (
    series_id    TEXT PRIMARY KEY,
    title        TEXT,
    frequency    TEXT,
    units        TEXT,
    use_tag      TEXT,
    note         TEXT,
    obs_start    TEXT,
    obs_end      TEXT,
    n_obs        INTEGER DEFAULT 0,
    last_updated TEXT
);

CREATE TABLE IF NOT EXISTS fred_observations (
    series_id  TEXT    NOT NULL,
    date       TEXT    NOT NULL,
    value      REAL,
    PRIMARY KEY (series_id, date)
);
CREATE INDEX IF NOT EXISTS idx_fred_obs ON fred_observations(series_id, date);
"""

# ---------------------------------------------------------------------------
# Low-level FRED API call
# ---------------------------------------------------------------------------

def _fred_get(endpoint: str, params: dict, api_key: str) -> dict:
    """Make one FRED API call. Raises on HTTP error."""
    params["api_key"]   = api_key
    params["file_type"] = "json"
    resp = requests.get(f"{FRED_BASE}/{endpoint}", params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()

def _parse_value(raw: str) -> float | None:
    """FRED uses '.' for missing values."""
    if raw == "." or raw is None:
        return None
    try:
        return float(raw)
    except (ValueError, TypeError):
        return None

# ---------------------------------------------------------------------------
# FredDataStore
# ---------------------------------------------------------------------------

class FredDataStore:
    """
    SQLite-backed FRED macro data store.
    Writes into the same market_data.db as MarketDataStore (Alpaca bars).
    """

    def __init__(self, db_path: str = "DATA/market_data.db") -> None:
        self.db_path = db_path
        self._api_key: str | None = None
        self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.db_path, timeout=10)
        c.row_factory = sqlite3.Row
        return c

    def _init_schema(self) -> None:
        with self._conn() as c:
            c.executescript(SCHEMA)

    def _get_api_key(self) -> str:
        if self._api_key:
            return self._api_key
        key = os.environ.get("FRED_API_KEY", "")
        if not key:
            raise EnvironmentError(
                "FRED_API_KEY not set. Free key at: "
                "https://fred.stlouisfed.org/docs/api/api_key.html"
            )
        self._api_key = key
        return key

    # ── Download ─────────────────────────────────────────────────────────────

    def download_series(
        self,
        series_id:  str,
        start_date: str = "2016-01-01",
        end_date:   str | None = None,
    ) -> int:
        """
        Download all observations for one series and store in SQLite.
        Returns the number of rows written.
        """
        api_key  = self._get_api_key()
        end_date = end_date or str(date.today())

        log.info("Downloading FRED %s (%s → %s)...", series_id, start_date, end_date)

        # Fetch series metadata
        try:
            meta = _fred_get("series", {"series_id": series_id}, api_key)
        except Exception as e:
            log.error("FRED series metadata failed for %s: %s", series_id, e)
            return 0
        time.sleep(FRED_RATE_LIMIT_DELAY)

        # Fetch observations
        try:
            data = _fred_get("series/observations", {
                "series_id":          series_id,
                "observation_start":  start_date,
                "observation_end":    end_date,
                "sort_order":         "asc",
            }, api_key)
        except Exception as e:
            log.error("FRED observations failed for %s: %s", series_id, e)
            return 0
        time.sleep(FRED_RATE_LIMIT_DELAY)

        obs = data.get("observations", [])
        rows = [(series_id, o["date"], _parse_value(o["value"])) for o in obs]

        with self._conn() as conn:
            # Upsert series metadata
            s_meta = SERIES.get(series_id, {})
            conn.execute(
                """INSERT OR REPLACE INTO fred_series
                   (series_id, title, frequency, units, use_tag, note,
                    obs_start, obs_end, n_obs, last_updated)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (series_id,
                 s_meta.get("title", series_id),
                 s_meta.get("frequency", ""),
                 s_meta.get("units", ""),
                 s_meta.get("use", ""),
                 s_meta.get("note", ""),
                 rows[0][1] if rows else None,
                 rows[-1][1] if rows else None,
                 len(rows),
                 datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds"))
            )
            # Upsert observations
            conn.executemany(
                "INSERT OR REPLACE INTO fred_observations (series_id, date, value) VALUES (?,?,?)",
                rows
            )

        log.info("  %s: %d observations stored", series_id, len(rows))
        return len(rows)

    def download_all(self, start_date: str = "2016-01-01") -> dict[str, int]:
        """Download all configured series. Returns {series_id: n_rows}."""
        results = {}
        for sid in SERIES:
            n = self.download_series(sid, start_date)
            results[sid] = n
        log.info("FRED download complete: %d series, %d total observations",
                 len(results), sum(results.values()))
        return results

    # ── Daily update ──────────────────────────────────────────────────────────

    def update(self) -> dict[str, int]:
        """
        Pull recent observations for all stored series.
        Fetches last 30 days to catch revisions.
        Called by runner.py at 16:30 EST daily.
        """
        start_date = (date.today() - timedelta(days=30)).isoformat()
        results = {}
        for sid in SERIES:
            # Only update if series is already downloaded
            with self._conn() as conn:
                exists = conn.execute(
                    "SELECT 1 FROM fred_series WHERE series_id=?", (sid,)
                ).fetchone()
            if not exists:
                log.info("FRED %s not downloaded yet — skipping update", sid)
                results[sid] = 0
                continue
            n = self.download_series(sid, start_date)
            results[sid] = n
        log.info("FRED update complete")
        return results

    # ── Query API ─────────────────────────────────────────────────────────────

    def get_latest(self, series_id: str, as_of: str | None = None) -> float | None:
        """
        Get the most recent non-null value for a series on or before as_of date.
        Returns None if no data.
        """
        as_of = as_of or str(date.today())
        with self._conn() as conn:
            row = conn.execute(
                """SELECT value FROM fred_observations
                   WHERE series_id=? AND date<=? AND value IS NOT NULL
                   ORDER BY date DESC LIMIT 1""",
                (series_id, as_of)
            ).fetchone()
        return float(row["value"]) if row else None

    def get_series(
        self, series_id: str, start: str, end: str
    ) -> list[dict]:
        """Return observations as list of {date, value} dicts."""
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT date, value FROM fred_observations
                   WHERE series_id=? AND date>=? AND date<=?
                   ORDER BY date""",
                (series_id, start, end)
            ).fetchall()
        return [dict(r) for r in rows]

    def get_macro_context(self, as_of: str | None = None) -> dict[str, Any]:
        """
        Return a macro snapshot for the given date.
        Used by morning_brief.py and session_analyser.py.

        Returns:
          vix             float   VIX level (CBOE via FRED)
          vix_regime      str     NORMAL / ELEVATED / HIGH / EXTREME
          dgs2            float   2Y Treasury yield %
          dgs10           float   10Y Treasury yield %
          yield_curve     float   10Y - 2Y spread (negative = inverted)
          hy_spread       float   HY OAS spread %
          fedfunds        float   Effective Fed Funds Rate %
          cpi_level       float   CPI index level
          cpi_yoy_pct     float   CPI year-over-year %
          breakeven_10y   float   10Y breakeven inflation %
          data_date       str     date the values are sourced from
        """
        as_of = as_of or str(date.today())

        vix    = self.get_latest("VIXCLS",       as_of)
        dgs2   = self.get_latest("DGS2",         as_of)
        dgs10  = self.get_latest("DGS10",        as_of)
        hy     = self.get_latest("BAMLH0A0HYM2", as_of)
        ff     = self.get_latest("FEDFUNDS",     as_of)
        cpi    = self.get_latest("CPIAUCSL",     as_of)
        bk10   = self.get_latest("T10YIE",       as_of)

        # VIX regime
        vix_regime = "UNKNOWN"
        if vix is not None:
            if vix > 35:   vix_regime = "EXTREME"
            elif vix > 25: vix_regime = "HIGH"
            elif vix > 18: vix_regime = "ELEVATED"
            else:          vix_regime = "NORMAL"

        # Yield curve spread
        yield_curve = round(dgs10 - dgs2, 3) if (dgs10 and dgs2) else None

        # CPI year-over-year
        cpi_yoy = None
        if cpi is not None:
            cpi_prev = self.get_latest(
                "CPIAUCSL",
                (date.fromisoformat(as_of) - timedelta(days=365)).isoformat()
            )
            if cpi_prev and cpi_prev > 0:
                cpi_yoy = round((cpi - cpi_prev) / cpi_prev * 100, 2)

        return {
            "vix":           round(vix,  2) if vix  else None,
            "vix_regime":    vix_regime,
            "dgs2":          round(dgs2, 3) if dgs2 else None,
            "dgs10":         round(dgs10,3) if dgs10 else None,
            "yield_curve":   yield_curve,
            "hy_spread":     round(hy,   2) if hy   else None,
            "fedfunds":      round(ff,   2) if ff   else None,
            "cpi_level":     round(cpi,  3) if cpi  else None,
            "cpi_yoy_pct":   cpi_yoy,
            "breakeven_10y": round(bk10, 3) if bk10 else None,
            "data_date":     as_of,
        }

    # ── Status ────────────────────────────────────────────────────────────────

    def status(self) -> dict:
        """Summary of what's stored."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT series_id, title, obs_start, obs_end, n_obs, last_updated "
                "FROM fred_series ORDER BY series_id"
            ).fetchall()
        return {"series": [dict(r) for r in rows]}


# ---------------------------------------------------------------------------
# Convenience function for morning_brief.py (no class needed)
# ---------------------------------------------------------------------------

def get_macro_context(
    db_path: str = "DATA/market_data.db",
    as_of:   str | None = None,
) -> dict[str, Any]:
    """
    Single-function interface for morning_brief.py.

    Falls back gracefully to yfinance if FRED data is not yet downloaded.
    """
    try:
        store   = FredDataStore(db_path)
        context = store.get_macro_context(as_of)

        # If VIX is missing from FRED (not downloaded yet), fall back
        if context["vix"] is None:
            raise ValueError("FRED VIX unavailable")

        context["source"] = "FRED (local)"
        return context

    except Exception as e:
        log.info("FRED store fallback to yfinance: %s", e)
        # Graceful fallback
        try:
            import yfinance as yf
            vix_df = yf.download("^VIX", period="5d", interval="1d",
                                  auto_adjust=True, progress=False)
            if isinstance(vix_df.columns, pd.MultiIndex):
                vix_df.columns = [c[0] if isinstance(c, tuple) else c
                                  for c in vix_df.columns]
            vix = float(vix_df["Close"].dropna().iloc[-1]) if not vix_df.empty else None
            vix_regime = ("EXTREME" if vix and vix > 35 else
                          "HIGH"     if vix and vix > 25 else
                          "ELEVATED" if vix and vix > 18 else "NORMAL") if vix else "UNKNOWN"
            return {
                "vix": round(vix, 2) if vix else None,
                "vix_regime": vix_regime,
                "dgs2": None, "dgs10": None, "yield_curve": None,
                "hy_spread": None, "fedfunds": None,
                "cpi_level": None, "cpi_yoy_pct": None, "breakeven_10y": None,
                "data_date": str(date.today()),
                "source": "yfinance (FRED not available)",
            }
        except Exception as e2:
            log.warning("Both FRED and yfinance VIX unavailable: %s", e2)
            return {"vix": None, "vix_regime": "UNKNOWN", "source": "unavailable",
                    "data_date": str(date.today())}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S"
    )

    p = argparse.ArgumentParser(description="FRED macro data store")
    p.add_argument("--download", action="store_true", help="Download all series (run once)")
    p.add_argument("--update",   action="store_true", help="Update with last 30 days")
    p.add_argument("--status",   action="store_true", help="Show stored series")
    p.add_argument("--date",     metavar="YYYY-MM-DD", help="Show macro context for date")
    p.add_argument("--db",       default="DATA/market_data.db")
    p.add_argument("--start",    default="2016-01-01", help="Download start date")
    args = p.parse_args()

    store = FredDataStore(args.db)

    if args.download:
        print(f"\nDownloading FRED macro series from {args.start}...")
        results = store.download_all(args.start)
        print(f"\nComplete:")
        for sid, n in results.items():
            print(f"  {sid:<20} {n:>6,} observations  — {SERIES[sid]['title']}")

    elif args.update:
        results = store.update()
        print("\nFRED update:")
        for sid, n in results.items():
            print(f"  {sid}: {n} new/revised observations")

    elif args.status:
        st = store.status()
        print(f"\n{'Series':<22} {'Title':<40} {'From':<12} {'To':<12} {'Rows':>6}")
        print("-" * 96)
        for s in st["series"]:
            print(f"  {s['series_id']:<20} {s['title'][:38]:<40} "
                  f"{s['obs_start'] or '':12} {s['obs_end'] or '':12} {s['n_obs']:>6,}")

    elif args.date:
        ctx = store.get_macro_context(args.date)
        print(f"\nMacro context for {args.date}:")
        print(f"  VIX:             {ctx['vix']}  ({ctx['vix_regime']})")
        print(f"  2Y Treasury:     {ctx['dgs2']}%")
        print(f"  10Y Treasury:    {ctx['dgs10']}%")
        print(f"  Yield curve:     {ctx['yield_curve']}%  ({'INVERTED ⚠️' if ctx['yield_curve'] and ctx['yield_curve'] < 0 else 'normal'})")
        print(f"  HY spread:       {ctx['hy_spread']}%")
        print(f"  Fed Funds:       {ctx['fedfunds']}%")
        print(f"  CPI YoY:         {ctx['cpi_yoy_pct']}%")
        print(f"  10Y breakeven:   {ctx['breakeven_10y']}%")

    else:
        p.print_help()
