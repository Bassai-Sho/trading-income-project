"""
sentiment_store.py
==================
Market sentiment and positioning data — same download-once, query-forever
pattern as market_data_store.py and fred_store.py.

TWO STREAMS
───────────
1. CBOE Put/Call Ratio (daily)
   Source A: Historical CSVs  2006-2019  cdn.cboe.com (confirmed free)
   Source B: Daily HTML scrape 2019+      cboe.com/us/options/market_statistics/daily/
   Series:   total_pc, equity_pc, index_pc
   Use:      Contrarian sentiment. High = fear (leans bullish). Low = complacency.
             Equity P/C is the cleanest signal (strips institutional hedging noise).

2. CFTC Commitment of Traders — S&P 500 (weekly, Tuesday positions)
   Source:   CFTC.gov annual ZIP files (free, no authentication)
   Series:   Leveraged Money net (hedge funds), Asset Manager net (institutions)
   Use:      Institutional positioning. Lev net heavily short = crowded trade.
             Divergence between lev and asset mgr often marks turning points.

INTEGRATION
───────────
  morning_brief.py  — get_sentiment_context(date) shows P/C + COT net
  session_context   — sentiment columns added to session_context table
  runner.py         — daily 16:40 update (P/C today), Saturday COT update

USAGE
─────
  python sentiment_store.py --download --db DATA/market_data.db
  python sentiment_store.py --update   --db DATA/market_data.db
  python sentiment_store.py --status   --db DATA/market_data.db
  python sentiment_store.py --date 2022-06-13 --db DATA/market_data.db
"""

from __future__ import annotations

import argparse
import csv
import io
import logging
import re
import sqlite3
import time
import zipfile
from datetime import date, datetime, timedelta
from typing import Any

import requests

log = logging.getLogger("sentiment_store")

REQUEST_DELAY = 1.0   # seconds between requests — be polite

# ---------------------------------------------------------------------------
# CBOE URLs
# ---------------------------------------------------------------------------

CBOE_CSV_URLS: dict[str, str] = {
    "total":  "https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/totalpc.csv",
    "equity": "https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/equitypc.csv",
    "index":  "https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/indexpc.csv",
    # Archive: 1995-2003
    "archive": "https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/pcratioarchive.csv",
}
CBOE_DAILY_STATS_URL = "https://www.cboe.com/us/options/market_statistics/daily/"

# ---------------------------------------------------------------------------
# CFTC URLs
# ---------------------------------------------------------------------------

CFTC_ZIP_URL = "https://www.cftc.gov/files/dea/history/fut_fin_txt_{year}.zip"
CFTC_SP500_MARKET_KEY = "S&P 500"  # matches "S&P 500 Consolidated - CHICAGO..."
CFTC_COLUMNS = {
    "market":       "Market_and_Exchange_Names",
    "date":         "Report_Date_as_YYYY-MM-DD",
    "open_int":     "Open_Interest_All",
    "lev_long":     "Lev_Money_Positions_Long_All",
    "lev_short":    "Lev_Money_Positions_Short_All",
    "asset_long":   "Asset_Mgr_Positions_Long_All",
    "asset_short":  "Asset_Mgr_Positions_Short_All",
    "lev_chg_long": "Change_in_Lev_Money_Long_All",
    "lev_chg_short":"Change_in_Lev_Money_Short_All",
}

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS cboe_pc_daily (
    date        TEXT    PRIMARY KEY,
    total_pc    REAL,
    equity_pc   REAL,
    index_pc    REAL,
    source      TEXT    DEFAULT 'csv'  -- 'csv' or 'html'
);
CREATE INDEX IF NOT EXISTS idx_cboe_date ON cboe_pc_daily(date);

CREATE TABLE IF NOT EXISTS cftc_cot_weekly (
    report_date     TEXT    PRIMARY KEY,
    open_interest   INTEGER,
    lev_long        INTEGER,
    lev_short       INTEGER,
    lev_net         INTEGER,
    lev_net_chg     INTEGER,
    lev_pct_long    REAL,
    asset_long      INTEGER,
    asset_short     INTEGER,
    asset_net       INTEGER,
    source_year     INTEGER
);
CREATE INDEX IF NOT EXISTS idx_cot_date ON cftc_cot_weekly(report_date);

CREATE TABLE IF NOT EXISTS sentiment_meta (
    id              INTEGER PRIMARY KEY DEFAULT 1,
    cboe_earliest   TEXT,
    cboe_latest     TEXT,
    cboe_n_rows     INTEGER,
    cot_earliest    TEXT,
    cot_latest      TEXT,
    cot_n_rows      INTEGER,
    last_updated    TEXT
);
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _conn(db_path: str) -> sqlite3.Connection:
    c = sqlite3.connect(db_path, timeout=10)
    c.row_factory = sqlite3.Row
    return c

def _int(s: str) -> int:
    try:
        return int(str(s).replace(",", "").strip())
    except (ValueError, TypeError):
        return 0

def _float(s: str) -> float | None:
    try:
        return float(str(s).replace(",", "").strip())
    except (ValueError, TypeError):
        return None

# ---------------------------------------------------------------------------
# CBOE Put/Call — historical CSV
# ---------------------------------------------------------------------------

def _parse_cboe_csv(text: str, series_name: str) -> list[tuple[str, float]]:
    """
    Parse a CBOE put/call CSV.
    Returns list of (date_str, ratio) tuples.
    CBOE CSVs have 4 header lines of boilerplate before the actual data.
    Column format: Date, PutVol, CallVol, TotalVol, PC_Ratio
    """
    rows: list[tuple[str, float]] = []
    lines = text.strip().splitlines()
    # Skip header lines — find the first line that starts with a date (MM/DD/YYYY)
    data_start = 0
    for i, line in enumerate(lines):
        if re.match(r'\d{1,2}/\d{1,2}/\d{4}', line.strip()):
            data_start = i
            break

    for line in lines[data_start:]:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        date_raw = parts[0]
        ratio_raw = parts[-1]   # last column is the ratio
        try:
            dt = datetime.strptime(date_raw, "%m/%d/%Y")
            date_str = dt.strftime("%Y-%m-%d")
            ratio = float(ratio_raw)
            if 0 < ratio < 5:   # sanity check
                rows.append((date_str, ratio))
        except (ValueError, IndexError):
            continue
    return rows


def download_cboe_historical(db_path: str) -> int:
    """
    Download historical CBOE P/C CSVs (2006-2019) and store in DB.
    Returns total rows written.
    """
    with _conn(db_path) as c:
        c.executescript(SCHEMA)

    all_rows: dict[str, dict[str, float | None]] = {}

    for series, url in CBOE_CSV_URLS.items():
        if series == "archive":
            continue   # archive (1995-2003) has different format; skip for now
        try:
            log.info("Downloading CBOE %s P/C from %s", series, url)
            r = requests.get(url, timeout=20,
                             headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            pairs = _parse_cboe_csv(r.text, series)
            for date_str, ratio in pairs:
                all_rows.setdefault(date_str, {})[series] = ratio
            log.info("  %s: %d rows", series, len(pairs))
            time.sleep(REQUEST_DELAY)
        except Exception as e:
            log.warning("CBOE %s CSV failed: %s", series, e)

    # Write merged rows
    written = 0
    with _conn(db_path) as c:
        for date_str, series_vals in sorted(all_rows.items()):
            c.execute(
                "INSERT OR REPLACE INTO cboe_pc_daily "
                "(date, total_pc, equity_pc, index_pc, source) VALUES (?,?,?,?,?)",
                (date_str,
                 series_vals.get("total"),
                 series_vals.get("equity"),
                 series_vals.get("index"),
                 "csv")
            )
            written += 1

    log.info("CBOE historical: %d dates stored", written)
    return written


def scrape_cboe_today(db_path: str) -> dict | None:
    """
    Scrape today's put/call ratios from the CBOE daily market statistics page.
    The page embeds data in both static HTML tables and escaped JSON.
    Stores the result in cboe_pc_daily table.
    Returns parsed dict or None on failure.
    """
    try:
        r = requests.get(CBOE_DAILY_STATS_URL, timeout=15,
                         headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        html = r.text

        # Parse HTML table: ">TOTAL PUT/CALL RATIO</td><td ...>0.88</td>"
        def _extract(label: str) -> float | None:
            m = re.search(
                rf">{re.escape(label)}</td><td[^>]*>([\d.]+)</td>",
                html, re.IGNORECASE
            )
            return float(m.group(1)) if m else None

        total_pc  = _extract("TOTAL PUT/CALL RATIO")
        equity_pc = _extract("EQUITY PUT/CALL RATIO")
        index_pc  = _extract("INDEX PUT/CALL RATIO")

        if total_pc is None:
            # Fallback: try JSON embedded data
            m = re.search(
                r'"TOTAL PUT/CALL RATIO\\\\",\\\\"value\\\\":\\\\"([\d.]+)\\\\"',
                html
            )
            total_pc = float(m.group(1)) if m else None

        if total_pc is None:
            log.warning("Could not parse CBOE P/C from daily stats page")
            return None

        today = date.today().isoformat()
        with _conn(db_path) as c:
            c.executescript(SCHEMA)
            c.execute(
                "INSERT OR REPLACE INTO cboe_pc_daily "
                "(date, total_pc, equity_pc, index_pc, source) VALUES (?,?,?,?,?)",
                (today, total_pc, equity_pc, index_pc, "html")
            )

        result = {
            "date": today,
            "total_pc": total_pc,
            "equity_pc": equity_pc,
            "index_pc": index_pc,
            "source": "html",
        }
        log.info("CBOE today scraped: %s", result)
        return result

    except Exception as e:
        log.warning("CBOE daily scrape failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# CFTC COT — annual ZIP files
# ---------------------------------------------------------------------------

def _parse_cftc_zip(content: bytes) -> list[dict]:
    """Extract S&P 500 rows from a CFTC annual ZIP."""
    rows: list[dict] = []
    z = zipfile.ZipFile(io.BytesIO(content))
    fname = z.namelist()[0]
    with z.open(fname) as f:
        text = io.TextIOWrapper(f, encoding="latin-1", errors="replace")
        reader = csv.DictReader(text)
        for row in reader:
            market = row.get(CFTC_COLUMNS["market"], "")
            if CFTC_SP500_MARKET_KEY not in market:
                continue
            try:
                report_date = row.get(CFTC_COLUMNS["date"], "").strip()
                if not report_date:
                    continue
                lev_long  = _int(row.get(CFTC_COLUMNS["lev_long"],  "0"))
                lev_short = _int(row.get(CFTC_COLUMNS["lev_short"], "0"))
                lev_net   = lev_long - lev_short
                lev_chg_l = _int(row.get(CFTC_COLUMNS["lev_chg_long"],  "0"))
                lev_chg_s = _int(row.get(CFTC_COLUMNS["lev_chg_short"], "0"))
                lev_net_chg = lev_chg_l - lev_chg_s
                tot_lev   = lev_long + lev_short
                lev_pct   = round(lev_long / tot_lev * 100, 2) if tot_lev > 0 else None
                asset_long  = _int(row.get(CFTC_COLUMNS["asset_long"],  "0"))
                asset_short = _int(row.get(CFTC_COLUMNS["asset_short"], "0"))
                rows.append({
                    "report_date":   report_date,
                    "open_interest": _int(row.get(CFTC_COLUMNS["open_int"], "0")),
                    "lev_long":      lev_long,
                    "lev_short":     lev_short,
                    "lev_net":       lev_net,
                    "lev_net_chg":   lev_net_chg,
                    "lev_pct_long":  lev_pct,
                    "asset_long":    asset_long,
                    "asset_short":   asset_short,
                    "asset_net":     asset_long - asset_short,
                })
            except Exception as e:
                log.debug("CFTC row parse error: %s", e)
    return rows


def download_cftc_year(year: int, db_path: str) -> int:
    """Download one year of CFTC Financial Futures data. Returns rows written."""
    url = CFTC_ZIP_URL.format(year=year)
    try:
        log.info("Downloading CFTC COT %d...", year)
        r = requests.get(url, timeout=30)
        if r.status_code == 404:
            log.info("CFTC %d: not available yet", year)
            return 0
        r.raise_for_status()
        rows = _parse_cftc_zip(r.content)
        if not rows:
            log.warning("CFTC %d: no S&P 500 rows found", year)
            return 0

        with _conn(db_path) as c:
            c.executescript(SCHEMA)
            c.executemany(
                """INSERT OR REPLACE INTO cftc_cot_weekly
                   (report_date, open_interest, lev_long, lev_short, lev_net,
                    lev_net_chg, lev_pct_long, asset_long, asset_short, asset_net,
                    source_year)
                   VALUES (:report_date,:open_interest,:lev_long,:lev_short,:lev_net,
                           :lev_net_chg,:lev_pct_long,:asset_long,:asset_short,:asset_net,
                           """ + str(year) + ")",
                rows
            )
        log.info("  CFTC %d: %d S&P 500 rows written", year, len(rows))
        return len(rows)
    except Exception as e:
        log.warning("CFTC %d failed: %s", year, e)
        return 0


def download_cftc_all(db_path: str, start_year: int = 2016) -> dict[int, int]:
    """Download CFTC COT for all years from start_year to current year."""
    current_year = date.today().year
    results: dict[int, int] = {}
    for year in range(start_year, current_year + 1):
        n = download_cftc_year(year, db_path)
        results[year] = n
        time.sleep(REQUEST_DELAY)
    return results


# ---------------------------------------------------------------------------
# Query API
# ---------------------------------------------------------------------------

class SentimentStore:
    """
    Unified query interface for CBOE P/C and CFTC COT data.
    Writes to the same market_data.db as MarketDataStore and FredDataStore.
    """

    def __init__(self, db_path: str = "DATA/market_data.db") -> None:
        self.db_path = db_path
        with _conn(self.db_path) as c:
            c.executescript(SCHEMA)

    def get_cboe_pc(self, as_of: str | None = None) -> dict | None:
        """Most recent CBOE P/C on or before as_of date."""
        as_of = as_of or date.today().isoformat()
        with _conn(self.db_path) as c:
            row = c.execute(
                "SELECT * FROM cboe_pc_daily WHERE date<=? ORDER BY date DESC LIMIT 1",
                (as_of,)
            ).fetchone()
        if not row:
            return None
        r = dict(row)
        # Interpret equity P/C
        eq = r.get("equity_pc")
        if eq is not None:
            if eq > 0.85:   sentiment = "FEAR (contrarian bullish)"
            elif eq < 0.55: sentiment = "COMPLACENCY (contrarian bearish)"
            else:           sentiment = "NEUTRAL"
            r["equity_sentiment"] = sentiment
        return r

    def get_cot(self, as_of: str | None = None) -> dict | None:
        """Most recent CFTC COT on or before as_of date."""
        as_of = as_of or date.today().isoformat()
        with _conn(self.db_path) as c:
            row = c.execute(
                "SELECT * FROM cftc_cot_weekly WHERE report_date<=? ORDER BY report_date DESC LIMIT 1",
                (as_of,)
            ).fetchone()
        if not row:
            return None
        r = dict(row)
        lev_pct = r.get("lev_pct_long")
        if lev_pct is not None:
            if lev_pct < 35:   r["lev_sentiment"] = "HEAVILY SHORT (crowded — potential squeeze)"
            elif lev_pct < 45: r["lev_sentiment"] = "NET SHORT"
            elif lev_pct > 65: r["lev_sentiment"] = "HEAVILY LONG (potential unwind)"
            else:              r["lev_sentiment"] = "BALANCED"
        return r

    def get_sentiment_context(self, as_of: str | None = None) -> dict:
        """
        Combined sentiment snapshot for morning_brief.py.
        Returns both P/C and COT, gracefully handling missing data.
        """
        as_of = as_of or date.today().isoformat()
        pc  = self.get_cboe_pc(as_of)
        cot = self.get_cot(as_of)
        return {
            "as_of":          as_of,
            "cboe_date":      pc["date"]         if pc  else None,
            "total_pc":       pc["total_pc"]     if pc  else None,
            "equity_pc":      pc["equity_pc"]    if pc  else None,
            "equity_sentiment": pc.get("equity_sentiment") if pc else None,
            "cot_date":       cot["report_date"] if cot else None,
            "lev_net":        cot["lev_net"]     if cot else None,
            "lev_pct_long":   cot["lev_pct_long"] if cot else None,
            "lev_sentiment":  cot.get("lev_sentiment") if cot else None,
            "asset_net":      cot["asset_net"]   if cot else None,
        }

    def update(self) -> dict:
        """Daily update: scrape today's CBOE P/C + refresh current-year CFTC."""
        results = {"cboe_today": None, "cftc_current_year": 0}

        # CBOE: scrape today's stats
        today_pc = scrape_cboe_today(self.db_path)
        results["cboe_today"] = today_pc

        # CFTC: re-download current year (cumulative ZIP gets latest Friday's data)
        current_year = date.today().year
        n = download_cftc_year(current_year, self.db_path)
        results["cftc_current_year"] = n

        return results

    def status(self) -> dict:
        with _conn(self.db_path) as c:
            cboe = c.execute(
                "SELECT COUNT(*) n, MIN(date) earliest, MAX(date) latest FROM cboe_pc_daily"
            ).fetchone()
            cot = c.execute(
                "SELECT COUNT(*) n, MIN(report_date) earliest, MAX(report_date) latest "
                "FROM cftc_cot_weekly"
            ).fetchone()
        return {
            "cboe_pc": dict(cboe) if cboe else {},
            "cftc_cot": dict(cot) if cot else {},
        }


# ---------------------------------------------------------------------------
# Convenience function for morning_brief.py
# ---------------------------------------------------------------------------

def get_sentiment_context(
    db_path: str = "DATA/market_data.db",
    as_of:   str | None = None,
) -> dict:
    """Single-call interface for morning_brief.py."""
    try:
        return SentimentStore(db_path).get_sentiment_context(as_of)
    except Exception as e:
        log.warning("Sentiment context unavailable: %s", e)
        return {}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S"
    )

    p = argparse.ArgumentParser(description="Sentiment data store (CBOE P/C + CFTC COT)")
    p.add_argument("--download", action="store_true", help="Download all historical data")
    p.add_argument("--update",   action="store_true", help="Daily update (scrape today + current COT year)")
    p.add_argument("--status",   action="store_true")
    p.add_argument("--date",     metavar="YYYY-MM-DD", help="Sentiment context for a date")
    p.add_argument("--db",       default="DATA/market_data.db")
    p.add_argument("--start-year", type=int, default=2016)
    args = p.parse_args()

    store = SentimentStore(args.db)

    if args.download:
        print(f"\nDownloading CBOE P/C historical (2006-2019)...")
        n_cboe = download_cboe_historical(args.db)
        print(f"CBOE: {n_cboe} dates stored")

        print(f"\nDownloading CFTC COT ({args.start_year} to present)...")
        cot_results = download_cftc_all(args.db, args.start_year)
        total_cot = sum(cot_results.values())
        print(f"CFTC: {total_cot} weekly observations across {len(cot_results)} years")

        print(f"\nScraping today's CBOE P/C...")
        today_pc = scrape_cboe_today(args.db)
        if today_pc:
            print(f"Today: total={today_pc['total_pc']} equity={today_pc['equity_pc']} index={today_pc['index_pc']}")

    elif args.update:
        r = store.update()
        print(f"Update: CBOE today={r['cboe_today']}, CFTC rows={r['cftc_current_year']}")

    elif args.status:
        st = store.status()
        pc = st["cboe_pc"]
        ct = st["cftc_cot"]
        print(f"\nCBOE P/C: {pc.get('n',0)} days  {pc.get('earliest','')} → {pc.get('latest','')}")
        print(f"CFTC COT: {ct.get('n',0)} weeks {ct.get('earliest','')} → {ct.get('latest','')}")

    elif args.date:
        ctx = store.get_sentiment_context(args.date)
        print(f"\nSentiment context for {args.date}:")
        print(f"  CBOE P/C ({ctx.get('cboe_date','N/A')}): "
              f"total={ctx.get('total_pc')} equity={ctx.get('equity_pc')} "
              f"→ {ctx.get('equity_sentiment','N/A')}")
        print(f"  CFTC COT ({ctx.get('cot_date','N/A')}): "
              f"lev_net={ctx.get('lev_net'):,} ({ctx.get('lev_pct_long')}% long) "
              f"→ {ctx.get('lev_sentiment','N/A')}"
              if ctx.get('lev_net') is not None else "  CFTC COT: N/A")
        print(f"  Asset Mgr net: {ctx.get('asset_net')}")
    else:
        p.print_help()
