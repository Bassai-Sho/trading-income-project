"""
universe_minutes.py  (P2-125 — box #2 universe, stage 3)
========================================================
Full-session 1-minute bars for the stock-days picked by universe_select
(top 20 per day by opening relative volume) — and nothing else.

Storage reuses MarketDataStore (same schema, calendar-close trimming, quality
checks, quality_ok flag) in a separate file, DATA/universe_minutes.db, so the
four-ticker store stays untouched. The stock-days are scattered (a stock picked
in March may next be picked in September), so bars are stored with
contiguous=False: no prior-day levels are carried between them.

One request per day covers that day's ~20 symbols (09:30-16:00 ET, SIP, raw).
*_DELISTED assets are requested under their base ticker (universe_store's
data_symbol) and stored under the asset symbol, matching the selection table.
Resumable per day; invalid symbols are dropped and the rest retried.

    python src/universe_minutes.py --download --start 2016-01-25 --end 2024-12-31
    python src/universe_minutes.py --status
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import time as _time
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from market_data_store import MarketDataStore                        # noqa: E402
from universe_store import SEALED, FatalDownloadError, UniverseStore, _now, data_symbol  # noqa: E402

log = logging.getLogger("universe_minutes")
NY = "America/New_York"

SCHEMA = """
CREATE TABLE IF NOT EXISTS universe_minutes_done (
    date TEXT PRIMARY KEY, n_symbols INTEGER, n_bars INTEGER, n_missing INTEGER, done_at TEXT);
"""


class MinuteLoader:
    def __init__(self, universe: UniverseStore, minutes: MarketDataStore):
        self.u, self.m = universe, minutes
        with self.u._conn() as c:
            c.executescript(SCHEMA)

    def download(self, client, start: date, end: date, pause: float = 0.2) -> dict:
        if end >= SEALED:
            end = date(2024, 12, 31)
        with self.u._conn() as c:
            sel = c.execute("SELECT date, symbol FROM universe_selection WHERE date BETWEEN ? AND ? "
                            "ORDER BY date, rank", (str(start), str(end))).fetchall()
            done = {r[0] for r in c.execute("SELECT date FROM universe_minutes_done")}
            known = {r[0] for r in c.execute("SELECT symbol FROM universe_assets")}
        by_day: dict[str, list[str]] = {}
        for d, s in sel:
            by_day.setdefault(d, []).append(s)
        todo = [d for d in sorted(by_day) if d not in done]
        failed, bars, missing = [], 0, []
        for k, d in enumerate(todo, 1):
            syms = by_day[d]
            try:
                frames, bad = self._fetch_day(client, d, syms, known)
            except FatalDownloadError:
                raise
            except Exception as e:
                failed.append({"date": d, "error": f"{type(e).__name__}: {e}"})
                log.error("%s failed: %s", d, e)
                continue
            n_day = 0
            for sym, df in frames.items():
                w, _ = self.m._store_bars(sym, df, {}, contiguous=False)
                n_day += w
            miss = [s for s in syms if s not in frames]
            missing += [(d, s) for s in miss]
            with self.u._conn() as c:
                c.execute("INSERT OR REPLACE INTO universe_minutes_done VALUES (?,?,?,?,?)",
                          (d, len(syms), n_day, len(miss), _now()))
            bars += n_day
            if k % 50 == 0 or k == len(todo):
                log.info("minutes %d/%d days (%s: %d symbols, %d bars)", k, len(todo), d, len(syms), n_day)
            _time.sleep(pause)
        return {"days": len(todo), "bars": bars, "missing_stock_days": len(missing),
                "missing_sample": missing[:10], "failed_days": failed}

    @staticmethod
    def _fetch_day(client, d: str, symbols: list[str], known: set[str]):
        """{asset_symbol: DataFrame(Open..Volume, NY index)} for one day."""
        from alpaca.common.exceptions import APIError
        from alpaca.data.enums import Adjustment, DataFeed
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
        by_data = {ds: s for s in symbols if (ds := data_symbol(s, known)) is not None}
        bad: list[str] = []
        t0 = pd.Timestamp(f"{d} 09:30", tz=NY)
        t1 = pd.Timestamp(f"{d} 15:59:59", tz=NY)
        while by_data:
            req = StockBarsRequest(symbol_or_symbols=list(by_data),
                                   timeframe=TimeFrame(1, TimeFrameUnit.Minute),
                                   start=t0.tz_convert("UTC").to_pydatetime(),
                                   end=t1.tz_convert("UTC").to_pydatetime(),
                                   adjustment=Adjustment.RAW, feed=DataFeed.SIP)
            try:
                df = client.get_stock_bars(req).df
                break
            except APIError as e:
                if getattr(e, "status_code", None) in (401, 403):
                    raise FatalDownloadError(f"Alpaca rejected the request: {e}") from e
                m = re.search(r"invalid symbol:\s*([^\s\"}]+)", str(e))
                if not m or m.group(1) not in by_data:
                    raise
                bad.append(by_data.pop(m.group(1)))
        else:
            return {}, bad
        if df is None or len(df) == 0:
            return {}, bad
        df = df.reset_index()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(NY)
        out = {}
        for ds, g in df.groupby("symbol"):
            if ds not in by_data:
                continue
            g = g.set_index("timestamp").sort_index()
            out[by_data[ds]] = g.rename(columns={"open": "Open", "high": "High", "low": "Low",
                                                 "close": "Close", "volume": "Volume"})[
                ["Open", "High", "Low", "Close", "Volume"]]
        return out, bad

    def status(self) -> dict:
        with self.u._conn() as c:
            done = c.execute("SELECT COUNT(*), SUM(n_symbols), SUM(n_missing), SUM(n_bars) "
                             "FROM universe_minutes_done").fetchone()
        with self.m._conn() as c:
            q = c.execute("SELECT COUNT(*), SUM(quality_ok) FROM session_context").fetchone()
            issues = dict(c.execute("SELECT issue_type, COUNT(*) FROM data_quality_log GROUP BY 1").fetchall())
        return {"days_done": done[0], "stock_days_requested": done[1], "stock_days_missing": done[2],
                "bars": done[3], "sessions_stored": q[0], "sessions_quality_ok": q[1],
                "quality_issues": issues}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")
    for noisy in ("market_data_store",):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="DATA/universe.db")
    p.add_argument("--minutes-db", default="DATA/universe_minutes.db")
    p.add_argument("--download", action="store_true")
    p.add_argument("--status", action="store_true")
    p.add_argument("--start", default="2016-01-25")
    p.add_argument("--end", default="2024-12-31")
    a = p.parse_args()
    loader = MinuteLoader(UniverseStore(a.db), MarketDataStore(a.minutes_db))
    if a.download:
        k, s = os.environ.get("ALPACA_API_KEY"), os.environ.get("ALPACA_SECRET_KEY")
        if not k or not s:
            sys.exit("Set ALPACA_API_KEY and ALPACA_SECRET_KEY (source .env first).")
        from alpaca.data.historical import StockHistoricalDataClient
        r = loader.download(StockHistoricalDataClient(k, s), date.fromisoformat(a.start),
                            date.fromisoformat(a.end))
        print({x: r[x] for x in ("days", "bars", "missing_stock_days")},
              f"failed days: {len(r['failed_days'])}")
        if r["missing_sample"]:
            print("   missing (first 10):", r["missing_sample"])
        for f in r["failed_days"][:10]:
            print("  ", f)
        if r["failed_days"]:
            print("Re-run the same command: finished days are skipped.")
            sys.exit(1)
    if a.status or not a.download:
        print(loader.status())
