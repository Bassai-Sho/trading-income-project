"""
universe_select.py  (P2-125 — box #2 universe, stage 2)
=======================================================
"Stocks in play" selection (Zarattini, Barbon & Aziz 2024) on top of
universe_store.py's daily eligibility:

  1. classify    each asset as stock / fund / noncommon.
                 Current listings: Nasdaq Trader symbol directory ETF flag, PLUS
                 the fund-name rules (the ETF flag is "N" for closed-end funds —
                 DSL leaked into the first selection, 26 Sep 2026).
                 Everything else (delisted): name rules. Measured on the
                 directory's current listings (26 Sep 2026): precision 0.95,
                 recall 0.98; the misses are mostly closed-end funds, ETNs and
                 preferreds (all excluded anyway); ~3 of 7,515 operating
                 companies misclassified. Units, warrants, rights, preferreds,
                 notes -> noncommon. ADRs are stocks.
  2. open5       the 09:30-09:35 bar (one 5-min bar) for every stock that is
                 eligible on day D or within the next 14 sessions — i.e. every
                 bar the relative-volume average will need, and nothing else.
                 Requested symbols with no trade in those 5 minutes are stored
                 with volume 0 (so "no trade" is never confused with "not
                 downloaded").
  3. select      RVOL(D) = opening volume(D) / mean opening volume over the
                 stock's 14 PRIOR sessions; keep eligible stocks with
                 RVOL >= 1.0; rank; top 20 per day.

Opening bars exist from 2016-01-04 (the start of Alpaca's history), so the
first day with a full 14-session window is 2016-01-25.

    python src/universe_select.py --classify
    python src/universe_select.py --open5 --start 2016-01-04 --end 2024-12-31
    python src/universe_select.py --select --start 2016-01-25 --end 2024-12-31
    python src/universe_select.py --status
"""
from __future__ import annotations

import argparse
import io
import logging
import os
import re
import sqlite3
import sys
import time as _time
import urllib.request
from datetime import date, datetime, time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from universe_store import (SEALED, FatalDownloadError, UniverseStore, _now,   # noqa: E402
                            data_symbol)

log = logging.getLogger("universe_select")
NY = "America/New_York"
DIRECTORY_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqtraded.txt"
LOOKBACK = 14

SCHEMA = """
CREATE TABLE IF NOT EXISTS universe_class (
    symbol TEXT PRIMARY KEY, kind TEXT, source TEXT, name TEXT);
CREATE TABLE IF NOT EXISTS universe_open5 (
    date TEXT, symbol TEXT, open REAL, high REAL, low REAL, close REAL, volume REAL,
    PRIMARY KEY (date, symbol));
CREATE TABLE IF NOT EXISTS universe_open5_done (
    date TEXT PRIMARY KEY, n_requested INTEGER, n_traded INTEGER, done_at TEXT);
CREATE TABLE IF NOT EXISTS universe_selection (
    date TEXT, symbol TEXT, rvol REAL, rank INTEGER, open5_volume REAL, avg14 REAL,
    PRIMARY KEY (date, symbol));
"""

# ── 1. Classification ─────────────────────────────────────────────────────────

_FUND_WORD = re.compile(r"\b(ETFs?|ETNs?|ETP|Exchange[- ]Traded|Fund|Portfolio|Index|Trust|Shares|"
                        r"Strategy|Income|Treasury|Bond)\b", re.I)
_GENERIC = re.compile(r"\b(ETFs?|ETNs?|ETP|Exchange[- ]Traded (Fund|Note)s?)\b|ADRhedged|UltraPro|"
                      r"\b(2X|3X|-1X|-2X|-3X)\b|\bLeveraged\b|\bInverse\b|\bIndex Fund\b|"
                      r"\b(Closed[- ]End )?Fund\b(?!.*\b(Corp|Inc|Ltd|plc)\b)", re.I)
_PURE_ISSUER = re.compile(r"\b(ProShares|iShares|SPDR|Direxion)\b")
_BRAND = re.compile(
    r"\b(Invesco|Vanguard|WisdomTree|VanEck|Global X|First Trust|Schwab|Xtrackers|Pacer|Defiance|"
    r"GraniteShares|Roundhill|Amplify|Tidal|Janus Henderson|Franklin|Fidelity|JPMorgan|Sprott|abrdn|"
    r"Grayscale|YieldMax|Tuttle|AdvisorShares|Innovator|Simplify|Harbor|Dimensional|Avantis|"
    r"Alpha Architect|Cambria|Nuveen|Columbia|Hartford|John Hancock|PIMCO|Principal|State Street|"
    r"American Century|BlackRock|Virtus|ETFMG|KraneShares|Teucrium|Bitwise|21Shares|Valkyrie|Hashdex|"
    r"CoinShares|ARK|Morgan Stanley|ETRACS|MicroSectors|iPath|"
    r"United States (Oil|Natural Gas|Gasoline|Commodity))\b")
_NONCOMMON = re.compile(r"\b(Warrants?|Units?|Rights?|Preferred|Pref\.?|Notes? due|Debentures?|"
                        r"Depositary Shares,? each representing|Subordinated)\b|\d+(\.\d+)?%", re.I)


def is_fund_name(name: str | None) -> bool:
    n = name or ""
    return bool(_GENERIC.search(n) or _PURE_ISSUER.search(n) or
                (_BRAND.search(n) and _FUND_WORD.search(n)))


def classify_name(name: str | None) -> str:
    if is_fund_name(name):
        return "fund"
    if _NONCOMMON.search(name or ""):
        return "noncommon"
    return "stock"


def load_directory(url: str = DIRECTORY_URL, text: str | None = None) -> pd.DataFrame:
    if text is None:
        with urllib.request.urlopen(url, timeout=60) as r:
            text = r.read().decode("utf-8", "replace")
    d = pd.read_csv(io.StringIO(text), sep="|", dtype=str)
    d = d[d["Test Issue"].ne("Y") & d["ETF"].isin(["Y", "N"])]
    return d[["Symbol", "Security Name", "ETF"]].rename(
        columns={"Symbol": "symbol", "Security Name": "dir_name", "ETF": "etf"})


class UniverseSelector:
    def __init__(self, store: UniverseStore):
        self.store = store
        with store._conn() as c:
            c.executescript(SCHEMA)

    def classify(self, directory: pd.DataFrame) -> dict:
        """Active assets found in the directory: its ETF flag (+ noncommon by
        name). Everything else: name rules."""
        with self.store._conn() as c:
            assets = pd.read_sql("SELECT symbol, name, status FROM universe_assets", c)
        dmap = directory.set_index("symbol")
        rows = []
        for sym, name, status in assets.itertuples(index=False):
            if status == "active" and sym in dmap.index:
                r = dmap.loc[sym]
                # the ETF flag misses closed-end funds and other non-ETF funds
                # (e.g. DSL, DoubleLine Income Solutions Fund), so name rules too
                kind = ("fund" if r["etf"] == "Y" or is_fund_name(r["dir_name"]) else
                        "noncommon" if _NONCOMMON.search(r["dir_name"] or "") else "stock")
                rows.append((sym, kind, "directory", name))
            else:
                rows.append((sym, classify_name(name), "name_rules", name))
        with self.store._conn() as c:
            c.execute("DELETE FROM universe_class")
            c.executemany("INSERT INTO universe_class VALUES (?,?,?,?)", rows)
        out = pd.DataFrame(rows, columns=["symbol", "kind", "source", "name"])
        return out.groupby(["source", "kind"]).size().to_dict()

    # ── 2. Opening 5-minute bars ────────────────────────────────────────────────

    def needed_by_day(self, sessions: list[str]) -> dict[str, list[str]]:
        """For each session D: stocks eligible on D or on any of the next
        LOOKBACK sessions (their RVOL needs D's opening volume)."""
        with self.store._conn() as c:
            e = pd.read_sql("""SELECT e.date, e.symbol FROM universe_eligibility e
                               JOIN universe_class k ON k.symbol = e.symbol AND k.kind = 'stock'""", c)
        by_date = e.groupby("date")["symbol"].apply(set).to_dict()
        out = {}
        for i, d in enumerate(sessions):
            need: set[str] = set()
            for dd in sessions[i:i + LOOKBACK + 1]:
                need |= by_date.get(dd, set())
            out[d] = sorted(need)
        return out

    def download_open5(self, client, start: date, end: date, batch: int = 400,
                       pause: float = 0.2) -> dict:
        import exchange_calendars as xc
        if end >= SEALED:
            end = date(2024, 12, 31)
        cal = xc.get_calendar("XNYS")
        # look ahead LOOKBACK sessions for eligibility, but only download up to `end`
        ahead = cal.sessions_in_range(str(start), str(min(SEALED, date(end.year + 1, 3, 1))))
        sessions = [str(s.date()) for s in ahead]
        need = self.needed_by_day(sessions)
        with self.store._conn() as c:
            done = {r[0] for r in c.execute("SELECT date FROM universe_open5_done")}
            known = {r[0] for r in c.execute("SELECT symbol FROM universe_assets")}
        todo = [d for d in sessions if str(start) <= d <= str(end) and d not in done]
        failed, n_rows = [], 0
        for k, d in enumerate(todo, 1):
            syms = need.get(d, [])
            try:
                got = self._fetch_open5(client, d, syms, known, batch)
            except FatalDownloadError:
                raise
            except Exception as e:
                failed.append({"date": d, "error": f"{type(e).__name__}: {e}"})
                log.error("%s failed: %s", d, e)
                continue
            rows = []
            for s in syms:                          # requested but no trade -> volume 0
                b = got.get(s)
                rows.append((d, s) + (b if b else (None, None, None, None, 0.0)))
            with self.store._conn() as c:
                c.executemany("INSERT OR REPLACE INTO universe_open5 VALUES (?,?,?,?,?,?,?)", rows)
                c.execute("INSERT OR REPLACE INTO universe_open5_done VALUES (?,?,?,?)",
                          (d, len(syms), len(got), _now()))
            n_rows += len(rows)
            if k % 50 == 0 or k == len(todo):
                log.info("open5 %d/%d days (%s, %d symbols)", k, len(todo), d, len(syms))
            _time.sleep(pause)
        return {"days": len(todo), "rows": n_rows, "failed_days": failed}

    @staticmethod
    def _fetch_open5(client, d: str, symbols: list[str], known: set[str], batch: int) -> dict:
        from alpaca.common.exceptions import APIError
        from alpaca.data.enums import Adjustment, DataFeed
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
        t0 = pd.Timestamp(f"{d} 09:30", tz=NY)
        out: dict = {}
        for i in range(0, len(symbols), batch):
            by_data = {ds: s for s in symbols[i:i + batch]
                       if (ds := data_symbol(s, known)) is not None}
            while by_data:
                req = StockBarsRequest(symbol_or_symbols=list(by_data),
                                       timeframe=TimeFrame(5, TimeFrameUnit.Minute),
                                       start=t0.tz_convert("UTC").to_pydatetime(),
                                       end=(t0 + pd.Timedelta(minutes=4, seconds=59)).tz_convert("UTC").to_pydatetime(),
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
                    by_data.pop(m.group(1))
            else:
                continue
            if df is None or len(df) == 0:
                continue
            df = df.reset_index()
            ts = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(NY)
            df = df[ts == t0]                                  # the 09:30 bar only
            for r in df.itertuples(index=False):
                out[by_data[r.symbol]] = (r.open, r.high, r.low, r.close, float(r.volume))
        return out

    # ── 3. Selection ────────────────────────────────────────────────────────────

    def select(self, start: date, end: date, top: int = 20, min_rvol: float = 1.0) -> dict:
        with self.store._conn() as c:
            o = pd.read_sql("SELECT date, symbol, volume FROM universe_open5", c)
            e = pd.read_sql("""SELECT e.date, e.symbol FROM universe_eligibility e
                               JOIN universe_class k ON k.symbol = e.symbol AND k.kind = 'stock'
                               WHERE e.date BETWEEN ? AND ?""", c, params=(str(start), str(end)))
        rv = rvol_frame(o)
        cand = e.merge(rv, on=["date", "symbol"], how="inner")
        cand = cand[cand["avg14"] > 0]
        cand["rvol"] = cand["open5_volume"] / cand["avg14"]
        cand = cand[cand["rvol"] >= min_rvol]
        cand["rank"] = cand.groupby("date")["rvol"].rank(ascending=False, method="first").astype(int)
        sel = cand[cand["rank"] <= top].sort_values(["date", "rank"])
        with self.store._conn() as c:
            c.execute("DELETE FROM universe_selection WHERE date BETWEEN ? AND ?", (str(start), str(end)))
            c.executemany("INSERT INTO universe_selection VALUES (?,?,?,?,?,?)",
                          list(sel[["date", "symbol", "rvol", "rank", "open5_volume", "avg14"]]
                               .itertuples(index=False, name=None)))
        per_day = sel.groupby("date").size()
        return {"days": int(per_day.size), "rows": len(sel),
                "days_with_full_20": int((per_day == top).sum()),
                "per_day_median": float(per_day.median()) if len(per_day) else 0.0,
                "median_rvol_of_selected": float(sel["rvol"].median()) if len(sel) else None}

    def status(self) -> dict:
        with self.store._conn() as c:
            q = lambda s: c.execute(s).fetchone()
            return {"class": dict(c.execute("SELECT kind, COUNT(*) FROM universe_class GROUP BY kind").fetchall()),
                    "open5_days": q("SELECT COUNT(*) FROM universe_open5_done")[0],
                    "open5_rows": q("SELECT COUNT(*) FROM universe_open5")[0],
                    "selection_days": q("SELECT COUNT(DISTINCT date) FROM universe_selection")[0],
                    "selection_rows": q("SELECT COUNT(*) FROM universe_selection")[0]}


def rvol_frame(open5: pd.DataFrame, lookback: int = LOOKBACK) -> pd.DataFrame:
    """Per (date, symbol): opening volume and the mean over the symbol's
    `lookback` PRIOR stored sessions (NaN until a full window exists)."""
    o = open5.sort_values(["symbol", "date"]).copy()
    o["avg14"] = (o.groupby("symbol")["volume"]
                  .transform(lambda v: v.rolling(lookback, min_periods=lookback).mean().shift(1)))
    return o.rename(columns={"volume": "open5_volume"})[["date", "symbol", "open5_volume", "avg14"]]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="DATA/universe.db")
    p.add_argument("--classify", action="store_true")
    p.add_argument("--open5", action="store_true")
    p.add_argument("--select", action="store_true")
    p.add_argument("--status", action="store_true")
    p.add_argument("--start", default="2016-01-04")
    p.add_argument("--end", default="2024-12-31")
    a = p.parse_args()
    sel = UniverseSelector(UniverseStore(a.db))
    if a.classify:
        for k, v in sorted(sel.classify(load_directory()).items()):
            print(f"  {k[0]:<11} {k[1]:<10} {v:>6}")
    if a.open5:
        k, s = os.environ.get("ALPACA_API_KEY"), os.environ.get("ALPACA_SECRET_KEY")
        if not k or not s:
            sys.exit("Set ALPACA_API_KEY and ALPACA_SECRET_KEY (source .env first).")
        from alpaca.data.historical import StockHistoricalDataClient
        r = sel.download_open5(StockHistoricalDataClient(k, s), date.fromisoformat(a.start),
                               date.fromisoformat(a.end))
        print({x: r[x] for x in ("days", "rows")}, f"failed days: {len(r['failed_days'])}")
        for f in r["failed_days"][:10]:
            print("  ", f)
        if r["failed_days"]:
            print("Re-run the same command: finished days are skipped.")
            sys.exit(1)
    if a.select:
        print(sel.select(date.fromisoformat(a.start), date.fromisoformat(a.end)))
    if a.status or not any((a.classify, a.open5, a.select)):
        print(sel.status())
