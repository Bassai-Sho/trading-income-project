"""
universe_store.py  (P2-125 — box #2 universe, stage 1)
======================================================
Survivorship-free daily universe for "stocks in play" selection
(Zarattini, Barbon & Aziz 2024). Stage 1 = assets + daily bars + eligibility.
Stage 2 (later) = first-5-minute relative volume -> top 20 per day.
Stage 3 (later) = full 1-min sessions only for the selected stock-days.

Data: Alpaca, free account, SIP feed (historical), active AND inactive US
equities on the main exchanges (the probe of 26 Sep 2026 showed delisted,
acquired and renamed stocks return full history).

SYMBOL MAPPING (probes on the NUC, 26 Sep 2026):
  * Default mapping follows the COMPANY: META returns Facebook's whole history
    (2021-07-01 close 354.39). The old symbol FB also still returns its own
    FB-era bars, so the same company appears twice — the 361 overlapping days
    of FB and META were identical row for row. `deduplicate()` removes such
    rows: for any two symbols sharing >= 20 identical (date, OHLCV) rows, the
    ACTIVE symbol's rows are kept and the matching rows of the other deleted.
  * asof="-" (no mapping) must NOT be used: it returns everything ever traded
    under a TICKER, so a reused ticker splices two companies together (META =
    the Roundhill Metaverse ETF from 2021-06-30, then Facebook from 2022-06-09).
  * Known minor gap: when a ticker is later reused by a different company, the
    earlier owner's pre-rename bars are reachable under neither symbol (the
    ETF's META-era months are missing from METV, which starts 2022-01-31).
    Rare, and such small funds are excluded anyway.

ELIGIBILITY on day D (paper rules; only data known at D's open):
    raw open(D) > $5
    ADV14(D) >= 1,000,000 shares   mean daily volume over the 14 sessions BEFORE D
    ATR14(D) > $0.50               mean true range over the 14 sessions BEFORE D
  Split-safe: raw prices make the split day's true range enormous; split-
  adjusted prices put old sessions on today's scale. So both are downloaded,
  the split points are read from raw/adjusted, and each window is rescaled to
  D's own share basis. ATR is a simple 14-day mean of true range.

    python src/universe_store.py --assets
    python src/universe_store.py --daily --start 2015-11-01 --end 2024-12-31
    python src/universe_store.py --dedupe
    python src/universe_store.py --eligibility --start 2016-01-01 --end 2024-12-31
    python src/universe_store.py --status
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import sqlite3
import sys
import time as _time
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("universe_store")

EXCHANGES = {"NYSE", "NASDAQ", "AMEX", "ARCA", "NYSEARCA", "BATS"}   # no OTC
TICKER_RE = re.compile(r"^[A-Z][A-Z.]{0,7}$")        # real tickers: letters, optional class suffix (BRK.B)
DELISTED_SUFFIX = "_DELISTED"


def data_symbol(asset_symbol: str, known: set[str]) -> str | None:
    """The symbol to request market data under, or None if unreachable.

    Alpaca's asset list contains two kinds of placeholder (run of 26 Sep 2026):
      * 'AET_DELISTED' — a delisted company renamed with a suffix. Its data is
        requested under the original ticker ('AET') — but only if no other
        listed asset uses that ticker now; otherwise the ticker belongs to a
        different company and the old one is unreachable (documented gap).
      * '0029900E0', '641ESC017', 'B002455' — identifier placeholders with no
        ticker and no market data: skipped.
    """
    if asset_symbol.endswith(DELISTED_SUFFIX):
        base = asset_symbol[: -len(DELISTED_SUFFIX)]
        return base if TICKER_RE.match(base) and base not in known else None
    return asset_symbol if TICKER_RE.match(asset_symbol) else None
SEALED = date(2025, 1, 1)

SCHEMA = """
CREATE TABLE IF NOT EXISTS universe_assets (
    symbol TEXT PRIMARY KEY, name TEXT, exchange TEXT, status TEXT,
    tradable INTEGER, asset_class TEXT, fetched_at TEXT);
CREATE TABLE IF NOT EXISTS universe_daily (
    symbol TEXT, date TEXT,
    open REAL, high REAL, low REAL, close REAL, volume REAL,
    adj_open REAL, adj_high REAL, adj_low REAL, adj_close REAL, adj_volume REAL,
    PRIMARY KEY (symbol, date));
CREATE INDEX IF NOT EXISTS idx_ud_date ON universe_daily(date);
CREATE TABLE IF NOT EXISTS universe_progress (
    batch_key TEXT PRIMARY KEY, n_symbols INTEGER, n_rows INTEGER, done_at TEXT);
CREATE TABLE IF NOT EXISTS universe_symbol_done (
    symbol TEXT, start TEXT, "end" TEXT, n_rows INTEGER, note TEXT, done_at TEXT,
    PRIMARY KEY (symbol, start, "end"));
CREATE TABLE IF NOT EXISTS universe_eligibility (
    date TEXT, symbol TEXT, open REAL, adv14 REAL, atr14 REAL,
    PRIMARY KEY (date, symbol));
"""


class FatalDownloadError(RuntimeError):
    pass


class _InvalidSymbol(Exception):
    def __init__(self, symbol: str):
        super().__init__(symbol)
        self.symbol = symbol


def _now() -> str:
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat()


class UniverseStore:
    def __init__(self, db_path: str = "DATA/universe.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        c = sqlite3.connect(self.db_path)
        try:
            yield c
            c.commit()
        finally:
            c.close()

    # ── Assets ───────────────────────────────────────────────────────────────

    def sync_assets(self, trading_client) -> dict:
        from alpaca.trading.enums import AssetClass, AssetStatus
        from alpaca.trading.requests import GetAssetsRequest
        rows = []
        for st in (AssetStatus.ACTIVE, AssetStatus.INACTIVE):
            for a in trading_client.get_all_assets(
                    GetAssetsRequest(status=st, asset_class=AssetClass.US_EQUITY)):
                ex = getattr(a.exchange, "value", a.exchange)
                rows.append((a.symbol, a.name, ex, getattr(a.status, "value", a.status),
                             int(bool(a.tradable)), "us_equity", _now()))
        with self._conn() as c:
            c.executemany("INSERT OR REPLACE INTO universe_assets VALUES (?,?,?,?,?,?,?)", rows)
        kept = sum(r[2] in EXCHANGES for r in rows)
        return {"assets": len(rows), "on_main_exchanges": kept}

    def symbols(self) -> list[str]:
        q = ",".join("?" * len(EXCHANGES))
        with self._conn() as c:
            return [r[0] for r in c.execute(
                f"SELECT symbol FROM universe_assets WHERE exchange IN ({q}) ORDER BY symbol",
                tuple(sorted(EXCHANGES)))]

    # ── Daily bars ───────────────────────────────────────────────────────────

    def download_daily(self, data_client, start: date, end: date, batch: int = 200,
                       pause: float = 0.3) -> dict:
        """Raw + split-adjusted daily bars for every reachable symbol, in batches.
        Resumable PER SYMBOL: a symbol is recorded as done once its batch
        succeeds (even with zero bars) and is skipped next time. One invalid
        symbol no longer sinks its batch: it is dropped, recorded, and the rest
        of the batch retried."""
        if end >= SEALED:
            end = date(2024, 12, 31)
            log.warning("end clamped to 2024-12-31 (sealed window)")
        assets = self.symbols()
        known = set(assets)
        mapped = {a: data_symbol(a, known) for a in assets}
        unreachable = sorted(a for a, d in mapped.items() if d is None)
        with self._conn() as c:
            done = {r[0] for r in c.execute(
                'SELECT symbol FROM universe_symbol_done WHERE start=? AND "end"=?',
                (str(start), str(end)))}
            if c.execute("SELECT COUNT(*) FROM universe_progress").fetchone()[0]:
                # batches finished by the previous (per-batch) version of this code
                done |= {r[0] for r in c.execute("SELECT DISTINCT symbol FROM universe_daily")}
        todo = [(a, d) for a, d in mapped.items() if d is not None and a not in done]
        failed, invalid, total = [], [], 0
        n_batches = -(-len(todo) // batch) if todo else 0
        for bi, i in enumerate(range(0, len(todo), batch), 1):
            chunk = todo[i:i + batch]
            try:
                raw, adj, bad = self._fetch_batch(data_client, chunk, start, end)
            except FatalDownloadError:
                raise
            except Exception as e:                      # transient: record and continue
                failed.append({"batch": f"{chunk[0][0]}..{chunk[-1][0]}",
                               "error": f"{type(e).__name__}: {e}"})
                log.error("batch %d/%d failed: %s", bi, n_batches, e)
                continue
            invalid += bad
            n = self._store_daily(raw, adj)
            total += n
            counts = raw.groupby("symbol").size().to_dict() if len(raw) else {}
            with self._conn() as c:
                c.executemany(
                    'INSERT OR REPLACE INTO universe_symbol_done VALUES (?,?,?,?,?,?)',
                    [(a, str(start), str(end), int(counts.get(a, 0)),
                      "invalid" if a in bad else "", _now()) for a, _ in chunk])
            log.info("batch %d/%d: %d rows%s", bi, n_batches, n,
                     f" ({len(bad)} invalid symbol(s) dropped)" if bad else "")
            _time.sleep(pause)
        return {"symbols": len(assets), "unreachable": len(unreachable),
                "delisted_mapped": sum(1 for a, d in mapped.items()
                                       if d and a.endswith(DELISTED_SUFFIX)),
                "skipped_done": len(assets) - len(unreachable) - len(todo),
                "fetched": len(todo), "rows": total, "invalid": invalid,
                "failed_batches": failed}

    def _fetch_batch(self, client, chunk, start, end):
        """chunk: [(asset_symbol, data_symbol)]. Returns raw, adj (with the ASSET
        symbol), and the asset symbols Alpaca rejected as invalid."""
        by_data = {d: a for a, d in chunk}
        bad: list[str] = []
        while by_data:
            try:
                raw = self._fetch(client, list(by_data), start, end, "raw")
                adj = self._fetch(client, list(by_data), start, end, "split")
                break
            except _InvalidSymbol as e:
                if e.symbol not in by_data:
                    raise
                bad.append(by_data.pop(e.symbol))
        else:
            empty = pd.DataFrame(columns=["symbol", "date", "open", "high", "low", "close", "volume"])
            return empty, empty, bad
        for df in (raw, adj):
            if len(df):
                df["symbol"] = df["symbol"].map(by_data)
        return raw, adj, bad

    @staticmethod
    def _fetch(client, symbols, start, end, adjustment) -> pd.DataFrame:
        from alpaca.data.enums import Adjustment, DataFeed
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        from alpaca.common.exceptions import APIError
        req = StockBarsRequest(symbol_or_symbols=list(symbols), timeframe=TimeFrame.Day,
                               start=datetime.combine(start, datetime.min.time()),
                               end=datetime.combine(end, datetime.max.time()),
                               adjustment=Adjustment(adjustment), feed=DataFeed.SIP)
        # default symbol mapping on purpose — see SYMBOL MAPPING in the module doc
        try:
            df = client.get_stock_bars(req).df
        except APIError as e:
            code = getattr(e, "status_code", None)
            if code in (401, 403):
                raise FatalDownloadError(f"Alpaca rejected the request (HTTP {code}): {e}") from e
            m = re.search(r"invalid symbol:\s*([^\s\"}]+)", str(e))
            if m:
                raise _InvalidSymbol(m.group(1)) from e
            raise
        if df is None or len(df) == 0:
            return pd.DataFrame(columns=["symbol", "date", "open", "high", "low", "close", "volume"])
        df = df.reset_index()
        ts = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("America/New_York")
        df["date"] = ts.dt.date.astype(str)
        return df[["symbol", "date", "open", "high", "low", "close", "volume"]]

    def _store_daily(self, raw: pd.DataFrame, adj: pd.DataFrame) -> int:
        if raw.empty:
            return 0
        m = raw.merge(adj, on=["symbol", "date"], how="left", suffixes=("", "_adj"))
        rows = list(m[["symbol", "date", "open", "high", "low", "close", "volume",
                       "open_adj", "high_adj", "low_adj", "close_adj", "volume_adj"]]
                    .itertuples(index=False, name=None))
        with self._conn() as c:
            c.executemany("INSERT OR REPLACE INTO universe_daily VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows)
        return len(rows)

    def duplicate_report(self, min_days: int = 20) -> pd.DataFrame:
        """Symbol pairs sharing >= min_days identical (date, OHLCV) rows (the
        same company under an old and a new symbol). Empty after deduplicate().
        Groups identical rows (one sort) instead of pairing every row with
        every other row on the same date (the first version, quadratic: hours)."""
        from collections import Counter
        from itertools import combinations
        with self._conn() as c:
            groups = c.execute(
                """SELECT GROUP_CONCAT(symbol, '|') FROM universe_daily
                   WHERE volume > 0
                   GROUP BY date, open, high, low, close, volume HAVING COUNT(*) > 1""").fetchall()
        pairs = Counter(p for (g,) in groups for p in combinations(sorted(g.split("|")), 2))
        rows = [(a, b, n) for (a, b), n in pairs.items() if n >= min_days]
        return (pd.DataFrame(rows, columns=["sym_a", "sym_b", "days"])
                .sort_values("days", ascending=False, ignore_index=True))

    def deduplicate(self, min_days: int = 20) -> dict:
        """Remove the same company's rows appearing under two symbols. For each
        pair with >= min_days identical rows, keep the active symbol (if both or
        neither are active: the one with the later last date, then alphabetical)
        and delete ONLY the other symbol's rows that are identical to it —
        its non-overlapping bars stay."""
        pairs = self.duplicate_report(min_days)
        if pairs.empty:
            return {"pairs": 0, "rows_deleted": 0, "dropped": []}
        with self._conn() as c:
            status = dict(c.execute("SELECT symbol, status FROM universe_assets").fetchall())
            last = dict(c.execute("SELECT symbol, MAX(date) FROM universe_daily GROUP BY symbol").fetchall())
            deleted, dropped = 0, []
            for a, b in pairs[["sym_a", "sym_b"]].itertuples(index=False):
                rank = lambda x: (status.get(x) == "active", last.get(x, ""), -ord(x[0]))
                keep, drop = (a, b) if rank(a) >= rank(b) else (b, a)
                n = c.execute(
                    """DELETE FROM universe_daily WHERE symbol=? AND date IN (
                         SELECT d.date FROM universe_daily d JOIN universe_daily k
                           ON k.symbol=? AND k.date=d.date AND k.open=d.open AND k.high=d.high
                          AND k.low=d.low AND k.close=d.close AND k.volume=d.volume
                         WHERE d.symbol=?)""", (drop, keep, drop)).rowcount
                deleted += n
                dropped.append({"kept": keep, "dropped": drop, "rows": n})
        return {"pairs": len(pairs), "rows_deleted": deleted, "dropped": dropped}

    # ── Eligibility ──────────────────────────────────────────────────────────

    def compute_eligibility(self, start: date, end: date, min_price: float = 5.0,
                            min_adv: float = 1e6, min_atr: float = 0.5,
                            window: int = 14) -> dict:
        with self._conn() as c:
            df = pd.read_sql("SELECT * FROM universe_daily ORDER BY symbol, date", c)
        out = [eligibility_frame(g, window) for _, g in df.groupby("symbol", sort=False)]
        e = pd.concat(out) if out else pd.DataFrame()
        if e.empty:
            return {"rows": 0}
        e = e[(e["date"] >= str(start)) & (e["date"] <= str(end))]
        e = e[(e["open"] > min_price) & (e["adv14"] >= min_adv) & (e["atr14"] > min_atr)]
        with self._conn() as c:
            c.execute("DELETE FROM universe_eligibility WHERE date BETWEEN ? AND ?", (str(start), str(end)))
            c.executemany("INSERT INTO universe_eligibility VALUES (?,?,?,?,?)",
                          list(e[["date", "symbol", "open", "adv14", "atr14"]]
                               .itertuples(index=False, name=None)))
        per_day = e.groupby("date").size()
        return {"rows": len(e), "days": int(per_day.size),
                "per_day_median": float(per_day.median()) if len(per_day) else 0.0,
                "per_day_min": int(per_day.min()) if len(per_day) else 0}

    def status(self) -> dict:
        with self._conn() as c:
            q = lambda s: c.execute(s).fetchone()
            return {"assets": q("SELECT COUNT(*) FROM universe_assets")[0],
                    "symbols_with_bars": q("SELECT COUNT(DISTINCT symbol) FROM universe_daily")[0],
                    "daily_rows": q("SELECT COUNT(*) FROM universe_daily")[0],
                    "date_range": q("SELECT MIN(date), MAX(date) FROM universe_daily"),
                    "batches_done": q("SELECT COUNT(*) FROM universe_progress")[0],
                    "eligible_rows": q("SELECT COUNT(*) FROM universe_eligibility")[0]}


def eligibility_frame(g: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """One symbol's daily rows (sorted by date) -> open, ADV14, ATR14 for each day,
    each measured on that day's own share basis from the `window` PRIOR sessions."""
    g = g.sort_values("date").reset_index(drop=True)
    # share-basis factor from raw/split-adjusted; a jump = a split between sessions
    s = (g["open"] / g["adj_open"]).to_numpy(float)
    ratio = np.r_[1.0, s[1:] / s[:-1]]
    ratio = np.where(np.isfinite(ratio) & (np.abs(ratio - 1.0) >= 0.01), ratio, 1.0)
    cum = np.cumprod(ratio)                       # constant between splits
    # prices / volumes on a single consistent basis (that of the first session)
    ph, pl, pc = (g[k].to_numpy(float) / cum for k in ("high", "low", "close"))
    vol = g["volume"].to_numpy(float) * cum
    prev_c = np.r_[np.nan, pc[:-1]]
    tr = np.nanmax(np.vstack([ph - pl, np.abs(ph - prev_c), np.abs(pl - prev_c)]), axis=0)
    tr[0] = ph[0] - pl[0]
    atr_base = pd.Series(tr).rolling(window, min_periods=window).mean().shift(1).to_numpy()
    adv_base = pd.Series(vol).rolling(window, min_periods=window).mean().shift(1).to_numpy()
    return pd.DataFrame({"date": g["date"], "symbol": g["symbol"], "open": g["open"],
                         "atr14": atr_base * cum,   # back to day D's own basis
                         "adv14": adv_base / cum})


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="DATA/universe.db")
    p.add_argument("--assets", action="store_true")
    p.add_argument("--daily", action="store_true")
    p.add_argument("--eligibility", action="store_true")
    p.add_argument("--duplicates", action="store_true", help="report same-company duplicates")
    p.add_argument("--dedupe", action="store_true", help="remove them (keep the active symbol)")
    p.add_argument("--status", action="store_true")
    p.add_argument("--start", default="2015-11-01")
    p.add_argument("--end", default="2024-12-31")
    a = p.parse_args()
    u = UniverseStore(a.db)
    if a.assets or a.daily:
        k, s = os.environ.get("ALPACA_API_KEY"), os.environ.get("ALPACA_SECRET_KEY")
        if not k or not s:
            sys.exit("Set ALPACA_API_KEY and ALPACA_SECRET_KEY (source .env first).")
    if a.assets:
        from alpaca.trading.client import TradingClient
        print(u.sync_assets(TradingClient(k, s, paper=True)))
    if a.daily:
        from alpaca.data.historical import StockHistoricalDataClient
        r = u.download_daily(StockHistoricalDataClient(k, s), date.fromisoformat(a.start),
                             date.fromisoformat(a.end))
        print({x: r[x] for x in ("symbols", "unreachable", "delisted_mapped", "skipped_done",
                                 "fetched", "rows")},
              f"invalid dropped: {len(r['invalid'])}, failed batches: {len(r['failed_batches'])}")
        if r["invalid"]:
            print("   invalid (first 20):", r["invalid"][:20])
        for f in r["failed_batches"][:10]:
            print("  ", f)
        if r["failed_batches"]:
            print("Re-run the same command: finished symbols are skipped.")
            sys.exit(1)
    if a.duplicates:
        d = u.duplicate_report()
        print("No duplicated series." if d.empty else d.head(30).to_string())
    if a.dedupe:
        r = u.deduplicate()
        print(f"pairs: {r['pairs']}, rows deleted: {r['rows_deleted']}")
        for d in r["dropped"][:20]:
            print(f"   kept {d['kept']:<8} dropped {d['dropped']:<8} ({d['rows']} rows)")
    if a.eligibility:
        print(u.compute_eligibility(date.fromisoformat(a.start), date.fromisoformat(a.end)))
    if a.status or not any((a.assets, a.daily, a.eligibility, a.duplicates, a.dedupe)):
        print(u.status())
