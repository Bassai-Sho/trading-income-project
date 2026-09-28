"""
seal_migrate.py — move post-seal rows out of the RESEARCH store into a LIVE store
================================================================================
Context (28 Sep 2026): DATA/market_data.db held SPY bars to 2026-06-30 -- written
by runner.py's daily update -- in the same table every research script reads.
This script moves every row on/after seal.SEAL_START into a separate LIVE
database (DATA/live_market_data.db) so forward/paper data keeps working and the
research store returns to <= 2024-12-31.

SAFETY DESIGN (each stage refuses to proceed if the previous one is not proven):
  1. DRY RUN is the default: prints what would move, changes nothing.
  2. --execute            copies rows to the live store (INSERT OR IGNORE, so it is
                          idempotent), then VERIFIES per-table row counts AND a
                          SHA-256 over the copied bars. Nothing is deleted.
  3. --delete-from-research  additionally requires --execute and
                          --confirm DELETE-SEALED-ROWS; it takes a full online
                          BACKUP of the research DB first, re-verifies, and only
                          then deletes. If any check fails it stops and deletes nothing.
Direct sqlite access is used on purpose (the store's own guard would refuse to
read sealed rows).

    python src/seal_migrate.py                                  # dry run
    python src/seal_migrate.py --execute                        # copy + verify
    python src/seal_migrate.py --execute --delete-from-research --confirm DELETE-SEALED-ROWS
"""
from __future__ import annotations

import argparse
import hashlib
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import seal                                                          # noqa: E402
from market_data_store import MarketDataStore                        # noqa: E402

CONFIRM_PHRASE = "DELETE-SEALED-ROWS"
SEAL = str(seal.SEAL_START)
# (table, date column). Only tables that hold dated market rows.
TABLES = (("market_bars", "ts_date"), ("session_context", "session_date"),
          ("data_quality_log", "session_date"))


def _ro(path: str) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def _cols(conn: sqlite3.Connection, table: str) -> list[str]:
    return [r[1] for r in conn.execute(f"PRAGMA table_info({table})") if r[1] != "id"]


def plan(research_path: str) -> dict:
    """Row counts on/after the seal date, per table, plus per-ticker bar ranges."""
    with _ro(research_path) as c:
        counts = {t: c.execute(f"SELECT COUNT(*) FROM {t} WHERE {d} >= ?", (SEAL,)).fetchone()[0]
                  for t, d in TABLES}
        per_ticker = c.execute(
            "SELECT ticker, MIN(ts_date), MAX(ts_date), COUNT(*) FROM market_bars "
            "WHERE ts_date >= ? GROUP BY ticker", (SEAL,)).fetchall()
    return {"rows_on_or_after_seal": counts, "per_ticker": per_ticker}


def bars_checksum(path: str) -> str:
    """SHA-256 over every sealed bar (ordered), independent of row ids."""
    h = hashlib.sha256()
    with _ro(path) as c:
        for row in c.execute(
                "SELECT ticker, ts, open, high, low, close, volume, bar_interval FROM market_bars "
                "WHERE ts_date >= ? ORDER BY ticker, bar_interval, ts", (SEAL,)):
            h.update(repr(row).encode())
    return h.hexdigest()


def copy_to_live(research_path: str, live_path: str) -> dict:
    """Copy sealed rows and the store settings needed for update() to continue."""
    MarketDataStore(live_path, sealed=False)          # creates schema, marks role='live'
    copied = {}
    src = _ro(research_path)
    dst = sqlite3.connect(live_path)
    try:
        for table, dcol in TABLES:
            cols = _cols(src, table)
            q = ",".join(cols)
            ph = ",".join("?" * len(cols))
            n = 0
            cur = src.execute(f"SELECT {q} FROM {table} WHERE {dcol} >= ?", (SEAL,))
            while True:
                rows = cur.fetchmany(5000)
                if not rows:
                    break
                dst.executemany(f"INSERT OR IGNORE INTO {table} ({q}) VALUES ({ph})", rows)
                n += len(rows)
            copied[table] = n
        # settings (e.g. 'adjustment:SPY') so update() keeps the same price basis; never the role
        for k, v in src.execute("SELECT key, value FROM store_settings WHERE key != 'role'"):
            dst.execute("INSERT OR REPLACE INTO store_settings(key, value) VALUES (?,?)", (k, v))
        dst.commit()
    finally:
        src.close()
        dst.close()
    return copied


def verify(research_path: str, live_path: str) -> dict:
    """Per-table counts and bar checksum must match exactly for sealed rows."""
    out = {"counts_match": True, "detail": {}}
    with _ro(research_path) as a, _ro(live_path) as b:
        for table, dcol in TABLES:
            na = a.execute(f"SELECT COUNT(*) FROM {table} WHERE {dcol} >= ?", (SEAL,)).fetchone()[0]
            nb = b.execute(f"SELECT COUNT(*) FROM {table} WHERE {dcol} >= ?", (SEAL,)).fetchone()[0]
            out["detail"][table] = (na, nb)
            if na != nb:
                out["counts_match"] = False
    out["checksum_match"] = bars_checksum(research_path) == bars_checksum(live_path)
    out["ok"] = out["counts_match"] and out["checksum_match"]
    return out


def backup(research_path: str) -> str:
    """Full online backup of the research DB (sqlite backup API), integrity-checked."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    dest = str(Path(research_path).with_suffix("")) + f".pre_seal_{stamp}.db"
    src = sqlite3.connect(research_path)
    dst = sqlite3.connect(dest)
    try:
        src.backup(dst)
    finally:
        src.close()
        dst.close()
    with sqlite3.connect(dest) as c:
        ok = c.execute("PRAGMA quick_check").fetchone()[0]
    if ok != "ok":
        raise RuntimeError(f"backup failed its integrity check: {ok}")
    return dest


def delete_from_research(research_path: str) -> dict:
    """Delete rows on/after the seal date and mark the store as the research store."""
    deleted = {}
    c = sqlite3.connect(research_path)
    try:
        for table, dcol in TABLES:
            deleted[table] = c.execute(f"DELETE FROM {table} WHERE {dcol} >= ?", (SEAL,)).rowcount
        c.execute("INSERT OR REPLACE INTO store_settings(key, value) VALUES ('role','research')")
        c.commit()
        left = c.execute("SELECT COUNT(*) FROM market_bars WHERE ts_date >= ?", (SEAL,)).fetchone()[0]
    finally:
        c.close()
    if left:
        raise RuntimeError(f"{left} sealed bars remain after delete")
    return deleted


def run(research: str, live: str, execute: bool, delete: bool, confirm: str | None) -> int:
    if delete and not execute:
        print("REFUSED: --delete-from-research requires --execute."); return 2
    if delete and confirm != CONFIRM_PHRASE:
        print(f"REFUSED: --delete-from-research requires --confirm {CONFIRM_PHRASE}"); return 2
    p = plan(research)
    print("sealed rows in research store:", p["rows_on_or_after_seal"])
    for t in p["per_ticker"]:
        print("  ", t)
    if not execute:
        print("\nDRY RUN: nothing changed. Re-run with --execute to copy and verify.")
        return 0
    print("\ncopied to live store:", copy_to_live(research, live))
    v = verify(research, live)
    print("verify:", v)
    if not v["ok"]:
        print("STOP: verification failed. Nothing deleted."); return 1
    if not delete:
        print("\nCopy verified. Research store untouched. To finish, add "
              f"--delete-from-research --confirm {CONFIRM_PHRASE}")
        return 0
    b = backup(research)
    print("backup written:", b)
    v2 = verify(research, live)                     # re-verify immediately before deleting
    if not v2["ok"]:
        print("STOP: re-verification failed. Nothing deleted."); return 1
    print("deleted from research store:", delete_from_research(research))
    print("DONE. Research store is now <= 2024-12-31 and marked 'research'.")
    return 0


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--research", default="DATA/market_data.db")
    a.add_argument("--live", default="DATA/live_market_data.db")
    a.add_argument("--execute", action="store_true")
    a.add_argument("--delete-from-research", action="store_true")
    a.add_argument("--confirm", default=None)
    args = a.parse_args()
    sys.exit(run(args.research, args.live, args.execute, args.delete_from_research, args.confirm))
