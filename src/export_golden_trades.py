"""
export_golden_trades.py  (Phase A, P2-123 — step 0)
===================================================
Freezes the LEGACY resting-fill backtest's trades as the regression fixture
that box #1 must reproduce through the new execution core (slice 2).

Runs trading_engine._backtest_orb_full_gate(fill_model="resting") exactly as
resting_fill_is_study.py did (quality_ok sessions, per-quarter ADV, frozen
PR-001 CONFIG, baseline exit, IS 2016-2022) and writes:

  tests/fixtures/golden_orb_resting_trades.json.gz
      every trade (entry/exit ts + price, stop, direction, exit reason,
      shares, net R), plus a fingerprint of the bars used and the commit.

Also reports how much buying power the legacy sizing implied. The legacy
backtest sizes every trade from 1% risk and never checks buying power; the
new core will. Knowing how many trades needed > 1x / 2x / 4x leverage tells
us whether parity can hold once buying power is enforced.

    python src/export_golden_trades.py            # all four tickers (~4 min)
    python src/export_golden_trades.py --tickers SPY
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pr001_instrument_test as pr                 # noqa: E402
import resting_fill_is_study as rs                 # noqa: E402
from market_data_store import MarketDataStore      # noqa: E402

OUT = Path("tests/fixtures/golden_orb_resting_trades.json.gz")
FIELDS = ("session_date", "direction", "entry_ts", "entry_fill_ts", "entry_price",
          "signal_close", "stop_price", "exit_ts", "exit_price", "exit_reason",
          "units", "actual_r", "adv_used")


def fingerprint(df: pd.DataFrame) -> str:
    """Cheap, order-sensitive hash of the bars the run used."""
    h = hashlib.sha256()
    h.update(str(len(df)).encode())
    h.update(pd.util.hash_pandas_object(df[["Open", "High", "Low", "Close", "Volume"]],
                                        index=True).values.tobytes())
    return h.hexdigest()[:16]


def leverage_report(trades: list[dict], account: float) -> dict:
    notional = [abs(t.get("units", 0)) * t["entry_price"] for t in trades]
    lev = [n / account for n in notional]
    return {"trades": len(lev),
            "median_leverage": float(pd.Series(lev).median()) if lev else None,
            "max_leverage": max(lev) if lev else None,
            "pct_over_1x": sum(x > 1 for x in lev) / len(lev) if lev else None,
            "pct_over_2x": sum(x > 2 for x in lev) / len(lev) if lev else None,
            "pct_over_4x": sum(x > 4 for x in lev) / len(lev) if lev else None}


def _commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, timeout=5).stdout.strip() or None
    except Exception:
        return None


def export(store: MarketDataStore, tickers: list[str], out: Path = OUT) -> dict:
    pr.assert_frozen(pr.ENGINE_CONFIG)
    doc = {"created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "commit": _commit(), "window": rs.IS_WINDOW, "fill_model": "resting",
           "entry_fill": "next_open", "exit_mode": pr.SPEC["exit_mode"],
           "account": pr.SPEC["account"], "risk_pct": pr.SPEC["risk_pct"], "tickers": {}}
    for tkr in tickers:
        df, q = pr.load_bars(store, tkr, *rs.IS_WINDOW)
        trades, skipped = rs.run_resting(df, tkr, "next_open")
        rows = [{k: (str(t[k]) if k.endswith("_ts") else t.get(k)) for k in FIELDS}
                for t in trades]
        doc["tickers"][tkr] = {"bars_fingerprint": fingerprint(df), "sessions": q,
                               "skipped_entries": skipped, "n": len(rows),
                               "leverage": leverage_report(trades, pr.SPEC["account"]),
                               "trades": rows}
        logging.info("%s: %d trades", tkr, len(rows))
    out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out, "wt") as f:
        json.dump(doc, f, default=str)
    return doc


def show(doc: dict) -> None:
    f = lambda x, s="{:.1%}": "—" if x is None else s.format(x)
    print(f"\n=== Golden trades frozen (commit {doc['commit']}) -> {OUT} ===")
    print(f"{'Ticker':<6}{'trades':>8}{'median lev':>12}{'max lev':>10}{'>1x':>8}{'>2x':>8}{'>4x':>8}")
    for t, v in doc["tickers"].items():
        L = v["leverage"]
        print(f"{t:<6}{v['n']:>8}{f(L['median_leverage'], '{:.1f}x'):>12}"
              f"{f(L['max_leverage'], '{:.0f}x'):>10}{f(L['pct_over_1x']):>8}"
              f"{f(L['pct_over_2x']):>8}{f(L['pct_over_4x']):>8}")
    print("\nLeverage = shares x entry price / $10,000 account. Commit the fixture:"
          "\n  git add tests/fixtures/golden_orb_resting_trades.json.gz")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")
    for noisy in ("trading_engine", "historical_sim", "market_data_store"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    p = argparse.ArgumentParser()
    p.add_argument("--tickers", nargs="+", default=pr.SPEC["tickers"])
    p.add_argument("--db", default="DATA/market_data.db")
    a = p.parse_args()
    show(export(MarketDataStore(a.db), a.tickers))
