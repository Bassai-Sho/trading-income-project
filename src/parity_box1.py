"""
parity_box1.py  (Phase A, P2-123 — slice 2 acceptance)
======================================================
Runs box #1 (boxes/orb_vwap_legacy) through the execution core on the same
bars as the frozen legacy golden fixture and compares every trade.

Known, documented quirks reproduced ON PURPOSE for parity (not for new boxes):
  * net R uses the legacy cost formula, charged in R per trade, with the
    legacy share count = account x risk / |fill - stop| (sized on the FILL,
    which a live order cannot know in advance; the box itself sizes its order
    on the signal close — size does not change R, only costs)
  * leverage unlimited (legacy never checked buying power)

    python src/parity_box1.py                 # all four tickers, ~5 min
    python src/parity_box1.py --tickers SPY
"""
from __future__ import annotations

import argparse
import gzip
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pr001_instrument_test as pr                          # noqa: E402
from boxes.orb_vwap_legacy import OrbVwapLegacyBox, Params  # noqa: E402
from core.backtest_runner import run_box                    # noqa: E402
from core.execution_core import AccountConfig               # noqa: E402
from trading_engine import CONFIG                           # noqa: E402

try:
    import trading_quant_toolkit_v2_4 as tk
except Exception:                                           # pragma: no cover
    tk = None

GOLDEN = Path("tests/fixtures/golden_orb_resting_trades.json.gz")
R_TOL = 1e-9
PX_TOL = 1e-4


def legacy_params() -> Params:
    return Params(orb_method=CONFIG["orb_method"], vwap_lookback=CONFIG.get("vwap_lookback", 3),
                  session_start=CONFIG["session_start"], session_end=CONFIG["session_end"],
                  exit_mode="baseline", account=pr.SPEC["account"], risk_pct=pr.SPEC["risk_pct"])


def legacy_cost_r(entry: float, stop: float, adv: float, p: Params) -> float:
    risk_dist = abs(entry - stop)
    units = (p.account * p.risk_pct) / risk_dist if risk_dist > 1e-6 else 0.0
    if tk is None:
        return 0.05
    try:
        return tk.realistic_backtest_cost(entry, units, p.account * p.risk_pct, adv,
                                          CONFIG.get("broker", "alpaca"))
    except Exception:
        return 0.05


def box_rows(journal, df_1m: pd.DataFrame, p: Params) -> list[dict]:
    adv = pr.quarterly_adv(df_1m)
    rows = []
    for t in journal:
        sgn = 1 if t.direction == "long" else -1
        risk = abs(t.entry_price - t.stop_price)
        gross = (t.exit_price - t.entry_price) / risk * sgn
        q = str(pd.Period(t.session_date, freq="Q"))
        cost = legacy_cost_r(t.entry_price, t.stop_price, adv.get(q, 150e6), p)
        rows.append({"session_date": t.session_date, "direction": t.direction,
                     "entry_ts": str(pd.Timestamp(t.signal_ts)),
                     "entry_fill_ts": str(pd.Timestamp(t.entry_fill_ts)),
                     "entry_price": t.entry_price, "stop_price": t.stop_price,
                     "exit_ts": str(pd.Timestamp(t.exit_ts)), "exit_price": t.exit_price,
                     "exit_reason": t.exit_reason, "actual_r": gross - cost})
    return rows


def compare(golden: list[dict], mine: list[dict]) -> dict:
    """Match by (session_date, entry_fill_ts); classify every difference."""
    key = lambda r: (r["session_date"], str(pd.Timestamp(r["entry_fill_ts"])))
    g = {key(r): r for r in golden}
    m = {key(r): r for r in mine}
    only_g = sorted(set(g) - set(m))
    only_m = sorted(set(m) - set(g))
    diffs = []
    for k in sorted(set(g) & set(m)):
        a, b = g[k], m[k]
        bad = []
        if a["direction"] != b["direction"]:
            bad.append("direction")
        for f in ("entry_price", "stop_price", "exit_price"):
            if abs(float(a[f]) - float(b[f])) > PX_TOL:
                bad.append(f)
        if str(pd.Timestamp(a["exit_ts"])) != b["exit_ts"]:
            bad.append("exit_ts")
        if a["exit_reason"] != b["exit_reason"]:
            bad.append("exit_reason")
        if abs(float(a["actual_r"]) - float(b["actual_r"])) > R_TOL:
            bad.append("actual_r")
        if bad:
            diffs.append({"key": k, "fields": bad, "golden": a, "box": b})
    return {"golden_n": len(g), "box_n": len(m), "matched": len(set(g) & set(m)),
            "identical": len(set(g) & set(m)) - len(diffs), "only_golden": only_g,
            "only_box": only_m, "diffs": diffs}


def run(store, tickers: list[str], golden_path: Path = GOLDEN) -> dict:
    gold = json.load(gzip.open(golden_path, "rt"))
    p = legacy_params()
    out = {}
    for tkr in tickers:
        df, _ = pr.load_bars(store, tkr, *gold["window"])
        res = run_box(OrbVwapLegacyBox(), p, df, tkr, AccountConfig(leverage=None))
        out[tkr] = compare(gold["tickers"][tkr]["trades"], box_rows(res.state.journal, df, p))
        logging.info("%s: %d/%d identical", tkr, out[tkr]["identical"], out[tkr]["golden_n"])
    return out


def show(rep: dict, n_examples: int = 5) -> bool:
    ok = True
    print("\n=== Box #1 parity vs frozen legacy golden trades ===")
    print(f"{'Ticker':<6}{'golden':>8}{'box':>8}{'identical':>11}{'differ':>8}{'only gold':>11}{'only box':>10}")
    for t, v in rep.items():
        print(f"{t:<6}{v['golden_n']:>8}{v['box_n']:>8}{v['identical']:>11}{len(v['diffs']):>8}"
              f"{len(v['only_golden']):>11}{len(v['only_box']):>10}")
        ok &= not (v["diffs"] or v["only_golden"] or v["only_box"])
    for t, v in rep.items():
        if v["diffs"]:
            fields = pd.Series([f for d in v["diffs"] for f in d["fields"]]).value_counts().to_dict()
            print(f"\n{t} differing fields: {fields}")
            for d in v["diffs"][:n_examples]:
                print(f"  {d['key']} {d['fields']}\n    golden {d['golden']}\n    box    {d['box']}")
        for label in ("only_golden", "only_box"):
            if v[label]:
                print(f"{t} {label}: {v[label][:n_examples]}")
    print("\nPARITY:", "PASS — every trade identical" if ok else "FAIL — see differences above")
    return ok


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")
    for noisy in ("trading_engine", "historical_sim", "market_data_store", "execution_core"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    from market_data_store import MarketDataStore
    a = argparse.ArgumentParser()
    a.add_argument("--tickers", nargs="+", default=pr.SPEC["tickers"])
    a.add_argument("--db", default="DATA/market_data.db")
    args = a.parse_args()
    sys.exit(0 if show(run(MarketDataStore(args.db), args.tickers)) else 1)
