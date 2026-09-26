"""
null_gate_box1.py  (P2-123 slice 3 — first real use of the null-alpha gate)
===========================================================================
Runs box #1 (the shelved PR-001 ORB) through the execution core on IS
2016-2022, then grades its trades with evaluation/null_baseline (direction +
timing diagnostics; gate = full random entry, 1,000 draws, alpha 0.05). Glue only: the gate
itself never sees the box.

Pre-registered expectation (review round 4, 25 Sep 2026): box #1 is
indistinguishable from chance — direction p > 0.40. Record the result either way.

    python src/null_gate_box1.py                  # ~2-3 min per ticker
    python src/null_gate_box1.py --tickers SPY
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pr001_instrument_test as pr                        # noqa: E402
from parity_box1 import box_rows, legacy_cost_r, legacy_params   # noqa: E402
from boxes.orb_vwap_legacy import OrbVwapLegacyBox         # noqa: E402
from core.backtest_runner import run_box                  # noqa: E402
from core.execution_core import AccountConfig             # noqa: E402
from evaluation.null_baseline import TradeIn, run_null_gate   # noqa: E402

IS = ("2016-01-01", "2022-12-31")


def run(store, tickers, draws=1000):
    p = legacy_params()
    out = {}
    for tkr in tickers:
        df, _ = pr.load_bars(store, tkr, *IS)
        res = run_box(OrbVwapLegacyBox(), p, df, tkr, AccountConfig(leverage=None))
        rows = box_rows(res.state.journal, df, p)
        trades = [TradeIn(t.session_date, t.direction, t.signal_ts, t.entry_price, t.stop_price,
                          r["actual_r"]) for t, r in zip(res.state.journal, rows)]
        adv = pr.quarterly_adv(df)
        cost = lambda e, s, d: legacy_cost_r(e, s, adv.get(str(pd.Period(d, freq="Q")), 150e6), p)
        rep = run_null_gate(trades, df, tkr, p.session_end, cost_fn=cost, draws=draws)
        out[tkr] = asdict(rep)
        logging.info("%s: gate p=%.3f direction p=%.3f timing p=%.3f", tkr, rep.gate_p,
                     rep.direction_p, rep.timing_p)
    return out


def show(rep):
    f = lambda x: "—" if x is None else f"{x:+.3f}"
    print("\n=== Null-alpha gate: box #1 (legacy ORB), IS 2016-2022, 1,000 draws, alpha 0.05 ===")
    print(f"{'Ticker':<6}{'n':>6}{'box exp':>9}{'common':>9}{'gate null':>10}{'gate p':>8}"
          f"{'dir p':>7}{'time p':>8}  gate   | leak check: side-kept null before / after signal")
    for t, v in rep.items():
        print(f"{t:<6}{v['n_trades']:>6}{f(v['box_expectancy']):>9}{f(v['common_exit_expectancy']):>9}"
              f"{f(v['gate_null_mean']):>10}{v['gate_p']:>8.3f}{v['direction_p']:>7.3f}"
              f"{v['timing_p']:>8.3f}  {'PASS' if v['passed'] else 'fail'}   | "
              f"{f(v['side_kept_timing_before'])} / {f(v['side_kept_timing_after'])}")
        for n in v["notes"]:
            print(f"       note: {n}")
    print("\n'common' = the box's trades under the harness's common exit (1R bracket, time exit);"
          "\nthe nulls use the same exit. Gate = random time + shuffled side. 'time p' is"
          "\ndirection-neutral. The leak check should show 'before' well above the gate null.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")
    for noisy in ("trading_engine", "historical_sim", "market_data_store", "execution_core"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    from market_data_store import MarketDataStore
    a = argparse.ArgumentParser()
    a.add_argument("--tickers", nargs="+", default=pr.SPEC["tickers"])
    a.add_argument("--db", default="DATA/market_data.db")
    a.add_argument("--draws", type=int, default=1000)
    args = a.parse_args()
    r = run(MarketDataStore(args.db), args.tickers, args.draws)
    show(r)
    Path("DATA/pr001").mkdir(parents=True, exist_ok=True)
    Path("DATA/pr001/null_gate_box1.json").write_text(json.dumps(r, indent=2, default=str))
    print("Saved: DATA/pr001/null_gate_box1.json")
