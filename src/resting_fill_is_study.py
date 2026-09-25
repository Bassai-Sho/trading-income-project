"""
resting_fill_is_study.py
========================
P2-122 in-sample study: the SAME strategy as PR-001 (frozen engine CONFIG,
baseline exit), filled as real resting orders on 1-minute bars
(fill_model="resting" in trading_engine._backtest_orb_full_gate).

IN-SAMPLE ONLY (2016-2022). Exploration under the Rulebook split rule v2 —
unlimited use, NOT evidence. Its purpose is to see whether an honest fill
model leaves anything worth pre-registering as PR-002 before the one-shot
2023-2024 window is spent.

Variants (entries/signals identical; only fills differ):
  resting            entry at next 1-min open, resting stop/target   <- candidate
  resting_sigclose   entry at signal close, resting stop/target      (attribution only:
                                                                        shows the entry-fill effect)

DSR uses N = 14 (PR-001's 10 + its 4 tickers), the trial count PR-002 inherits.

    python src/resting_fill_is_study.py                 # both variants (~15 min)
    python src/resting_fill_is_study.py --variants resting
"""
from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pr001_instrument_test as pr                              # noqa: E402  (data + stats helpers)
from trading_engine import _backtest_orb_full_gate, CONFIG      # noqa: E402
from historical_sim import _to_5min                             # noqa: E402
from market_data_store import MarketDataStore                   # noqa: E402

log = logging.getLogger("resting_is")
N_TRIALS = 14
IS_WINDOW = ("2016-01-01", "2022-12-31")
VARIANTS = {"resting": "next_open", "resting_sigclose": "signal_close"}


def run_resting(df_1m: pd.DataFrame, ticker: str, entry_fill: str) -> tuple[list[dict], int]:
    """Quarter-by-quarter (per-quarter ADV, as in PR-001), resting fills."""
    adv = pr.quarterly_adv(df_1m)
    q1 = pd.PeriodIndex(df_1m.index.tz_localize(None), freq="Q").astype(str)
    trades, skipped = [], 0
    for qtr in sorted(set(q1)):
        m1 = df_1m[q1 == qtr]
        cfg = {**CONFIG, "ticker": ticker, "account_balance": pr.SPEC["account"],
               "risk_pct": pr.SPEC["risk_pct"],
               "avg_daily_volume": float(adv.get(qtr, CONFIG.get("avg_daily_volume", 150e6)))}
        res = _backtest_orb_full_gate(_to_5min(m1), cfg, exit_mode=pr.SPEC["exit_mode"],
                                      fill_model="resting", df_1m=m1, entry_fill=entry_fill)
        trades += res.get("trades", [])
        skipped += res.get("skipped_entries", 0)
    return trades, skipped


def summarise(trades: list[dict], skipped: int) -> dict:
    rs = [t["actual_r"] for t in trades]
    st = pr.stats(rs, N_TRIALS)
    daily = pd.Series(rs, index=[t["session_date"] for t in trades]).groupby(level=0).sum().tolist()
    dst = pr.stats(daily, N_TRIALS)
    rg = pr.regime_breakdown(trades)
    exits = pd.Series([t["exit_reason"] for t in trades]).value_counts().to_dict() if trades else {}
    slip = [abs(t["entry_price"] - t["signal_close"]) / abs(t["entry_price"] - t["stop_price"])
            for t in trades if "signal_close" in t and t["entry_price"] != t["stop_price"]]
    return {"stats": st, "daily": {"sharpe": dst["sharpe"], "dsr": dst["dsr"], "days": dst["n"]},
            "regimes": rg, "exits": exits, "skipped_entries": skipped,
            "mean_entry_gap_R": statistics.mean(slip) if slip else None,
            "pr001_stage1_rules": pr.stage1_verdict(st, rg)}


def run(store: MarketDataStore, variants: list[str]) -> dict:
    pr.assert_frozen(CONFIG)
    assert IS_WINDOW[1] <= "2022-12-31", "in-sample only (Rulebook split v2)"
    rep = {"window": IS_WINDOW, "n_trials": N_TRIALS, "tickers": {}}
    for tkr in pr.SPEC["tickers"]:
        df, q = pr.load_bars(store, tkr, *IS_WINDOW)
        rep["tickers"][tkr] = {"sessions": q}
        for v in variants:
            trades, skipped = run_resting(df, tkr, VARIANTS[v])
            rep["tickers"][tkr][v] = summarise(trades, skipped)
            log.info("%s %s: n=%d", tkr, v, len(trades))
    return rep


def show(rep: dict, variants: list[str]) -> None:
    f = lambda x, s="{:+.3f}": "—" if x is None else s.format(x)
    print(f"\n=== Resting-order fills, IS {IS_WINDOW[0]}..{IS_WINDOW[1]} (exploration, DSR N={N_TRIALS}) ===")
    print("PR-001 booked for reference: SPY +0.064  QQQ +0.052  NVDA +0.042  TSLA +0.044 (not achievable)")
    print(f"{'Ticker':<6}{'variant':<18}{'n':>6}{'Exp R':>8}{'Sharpe':>8}{'DSR':>7}{'dayDSR':>8}"
          f"{'+reg':>6}{'skip':>6}")
    for tkr, v in rep["tickers"].items():
        for k in variants:
            m = v[k]; st = m["stats"]
            print(f"{tkr:<6}{k:<18}{st['n']:>6}{f(st['expectancy']):>8}{f(st['sharpe']):>8}"
                  f"{f(st['dsr'], '{:.3f}'):>7}{f(m['daily']['dsr'], '{:.3f}'):>8}"
                  f"{m['pr001_stage1_rules']['positive_regimes']:>4}/4{m['skipped_entries']:>6}")
        m = v[variants[0]]
        print(f"       exits {m['exits']}  entry gap vs signal close: "
              f"{f(m['mean_entry_gap_R'], '{:.3f}')}R")
        print("       regimes: " + "  ".join(f"{g['label'].split(' ', 1)[0]} {f(g['expectancy'])} ({g['n']})"
                                          for g in m["regimes"].values()) + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")
    for noisy in ("trading_engine", "historical_sim", "market_data_store"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    p = argparse.ArgumentParser()
    p.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    p.add_argument("--db", default="DATA/market_data.db")
    a = p.parse_args()
    rep = run(MarketDataStore(a.db), a.variants)
    show(rep, a.variants)
    Path("DATA/pr001").mkdir(parents=True, exist_ok=True)
    Path("DATA/pr001/resting_is_study.json").write_text(json.dumps(rep, indent=2, default=str))
    print("Saved: DATA/pr001/resting_is_study.json")
