"""
pr001_fill_diagnostic.py
========================
Diagnostic for PR-001 Stage 1 (IN-SAMPLE ONLY — never reads 2023+).

The canonical backtest decides a stop/target on a 5-min bar's CLOSE but books
the exit AT the stop/target price. No real order gets that fill: a resting stop
fills at the stop but also triggers on intrabar wicks; a close-based decision
fills near the close. This re-prices every Stage-1 exit under two consistent
models, from the same trades (entries and exit decisions unchanged):

  booked      : as the backtest records it (stop/target price)          [PR-001]
  close_fill  : every exit filled at the triggering bar's close
  next_open   : every exit filled at the next 5-min bar's open (decision on the
                close, order sent after it — what the live engine can actually do)

Costs are kept exactly as charged per trade. Also reports how many trades per
session there are and a per-DAY Sharpe/DSR, because several trades on one day
are not independent observations.

This changes nothing in PR-001: it measures how much of the Stage-1 edge
depends on the booked fill price, before the one-shot OOS window is spent.

    python src/pr001_fill_diagnostic.py
"""
from __future__ import annotations

import json
import logging
import statistics
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pr001_instrument_test as pr                      # noqa: E402
from historical_sim import _to_5min                     # noqa: E402
from market_data_store import MarketDataStore           # noqa: E402

log = logging.getLogger("pr001_fill")


def reprice(trades: list[dict], df5: pd.DataFrame) -> dict[str, list[float]]:
    """Return net R per trade under each fill model."""
    out = {"booked": [], "close_fill": [], "next_open": []}
    idx = df5.index
    for t in trades:
        sign = 1 if t["direction"] == "long" else -1
        entry, stop0 = t["entry_price"], t["stop_price"]           # stop_price = initial stop
        risk = abs(entry - stop0)
        if risk <= 1e-9:
            continue
        booked_px = t["exit_price"]
        cost = (booked_px - entry) / risk * sign - t["actual_r"]   # cost charged by the backtest
        pos = idx.get_indexer([t["exit_ts"]])[0]
        close_px = float(df5["Close"].iloc[pos])
        nxt = pos + 1
        same_day = nxt < len(idx) and idx[nxt].date() == idx[pos].date()
        open_px = float(df5["Open"].iloc[nxt]) if same_day else close_px
        for k, px in (("booked", booked_px), ("close_fill", close_px), ("next_open", open_px)):
            out[k].append((px - entry) / risk * sign - cost)
    return out


def per_day(trades: list[dict], rs: list[float]) -> list[float]:
    s = pd.Series(rs, index=[t["session_date"] for t in trades[:len(rs)]])
    return s.groupby(level=0).sum().tolist()


def run(store: MarketDataStore) -> dict:
    pr.assert_frozen(pr.ENGINE_CONFIG)
    start, end = pr.SPEC["is_window"]
    assert end <= "2022-12-31", "diagnostic is in-sample only"
    report = {}
    for tkr in pr.SPEC["tickers"]:
        df, _ = pr.load_bars(store, tkr, start, end)
        trades = pr.run_trades(df, tkr)
        df5 = _to_5min(df)
        models = reprice(trades, df5)
        exits = pd.Series([t["exit_reason"] for t in trades]).value_counts().to_dict()
        rows = {}
        for k, rs in models.items():
            st = pr.stats(rs, pr.SPEC["dsr_n_trials"])
            daily = per_day(trades, rs)
            dst = pr.stats(daily, pr.SPEC["dsr_n_trials"])
            rows[k] = {"exp": st["expectancy"], "sharpe": st["sharpe"], "dsr": st["dsr"],
                       "daily_sharpe": dst["sharpe"], "daily_dsr": dst["dsr"], "days": dst["n"]}
        report[tkr] = {"n": len(trades), "exits": exits,
                       "trades_per_day": len(trades) / max(rows["booked"]["days"], 1),
                       "models": rows}
        log.info("%s done", tkr)
    return report


def show(r: dict) -> None:
    f = lambda x, s="{:+.3f}": "—" if x is None else s.format(x)
    print("\n=== PR-001 fill-model diagnostic (IS 2016-2022, same trades, exits re-priced) ===")
    print(f"{'Ticker':<6}{'model':<12}{'Exp R':>8}{'Sharpe':>8}{'DSR':>7}{'dayShp':>8}{'dayDSR':>8}")
    for tkr, v in r.items():
        for k, m in v["models"].items():
            print(f"{tkr:<6}{k:<12}{f(m['exp']):>8}{f(m['sharpe']):>8}{f(m['dsr'],'{:.3f}'):>7}"
                  f"{f(m['daily_sharpe']):>8}{f(m['daily_dsr'],'{:.3f}'):>8}")
        print(f"       exits: {v['exits']}   trades/day-with-trades: {v['trades_per_day']:.2f}\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")
    for noisy in ("trading_engine", "historical_sim", "market_data_store"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    rep = run(MarketDataStore("DATA/market_data.db"))
    show(rep)
    Path("DATA/pr001").mkdir(parents=True, exist_ok=True)
    Path("DATA/pr001/fill_diagnostic.json").write_text(json.dumps(rep, indent=2, default=str))
    print("Saved: DATA/pr001/fill_diagnostic.json")
