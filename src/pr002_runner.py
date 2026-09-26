"""
pr002_runner.py — PR-002 Stage 1 (in-sample)   (Notion P2-127, frozen 26 Sep 2026)
=================================================================================
Zarattini ORB on the top-20 stocks in play (box #2), IS 2016-01-25..2022-12-31.
Glue only: the strategy is boxes/orb_stocks_in_play, fills are the execution
core's, grading is evaluation/*. Nothing here may be tuned after results are
seen — SPEC is the pre-registration.

    python src/pr002_runner.py                  # full Stage 1 (null gate ~20-40 min)
    python src/pr002_runner.py --skip-null      # everything except G1 (quick look)
    python src/pr002_runner.py --optimistic-ties
        DIAGNOSTIC (pre-registered in P2-127): the stop loss goes live the minute
        AFTER the entry instead of in the entry minute — the favourable reading
        of the intrabar-ordering ambiguity. Not a gate; never replaces the frozen
        (conservative) result. Output: stage1_optimistic_<utc>.json.

Output: DATA/pr002/stage1_<utc>.json and a printed report.
"""
from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from boxes.orb_stocks_in_play import OrbStocksInPlayBox, Params            # noqa: E402
from core.backtest_runner import run_box                                   # noqa: E402
from core.execution_core import AccountConfig                              # noqa: E402
from evaluation.costs import COMMISSION, abdi_ranaldo, breakeven_bps, cost_r  # noqa: E402
from evaluation.null_baseline import TradeIn, run_null_gate                # noqa: E402
from market_data_store import REGIMES, MarketDataStore                     # noqa: E402

log = logging.getLogger("pr002")

SPEC = {
    "id": "PR-002", "notion": "P2-127", "frozen": "2026-09-26",
    "is_window": ("2016-01-25", "2022-12-31"),
    "top_n": 20, "stop_atr_frac": 0.10,
    "commission_per_share": COMMISSION,
    "h2_bps_per_side": 5.0,
    "cost_stress": 1.2,
    "data_error_tolerance": 0.01,
    "dsr_n_trials": 15, "dsr_min": 0.95,
    "min_trades": 100, "min_positive_regimes": 3, "min_positive_years": 5,
    "null": {"draws": 1000, "alpha": 0.05, "n_candidates": 12, "window_start": "09:30",
             "target_r": None, "exit": "close"},
    "spread_check": {"window": ("09:30", "10:30"), "spy_max_half_bps": 2.0},
}
IS_REGIMES = ["normal_bull", "covid_crash", "covid_recovery", "rate_hike_bear"]


@dataclass
class StockDay:
    symbol: str
    date: str
    rank: int
    atr: float
    day_high: float
    day_low: float


# ── Inputs ─────────────────────────────────────────────────────────────────────

def load_stock_days(universe_db: str, start: str, end: str, top_n: int) -> list[StockDay]:
    with sqlite3.connect(universe_db) as c:
        rows = c.execute(
            """SELECT s.symbol, s.date, s.rank, e.atr14, d.high, d.low
               FROM universe_selection s
               JOIN universe_eligibility e ON e.date = s.date AND e.symbol = s.symbol
               JOIN universe_daily d ON d.date = s.date AND d.symbol = s.symbol
               WHERE s.date BETWEEN ? AND ? AND s.rank <= ? ORDER BY s.date, s.rank""",
            (start, end, top_n)).fetchall()
    return [StockDay(*r) for r in rows]


def data_error(df: pd.DataFrame, day_high: float, day_low: float, tol: float) -> str | None:
    """Only genuine data errors (PR-002): invalid prices, or a minute outside the
    day's official high/low. Thin-trading flags are NOT exclusions."""
    if df.empty:
        return "no_bars"
    if (df[["Open", "High", "Low", "Close"]] <= 0).any().any() or (df["High"] < df["Low"]).any():
        return "invalid_price"
    if df["High"].max() > day_high + tol or df["Low"].min() < day_low - tol:
        return "outside_daily_range"
    return None


# ── Trades ─────────────────────────────────────────────────────────────────────

def run_trades(stock_days: list[StockDay], minutes: MarketDataStore, stop_atr_frac: float,
               tol: float, stop_from_next_minute: bool = False) -> tuple[list, dict, dict]:
    """Returns (trades, excluded counts, {(symbol, date): bars} for the null gate)."""
    box = OrbStocksInPlayBox()
    trades, excluded, bars = [], {}, {}
    for k, sd in enumerate(stock_days, 1):
        df = minutes.get_session_bars(sd.symbol, sd.date)
        why = data_error(df, sd.day_high, sd.day_low, tol)
        if why:
            excluded[why] = excluded.get(why, 0) + 1
            continue
        df = df[["Open", "High", "Low", "Close", "Volume"]]
        res = run_box(box, Params(atr14={sd.date: sd.atr}, stop_atr_frac=stop_atr_frac,
                                  stop_from_next_minute=stop_from_next_minute),
                      df, sd.symbol, AccountConfig(leverage=None))
        trades += list(res.state.journal)
        if res.state.journal:
            bars[(sd.symbol, sd.date)] = df
        if k % 5000 == 0:
            log.info("stock-days %d/%d, trades %d", k, len(stock_days), len(trades))
    return trades, excluded, bars


def trade_frame(trades) -> pd.DataFrame:
    t = pd.DataFrame([asdict(x) for x in trades])
    if t.empty:
        return t
    sgn = np.where(t["direction"] == "long", 1.0, -1.0)
    t["stop_dist"] = (t["entry_price"] - t["stop_price"]).abs()
    t["gross_r"] = (t["exit_price"] - t["entry_price"]) / t["stop_dist"] * sgn
    t["entry_minute_stop"] = (t["exit_reason"] == "stop") & (t["exit_ts"] == t["entry_fill_ts"])
    t["gapped_entry"] = (t["entry_price"] - t["entry_order_price"]).abs() > 1e-9
    return t


def net(t: pd.DataFrame, bps: float, mult: float = 1.0) -> pd.Series:
    return t["gross_r"] - mult * cost_r(t["entry_price"], t["exit_price"], t["stop_dist"],
                                        SPEC["commission_per_share"], bps)


# ── Statistics and gates ───────────────────────────────────────────────────────

def summarise(t: pd.DataFrame, r: pd.Series) -> dict:
    import purgedcv
    daily = r.groupby(t["session_date"]).sum()
    n_days = len(daily)
    sr = float(daily.mean() / daily.std(ddof=1)) if n_days > 2 and daily.std(ddof=1) > 0 else 0.0
    dsr = (float(purgedcv.deflated_sharpe_ratio(daily.to_numpy(), SPEC["dsr_n_trials"], 1 / (n_days - 1)))
           if n_days > 2 else None)
    regimes = {}
    for k in IS_REGIMES:
        g = REGIMES[k]
        m = (t["session_date"] >= str(g["start"])) & (t["session_date"] <= str(g["end"]))
        regimes[k] = {"n": int(m.sum()), "expectancy": float(r[m].mean()) if m.any() else None}
    years = r.groupby(t["session_date"].str[:4]).mean().to_dict()
    return {"n": len(r), "expectancy": float(r.mean()), "win_rate": float((r > 0).mean()),
            "daily_sharpe": sr, "days": n_days, "dsr": dsr, "regimes": regimes,
            "years": {k: float(v) for k, v in years.items()}}


def spread_check(stock_days, minutes, market_db: str | None) -> dict:
    """Abdi-Ranaldo on 09:30-10:30 minute bars. Validated on SPY first."""
    a, b = (pd.Timestamp(x).time() for x in SPEC["spread_check"]["window"])
    def est(df):
        if df.empty:
            return np.nan
        w = df[(df.index.time >= a) & (df.index.time < b)]
        return abdi_ranaldo(w["High"], w["Low"], w["Close"]) if len(w) > 10 else np.nan
    spy = None
    if market_db and Path(market_db).exists():
        m = MarketDataStore(market_db)
        days = m.get_date_range("SPY", date.fromisoformat(SPEC["is_window"][0]),
                                date.fromisoformat(SPEC["is_window"][1]), quality_ok_only=True)
        vals = [est(m.get_session_bars("SPY", d)) for d in days[::5]]
        spy = float(np.nanmedian(vals) / 2 * 1e4) if vals else None
    sample = stock_days[::10]
    vals = [est(minutes.get_session_bars(s.symbol, s.date)) for s in sample]
    sel = float(np.nanmedian(vals) / 2 * 1e4)
    validated = spy is not None and spy <= SPEC["spread_check"]["spy_max_half_bps"]
    return {"spy_median_half_bps": spy, "validated_on_spy": validated,
            "selected_median_half_bps": sel, "sample_stock_days": len(sample)}


def run_stage1(universe_db, minutes_db, market_db, skip_null=False, out_dir=Path("DATA/pr002"),
               optimistic_ties=False):
    start, end = SPEC["is_window"]
    minutes = MarketDataStore(minutes_db)
    sds = load_stock_days(universe_db, start, end, SPEC["top_n"])
    log.info("stock-days: %d", len(sds))
    if optimistic_ties:
        skip_null = True                                   # diagnostic only: no gates from it
    trades, excluded, bars = run_trades(sds, minutes, SPEC["stop_atr_frac"],
                                        SPEC["data_error_tolerance"], optimistic_ties)
    t = trade_frame(trades)
    if t.empty:
        raise SystemExit("no trades")
    spread = spread_check(sds, minutes, market_db)
    bps = SPEC["h2_bps_per_side"]
    if spread["validated_on_spy"] and spread["selected_median_half_bps"] > bps:
        bps = spread["selected_median_half_bps"]          # pre-committed rule (P2-127)
    h1 = summarise(t, net(t, 0.0))
    h2 = summarise(t, net(t, bps))
    stress = float(net(t, bps, SPEC["cost_stress"]).mean())
    be = breakeven_bps(t["gross_r"], t["entry_price"], t["exit_price"], t["stop_dist"],
                       SPEC["commission_per_share"])
    diag = {"stock_days": len(sds), "excluded": excluded, "trades": len(t),
            "trade_rate": len(t) / max(1, len(sds) - sum(excluded.values())),
            "long_share": float((t["direction"] == "long").mean()),
            "entry_minute_stop_share": float(t["entry_minute_stop"].mean()),
            "h2_excl_entry_minute_stops": float(net(t, bps)[~t["entry_minute_stop"]].mean()),
            "gapped_entry_share": float(t["gapped_entry"].mean()),
            "h2_long": float(net(t, bps)[t["direction"] == "long"].mean()),
            "h2_short": float(net(t, bps)[t["direction"] == "short"].mean()),
            "gross_expectancy": float(t["gross_r"].mean()),
            "median_stop_bps_of_price": float((t["stop_dist"] / t["entry_price"]).median() * 1e4)}
    g2 = {"min_trades": h2["n"] >= SPEC["min_trades"], "expectancy": h2["expectancy"] > 0,
          "dsr": (h2["dsr"] or 0) >= SPEC["dsr_min"],
          "regimes": sum(1 for v in h2["regimes"].values()
                         if v["expectancy"] is not None and v["expectancy"] > 0) >= SPEC["min_positive_regimes"],
          "years": sum(1 for v in h2["years"].values() if v > 0) >= SPEC["min_positive_years"]}
    g3 = stress > 0
    null = None
    if not skip_null:
        nk = SPEC["null"]
        tin = [TradeIn(r.session_date, r.direction,
                       pd.Timestamp(r.entry_fill_ts) - pd.Timedelta(minutes=5), r.entry_price,
                       r.stop_price, float(n), r.symbol)
               for r, n in zip(t.itertuples(), net(t, bps))]
        cfn = lambda e, s, d: float(cost_r(e, e, abs(e - s), SPEC["commission_per_share"], bps))
        rep = run_null_gate(tin, bars, "", time(16, 0), cost_fn=cfn,
                            window_start=pd.Timestamp(nk["window_start"]).time(), draws=nk["draws"],
                            alpha=nk["alpha"], target_r=nk["target_r"], exit=nk["exit"],
                            n_candidates=nk["n_candidates"])
        null = asdict(rep)
    gates = {"G1_null": None if null is None else null["passed"],
             "G2_in_sample": all(g2.values()), "G3_cost_stress": g3}
    order = ["G1_null", "G2_in_sample", "G3_cost_stress"]
    first_fail = next((g for g in order if gates[g] is False), None)
    verdict = ("FAIL at " + first_fail) if first_fail else (
        "PASS G1-G3 -> run G4 robustness audit" if gates["G1_null"] else "G2/G3 pass; G1 not run")
    if optimistic_ties:
        verdict = "DIAGNOSTIC ONLY (optimistic intrabar ordering) — not a gate; would-be: " + verdict
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                                text=True, timeout=5).stdout.strip()
    except Exception:
        commit = None
    report = {"spec": SPEC, "mode": "optimistic_ties" if optimistic_ties else "frozen",
              "run_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
              "commit": commit, "h2_bps_used": bps, "spread": spread, "H1": h1, "H2": h2,
              "H2_cost_stress_expectancy": stress, "breakeven_bps_per_side": be,
              "G2_detail": g2, "null": null, "gates": gates, "verdict": verdict, "diagnostics": diag}
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = "stage1_optimistic" if optimistic_ties else "stage1"
    path = out_dir / f"{stem}_{report['run_at'].replace(':', '')}.json"
    path.write_text(json.dumps(report, indent=2, default=str))
    report["path"] = str(path)
    return report


def show(r: dict) -> None:
    f = lambda x, s="{:+.3f}": "—" if x is None else s.format(x)
    d = r["diagnostics"]
    mode = "  [DIAGNOSTIC: optimistic intrabar ordering]" if r.get("mode") == "optimistic_ties" else ""
    print(f"\n=== PR-002 Stage 1 (IS {SPEC['is_window'][0]}..{SPEC['is_window'][1]}) commit={r['commit']}{mode} ===")
    print(f"stock-days {d['stock_days']}, excluded {d['excluded']}, trades {d['trades']} "
          f"(trigger rate {d['trade_rate']:.0%}, long {d['long_share']:.0%})")
    print(f"median stop = {d['median_stop_bps_of_price']:.0f} bps of price; gross expectancy {f(d['gross_expectancy'])}R")
    s = r["spread"]
    print(f"spread check: SPY half-spread est {f(s['spy_median_half_bps'], '{:.2f}')} bps "
          f"(validated={s['validated_on_spy']}); selected stocks median half-spread "
          f"{s['selected_median_half_bps']:.1f} bps -> H2 uses {r['h2_bps_used']:.1f} bps/side")
    print(f"\n{'':<34}{'H1 comm only':>14}{'H2 viability':>14}")
    for k, lab, fmt in (("expectancy", "expectancy (R/trade)", "{:+.3f}"), ("win_rate", "win rate", "{:.1%}"),
                        ("daily_sharpe", "daily Sharpe", "{:+.3f}"),
                        ("dsr", f"DSR (N={SPEC['dsr_n_trials']})", "{:.3f}")):
        print(f"{lab:<34}{f(r['H1'][k], fmt):>14}{f(r['H2'][k], fmt):>14}")
    print(f"{'H2 at costs x1.2':<34}{'':>14}{f(r['H2_cost_stress_expectancy']):>14}")
    print(f"\nBREAK-EVEN COST: {r['breakeven_bps_per_side']:.1f} bps per side")
    print("H2 regimes: " + "  ".join(f"{k} {f(v['expectancy'])} ({v['n']})" for k, v in r["H2"]["regimes"].items()))
    print("H2 years:   " + "  ".join(f"{k} {f(v)}" for k, v in r["H2"]["years"].items()))
    print(f"diagnostics: entry-minute stops {d['entry_minute_stop_share']:.0%} "
          f"(H2 without them {f(d['h2_excl_entry_minute_stops'])}), gapped entries {d['gapped_entry_share']:.0%}, "
          f"H2 long {f(d['h2_long'])} / short {f(d['h2_short'])}")
    if r["null"]:
        n = r["null"]
        print(f"G1 null: gate p={n['gate_p']:.3f} (null mean {f(n['gate_null_mean'])}), "
              f"direction p={n['direction_p']:.3f}, timing p={n['timing_p']:.3f}")
    print(f"G2 detail: {r['G2_detail']}")
    print(f"GATES: {r['gates']}\nVERDICT: {r['verdict']}\nSaved: {r['path']}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")
    for noisy in ("market_data_store", "execution_core"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    p = argparse.ArgumentParser()
    p.add_argument("--universe-db", default="DATA/universe.db")
    p.add_argument("--minutes-db", default="DATA/universe_minutes.db")
    p.add_argument("--market-db", default="DATA/market_data.db")
    p.add_argument("--skip-null", action="store_true")
    p.add_argument("--optimistic-ties", action="store_true",
                   help="diagnostic: stop live from the minute after entry (not a gate)")
    a = p.parse_args()
    show(run_stage1(a.universe_db, a.minutes_db, a.market_db, skip_null=a.skip_null,
                    optimistic_ties=a.optimistic_ties))
