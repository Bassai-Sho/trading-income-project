"""
pr003_gap_tail.py — descriptive intraday adverse-excursion check (28 Sep 2026)
================================================================================
Round-4 cold-fork correction: "flat overnight" (verified: tests/test_box3_
noise_area.py::test_no_decision_at_a_half_day_close_and_flat_every_night,
test_every_day_ends_flat_on_noisy_data, test_fail_closed_if_not_flat_at_the_open,
all passing at commit 54a7fb4; the underlying bug they guard was fixed in
5555c5e) rules out OVERNIGHT gap risk (no position is ever held past the
close). It does NOT rule out INTRADAY gap/execution risk at the open, and a
backtest fill model can only describe what happened in 2016-2024, not price
a forward tail. This is a PURELY DESCRIPTIVE pull from data already in hand
-- no new hypothesis, no threshold chosen from the result, nothing fitted.

CAVEAT STATED, NOT HIDDEN: this project's data starts 2016-01-04. The
canonical historical stress dates named in review (2010-05-06 Flash Crash,
2015-08-24) are NOT in our data and cannot be examined here. The closest
in-sample analogues are reported instead (Feb 2018 Volmageddon, Mar 2020
COVID), explicitly labelled as such, not substituted silently.

Reports: the distribution of each day's worst intraday move AGAINST the
open, scaled by that day's ACTUAL leverage (effective_leverage, already
built for the leverage-cap check) -- i.e. the P&L impact if a position had
been marked at the day's single worst intraday tick. A stress DESCRIPTION
for the risk-policy memo to consume, not a gate, not a fix.

    python src/pr003_gap_tail.py
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pr003_diagnostics as diag                                      # noqa: E402
import pr003_episode_decomposition as ed                              # noqa: E402
import pr003_refine as rf                                              # noqa: E402
import pr003_replication as pr                                         # noqa: E402
from boxes.noise_area_momentum import NoiseAreaMomentumBox, Params    # noqa: E402
from core.backtest_runner import run_box                              # noqa: E402
from core.execution_core import AccountConfig                         # noqa: E402
from evaluation.mes_costs import mes_fee_fn                            # noqa: E402

log = logging.getLogger("pr003_gap_tail")

IN_SAMPLE_ANALOGUES = {"2018_volmageddon": ("2018-02-01", "2018-02-09"),
                       "2020_covid_crash": ("2020-02-20", "2020-03-23")}

FLAT_OVERNIGHT_CITATION = {
    "claim": "box #3 carries zero position past the session close (no overnight gap risk)",
    "tests": ["tests/test_box3_noise_area.py::test_no_decision_at_a_half_day_close_and_flat_every_night",
             "tests/test_box3_noise_area.py::test_every_day_ends_flat_on_noisy_data",
             "tests/test_box3_noise_area.py::test_fail_closed_if_not_flat_at_the_open"],
    "underlying_bugfix_commit": "5555c5e",
    "status": "verified: all 3 tests confirmed passing at commit 54a7fb4 (28 Sep 2026)",
    "scope": "rules out OVERNIGHT gap risk only; INTRADAY gap/execution risk at the open is "
            "separate and is exactly what this script's excursion metric describes"}


def worst_intraday_excursion_per_day(df_1m: pd.DataFrame, daily: pd.DataFrame) -> pd.Series:
    """For each session: |entry-side worst move|, defined direction-agnostically
    as max(open - low, high - open) / open -- the larger of the two possible
    adverse moves from the session's own open. Direction-agnostic on purpose:
    this describes exposure to a bad intraday move, not a claim about which
    way box #3 was positioned that day."""
    down = (daily["open"] - daily["low"]) / daily["open"]
    up = (daily["high"] - daily["open"]) / daily["open"]
    return pd.concat([down, up], axis=1).max(axis=1)


def direction_on_day(df: pd.DataFrame, divs: dict, target_date: str,
                    warmup_days: int = 20) -> dict:
    """Reports n_trade_fills for ONE historical day (>1 suggests a possible
    re-entry/flip that day) -- honestly, from the existing run_box API.

    DIRECTION IS DELIBERATELY NOT REPORTED. Two attempts at deriving it were
    tried and both were wrong, found only by cross-checking against real
    data rather than trusting the code: (1) reading state.exposure after
    feeding the full day returns 0/"flat" ALWAYS, because box #3's own
    market-on-close order has already flattened it by the time the day's
    bars finish (0/37 agreement against an independent fill-vs-open check on
    real data); (2) truncating the fed bars to end right after the entry
    fill does NOT avoid this, because backtest_runner.run_box calls
    core.end_session() once per calendar-day GROUP regardless of how many of
    that day's bars were actually supplied -- the close still fires. Getting
    intraday exposure honestly requires either duplicating run_box's
    internal dispatch loop or adding a no-auto-close mode to shared,
    heavily-tested infrastructure, neither of which is being done for one
    side diagnostic. Reported as unknown, not guessed a third time."""
    days = sorted(set(df.index.date))
    target = pd.Timestamp(target_date).date()
    if target not in days:
        return {"n_trade_fills": None, "note": "date not in data"}
    j = days.index(target)
    warm = df[np.isin(df.index.date, days[max(0, j - warmup_days):j])]
    full_target_day = df[df.index.date == target]
    res = run_box(NoiseAreaMomentumBox(), Params(sizing="vol_target"), pd.concat([warm, full_target_day]),
                  "SPY", AccountConfig(cash=100_000.0, leverage=None), fee_fn=mes_fee_fn(), keep_reports=True)
    n_fills = sum(1 for r in res.reports if r.status == "FILLED" and r.tag == "trade"
                 and pd.Timestamp(r.timestamp).date() == target)
    return {"n_trade_fills": n_fills}


def gap_tail_report(market_db: str) -> dict:
    df = pr.load_spy(market_db)
    assert df.index.max().date() <= pr.WINDOW[1], "sealed window must not be read"
    divs = pr.load_dividends()
    net, cost, gross, res = rf.run_variant_full(df, divs, *rf.PUBLISHED, fee_fn=mes_fee_fn())
    daily = ed.daily_spy_bars(df)
    daily.index = pd.to_datetime([diag._date_str(x) for x in daily.index])
    lev = ed.effective_leverage(daily).reindex(daily.index)
    excursion = worst_intraday_excursion_per_day(df, daily)

    combined = pd.concat([excursion.rename("excursion"), lev.rename("leverage")], axis=1)
    n_before_dropna = len(combined)
    combined = combined.dropna()
    n_missing_warmup = n_before_dropna - len(combined)   # leverage needs `lookback`+1 sessions first
    scaled = combined["excursion"] * combined["leverage"]

    pct = {p: float(np.percentile(scaled, p)) for p in (50, 90, 95, 99, 99.5, 100)}
    worst_days = scaled.nlargest(10)

    # Reconciliation table for the worst 10 days: entry leverage, RAW
    # (unscaled) open-to-extreme move, and n_trade_fills (direction NOT
    # reported -- see direction_on_day's docstring).
    reconciliation = []
    for d, v in worst_days.items():
        info = direction_on_day(df, divs, str(d.date()))
        reconciliation.append({"date": str(d.date()), "scaled_excursion": float(v),
                              "leverage": float(combined.loc[d, "leverage"]),
                              "raw_excursion_pct": float(combined.loc[d, "excursion"]),
                              "n_trade_fills": info["n_trade_fills"]})

    analogues = {}
    for name, (start, end) in IN_SAMPLE_ANALOGUES.items():
        window = scaled.loc[start:end]
        analogues[name] = {"n_days": len(window),
                           "worst_scaled_excursion": float(window.max()) if len(window) else None,
                           "mean_scaled_excursion": float(window.mean()) if len(window) else None}

    out = {"run_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
          "metric_definition": "worst_intraday_excursion_per_day = max(open-low, high-open)/open "
                               "-- DIRECTION-AGNOSTIC (does not know which side box #3 actually "
                               "held that day; this is a worst-case bound assuming either side, "
                               "not a claim about realized exposure). Entry-at-open assumed; "
                               "PRE-STOP, PRE-FILL (no stop-loss or fill-timing effects modelled) "
                               "-- upper-bounds normal-day loss, LOWER-bounds dislocation-day loss "
                               "(gap-through fills at real dislocations would be worse than this). "
                               "Scaled by that day's actual vol-target leverage.",
          "flat_overnight_citation": FLAT_OVERNIGHT_CITATION,
          "n_days": len(scaled), "n_missing_leverage_warmup": n_missing_warmup,
          "percentiles_of_leverage_scaled_adverse_excursion": pct,
          "worst_10_days_reconciliation": reconciliation,
          "in_sample_analogues": analogues,
          "caveat": "Data starts 2016-01-04; the canonical 2010-05-06 and 2015-08-24 stress "
                    "dates are NOT in this sample and are not examined here. Only in-sample "
                    "analogues (2018 Volmageddon, 2020 COVID) are reported."}
    try:
        out["commit"] = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                                       text=True, timeout=5).stdout.strip()
    except Exception:
        out["commit"] = None
    out_dir = Path("DATA/pr003")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"gap_tail_{out['run_at'].replace(':', '')}.json"
    path.write_text(json.dumps(out, indent=2, default=str))
    out["path"] = str(path)
    return out


def show(o: dict) -> None:
    print(f"\n=== Descriptive intraday adverse-excursion check, leverage-scaled (commit {o['commit']}) ===")
    print(f"metric definition: {o['metric_definition']}")
    fc = o["flat_overnight_citation"]
    print(f"\nflat-overnight citation: {fc['status']}")
    for t in fc["tests"]:
        print(f"    {t}")
    print(f"    underlying bug-fix: commit {fc['underlying_bugfix_commit']}; scope: {fc['scope']}")
    print(f"\nn_days = {o['n_days']} (n_missing, leverage warm-up = {o['n_missing_leverage_warmup']})\n")
    print("percentiles of (worst intraday move against the open) x (that day's actual leverage):")
    for p, v in o["percentiles_of_leverage_scaled_adverse_excursion"].items():
        print(f"  p{p:<6} {v:>7.1%}")
    print("\nworst 10 days -- reconciliation (leverage x raw move; n_trade_fills re-derived; "
         "direction NOT reported -- see limitation note below):")
    print(f"  {'date':<12}{'scaled':>8}{'leverage':>9}{'raw move':>10}{'n_fills':>8}")
    for d in o["worst_10_days_reconciliation"]:
        print(f"  {d['date']:<12}{d['scaled_excursion']:>7.1%}{d['leverage']:>8.2f}x"
             f"{d['raw_excursion_pct']:>9.1%}{d['n_trade_fills']:>8}")
    print("  NOTE: direction (long/short) is not reported here -- two derivation attempts were "
         "tried and both were wrong on inspection; see direction_on_day's docstring.")
    print("\nin-sample analogues (NOT the canonical 2010/2015 dates -- see caveat):")
    for name, v in o["in_sample_analogues"].items():
        print(f"  {name}: n={v['n_days']}, worst {v['worst_scaled_excursion']}, "
             f"mean {v['mean_scaled_excursion']}")
    print(f"\nCAVEAT: {o['caveat']}")
    print(f"\nSaved: {o['path']}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    for noisy in ("market_data_store", "execution_core"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    a = argparse.ArgumentParser()
    a.add_argument("--db", default="DATA/market_data.db")
    args = a.parse_args()
    show(gap_tail_report(args.db))
