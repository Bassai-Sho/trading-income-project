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
entered position, scaled by that day's ACTUAL leverage (effective_leverage,
already built for the leverage-cap check) -- i.e. the realised P&L impact if
the entry had been marked at the day's single worst intraday tick rather
than its actual exit. This is a stress DESCRIPTION for the risk-policy memo
to consume, not a gate, not a fix, not a re-run of anything.

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
from evaluation.mes_costs import mes_fee_fn                            # noqa: E402

log = logging.getLogger("pr003_gap_tail")

IN_SAMPLE_ANALOGUES = {"2018_volmageddon": ("2018-02-01", "2018-02-09"),
                       "2020_covid_crash": ("2020-02-20", "2020-03-23")}


def worst_intraday_excursion_per_day(df_1m: pd.DataFrame, daily: pd.DataFrame) -> pd.Series:
    """For each session: |entry-side worst move|, defined direction-agnostically
    as max(open - low, high - open) / open -- the larger of the two possible
    adverse moves from the session's own open, whichever side a position
    entered near the open. Direction-agnostic on purpose: this describes
    exposure to a bad intraday move, not a claim about which way box #3 was
    positioned that day (that would require re-deriving its live signal,
    which this descriptive pull does not do)."""
    down = (daily["open"] - daily["low"]) / daily["open"]
    up = (daily["high"] - daily["open"]) / daily["open"]
    return pd.concat([down, up], axis=1).max(axis=1)


def gap_tail_report(market_db: str) -> dict:
    df = pr.load_spy(market_db)
    assert df.index.max().date() <= pr.WINDOW[1], "sealed window must not be read"
    divs = pr.load_dividends()
    net, cost, gross, res = rf.run_variant_full(df, divs, *rf.PUBLISHED, fee_fn=mes_fee_fn())
    daily = ed.daily_spy_bars(df)
    daily.index = pd.to_datetime([diag._date_str(x) for x in daily.index])
    lev = ed.effective_leverage(daily).reindex(daily.index)
    excursion = worst_intraday_excursion_per_day(df, daily)
    scaled = (excursion * lev).dropna()      # scaled by the ACTUAL leverage carried that day

    pct = {p: float(np.percentile(scaled, p)) for p in (50, 90, 95, 99, 99.5, 100)}
    worst_days = scaled.nlargest(10)

    analogues = {}
    for name, (start, end) in IN_SAMPLE_ANALOGUES.items():
        window = scaled.loc[start:end]
        analogues[name] = {"n_days": len(window),
                           "worst_scaled_excursion": float(window.max()) if len(window) else None,
                           "mean_scaled_excursion": float(window.mean()) if len(window) else None}

    out = {"run_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
          "n_days": len(scaled), "percentiles_of_leverage_scaled_adverse_excursion": pct,
          "worst_10_days": [{"date": str(d.date()), "scaled_excursion": float(v)}
                            for d, v in worst_days.items()],
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
    print(f"n_days = {o['n_days']}\n")
    print("percentiles of (worst intraday move against the open) x (that day's actual leverage):")
    for p, v in o["percentiles_of_leverage_scaled_adverse_excursion"].items():
        print(f"  p{p:<6} {v:>7.1%}")
    print("\nworst 10 individual days:")
    for d in o["worst_10_days"]:
        print(f"  {d['date']}  {d['scaled_excursion']:>7.1%}")
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
