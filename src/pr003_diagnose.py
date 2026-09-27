"""
pr003_diagnose.py — why does box #3 disagree with the authors' code on real SPY data?
Runs both on the same data (vol-target, paper costs) and prints:
  1. days where box #3 ends NOT flat (it must be flat every night)
  2. the largest gaps between a fill price (next-minute OPEN) and the previous
     minute's CLOSE — a bad print in an open would hit box #3 but not the
     reference, which uses closes only
  3. leverage at entry (must be <= 4x)
  4. the 8 days with the biggest box-vs-reference difference, with their fills
Read-only diagnostics; no gates. Uses the replication window only (no 2025+).
    python src/pr003_diagnose.py
"""
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pr003_replication as pr
from boxes.noise_area_momentum import NoiseAreaMomentumBox, Params
from core.backtest_runner import run_box
from core.execution_core import AccountConfig
from evaluation.reference_noise_area import reference_backtest

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
for n in ("market_data_store", "execution_core"):
    logging.getLogger(n).setLevel(logging.WARNING)

df = pr.load_spy("DATA/market_data.db")
divs = pr.load_dividends()
print(f"bars {len(df):,}, days {len(set(df.index.date))}, index tz {df.index.tz}, "
      f"first {df.index[0]}, last {df.index[-1]}")
print(f"bar times per day: min first {df.groupby(df.index.date).apply(lambda g: g.index[0].time()).min()}, "
      f"max last {df.groupby(df.index.date).apply(lambda g: g.index[-1].time()).max()}")
res = run_box(NoiseAreaMomentumBox(), Params(dividends=divs), df, "SPY",
              AccountConfig(cash=pr.AUM0, leverage=None), fee_fn=pr.fee_fn("fidelity"), keep_reports=True)
box = pr.daily_returns(res.equity_by_day)
ref = reference_backtest(df, divs)

pos = pd.Series(res.position_by_day)
notflat = pos[pos != 0]
print(f"\n1) days ending NOT flat: {len(notflat)}")
print(notflat.head(10).to_string())

fills = [r for r in res.reports if r.status == "FILLED"]
prev_close = df["Close"].shift(1)
rows = []
for r in fills:
    ts = pd.Timestamp(r.timestamp)
    if r.tag == "trade" and ts in df.index:
        pc = prev_close.loc[ts]
        rows.append((ts, r.tag, r.fill_qty, r.fill_price, pc, abs(r.fill_price / pc - 1) * 1e4))
f = pd.DataFrame(rows, columns=["ts", "tag", "qty", "fill", "prev_close", "gap_bps"])
print(f"\n2) trade fills {len(f)}; fill-vs-previous-close gap: median {f.gap_bps.median():.1f} bps, "
      f"99th pct {f.gap_bps.quantile(.99):.1f} bps, max {f.gap_bps.max():.1f} bps")
print(f.nlargest(8, "gap_bps").to_string(index=False))

eq = pd.Series(res.equity_by_day)
first = f.groupby(f.ts.dt.date).first()
lev = first.apply(lambda r: r.qty * r.fill / eq.shift(1).get(r.name, pr.AUM0), axis=1)
print(f"\n3) leverage at first entry: median {lev.median():.2f}x, max {lev.max():.2f}x (must be <= 4x)")

j = pd.concat([box.rename("box"), ref.rename("ref")], axis=1).dropna().iloc[20:]
j["diff"] = (j.box - j.ref).abs()
print("\n4) biggest box-vs-reference daily differences:")
for d, r in j.nlargest(8, "diff").iterrows():
    print(f"  {d}: box {r.box:+.2%}  ref {r.ref:+.2%}  end pos {pos.get(d, 0):+.0f}")
    for x in [x for x in fills if pd.Timestamp(x.timestamp).date() == d]:
        print(f"       {pd.Timestamp(x.timestamp).time()} {x.tag:<6} qty {x.fill_qty:>8.0f} @ {x.fill_price:.2f}")
