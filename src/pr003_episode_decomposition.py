"""
pr003_episode_decomposition.py — PR-003 pre-registered diagnostic (28 Sep 2026)
================================================================================
Cold-fork round 2 (corrected protocol, point 1): decompose the drawdown_
attribution's worst episode (2016-04-07 .. 2020-09-03, -39.5%, 73% of the
total drawdown) by sub-period and by SESSION-AHEAD observables, BEFORE any
filter is designed. Written up regardless of result. No parameter search,
no threshold tuning, no filter built here -- this only answers: is the
loss diffuse bleed, or concentrated in identifiable, tradable conditions?

Session-ahead observables ONLY (point 5 of the corrected protocol -- a gate
conditioning on TODAY's close-to-close vol to trade TODAY is inadmissible;
everything here is known before today's open):
  prior_day_abs_return   |yesterday's close-to-close return|
  prior_day_range_pct    yesterday's (high-low)/open
  overnight_gap_pct      |today's open / yesterday's close - 1|
  vix_level              yesterday's VIX close (FRED VIXCLS if downloaded,
                         else [MISSING] -- reported, never silently substituted)

Bucketing: terciles of each observable, computed on THIS strategy's full
2016-2024 sample (not on the episode alone, to avoid defining "high/low"
using the very data being explained).

    python src/pr003_episode_decomposition.py
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
import pr003_diagnostics as diag                                     # noqa: E402
import pr003_refine as rf                                             # noqa: E402
import pr003_replication as pr                                        # noqa: E402
from evaluation.mes_costs import mes_fee_fn                           # noqa: E402
from market_data_store import REGIMES                                 # noqa: E402

log = logging.getLogger("pr003_episode_decomp")

# From the completed drawdown_attribution run (27 Sep 2026, commit 34e1ccd):
EPISODE = {"start": "2016-04-07", "trough": "2018-01-09", "end": "2020-09-03"}


def daily_spy_bars(df_1m: pd.DataFrame) -> pd.DataFrame:
    g = df_1m.groupby(df_1m.index.date)
    return pd.DataFrame({"open": g["Open"].first(), "high": g["High"].max(),
                         "low": g["Low"].min(), "close": g["Close"].last()})


def session_ahead_observables(daily: pd.DataFrame) -> pd.DataFrame:
    """Everything here is known strictly BEFORE today's session (point 5:
    a gate must not condition on today's own close-to-close move)."""
    prev_close = daily["close"].shift(1)
    prior_return = daily["close"].pct_change()          # yesterday's ret, known at today's open
    prior_range = (daily["high"] - daily["low"]) / daily["open"]
    return pd.DataFrame({
        "prior_day_abs_return": prior_return.abs().shift(1),
        "prior_day_range_pct": prior_range.shift(1),
        "overnight_gap_pct": (daily["open"] / prev_close - 1).abs(),
    }, index=daily.index)


def load_vix(market_db: str, start: str, end: str) -> pd.Series | None:
    try:
        from fred_store import FredDataStore
        rows = FredDataStore(market_db).get_series("VIXCLS", start, end)
    except Exception as e:
        log.warning("FRED store unavailable (%s) -- vix_level will be [MISSING]", e)
        return None
    if not rows:
        return None
    s = pd.Series({r["date"]: r["value"] for r in rows})
    s.index = pd.to_datetime(s.index)
    return s.shift(1)          # yesterday's VIX close, known at today's open


def tercile_labels(x: pd.Series) -> pd.Series:
    """Terciles on the FULL sample the series was fit on, not the episode
    alone -- 'high/low' must not be defined using the data being explained."""
    q = x.quantile([1 / 3, 2 / 3])
    return pd.cut(x, [-np.inf, q.iloc[0], q.iloc[1], np.inf], labels=["low", "mid", "high"])


def bucket_contribution(net: pd.Series, obs: pd.Series, mask: np.ndarray) -> dict:
    labels = tercile_labels(obs)                        # terciles from the FULL sample
    out = {}
    for lab in ("low", "mid", "high"):
        sel = mask & (labels.reindex(net.index) == lab).to_numpy()
        n = int(sel.sum())
        out[lab] = {"n_days": n, "sum_net_r": float(net[sel].sum()) if n else 0.0,
                   "mean_net_r": float(net[sel].mean()) if n else None,
                   "share_of_days": n / max(1, int(mask.sum()))}
    return out


def sub_period_contribution(net: pd.Series, dates: pd.Index) -> list[dict]:
    """Calendar-quarter breakdown of the episode, plus each quarter's REGIMES tag."""
    out = []
    for q, g in net.groupby(dates.to_period("Q")):
        out.append({"quarter": str(q), "n_days": len(g), "sum_net_r": float(g.sum()),
                   "mean_net_r": float(g.mean()), "regime": diag._regime_of(diag._date_str(g.index[len(g) // 2]))})
    return out


def decompose(market_db: str, out_dir: Path) -> dict:
    df = pr.load_spy(market_db)
    assert df.index.max().date() <= pr.WINDOW[1], "sealed window must not be read"
    divs = pr.load_dividends()
    net, cost, gross, res = rf.run_variant_full(df, divs, *rf.PUBLISHED, fee_fn=mes_fee_fn())
    net_s = net.copy(); net_s.index = pd.to_datetime([diag._date_str(x) for x in net.index])
    daily = daily_spy_bars(df)
    daily.index = pd.to_datetime([diag._date_str(x) for x in daily.index])
    obs = session_ahead_observables(daily)
    vix = load_vix(market_db, pr.WINDOW[0].isoformat(), pr.WINDOW[1].isoformat())
    if vix is not None:
        obs["vix_level"] = vix.reindex(obs.index)

    ep_start, ep_end = pd.Timestamp(EPISODE["start"]), pd.Timestamp(EPISODE["end"])
    in_ep = (net_s.index >= ep_start) & (net_s.index <= ep_end)

    result = {"run_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
              "episode": EPISODE, "n_days_in_episode": int(in_ep.sum()),
              "episode_sum_net_r": float(net_s[in_ep].sum()),
              "sub_periods": sub_period_contribution(net_s[in_ep], net_s.index[in_ep]),
              "buckets": {}, "vix_available": vix is not None}
    for col in obs.columns:
        if obs[col].notna().sum() < 30:
            result["buckets"][col] = "[MISSING] insufficient data"
            continue
        result["buckets"][col] = {
            "in_episode": bucket_contribution(net_s, obs[col], in_ep),
            "rest_of_sample": bucket_contribution(net_s, obs[col], ~in_ep)}
    try:
        result["commit"] = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                                          text=True, timeout=5).stdout.strip()
    except Exception:
        result["commit"] = None
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"episode_decomposition_{result['run_at'].replace(':', '')}.json"
    path.write_text(json.dumps(result, indent=2, default=str))
    result["path"] = str(path)
    return result


def show(r: dict) -> None:
    print(f"\n=== Episode decomposition: {r['episode']['start']} .. {r['episode']['end']} "
         f"(commit {r['commit']}) ===")
    print(f"{r['n_days_in_episode']} days in episode, sum net R {r['episode_sum_net_r']:+.1%}\n")
    print("Sub-period breakdown (calendar quarter):")
    for sp in r["sub_periods"]:
        print(f"  {sp['quarter']}  n={sp['n_days']:>3}  sum {sp['sum_net_r']:>+8.2%}  "
             f"mean/day {sp['mean_net_r']:>+8.3%}  ({sp['regime']})")
    print(f"\nVIX data: {'available (FRED VIXCLS)' if r['vix_available'] else '[MISSING] -- not downloaded; run fred_store.py --download'}")
    print("\nBucketed by session-ahead observable (terciles on the FULL 2016-2024 sample):")
    for col, b in r["buckets"].items():
        if isinstance(b, str):
            print(f"\n  {col}: {b}")
            continue
        print(f"\n  {col}:")
        for scope, buckets in b.items():
            print(f"    {scope}:")
            for lab, v in buckets.items():
                m = "—" if v["mean_net_r"] is None else f"{v['mean_net_r']:+.3%}"
                print(f"      {lab:<5} n={v['n_days']:>4} ({v['share_of_days']:.0%})  "
                     f"sum {v['sum_net_r']:>+8.2%}  mean/day {m}")
    print(f"\nSaved: {r['path']}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    for noisy in ("market_data_store", "execution_core"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    a = argparse.ArgumentParser()
    a.add_argument("--db", default="DATA/market_data.db")
    args = a.parse_args()
    show(decompose(args.db, Path("DATA/pr003")))
