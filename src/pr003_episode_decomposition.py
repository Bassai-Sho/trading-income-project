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

SEVEN-INDICATOR SURVEY (28 Sep 2026, per DAC "survey all indicators" audit):
addresses the corrected cold-fork protocol's point 4 -- a single indicator's
result (VIX, run 27 Sep 2026) is exactly the "K=1 of some larger menu"
pattern the protocol warns is snoop-shaped even when the one indicator is
externally anchored. MENU, PRE-COMMITTED BEFORE RUNNING, never edited after
seeing results:
    vix_level              yesterday's VIX close
    vix_change             yesterday's VIX close minus the day before's
    prior_day_abs_return    (already had)
    prior_day_range_pct     (already had)
    overnight_gap_pct       (already had)
    term_spread             yesterday's 10Y minus 2Y Treasury yield (DGS10-DGS2)
    hy_spread               yesterday's high-yield OAS credit spread (BAMLH0A0HYM2)
EXCLUDED ON PRINCIPLE, stated before running (not after seeing results):
    Fed Funds rate, CPI -- both update monthly and barely move within a
    quarter; including them would just re-encode calendar time, which the
    sub-period table already reports directly.
PASS CRITERION, fixed before running: within the episode alone, an
indicator "passes" if its high-tercile mean daily return > 0 AND its
low-tercile mean daily return < 0 (the direction the paper's own VIX
finding predicts). Report pass/fail per indicator and the K-of-N fraction.
This is still pure diagnosis -- no threshold, no gate, no action decided.

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


def load_fred_raw(market_db: str, series_id: str, start: str, end: str) -> pd.Series | None:
    """Unshifted FRED series, sorted by date. None (not a substitute) if the
    series isn't downloaded -- callers must report [MISSING], never fall
    back to a proxy silently."""
    try:
        from fred_store import FredDataStore
        rows = FredDataStore(market_db).get_series(series_id, start, end)
    except Exception as e:
        log.warning("FRED store unavailable for %s (%s)", series_id, e)
        return None
    if not rows:
        return None
    s = pd.Series({r["date"]: r["value"] for r in rows})
    s.index = pd.to_datetime(s.index)
    return s.sort_index()


def load_vix(market_db: str, start: str, end: str) -> pd.Series | None:
    raw = load_fred_raw(market_db, "VIXCLS", start, end)
    return None if raw is None else raw.shift(1)   # yesterday's VIX close, known at today's open


def load_vix_change(market_db: str, start: str, end: str) -> pd.Series | None:
    raw = load_fred_raw(market_db, "VIXCLS", start, end)
    return None if raw is None else raw.diff().shift(1)   # yesterday's VIX close minus the day before's


def load_term_spread(market_db: str, start: str, end: str) -> pd.Series | None:
    d10 = load_fred_raw(market_db, "DGS10", start, end)
    d2 = load_fred_raw(market_db, "DGS2", start, end)
    if d10 is None or d2 is None:
        return None
    return (d10 - d2).shift(1)


def load_hy_spread(market_db: str, start: str, end: str) -> pd.Series | None:
    raw = load_fred_raw(market_db, "BAMLH0A0HYM2", start, end)
    return None if raw is None else raw.shift(1)


def indicator_passes(bucket_in_episode: dict) -> bool | None:
    """Fixed pass criterion (set BEFORE running): high-tercile mean daily
    return > 0 AND low-tercile mean daily return < 0, within the episode
    alone -- the direction the paper's own VIX finding predicts. None if
    either bucket has no observations."""
    hi, lo = bucket_in_episode["high"]["mean_net_r"], bucket_in_episode["low"]["mean_net_r"]
    if hi is None or lo is None:
        return None
    return hi > 0 and lo < 0


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
    fred_start, fred_end = pr.WINDOW[0].isoformat(), pr.WINDOW[1].isoformat()
    fred_loaders = {"vix_level": load_vix, "vix_change": load_vix_change,
                    "term_spread": load_term_spread, "hy_spread": load_hy_spread}
    fred_missing = []
    for name, loader in fred_loaders.items():
        series = loader(market_db, fred_start, fred_end)
        if series is None:
            fred_missing.append(name)
        else:
            obs[name] = series.reindex(obs.index)
    vix = obs.get("vix_level")   # kept for the existing vix_available flag below

    ep_start, ep_end = pd.Timestamp(EPISODE["start"]), pd.Timestamp(EPISODE["end"])
    in_ep = (net_s.index >= ep_start) & (net_s.index <= ep_end)

    result = {"run_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
              "episode": EPISODE, "n_days_in_episode": int(in_ep.sum()),
              "episode_sum_net_r": float(net_s[in_ep].sum()),
              "sub_periods": sub_period_contribution(net_s[in_ep], net_s.index[in_ep]),
              "buckets": {}, "vix_available": vix is not None, "fred_missing": fred_missing}
    for col in obs.columns:
        if obs[col].notna().sum() < 30:
            result["buckets"][col] = "[MISSING] insufficient data"
            continue
        result["buckets"][col] = {
            "in_episode": bucket_contribution(net_s, obs[col], in_ep),
            "rest_of_sample": bucket_contribution(net_s, obs[col], ~in_ep)}

    # Seven-indicator survey: pass/fail per indicator (fixed criterion, set
    # before running), plus the K-of-N pass fraction -- never edited after
    # seeing results.
    survey, survey_notes = {}, {}
    for col, b in result["buckets"].items():
        if isinstance(b, str):
            survey[col] = None
            survey_notes[col] = "[MISSING] insufficient data over the full sample"
            continue
        n_in_episode = sum(v["n_days"] for v in b["in_episode"].values())
        # A column can clear the FULL-SAMPLE 30-day minimum (used to fit
        # terciles) while having almost no overlap with the episode itself --
        # exactly what happened with hy_spread (BAMLH0A0HYM2 only starts
        # partway into the window: 0 in-episode days, real run 27 Sep 2026).
        # That must not count as an evaluated indicator either way.
        if n_in_episode < 30:
            survey[col] = None
            survey_notes[col] = f"[MISSING] only {n_in_episode} in-episode days (need >= 30)"
            continue
        survey[col] = indicator_passes(b["in_episode"])
    evaluated = {k: v for k, v in survey.items() if v is not None}
    result["survey"] = {"per_indicator": survey, "notes": survey_notes,
                        "k_pass": sum(evaluated.values()), "n_evaluated": len(evaluated),
                        "excluded_by_design": ["FEDFUNDS", "CPIAUCSL"]}
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
    if r.get("fred_missing"):
        print(f"FRED series not available: {r['fred_missing']} -- excluded from the survey below, not treated as fails")
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
    sv = r.get("survey")
    if sv:
        print(f"\n=== Seven-indicator survey (pass = high-tercile mean > 0 AND low-tercile mean < 0, "
             f"within the episode) ===")
        for name, passed in sv["per_indicator"].items():
            if passed is None:
                label = sv.get("notes", {}).get(name, "n/a [MISSING]")
            else:
                label = "PASS" if passed else "fail"
            print(f"  {name:<22} {label}")
        print(f"  excluded by design (pre-committed, not post-hoc): {sv['excluded_by_design']}")
        print(f"\n  K/N = {sv['k_pass']}/{sv['n_evaluated']} indicators corroborate the VIX-style direction")
    print(f"\nSaved: {r['path']}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    for noisy in ("market_data_store", "execution_core"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    a = argparse.ArgumentParser()
    a.add_argument("--db", default="DATA/market_data.db")
    args = a.parse_args()
    show(decompose(args.db, Path("DATA/pr003")))
