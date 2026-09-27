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

CHANCE BASELINE (28 Sep 2026): real result was K/N = 3/6. Fifty percent is
ambiguous on its own -- the six indicators are not independent of each
other, so a textbook binomial test on them would be the wrong tool. Instead:
generate many purely random "indicators" (same length as the real series,
run through the IDENTICAL tercile_labels/bucket_contribution/indicator_passes
pipeline -- no separate logic), and measure how often a MEANINGLESS
indicator passes this test on this episode by chance alone. That rate is
directly comparable to 3/6, and a rough (indicators-independent-assumption)
binomial comparison is reported alongside it, explicitly flagged [APPROX].

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
    """Buckets are FULL-SAMPLE terciles (labelled 'low'/'mid'/'high' relative
    to the whole 2016-2024 sample, NOT re-cut within the masked scope -- a
    masked scope's own bucket sizes need not be equal thirds of itself).
    n_missing accounts for masked days excluded because their observable was
    NaN (warmup shift, or a genuine gap in the source data e.g. FRED) -- this
    is reported explicitly rather than left as silent subtraction (round-3
    cold-fork review, 28 Sep 2026: 1,112 episode days vs 1,073 accounted for
    vix_level was real FRED data gaps, not a bug -- but needed stating)."""
    labels = tercile_labels(obs)                        # terciles from the FULL sample
    out = {}
    for lab in ("low", "mid", "high"):
        sel = mask & (labels.reindex(net.index) == lab).to_numpy()
        n = int(sel.sum())
        out[lab] = {"n_days": n, "sum_net_r": float(net[sel].sum()) if n else 0.0,
                   "mean_net_r": float(net[sel].mean()) if n else None,
                   "share_of_days": n / max(1, int(mask.sum()))}
    accounted = sum(v["n_days"] for v in out.values())
    out["n_missing"] = int(mask.sum()) - accounted
    return out


def _real_buckets(b: dict) -> dict:
    """b, minus the 'n_missing' accounting entry -- the three actual
    low/mid/high buckets. Every caller that loops over a bucket_contribution
    result's buckets (not just displays n_missing) must go through this."""
    return {k: v for k, v in b.items() if k != "n_missing"}


def sub_period_contribution(net: pd.Series, dates: pd.Index) -> list[dict]:
    """Calendar-quarter breakdown of the episode, plus each quarter's REGIMES tag."""
    out = []
    for q, g in net.groupby(dates.to_period("Q")):
        out.append({"quarter": str(q), "n_days": len(g), "sum_net_r": float(g.sum()),
                   "mean_net_r": float(g.mean()), "regime": diag._regime_of(diag._date_str(g.index[len(g) // 2]))})
    return out


def effective_leverage(daily: pd.DataFrame, lookback: int = 14, target_vol: float = 0.02,
                       max_leverage: float = 4.0) -> pd.Series:
    """Approximates box #3's own vol-target sizing formula
    (min(max_leverage, target_vol / sigma_spy)) directly from daily closes --
    NOT a rerun of the box, just its documented formula applied analytically.
    Tests a mechanical alternative to any 'low vol -> bad signal' story: once
    realised vol falls low enough, sizing hits the 4x cap and stops shrinking
    further, so the very lowest-vol days could carry structurally maximal
    leverage relative to their own risk, with no signal-quality claim needed
    at all (added per round-3 cold-fork review, 28 Sep 2026)."""
    vol = daily["close"].pct_change().rolling(lookback).std(ddof=1).shift(1)
    return np.minimum(max_leverage, target_vol / vol)


def leverage_cap_check(daily: pd.DataFrame, obs_col: pd.Series, in_ep: np.ndarray,
                       cap_threshold: float = 3.9) -> dict:
    """Mean effective leverage and the share of days AT the cap, by
    full-sample tercile of obs_col, within the episode."""
    lev = effective_leverage(daily).reindex(obs_col.index)
    labels = tercile_labels(obs_col)
    out = {}
    for lab in ("low", "mid", "high"):
        sel = in_ep & (labels.reindex(lev.index) == lab).to_numpy() & lev.notna().to_numpy()
        n = int(sel.sum())
        out[lab] = {"n_days": n, "mean_leverage": float(lev[sel].mean()) if n else None,
                   "share_at_cap": float((lev[sel] >= cap_threshold).mean()) if n else None}
    return out


def random_indicator_pass_rate(net_s: pd.Series, in_ep: np.ndarray, draws: int = 2000,
                              seed: int = 0) -> dict:
    """What fraction of MEANINGLESS random series would 'pass' this exact
    test on this exact episode, by chance alone? Reuses tercile_labels /
    bucket_contribution / indicator_passes verbatim -- no parallel logic."""
    rng = np.random.default_rng(seed)
    passed, evaluated = 0, 0
    for _ in range(draws):
        rand_obs = pd.Series(rng.normal(0, 1, len(net_s)), index=net_s.index)
        b = bucket_contribution(net_s, rand_obs, in_ep)
        if sum(v["n_days"] for v in _real_buckets(b).values()) < 30:
            continue
        r = indicator_passes(b)
        if r is not None:
            evaluated += 1
            passed += int(r)
    rate = passed / evaluated if evaluated else float("nan")
    return {"draws": draws, "evaluated": evaluated, "chance_pass_rate": rate}


def common_shift_null(net_s: pd.Series, in_ep: np.ndarray, real_obs: dict[str, pd.Series],
                      draws: int = 2000, seed: int = 0) -> dict:
    """Round-3 cold-fork correction: the six real indicators are NOT mutually
    independent (VIX level/change/prior-day range/overnight gap proxy one
    volatility factor; term/HY spread a separate macro factor), so testing
    against six INDEPENDENT random draws (random_indicator_pass_rate) likely
    UNDERSTATES the true chance rate. Fix: a common-offset circular shift --
    each draw picks ONE random offset and applies it to ALL SIX real series
    together (np.roll). This preserves each series' own autocorrelation AND
    the real cross-series correlation structure exactly, while destroying
    their alignment with the (fixed) net-return series and episode dates --
    exactly what a genuine chance mechanism should do. Reuses
    tercile_labels/bucket_contribution/indicator_passes verbatim."""
    rng = np.random.default_rng(seed)
    cols = list(real_obs.keys())
    arrays = {c: real_obs[c].reindex(net_s.index).to_numpy() for c in cols}
    n = len(net_s)
    k_pass_counts = []
    for _ in range(draws):
        k = 0
        for c in cols:
            offset = int(rng.integers(1, n))
            shifted = pd.Series(np.roll(arrays[c], offset), index=net_s.index)
            b = bucket_contribution(net_s, shifted, in_ep)
            if sum(v["n_days"] for v in _real_buckets(b).values()) < 30:
                continue
            r = indicator_passes(b)
            if r:
                k += 1
        k_pass_counts.append(k)
    k_arr = np.array(k_pass_counts)
    return {"draws": draws, "n_indicators": len(cols),
           "mean_k_pass": float(k_arr.mean()),
           "p_at_least_3_of_n": float((k_arr >= 3).mean())}


def year_dummy_fork(net_s: pd.Series, obs_col: pd.Series, in_ep: np.ndarray,
                    years: tuple = (2017, 2018, 2019, 2020)) -> dict:
    """Separates P (a dead zone exists) from Q (low vol specifically caused
    it) using data already in hand: does the low-vs-high VIX contrast hold
    WITHIN each individual year of the episode, not just across years? If Q
    is real it should survive a year control; if the pattern is really just
    'this was a calm multi-year stretch', within-year contrasts (which hold
    the calendar fixed) should be flat or inconsistent. Full-sample tercile
    labels (fit once, as everywhere else) are reused, only the day-mask
    changes per year."""
    labels = tercile_labels(obs_col)
    out = {}
    for y in years:
        year_mask = in_ep & (net_s.index.year == y)
        n_year = int(year_mask.sum())
        if n_year < 20:
            out[y] = {"n_days": n_year, "note": "[MISSING] too few episode days this year"}
            continue
        within_year_vals = obs_col.reindex(net_s.index)[year_mask]
        low_r = float(net_s[year_mask & (labels.reindex(net_s.index) == "low").to_numpy()].mean()) \
            if (year_mask & (labels.reindex(net_s.index) == "low").to_numpy()).sum() else None
        high_r = float(net_s[year_mask & (labels.reindex(net_s.index) == "high").to_numpy()].mean()) \
            if (year_mask & (labels.reindex(net_s.index) == "high").to_numpy()).sum() else None
        out[y] = {"n_days": n_year, "within_year_obs_std": float(within_year_vals.std()),
                 "low_mean_net_r": low_r, "high_mean_net_r": high_r,
                 "q_direction_holds": (low_r is not None and high_r is not None and low_r < high_r)}
    informative = [y for y, v in out.items() if v.get("q_direction_holds") is not None]
    supporting = [y for y in informative if out[y]["q_direction_holds"]]
    return {"per_year": out, "informative_years": informative,
           "years_supporting_q": supporting,
           "fork_result": "Q survives year control" if len(supporting) >= max(2, len(informative) // 2 + 1)
                          else "calendar/drift explains the pattern at least as well as Q"}


def post_episode_bucket_check(net_s: pd.Series, obs_col: pd.Series, start: str, end: str) -> dict:
    """Same full-sample tercile boundaries, applied OUTSIDE the episode that
    defined them -- here, Jan 2023-Dec 2024. Not a clean statistical
    hold-out (this window already informed Stage R/Refine/V), so weight as
    corroboration only, per round-3 review."""
    mask = (net_s.index >= pd.Timestamp(start)) & (net_s.index <= pd.Timestamp(end))
    return bucket_contribution(net_s, obs_col, mask)


def binomial_at_least_k(n: int, k: int, p: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p). [APPROX]: assumes independence
    across the n real indicators, which is not quite true here -- reported
    as a rough comparison, not a rigorous p-value."""
    from math import comb
    return sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


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
        n_in_episode = sum(v["n_days"] for v in _real_buckets(b["in_episode"]).values())
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

    chance = random_indicator_pass_rate(net_s, in_ep)
    k, n = result["survey"]["k_pass"], result["survey"]["n_evaluated"]
    chance["binomial_p_at_least_k_APPROX"] = (
        binomial_at_least_k(n, k, chance["chance_pass_rate"])
        if not np.isnan(chance["chance_pass_rate"]) else None)
    result["chance_baseline"] = chance

    # Cluster-aware chance baseline (round 3): only the columns actually
    # evaluated in the survey, real series, common-offset shift.
    real_series = {col: obs[col] for col in evaluated}
    if len(real_series) >= 2:
        result["chance_baseline_common_shift"] = common_shift_null(net_s, in_ep, real_series)
    else:
        result["chance_baseline_common_shift"] = None

    # Year-dummy fork (round 3): only on vix_level, the strongest surviving
    # candidate, and only if it was evaluable at all.
    if "vix_level" in obs.columns and evaluated.get("vix_level") is not None:
        result["year_fork"] = year_dummy_fork(net_s, obs["vix_level"], in_ep)
    else:
        result["year_fork"] = None

    # Post-episode (2023-2024) check on the same indicators, corroboration only.
    result["post_episode_2023_2024"] = {}
    for col in evaluated:
        result["post_episode_2023_2024"][col] = post_episode_bucket_check(
            net_s, obs[col], "2023-01-01", "2024-12-31")

    # Leverage-cap mechanical check, on vix_level if available.
    if "vix_level" in obs.columns and evaluated.get("vix_level") is not None:
        result["leverage_cap_check"] = leverage_cap_check(daily, obs["vix_level"], in_ep)
    else:
        result["leverage_cap_check"] = None
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
            print(f"    {scope}: (n_missing={buckets['n_missing']})")
            for lab, v in _real_buckets(buckets).items():
                m = "—" if v["mean_net_r"] is None else f"{v['mean_net_r']:+.3%}"
                print(f"      {lab:<5} n={v['n_days']:>4} ({v['share_of_days']:.0%})  "
                     f"sum {v['sum_net_r']:>+8.2%}  mean/day {m}")
    cb = r.get("chance_baseline")
    if cb:
        print(f"\n=== Chance baseline: how often does a MEANINGLESS random indicator pass this "
             f"exact test on this episode? ({cb['draws']} random draws) ===")
        print(f"  chance pass rate: {cb['chance_pass_rate']:.1%}  ({cb['evaluated']}/{cb['draws']} evaluable draws)")
        if cb["binomial_p_at_least_k_APPROX"] is not None:
            print(f"  P(>= {r['survey']['k_pass']} of {r['survey']['n_evaluated']} real indicators pass "
                 f"| chance alone) = {cb['binomial_p_at_least_k_APPROX']:.1%}  [APPROX -- assumes independence, "
                 f"which the real indicators do not fully have]")
    csn = r.get("chance_baseline_common_shift")
    if csn:
        print(f"\n=== Cluster-aware chance baseline (common-offset shift, preserves cross-indicator "
             f"correlation) ===")
        print(f"  mean indicators passing per shifted draw: {csn['mean_k_pass']:.2f} of {csn['n_indicators']}")
        print(f"  P(>= 3 of {csn['n_indicators']} pass | correlated chance): {csn['p_at_least_3_of_n']:.1%}")

    yf = r.get("year_fork")
    if yf:
        print(f"\n=== Year-dummy fork (does the VIX contrast hold WITHIN each year, or only across years?) ===")
        for y, v in yf["per_year"].items():
            if "note" in v:
                print(f"  {y}: {v['note']}")
                continue
            lo = "—" if v["low_mean_net_r"] is None else f"{v['low_mean_net_r']:+.3%}"
            hi = "—" if v["high_mean_net_r"] is None else f"{v['high_mean_net_r']:+.3%}"
            print(f"  {y}: n={v['n_days']:>3}  low {lo}  high {hi}  "
                 f"within-year VIX std {v['within_year_obs_std']:.2f}  "
                 f"Q holds: {v['q_direction_holds']}")
        print(f"  informative years: {yf['informative_years']}, supporting Q: {yf['years_supporting_q']}")
        print(f"  FORK RESULT: {yf['fork_result']}")

    pe = r.get("post_episode_2023_2024")
    if pe:
        print(f"\n=== Post-episode check, Jan 2023-Dec 2024 (same tercile boundaries; "
             f"corroboration only, NOT a clean hold-out) ===")
        for col, b in pe.items():
            rb = _real_buckets(b)
            lo, hi = rb["low"]["mean_net_r"], rb["high"]["mean_net_r"]
            lo_s = "—" if lo is None else f"{lo:+.3%}"
            hi_s = "—" if hi is None else f"{hi:+.3%}"
            print(f"  {col:<22} low {lo_s}  high {hi_s}  (n_missing={b['n_missing']})")

    lc = r.get("leverage_cap_check")
    if lc:
        print(f"\n=== Leverage-cap mechanical check (by VIX tercile, within the episode) ===")
        for lab, v in lc.items():
            ml = "—" if v["mean_leverage"] is None else f"{v['mean_leverage']:.2f}x"
            sc = "—" if v["share_at_cap"] is None else f"{v['share_at_cap']:.0%}"
            print(f"  {lab:<5} n={v['n_days']:>4}  mean leverage {ml}  share at/near 4x cap {sc}")

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
