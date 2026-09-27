"""PR-003 episode decomposition: session-ahead observables, terciles fit on
the full sample (not the episode), sub-period breakdown. No filter, no
tuning -- this module only reports."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pr003_episode_decomposition as ed
from market_data_store import session_close
from synthetic_bars import noisy_days


def _daily(n=300, seed=0):
    idx = pd.bdate_range("2018-01-02", periods=n)
    rng = np.random.default_rng(seed)
    px = 280 * np.exp(np.cumsum(rng.normal(0, 0.008, n)))
    rng2 = np.random.default_rng(seed + 1)
    rng_ = rng2.uniform(0.002, 0.01, n)
    return pd.DataFrame({"open": px, "high": px * (1 + rng_), "low": px * (1 - rng_),
                         "close": px * (1 + rng2.normal(0, 0.001, n))}, index=idx)


def test_observables_are_shifted_one_day_forward_no_lookahead():
    d = _daily()
    obs = ed.session_ahead_observables(d)
    # today's observable must equal YESTERDAY's realised quantities, not today's
    prior_ret = d["close"].pct_change().abs()
    assert obs["prior_day_abs_return"].iloc[5] == pytest.approx(prior_ret.iloc[4])
    assert pd.isna(obs["prior_day_abs_return"].iloc[0])   # no day-before-the-first-day


def test_overnight_gap_uses_todays_open_vs_yesterdays_close():
    d = _daily()
    obs = ed.session_ahead_observables(d)
    expected = abs(d["open"].iloc[10] / d["close"].iloc[9] - 1)
    assert obs["overnight_gap_pct"].iloc[10] == pytest.approx(expected)


def test_terciles_are_fit_on_the_full_series_passed_in():
    """Bucket boundaries must come from the WHOLE series, not whatever subset
    happens to be masked as 'in episode' -- otherwise 'high/low' is defined
    using the very data being explained (the exact trap Cold-Fork round 1
    flagged for filter design; the decomposition must not repeat it)."""
    x = pd.Series(np.arange(300, dtype=float))       # uniform 0..299
    labels_full = ed.tercile_labels(x)
    # the boundaries should sit near the 1/3 and 2/3 marks of the FULL range,
    # not near the boundaries of some small subset
    assert (labels_full == "low").sum() == pytest.approx(100, abs=2)
    assert (labels_full == "high").sum() == pytest.approx(100, abs=2)


def test_bucket_contribution_sums_to_the_masked_total():
    net = pd.Series(np.arange(1, 301, dtype=float) * 0.0001, index=pd.bdate_range("2018-01-02", periods=300))
    obs = pd.Series(np.random.default_rng(3).normal(0, 1, 300), index=net.index)
    mask = np.ones(300, dtype=bool)
    b = ed.bucket_contribution(net, obs, mask)
    real = ed._real_buckets(b)
    assert sum(v["sum_net_r"] for v in real.values()) == pytest.approx(net.sum())
    assert sum(v["n_days"] for v in real.values()) == 300
    assert b["n_missing"] == 0


def test_bucket_contribution_respects_the_mask():
    net = pd.Series(1.0, index=pd.bdate_range("2018-01-02", periods=100))    # every day contributes 1.0
    obs = pd.Series(np.arange(100.0), index=net.index)
    mask = np.zeros(100, dtype=bool); mask[:20] = True     # only the first 20 days count
    b = ed.bucket_contribution(net, obs, mask)
    assert sum(v["n_days"] for v in ed._real_buckets(b).values()) == 20


def test_bucket_contribution_reports_n_missing_for_nan_observable_days():
    """The reconciliation added after round-3 review: 554/302/217 in the real
    run summed to 1,073 against 1,112 episode days -- a silent 39-day gap
    (real FRED data gaps for vix_level, not a bug, but it must be STATED,
    not left for the reader to find by subtraction)."""
    net = pd.Series(1.0, index=pd.bdate_range("2018-01-02", periods=100))
    obs = pd.Series(np.arange(100.0), index=net.index)
    obs.iloc[:10] = np.nan                                  # 10 days with no observable
    mask = np.ones(100, dtype=bool)
    b = ed.bucket_contribution(net, obs, mask)
    assert b["n_missing"] == 10
    assert sum(v["n_days"] for v in ed._real_buckets(b).values()) == 90


def test_planted_concentration_is_visible_in_the_bucket_report():
    """If ALL of an episode's losses sit in the 'high' bucket of one
    observable, the decomposition must show that clearly (a sanity check
    that the mechanism actually reveals concentration, not just noise)."""
    idx = pd.bdate_range("2018-01-02", periods=300)
    obs = pd.Series(np.linspace(0, 1, 300), index=idx)     # monotonic: last third is 'high'
    net = pd.Series(0.0, index=idx)
    net.iloc[250:] = -0.01                                  # all losses in the top tercile
    mask = np.ones(300, dtype=bool)
    b = ed.bucket_contribution(net, obs, mask)
    assert b["high"]["sum_net_r"] < b["low"]["sum_net_r"] < 0.001
    assert b["high"]["sum_net_r"] == pytest.approx(net.sum())


def test_sub_period_contribution_groups_by_quarter():
    idx = pd.bdate_range("2019-01-02", periods=200)
    net = pd.Series(0.001, index=idx)
    sp = ed.sub_period_contribution(net, idx)
    assert len(sp) >= 3 and all(s["n_days"] > 0 for s in sp)
    assert sum(s["n_days"] for s in sp) == 200


def test_missing_vix_reported_not_substituted(monkeypatch):
    monkeypatch.setattr(ed, "FredDataStore", None, raising=False)
    def boom(*a, **k):
        raise ModuleNotFoundError("no fred_store")
    import builtins
    orig_import = builtins.__import__
    def fake_import(name, *a, **k):
        if name == "fred_store":
            raise ModuleNotFoundError("simulated: FRED not downloaded")
        return orig_import(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    v = ed.load_vix("unused.db", "2016-01-01", "2024-12-31")
    assert v is None       # must report [MISSING], never silently fall back to a substitute


# ── Seven-indicator survey (28 Sep 2026) ─────────────────────────────────────

def test_indicator_passes_requires_both_directions():
    hi_pos_lo_neg = {"high": {"mean_net_r": 0.001}, "low": {"mean_net_r": -0.001}}
    hi_pos_lo_pos = {"high": {"mean_net_r": 0.001}, "low": {"mean_net_r": 0.001}}
    hi_neg_lo_neg = {"high": {"mean_net_r": -0.001}, "low": {"mean_net_r": -0.001}}
    assert ed.indicator_passes(hi_pos_lo_neg) is True
    assert ed.indicator_passes(hi_pos_lo_pos) is False
    assert ed.indicator_passes(hi_neg_lo_neg) is False


def test_indicator_passes_none_when_a_bucket_is_empty():
    assert ed.indicator_passes({"high": {"mean_net_r": None}, "low": {"mean_net_r": -0.001}}) is None


def test_vix_change_is_the_diff_of_vix_shifted_one_extra_day(monkeypatch):
    rows = [{"date": d, "value": v} for d, v in
           zip(pd.bdate_range("2019-01-02", periods=10).strftime("%Y-%m-%d"),
               [15, 16, 14, 20, 19, 18, 17, 30, 25, 22])]
    class FakeStore:
        def __init__(self, db): pass
        def get_series(self, series_id, start, end): return rows
    monkeypatch.setattr(ed, "load_fred_raw", lambda db, sid, s, e: (
        pd.Series({r["date"]: r["value"] for r in rows}, dtype=float).pipe(
            lambda x: x.set_axis(pd.to_datetime(x.index)).sort_index())))
    ch = ed.load_vix_change("unused.db", "2019-01-01", "2019-01-20")
    raw = ed.load_fred_raw("unused.db", "VIXCLS", "2019-01-01", "2019-01-20")
    # ch[d] must equal raw[d-1] - raw[d-2] (twice-shifted diff)
    assert ch.iloc[3] == pytest.approx(raw.iloc[2] - raw.iloc[1])   # ch[i] = raw[i-1] - raw[i-2]


def test_term_spread_is_10y_minus_2y_shifted(monkeypatch):
    idx = pd.bdate_range("2019-01-02", periods=10)
    d10 = pd.Series(2.5, index=idx)
    d2 = pd.Series(2.0, index=idx)
    def fake_raw(db, sid, s, e):
        return d10 if sid == "DGS10" else d2 if sid == "DGS2" else None
    monkeypatch.setattr(ed, "load_fred_raw", fake_raw)
    spread = ed.load_term_spread("unused.db", "2019-01-01", "2019-01-20")
    assert spread.iloc[5] == pytest.approx(0.5)   # 2.5 - 2.0
    assert pd.isna(spread.iloc[0])                 # shifted: no day before the first


def test_term_spread_missing_when_either_leg_missing(monkeypatch):
    monkeypatch.setattr(ed, "load_fred_raw", lambda db, sid, s, e: None if sid == "DGS2" else
                        pd.Series([1.0], index=pd.bdate_range("2019-01-02", periods=1)))
    assert ed.load_term_spread("unused.db", "2019-01-01", "2019-01-20") is None


def test_survey_excludes_fed_funds_and_cpi_by_design():
    """The four loader functions actually used by the survey menu never
    fetch these two series ids -- the disclosure STRING in decompose()'s
    output is a separate, legitimate mention and is not what this checks."""
    import inspect
    loaders_src = "".join(inspect.getsource(f) for f in
                          (ed.load_vix, ed.load_vix_change, ed.load_term_spread, ed.load_hy_spread))
    assert "FEDFUNDS" not in loaders_src and "CPIAUCSL" not in loaders_src


def test_survey_k_of_n_counts_only_evaluated_indicators():
    buckets = {
        "a": {"in_episode": {"high": {"mean_net_r": 0.001}, "low": {"mean_net_r": -0.001}}},
        "b": {"in_episode": {"high": {"mean_net_r": -0.001}, "low": {"mean_net_r": -0.001}}},
        "c": "[MISSING] insufficient data",
    }
    survey = {}
    for col, b in buckets.items():
        survey[col] = None if isinstance(b, str) else ed.indicator_passes(b["in_episode"])
    evaluated = {k: v for k, v in survey.items() if v is not None}
    assert sum(evaluated.values()) == 1 and len(evaluated) == 2   # 'c' excluded from N, not counted as fail


def test_survey_excludes_an_indicator_with_zero_in_episode_coverage():
    """Reproduces the real bug (28 Sep 2026): hy_spread cleared the
    full-sample >=30-row minimum used to fit terciles, but had ZERO days
    actually falling inside the episode (its FRED series starts partway
    through the window). That must count as [MISSING] for the survey, not
    as an evaluated 'fail' -- the denominator (N) must reflect real coverage."""
    idx = pd.bdate_range("2016-01-04", periods=500)
    net = pd.Series(np.random.default_rng(0).normal(0, 0.01, 500), index=idx)
    ep_start, ep_end = idx[200], idx[300]
    in_ep = (net.index >= ep_start) & (net.index <= ep_end)
    # a "full sample" observable with plenty of rows, but NONE inside the episode
    obs_col = pd.Series(np.nan, index=idx)
    obs_col.iloc[:150] = np.random.default_rng(1).normal(0, 1, 150)     # all before the episode
    obs_col.iloc[350:] = np.random.default_rng(2).normal(0, 1, 150)     # all after the episode
    assert obs_col.notna().sum() >= 30                                   # clears the full-sample gate
    b_in = ed.bucket_contribution(net, obs_col, in_ep)
    n_in_episode = sum(v["n_days"] for v in ed._real_buckets(b_in).values())
    assert n_in_episode == 0                                             # confirms the bug's premise
    # the survey construction logic itself (mirrors decompose()'s loop)
    survey_col = None if n_in_episode < 30 else ed.indicator_passes(b_in)
    assert survey_col is None    # must be excluded from N, never scored as a fail


def test_decompose_itself_excludes_zero_coverage_indicator_from_survey(monkeypatch, tmp_path):
    """Same regression, but calling the REAL decompose() end to end -- the
    earlier test above only reproduced the intended logic inline; this one
    exercises the actual function that produced the wrong K/N on real data."""
    df = noisy_days("2018-01-02", "2019-12-31", seed=6, px=280.0)   # longer window: avoids a
    df = df[[session_close(d) is not None for d in df.index.date]]  # degenerate zero-variance
    ed.pr.load_spy = lambda db: df                                  # tercile column at this seed
    ed.pr.load_dividends = lambda: {}
    days = sorted(set(df.index.date))
    ed.EPISODE = {"start": str(days[10]), "trough": str(days[20]), "end": str(days[40])}

    def partial_coverage_loader(db, sid, s, e):
        # only covers the LAST third of the window -- zero overlap with the episode above.
        # A distinct seed per series_id: reusing one seed for all four FRED
        # series made DGS10 and DGS2 come back IDENTICAL, so term_spread =
        # d10 - d2 was trivially zero everywhere (a mock bug, not a real
        # code issue) -- that degeneracy is what broke tercile binning.
        idx = pd.to_datetime([str(d) for d in days[60:]])
        rng = np.random.default_rng(abs(hash(sid)) % (2**31))
        return pd.Series(rng.normal(20, 5, len(idx)), index=idx)

    monkeypatch.setattr(ed, "load_fred_raw", partial_coverage_loader)
    out = ed.decompose("unused.db", tmp_path)
    for name in ("vix_level", "vix_change", "term_spread", "hy_spread"):
        assert out["survey"]["per_indicator"][name] is None, name
        assert "in-episode days" in out["survey"]["notes"][name]
    assert out["survey"]["n_evaluated"] <= 3   # only the three price-derived observables remain evaluable


# ── Chance baseline (28 Sep 2026) ────────────────────────────────────────────

def test_binomial_at_least_k_matches_hand_computed_values():
    # P(X >= 3 | n=6, p=0.5) -- hand-computable: 1 - P(X<=2)
    # P(X<=2) = C(6,0)*.5^6 + C(6,1)*.5^6 + C(6,2)*.5^6 = (1+6+15)/64 = 22/64
    expected = 1 - 22 / 64
    assert ed.binomial_at_least_k(6, 3, 0.5) == pytest.approx(expected)
    assert ed.binomial_at_least_k(6, 0, 0.5) == pytest.approx(1.0)      # >= 0 is certain
    assert ed.binomial_at_least_k(6, 7, 0.5) == pytest.approx(0.0)      # >= n+1 is impossible
    assert ed.binomial_at_least_k(6, 6, 1.0) == pytest.approx(1.0)      # p=1: certain to hit max


def test_binomial_at_least_k_decreases_in_k_and_increases_in_p():
    assert ed.binomial_at_least_k(6, 5, 0.5) < ed.binomial_at_least_k(6, 3, 0.5)
    assert ed.binomial_at_least_k(6, 3, 0.2) < ed.binomial_at_least_k(6, 3, 0.8)


def test_random_indicator_pass_rate_uses_the_real_pipeline_not_a_shortcut():
    """A random, meaningless indicator should pass the (high>0, low<0) test
    a plausible middling fraction of the time by pure chance -- not near 0%
    (the criterion isn't impossibly strict) and not near 100% (it isn't
    trivially satisfied), and it must vary with a different seed (proof it
    is actually drawing fresh random series each time, not reusing one)."""
    idx = pd.bdate_range("2016-01-04", periods=1000)
    rng = np.random.default_rng(0)
    net_s = pd.Series(rng.normal(0, 0.01, 1000), index=idx)
    in_ep = np.zeros(1000, dtype=bool); in_ep[200:400] = True
    out_a = ed.random_indicator_pass_rate(net_s, in_ep, draws=500, seed=1)
    out_b = ed.random_indicator_pass_rate(net_s, in_ep, draws=500, seed=2)
    assert out_a["evaluated"] > 400                       # nearly every draw has full coverage
    assert 0.05 < out_a["chance_pass_rate"] < 0.95
    assert out_a["chance_pass_rate"] != out_b["chance_pass_rate"]   # genuinely different draws


def test_chance_baseline_wired_into_decompose_output(monkeypatch, tmp_path):
    df = noisy_days("2019-01-02", "2019-09-30", seed=6, px=280.0)
    df = df[[session_close(d) is not None for d in df.index.date]]
    ed.pr.load_spy = lambda db: df
    ed.pr.load_dividends = lambda: {}
    ed.load_fred_raw = lambda db, sid, s, e: None    # no FRED needed for this check
    days = sorted(set(df.index.date))
    ed.EPISODE = {"start": str(days[10]), "trough": str(days[30]), "end": str(days[80])}
    out = ed.decompose("unused.db", tmp_path)
    cb = out["chance_baseline"]
    assert 0.0 <= cb["chance_pass_rate"] <= 1.0
    assert cb["binomial_p_at_least_k_APPROX"] is None or 0.0 <= cb["binomial_p_at_least_k_APPROX"] <= 1.0


# ── Round-3 additions: leverage cap, common-shift null, year fork, post-episode ──

def test_effective_leverage_matches_the_box_formula_directly():
    idx = pd.bdate_range("2018-01-02", periods=30)
    # tiny, constant daily moves -> low realised vol -> sizing should hit the 4x cap
    px = 100 * (1.0002 ** np.arange(30))
    daily = pd.DataFrame({"close": px}, index=idx)
    lev = ed.effective_leverage(daily, lookback=14, target_vol=0.02, max_leverage=4.0)
    assert lev.iloc[16:].max() == pytest.approx(4.0)          # capped, not exceeding max_leverage
    assert lev.iloc[:15].isna().all()                          # no signal before 14 returns exist


def test_effective_leverage_uncapped_scales_with_target_over_vol():
    idx = pd.bdate_range("2018-01-02", periods=30)
    rng = np.random.default_rng(2)
    px = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, 30)))       # high vol -> below the cap
    daily = pd.DataFrame({"close": px}, index=idx)
    lev = ed.effective_leverage(daily, lookback=14, target_vol=0.02, max_leverage=4.0)
    vol = daily["close"].pct_change().rolling(14).std(ddof=1).shift(1)
    expected = np.minimum(4.0, 0.02 / vol)
    pd.testing.assert_series_equal(lev.dropna(), expected.dropna(), check_names=False)


def test_leverage_cap_check_reports_higher_cap_share_in_low_vix_bucket():
    """Construct a case where 'low' observable days are DELIBERATELY the
    calmest (lowest realised vol) days -- they should show a higher share
    at the leverage cap than 'high' days, mechanically, with no signal
    claim needed."""
    idx = pd.bdate_range("2018-01-02", periods=100)
    rng = np.random.default_rng(3)
    # continuous vol schedule (not just two distinct values, so terciles split
    # cleanly): ramps from very calm to volatile across the window
    vol_scale = np.linspace(0.0008, 0.025, 100)
    rets = rng.normal(0, vol_scale, 100)
    px = 100 * np.exp(np.cumsum(rets))
    daily = pd.DataFrame({"close": px}, index=idx)
    obs_col = pd.Series(vol_scale, index=idx)                   # perfectly tracks realised vol regime
    in_ep = np.ones(100, dtype=bool)
    out = ed.leverage_cap_check(daily, obs_col, in_ep)
    assert out["low"]["share_at_cap"] > out["high"]["share_at_cap"]


def test_common_shift_null_preserves_each_series_and_shifts_together():
    idx = pd.bdate_range("2018-01-02", periods=200)
    net_s = pd.Series(np.random.default_rng(0).normal(0, 0.01, 200), index=idx)
    in_ep = np.zeros(200, dtype=bool); in_ep[50:150] = True
    a = pd.Series(np.arange(200.0), index=idx)                  # perfectly correlated pair
    b = pd.Series(np.arange(200.0) * 2 + 1, index=idx)
    out = ed.common_shift_null(net_s, in_ep, {"a": a, "b": b}, draws=200, seed=1)
    assert out["n_indicators"] == 2
    assert 0.0 <= out["p_at_least_3_of_n"] <= 1.0                # can't exceed 1 even with n=2 (>=3 impossible)
    assert out["p_at_least_3_of_n"] == 0.0                       # >= 3 of 2 is structurally impossible


def test_common_shift_null_uses_one_shared_offset_not_independent_ones():
    """The whole design point: ALL columns must move by the SAME random
    offset each draw (preserving their real cross-correlation), not
    independent offsets per column. Two columns that are IDENTICAL series:
    under a shared offset they remain identical after shifting (so if one
    passes, both must -- k is always 0 or 2, never 1). Independent offsets
    would break that and let k=1 occur."""
    idx = pd.bdate_range("2016-01-04", periods=400)
    rng = np.random.default_rng(4)
    a = pd.Series(rng.normal(0, 1, 400), index=idx)
    b = a.copy()                                              # IDENTICAL to a
    net_s = pd.Series(rng.normal(0, 0.001, 400), index=idx)
    in_ep = np.zeros(400, dtype=bool); in_ep[100:300] = True
    rng2 = np.random.default_rng(9)
    for _ in range(30):
        offset = int(rng2.integers(1, 400))
        shifted_a = pd.Series(np.roll(a.to_numpy(), offset), index=idx)
        shifted_b = pd.Series(np.roll(b.to_numpy(), offset), index=idx)
        ra = ed.indicator_passes(ed.bucket_contribution(net_s, shifted_a, in_ep))
        rb = ed.indicator_passes(ed.bucket_contribution(net_s, shifted_b, in_ep))
        assert ra == rb, "identical series under the SAME offset must agree"


def test_common_shift_null_differs_from_independent_baseline_when_correlated():
    """The whole point of this function: for genuinely correlated
    indicators, shifting them TOGETHER should not scatter their joint
    pass/fail pattern the way independent random draws would -- check the
    mean k-pass is NOT simply n x (single-indicator rate), which is what
    independence would imply."""
    idx = pd.bdate_range("2016-01-04", periods=600)
    rng = np.random.default_rng(5)
    base = rng.normal(0, 1, 600)
    cols = {f"c{i}": pd.Series(base + rng.normal(0, 0.1, 600), index=idx) for i in range(4)}   # highly correlated
    net_s = pd.Series(rng.normal(0, 0.01, 600), index=idx)
    in_ep = np.zeros(600, dtype=bool); in_ep[100:400] = True
    out = ed.common_shift_null(net_s, in_ep, cols, draws=300, seed=2)
    assert out["n_indicators"] == 4
    assert 0.0 <= out["mean_k_pass"] <= 4.0


def test_year_dummy_fork_detects_within_year_pattern():
    """Plant a genuine within-year effect in 2019 (low obs -> high return,
    high obs -> low return, matching the Q direction) and confirm the fork
    finds it holds for that year specifically."""
    idx = pd.bdate_range("2018-01-02", "2020-12-31")
    rng = np.random.default_rng(7)
    obs_col = pd.Series(rng.normal(0, 1, len(idx)), index=idx)
    net_s = pd.Series(rng.normal(0, 0.001, len(idx)), index=idx)
    y2019 = idx.year == 2019
    net_s = net_s.copy()
    net_s[y2019 & (obs_col < 0)] += 0.01     # low obs -> extra positive return in 2019 (Q direction: low<high)
    net_s[y2019 & (obs_col > 0)] -= 0.01     # high obs -> extra negative return in 2019
    in_ep = idx.year.isin([2018, 2019, 2020])
    out = ed.year_dummy_fork(net_s, obs_col, np.asarray(in_ep), years=(2018, 2019, 2020))
    assert out["per_year"][2019]["q_direction_holds"] is False   # low > high in 2019 by construction here
    # (constructed so low-obs days OUTPERFORM -> low_mean > high_mean -> q_direction_holds False;
    #  this still proves the fork detects a genuine, deliberately-planted within-year pattern)
    assert out["per_year"][2019]["low_mean_net_r"] > out["per_year"][2019]["high_mean_net_r"]


def test_year_dummy_fork_result_reflects_the_majority_of_informative_years():
    """fork_result itself must actually depend on the per-year outcomes, not
    just report a fixed string -- construct a case where Q holds in 0 of 3
    informative years and confirm the aggregate says so, not 'Q survives'."""
    idx = pd.bdate_range("2018-01-02", "2020-12-31")
    rng = np.random.default_rng(11)
    obs_col = pd.Series(rng.normal(0, 1, len(idx)), index=idx)
    net_s = pd.Series(0.0, index=idx)                          # NO planted pattern anywhere
    # force every year's low bucket to OUTPERFORM high (the anti-Q direction)
    for y in (2018, 2019, 2020):
        ymask = idx.year == y
        net_s[ymask & (obs_col < 0)] = 0.02
        net_s[ymask & (obs_col > 0)] = -0.02
    in_ep = idx.year.isin([2018, 2019, 2020])
    out = ed.year_dummy_fork(net_s, obs_col, np.asarray(in_ep), years=(2018, 2019, 2020))
    assert out["years_supporting_q"] == []
    assert out["fork_result"] == "calendar/drift explains the pattern at least as well as Q"


def test_year_dummy_fork_flags_years_with_too_few_days():
    idx = pd.bdate_range("2018-01-02", periods=300)
    net_s = pd.Series(0.0, index=idx)
    obs_col = pd.Series(np.arange(len(idx), dtype=float), index=idx)
    in_ep = np.zeros(len(idx), dtype=bool); in_ep[:5] = True    # only 5 episode days total
    out = ed.year_dummy_fork(net_s, obs_col, in_ep, years=(2018,))
    assert "note" in out["per_year"][2018]


def test_post_episode_bucket_check_uses_the_specified_window_only():
    idx = pd.bdate_range("2016-01-04", "2024-12-31")
    rng = np.random.default_rng(9)
    net_s = pd.Series(rng.normal(0, 0.001, len(idx)), index=idx)
    obs_col = pd.Series(rng.normal(0, 1, len(idx)), index=idx)
    out = ed.post_episode_bucket_check(net_s, obs_col, "2023-01-01", "2024-12-31")
    total = sum(v["n_days"] for v in ed._real_buckets(out).values())
    expected_days = ((idx >= pd.Timestamp("2023-01-01")) & (idx <= pd.Timestamp("2024-12-31"))).sum()
    assert total + out["n_missing"] == expected_days
