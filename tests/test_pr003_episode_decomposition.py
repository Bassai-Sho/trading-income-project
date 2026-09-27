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
    assert sum(v["sum_net_r"] for v in b.values()) == pytest.approx(net.sum())
    assert sum(v["n_days"] for v in b.values()) == 300


def test_bucket_contribution_respects_the_mask():
    net = pd.Series(1.0, index=pd.bdate_range("2018-01-02", periods=100))    # every day contributes 1.0
    obs = pd.Series(np.arange(100.0), index=net.index)
    mask = np.zeros(100, dtype=bool); mask[:20] = True     # only the first 20 days count
    b = ed.bucket_contribution(net, obs, mask)
    assert sum(v["n_days"] for v in b.values()) == 20


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
