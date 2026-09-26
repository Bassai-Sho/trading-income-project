"""P2-123 slice 3b: grid-level gates (N_eff, DSR, PBO, walk-forward, plateau,
cost stress). Synthetic daily-R grids with known answers."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from evaluation.robustness import Grid, evaluate_grid, n_eff

GRID = Grid({"a": (1, 2, 3), "b": (1, 2, 3), "c": (1, 2, 3)})
DAYS = pd.bdate_range("2016-01-04", "2022-12-30")


def _grid_returns(edge_fn, seed=0, trade_prob=0.5, common=0.6, sd=1.0):
    """Daily net R per variant: shared market noise + own noise + edge_fn(key, dates).
    Days without a trade are exactly 0 for every variant (zero inflation)."""
    rng = np.random.default_rng(seed)
    traded = rng.random(len(DAYS)) < trade_prob
    shared = rng.normal(0, sd, len(DAYS))
    out, cost = {}, {}
    for k in GRID.keys():
        x = common * shared + np.sqrt(1 - common ** 2) * rng.normal(0, sd, len(DAYS))
        x = (x + edge_fn(k, DAYS)) * traded
        out[k] = pd.Series(x, index=DAYS)
        cost[k] = pd.Series(0.05 * traded, index=DAYS)
    return out, cost


def plateau_edge(k, d):             # strongest at the centre, positive for all neighbours
    return 0.25 - 0.04 * sum(abs(v - 2) for v in k)


def test_planted_plateau_edge_passes_every_gate():
    r, c = _grid_returns(plateau_edge)
    rep = evaluate_grid(GRID, r, c, prior_trials=14, pbo_splits=10)
    assert rep.passed, rep.gates
    assert rep.best == (2, 2, 2) and rep.pbo <= 0.15 and rep.dsr >= 0.95


def test_pure_noise_fails():
    r, c = _grid_returns(lambda k, d: 0.0, seed=4)
    rep = evaluate_grid(GRID, r, c, prior_trials=14, pbo_splits=10)
    assert not rep.passed and not rep.gates["dsr"]


def test_lone_spike_fails_plateau():
    r, c = _grid_returns(lambda k, d: 0.30 if k == (3, 1, 2) else -0.02, seed=5)
    rep = evaluate_grid(GRID, r, c, pbo_splits=10)
    assert rep.best == (3, 1, 2) and not rep.gates["plateau"] and not rep.passed


def test_edge_that_stops_after_2018_fails_walk_forward():
    r, c = _grid_returns(lambda k, d: np.where(d.year <= 2018, 0.4 * plateau_edge(k, d) / 0.25, -0.05),
                         seed=6)
    rep = evaluate_grid(GRID, r, c, pbo_splits=10)
    assert not rep.gates["walk_forward"] and not rep.passed


def test_thin_edge_fails_cost_stress():
    r, c = _grid_returns(lambda k, d: 0.005, seed=7, sd=0.02)
    c = {k: v * 1.0 for k, v in c.items()}            # costs 0.05/trade day vs edge 0.005
    rep = evaluate_grid(GRID, r, c, pbo_splits=10)
    assert rep.cost_stress_mean < 0 and not rep.gates["cost_stress"]


def test_missing_cost_series_fails_closed():
    r, _ = _grid_returns(plateau_edge)
    rep = evaluate_grid(GRID, r, None, pbo_splits=10)
    assert not rep.gates["cost_stress"] and not rep.passed


# ── N_eff ─────────────────────────────────────────────────────────────────────

def test_n_eff_independent_vs_near_identical_and_floor():
    rng = np.random.default_rng(1)
    indep = rng.normal(0, 1, (27, 1500))
    base = rng.normal(0, 1, 1500)
    same = np.array([base + rng.normal(0, 0.05, 1500) for _ in range(27)])
    assert n_eff(indep) > 22 and n_eff(same) < 1.5
    assert GRID.floor() == 7


def test_n_eff_ignores_days_on_which_no_variant_traded():
    rng = np.random.default_rng(2)
    x = rng.normal(0, 1, (10, 800))
    padded = np.concatenate([x, np.zeros((10, 1200))], axis=1)
    assert n_eff(padded) == pytest.approx(n_eff(x))


def test_n_eff_used_in_dsr_is_floored_and_adds_prior_trials():
    r, c = _grid_returns(plateau_edge, common=0.99)     # near-identical variants
    rep = evaluate_grid(GRID, r, c, prior_trials=14, pbo_splits=10)
    assert rep.n_eff_raw < GRID.floor() and rep.n_trials_used == GRID.floor() + 14


def test_neighbours_are_one_step_on_one_axis():
    assert sorted(GRID.neighbours((2, 2, 2))) == sorted(
        [(1, 2, 2), (3, 2, 2), (2, 1, 2), (2, 3, 2), (2, 2, 1), (2, 2, 3)])
    assert sorted(GRID.neighbours((1, 1, 1))) == [(1, 1, 2), (1, 2, 1), (2, 1, 1)]


# ── DSR: library vs toolkit ──────────────────────────────────────────────────

def test_purgedcv_dsr_matches_toolkit_given_same_inputs():
    import purgedcv
    from scipy import stats
    import trading_quant_toolkit_v2_4 as tk
    for seed, (mu, n, N) in enumerate([(0.08, 800, 14), (0.12, 1500, 100), (0.03, 500, 10)]):
        r = np.random.default_rng(seed).standard_t(5, n) * 0.9 + mu
        sr = r.mean() / r.std(ddof=1)
        a = tk.deflated_sharpe_ratio(sr, n, N, skew=stats.skew(r),
                                     kurtosis=stats.kurtosis(r, fisher=False))
        assert purgedcv.deflated_sharpe_ratio(r, N, 1 / (n - 1)) == pytest.approx(a, abs=0.01)


def test_walk_forward_never_sees_the_test_year():
    """Each year a DIFFERENT variant has a spectacular year. Choosing on prior
    years cannot know which, so out-of-fold performance must stay near zero; a
    selection that peeked at the test year would pick the winner every time."""
    from evaluation.robustness import walk_forward
    rng = np.random.default_rng(9)
    keys = GRID.keys()
    mat = rng.normal(0, 1, (len(keys), len(DAYS)))
    for i, y in enumerate(sorted(set(DAYS.year))):
        mat[i, DAYS.year == y] += 3.0
    wfe, oos_mean, folds = walk_forward(mat, DAYS, keys, min_train_years=3)
    assert abs(oos_mean) < 0.2
    assert all(f["pick"] != keys[i] for i, f in
               zip(range(3, 3 + len(folds)), folds))
