"""
evaluation/robustness.py  (Phase A, P2-123 — slice 3b)
======================================================
Grades a GRID of parameter variants of one box, AFTER the chosen variant has
passed the null-alpha gate (evaluation/null_baseline.py). Box-agnostic: input
is each variant's daily net-R series (and optionally its daily cost series).

Reused, not rebuilt: PBO (CSCV) and the Deflated Sharpe Ratio come from
`purgedcv` (MIT, tested; PBO verified here to average 0.48 on noise and ~0.12
with a planted edge). Its DSR agrees with trading_quant_toolkit's to < 0.01
when given the same inputs (test), and uses the true cross-trial Sharpe
variance, which a grid provides.

Ours (not in the library): N_eff, walk-forward by calendar year, plateau and
cost stress.

GATES (design record: handover doc section 10; review rounds 1-3)
  dsr       >= 0.95  DSR of the in-sample best variant, with n_trials =
                     max(N_eff, floor) + prior trials from the ledger
  pbo       <= 0.15  (0.50 = no information)
  walk-fwd  WFE >= 0.5 and mean out-of-fold daily R > 0 (primary out-of-fold gate)
  plateau   every one-step grid neighbour of the best variant has mean daily R > 0
  cost      the best variant's mean daily R stays > 0 with costs 20% higher

N_eff = participation ratio (sum lambda)^2 / sum lambda^2 of the eigenvalues
of the variants' daily-return correlation matrix, computed on ACTIVE days only
(days on which at least one variant traded — review round 2, zero-inflation).
Floor: sum over axes of (levels - 1) + 1, the main-effect degrees of freedom
of the grid (7 for a 3x3x3 grid). CORRECTION: the decision record's wording
"not below the number of distinct grid settings" would equal the raw count and
make N_eff meaningless; this is the intended floor.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Hashable

import numpy as np
import pandas as pd
import purgedcv

Key = tuple


@dataclass(frozen=True)
class Grid:
    """Ordered parameter axes, e.g. {"orb_min": (5, 15, 30), "stop_atr": (0.05, 0.1, 0.2)}.
    A variant key is the tuple of values in axis order."""
    axes: dict[str, tuple]

    def keys(self) -> list[Key]:
        return list(product(*self.axes.values()))

    def floor(self) -> int:
        return sum(len(v) - 1 for v in self.axes.values()) + 1

    def neighbours(self, key: Key) -> list[Key]:
        out = []
        for i, vals in enumerate(self.axes.values()):
            j = vals.index(key[i])
            for k in (j - 1, j + 1):
                if 0 <= k < len(vals):
                    out.append(key[:i] + (vals[k],) + key[i + 1:])
        return out


@dataclass
class GridReport:
    best: Key
    n_variants: int
    n_eff_raw: float
    n_eff_floor: int
    prior_trials: int
    n_trials_used: int
    best_mean_daily_r: float
    best_sharpe_daily: float
    dsr: float
    pbo: float
    wfe: float | None
    wf_oos_mean: float | None
    wf_folds: list[dict]
    plateau_ok: bool
    plateau_neighbours: dict
    cost_stress_mean: float | None
    gates: dict[str, bool]
    passed: bool
    notes: list[str] = field(default_factory=list)


def _sharpe(x: np.ndarray) -> float:
    s = x.std(ddof=1)
    return float(x.mean() / s) if s > 0 else 0.0


def n_eff(returns: np.ndarray) -> float:
    """Participation ratio of the correlation eigenvalues on active days."""
    active = (returns != 0).any(axis=0)
    r = returns[:, active]
    r = r[r.std(axis=1) > 0]                    # a variant that never trades adds no trial
    if len(r) < 2:
        return float(len(r))
    lam = np.clip(np.linalg.eigvalsh(np.corrcoef(r)), 0.0, None)
    return float(lam.sum() ** 2 / (lam ** 2).sum())


def walk_forward(mat: np.ndarray, dates: pd.DatetimeIndex, keys: list[Key],
                 min_train_years: int = 3) -> tuple[float | None, float | None, list[dict]]:
    """Expanding window by calendar year: pick the best variant (daily Sharpe)
    on all years before Y, record its Sharpe in-sample and in year Y."""
    years = sorted(set(dates.year))
    folds, oos_all = [], []
    for y in years[min_train_years:]:
        tr, te = dates.year < y, dates.year == y
        is_sr = np.array([_sharpe(m) for m in mat[:, tr]])
        b = int(is_sr.argmax())
        oos = mat[b, te]
        oos_all.append(oos)
        folds.append({"test_year": y, "pick": keys[b], "is_sharpe": float(is_sr[b]),
                      "oos_sharpe": _sharpe(oos), "oos_mean": float(oos.mean())})
    if not folds:
        return None, None, []
    is_m = np.mean([f["is_sharpe"] for f in folds])
    wfe = float(np.mean([f["oos_sharpe"] for f in folds]) / is_m) if is_m > 0 else None
    return wfe, float(np.concatenate(oos_all).mean()), folds


def evaluate_grid(grid: Grid, daily_r: dict[Key, pd.Series],
                  daily_cost: dict[Key, pd.Series] | None = None, *, prior_trials: int = 0,
                  pbo_splits: int = 16, min_train_years: int = 3,
                  dsr_min: float = 0.95, pbo_max: float = 0.15, wfe_min: float = 0.5,
                  cost_mult: float = 1.2) -> GridReport:
    """daily_r[key]: net R per session for that variant (0 on no-trade days).
    Every key in the grid must be present. Dates are aligned on their union."""
    keys = grid.keys()
    missing = [k for k in keys if k not in daily_r]
    if missing:
        raise ValueError(f"grid variants without results: {missing[:3]}")
    frame = pd.DataFrame({k: daily_r[k] for k in keys}).fillna(0.0).sort_index()
    dates = pd.DatetimeIndex(frame.index)
    mat = frame.to_numpy().T                     # (n_variants, n_days)
    notes = []

    sr = np.array([_sharpe(m) for m in mat])
    b = int(sr.argmax())
    best = keys[b]
    ne = n_eff(mat)
    floor = grid.floor()
    n_used = int(round(max(ne, floor))) + prior_trials
    var_sr = float(np.var(sr, ddof=1)) if len(sr) > 1 else 0.0
    if var_sr <= 0:
        notes.append("zero Sharpe variance across variants; DSR uses 1/(n-1)")
        var_sr = 1.0 / (mat.shape[1] - 1)
    dsr = float(purgedcv.deflated_sharpe_ratio(mat[b], n_used, var_sr))
    pbo = float(purgedcv.probability_of_backtest_overfitting(mat, n_splits=pbo_splits).pbo)
    wfe, wf_mean, folds = walk_forward(mat, dates, keys, min_train_years)

    means = dict(zip(keys, mat.mean(axis=1)))
    neigh = {k: float(means[k]) for k in grid.neighbours(best)}
    plateau_ok = bool(neigh) and all(v > 0 for v in neigh.values())
    stress = None
    if daily_cost is not None and best in daily_cost:
        c = daily_cost[best].reindex(frame.index).fillna(0.0).to_numpy()
        stress = float((mat[b] - (cost_mult - 1.0) * c).mean())
    else:
        notes.append("no cost series supplied: cost stress not evaluated (gate fails)")

    gates = {"dsr": dsr >= dsr_min, "pbo": pbo <= pbo_max,
             "walk_forward": wfe is not None and wfe >= wfe_min and (wf_mean or 0) > 0,
             "plateau": plateau_ok, "cost_stress": stress is not None and stress > 0}
    return GridReport(best=best, n_variants=len(keys), n_eff_raw=ne, n_eff_floor=floor,
                      prior_trials=prior_trials, n_trials_used=n_used,
                      best_mean_daily_r=float(mat[b].mean()), best_sharpe_daily=float(sr[b]),
                      dsr=dsr, pbo=pbo, wfe=wfe, wf_oos_mean=wf_mean, wf_folds=folds,
                      plateau_ok=plateau_ok, plateau_neighbours=neigh, cost_stress_mean=stress,
                      gates=gates, passed=all(gates.values()), notes=notes)
