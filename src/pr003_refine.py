"""
pr003_refine.py — PR-003 Stage REFINE  (Notion P2-130 / P2-126, unblocked 27 Sep 2026)
======================================================================================
Runs a coarse grid of box #3 variants on 2016-2024 ONLY (2025+ is never read —
enforced by pr003_replication.WINDOW) and grades it with the box-agnostic
evaluate_grid (src/evaluation/robustness.py, slice 3b) — reused, not
duplicated: PBO/DSR from purgedcv, N_eff/walk-forward/plateau/cost-stress ours.

GRID (brackets the published values; sizing fixed at "vol_target", the paper's
headline variant — not itself a grid axis here):
    lookback (days):        7, 14, 28, 56      (published: 14)
    band multiplier (vm):   0.75, 1.0, 1.25, 1.5  (published: 1.0)
    decision interval (min): 15, 30, 60        (published: 30)
48 variants. Costs: VIABILITY (commission + spread + slippage) throughout —
refinement is judged under realistic costs, not the paper's optimistic ones.

ADOPTION RULE (frozen in P2-130): a refined variant replaces the published one
ONLY if ALL of:
  1. its mean daily return beats the published variant's;
  2. every one-step grid neighbour of the best variant is ALSO net positive
     (plateau — no lone spike);
  3. PBO <= 0.15;
  4. it survives cost stress (+20%) with mean daily return > 0.
Otherwise the published (14, 1.0, 30) stands. The paper's own reported
"optimal" (lookback 90, vm 1.5) is never adopted: 90 sits outside this grid by
design (found on data overlapping our 2023-24 window), and 1.5 must earn its
place through these gates like every other value, not by citation.

    python src/pr003_refine.py
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from boxes.noise_area_momentum import NoiseAreaMomentumBox, Params   # noqa: E402
from core.backtest_runner import run_box                             # noqa: E402
from core.execution_core import AccountConfig                        # noqa: E402
from evaluation.robustness import Grid, evaluate_grid                # noqa: E402
import pr003_replication as pr                                       # noqa: E402

log = logging.getLogger("pr003_refine")

GRID = Grid({"lookback": (7, 14, 28, 56), "vm": (0.75, 1.0, 1.25, 1.5),
            "trade_every_min": (15, 30, 60)})
PUBLISHED: tuple = (14, 1.0, 30)
# Trials already spent before this grid (P2-127 DSR N=15; +2 box #3 cost/sizing
# variants tested in Stage R = 18) — carried forward per the Rulebook's DSR rule.
PRIOR_TRIALS = 18


def run_variant(df: pd.DataFrame, divs: dict, lookback: int, vm: float,
                interval: int) -> tuple[pd.Series, pd.Series]:
    """Daily net return and daily cost-as-fraction-of-prior-equity for one variant."""
    p = Params(lookback=lookback, vm=vm, trade_every_min=interval,
              sizing="vol_target", dividends=divs)
    res = run_box(NoiseAreaMomentumBox(), p, df, "SPY", AccountConfig(cash=pr.AUM0, leverage=None),
                  fee_fn=pr.fee_fn("viability"), keep_reports=True)
    r = pr.daily_returns(res.equity_by_day)
    eq_prev = pd.Series(res.equity_by_day).sort_index().shift(1).fillna(pr.AUM0)
    fees = {}
    for rep in res.reports:
        if rep.status == "FILLED":
            d = pd.Timestamp(rep.timestamp).date()
            fees[d] = fees.get(d, 0.0) + rep.fee
    cost = pd.Series({d: fees.get(pd.Timestamp(d).date(), 0.0) / eq_prev.get(d, pr.AUM0)
                      for d in r.index})
    return r, cost


def run(market_db: str, out_dir: Path = Path("DATA/pr003")) -> dict:
    df = pr.load_spy(market_db)
    assert df.index.max().date() <= pr.WINDOW[1], "sealed window must not be read"
    divs = pr.load_dividends()
    daily_r, daily_cost = {}, {}
    keys = GRID.keys()
    for i, k in enumerate(keys, 1):
        log.info("variant %d/%d: lookback=%s vm=%s interval=%s", i, len(keys), *k)
        daily_r[k], daily_cost[k] = run_variant(df, divs, *k)
    rep = evaluate_grid(GRID, daily_r, daily_cost, prior_trials=PRIOR_TRIALS)
    pub_mean = float(daily_r[PUBLISHED].mean())
    best_mean = rep.best_mean_daily_r
    adopt = (best_mean > pub_mean and rep.gates["plateau"] and rep.gates["pbo"]
            and rep.gates["cost_stress"])
    out = {"run_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
          "grid": {k: list(v) for k, v in GRID.axes.items()}, "published": PUBLISHED,
          "published_mean_daily_r": pub_mean, "prior_trials": PRIOR_TRIALS,
          "report": asdict(rep), "adopt_refinement": adopt,
          "final_variant": rep.best if adopt else PUBLISHED}
    try:
        out["commit"] = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                                       text=True, timeout=5).stdout.strip()
    except Exception:
        out["commit"] = None
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"refine_{out['run_at'].replace(':', '')}.json"
    path.write_text(json.dumps(out, indent=2, default=str))
    out["path"] = str(path)
    return out


def show(o: dict) -> None:
    r = o["report"]
    print(f"\n=== PR-003 Stage REFINE: {len(GRID.keys())}-variant grid on SPY 2016-2024 (commit {o['commit']}) ===")
    print(f"grid: lookback {GRID.axes['lookback']}, vm {GRID.axes['vm']}, "
          f"interval(min) {GRID.axes['trade_every_min']}")
    print(f"published (14, 1.0, 30): mean daily r {o['published_mean_daily_r']:+.4%}")
    print(f"grid best {r['best']}: mean daily r {r['best_mean_daily_r']:+.4%}, "
          f"daily Sharpe {r['best_sharpe_daily']:.2f}")
    print(f"N_eff {r['n_eff_raw']:.1f} (floor {r['n_eff_floor']}), trials used {r['n_trials_used']} "
          f"({o['prior_trials']} prior + this grid)")
    print(f"DSR {r['dsr']:.3f} (gate >= 0.95)   PBO {r['pbo']:.3f} (gate <= 0.15)")
    wf = "n/a (< min_train_years)" if r["wfe"] is None else f"{r['wfe']:.2f}"
    print(f"walk-forward efficiency {wf} (gate >= 0.5), OOS-fold mean {r['wf_oos_mean']}")
    print(f"plateau (best's neighbours all > 0): {r['plateau_ok']} -> {r['plateau_neighbours']}")
    print(f"cost stress (+20%): best mean daily r {r['cost_stress_mean']:+.4%} (gate > 0)")
    print(f"\nGATES: {r['gates']}")
    print(f"ADOPT REFINEMENT: {o['adopt_refinement']}  ->  FINAL VARIANT for Stage V: {o['final_variant']}")
    print(f"Saved: {o['path']}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    for noisy in ("market_data_store", "execution_core"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    a = argparse.ArgumentParser()
    a.add_argument("--db", default="DATA/market_data.db")
    args = a.parse_args()
    show(run(args.db))
