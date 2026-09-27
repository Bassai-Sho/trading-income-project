"""
pr003_stage_v.py — PR-003 Stage V: viability  (Notion P2-130, frozen 27 Sep 2026)
=================================================================================
Runs the FINAL variant from Stage REFINE (the published (14, 1.0, 30), since
refinement was rejected on 27 Sep 2026 — PBO 0.405) under MES-equivalent
futures costs, and checks the three Stage V requirements:

  GATE 1  null-alpha (exposure form, evaluation/null_exposure.py): does the
          strategy's chosen DAILY DIRECTION beat a coin flip with the same
          trade timing, sizing and costs? p < 0.05.
  GATE 2  correlation with plain buy-and-hold SPY (daily returns) <= 0.30 —
          the guardrail that this book adds to, rather than duplicates,
          Adam's existing long portfolio.
  GATE 3  positive net expectancy under MES costs (viability itself).

REPORTED, NOT GATED (per the 27 Sep Rulebook amendment — risk/capital
settings come after viability, not as a research gate): max-drawdown /
annualised-volatility ratio, at the paper's own 14.3% vol and rescaled to a
10% target for reference. This is a deliberate, stated design choice: Stage V
answers "is there a real, useful edge?", not "how much should be risked?".

Window: 2016-2024 only (2025+ enforced sealed via pr003_replication.WINDOW).

    python src/pr003_stage_v.py
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
import pr003_refine as rf                                            # noqa: E402
import pr003_replication as pr                                        # noqa: E402
from evaluation.mes_costs import (COST_PER_CONTRACT_SIDE, breakeven_cost_per_contract_side,  # noqa: E402
                                  mes_fee_fn)
from evaluation.null_exposure import null_alpha_exposure               # noqa: E402

log = logging.getLogger("pr003_stage_v")
FINAL_VARIANT = (14, 1.0, 30)          # from the Stage REFINE result (refinement rejected)
CORR_MAX = 0.30
MDD_VOL_REF = 1.75                     # the paper's own MDD/annualised-vol ratio, for context only


def spy_buy_hold_daily(df: pd.DataFrame) -> pd.Series:
    close = df.groupby(df.index.date)["Close"].last()
    close.index = pd.to_datetime(close.index)
    return close.pct_change().dropna()


def stats(r: pd.Series) -> dict:
    eq = (1 + r).cumprod()
    vol = float(r.std(ddof=1) * np.sqrt(252))
    mdd = float((eq / eq.cummax() - 1).min())
    return {"sharpe": float(r.mean() / r.std(ddof=1) * np.sqrt(252)) if r.std(ddof=1) > 0 else float("nan"),
           "ann_vol": vol, "cagr": float(eq.iloc[-1] ** (252 / len(r)) - 1), "max_dd": mdd,
           "mdd_vol_ratio": abs(mdd) / vol if vol > 0 else float("nan"),
           "max_dd_at_10pct_vol": mdd / vol * 0.10 if vol > 0 else float("nan")}


def run(market_db: str, out_dir: Path = Path("DATA/pr003")) -> dict:
    df = pr.load_spy(market_db)
    assert df.index.max().date() <= pr.WINDOW[1], "sealed window must not be read"
    divs = pr.load_dividends()
    lb, vm, iv = FINAL_VARIANT
    net, cost, gross, res = rf.run_variant_full(df, divs, lb, vm, iv, fee_fn=mes_fee_fn())

    null = null_alpha_exposure(gross, cost)
    spy = spy_buy_hold_daily(df).reindex(net.index).fillna(0.0)
    corr = float(pd.Series(net.to_numpy()).corr(pd.Series(spy.to_numpy())))
    strat_stats = stats(net)
    be = breakeven_cost_per_contract_side(float(gross.sum()), float(cost.sum()))

    gates = {"null_alpha": null["passed"], "correlation": abs(corr) <= CORR_MAX,
            "expectancy": strat_stats["sharpe"] > 0}
    out = {"run_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
          "variant": FINAL_VARIANT, "n_days": int(len(net)),
          "cost_model": {"cost_per_contract_side": COST_PER_CONTRACT_SIDE,
                         "breakeven_cost_per_contract_side": be},
          "strategy": strat_stats, "correlation_with_spy_buy_hold": corr,
          "null_alpha": null, "gates": gates, "stage_v_pass": all(gates.values()),
          "mdd_report_only": {"mdd_vol_ratio": strat_stats["mdd_vol_ratio"],
                              "paper_reference_ratio": MDD_VOL_REF,
                              "max_dd_at_10pct_vol_target": strat_stats["max_dd_at_10pct_vol"]}}
    try:
        out["commit"] = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                                       text=True, timeout=5).stdout.strip()
    except Exception:
        out["commit"] = None
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"stage_v_{out['run_at'].replace(':', '')}.json"
    path.write_text(json.dumps(rf._tuple_keys_to_str(out), indent=2, default=str))
    out["path"] = str(path)
    return out


def show(o: dict) -> None:
    s, n = o["strategy"], o["null_alpha"]
    print(f"\n=== PR-003 Stage V: viability, variant {o['variant']}, MES-equivalent costs (commit {o['commit']}) ===")
    print(f"days {o['n_days']}, cost/contract/side ${o['cost_model']['cost_per_contract_side']:.2f} "
          f"[ASSUMED], breakeven ${o['cost_model']['breakeven_cost_per_contract_side']:.2f}")
    print(f"\nSharpe {s['sharpe']:.2f}  vol {s['ann_vol']:.1%}  CAGR {s['cagr']:+.1%}  maxDD {s['max_dd']:+.1%}")
    print(f"correlation with SPY buy-and-hold (daily): {o['correlation_with_spy_buy_hold']:+.2f}  (gate |.| <= {CORR_MAX})")
    print(f"\nGATE 1 null-alpha (exposure form): observed {n['observed']:+.4%}/day vs null mean "
          f"{n['null_mean']:+.4%} (p95 {n['null_p95']:+.4%}), p={n['p_value']:.3f} (gate < 0.05)")
    print(f"GATE 2 correlation: {o['gates']['correlation']}")
    print(f"GATE 3 expectancy (Sharpe > 0): {o['gates']['expectancy']}")
    m = o["mdd_report_only"]
    print(f"\n[REPORT ONLY, not gated] MDD/vol ratio {m['mdd_vol_ratio']:.2f} (paper's own: {m['paper_reference_ratio']:.2f}); "
          f"drawdown rescaled to a 10% vol target: {m['max_dd_at_10pct_vol_target']:+.1%}")
    print(f"\nGATES: {o['gates']}")
    print(f"STAGE V: {'PASS' if o['stage_v_pass'] else 'FAIL'}")
    print(f"Saved: {o['path']}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    for noisy in ("market_data_store", "execution_core"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    a = argparse.ArgumentParser()
    a.add_argument("--db", default="DATA/market_data.db")
    args = a.parse_args()
    show(run(args.db))
