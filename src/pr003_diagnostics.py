"""
pr003_diagnostics.py — PR-003 pre-unseal diagnostics (26/27 Sep 2026 cold-fork response)
=========================================================================================
Three checks on 2016-2024 ONLY (2025+ never read), requested after Stage V's
2.75x MDD/vol ratio (vs the paper's own 1.75x) went undiagnosed:

  1. drawdown_attribution   which specific episodes drive the excess drawdown,
                            and how much of it one episode alone explains
                            (a leave-one-episode-out counterfactual).
  2. cost_sensitivity_curve Sharpe/CAGR/maxDD as MES cost per contract/side
                            rises from the assumed $1.125 towards and past the
                            $2.62 breakeven — a curve, not one point, so a
                            cliff (vs a smooth decline) would be visible.
  3. intraday_shuffle_placebo  on a sample of days, keep the first half-hour
                            (which anchors the day's open/band/VWAP) fixed but
                            SHUFFLE THE ORDER of the remaining half-hour blocks
                            within that day (each block's own bars stay intact
                            -- only their position in the day is randomised),
                            using the real previous day for prev_close. If the
                            edge survives this, it is harvesting generic
                            daily volatility, not genuine intraday sequence.

    python src/pr003_diagnostics.py --drawdown
    python src/pr003_diagnostics.py --cost-curve
    python src/pr003_diagnostics.py --placebo --days 200 --draws 30
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pr003_refine as rf                                             # noqa: E402
import pr003_replication as pr                                        # noqa: E402
import pr003_stage_v as sv                                            # noqa: E402
from boxes.noise_area_momentum import NoiseAreaMomentumBox, Params    # noqa: E402
from core.backtest_runner import run_box                              # noqa: E402
from core.execution_core import AccountConfig                         # noqa: E402
from evaluation.mes_costs import COST_PER_CONTRACT_SIDE, mes_fee_fn   # noqa: E402
from market_data_store import REGIMES                                 # noqa: E402

log = logging.getLogger("pr003_diagnostics")
HALF_HOUR = pd.Timedelta(minutes=30)
WARMUP_DAYS = 20   # >= box #3's Params.lookback (14), real unshuffled history


def _date_str(x) -> str:
    """Index items may be pandas Timestamp (has .date()) OR plain
    datetime.date (does not) -- the runner's equity_by_day/position_by_day
    keys are plain date objects, which crashed the first version of this
    module (27 Sep 2026, real-data run)."""
    d = x.date() if hasattr(x, "date") and callable(getattr(x, "date")) else x
    return str(d)


def _get_final_variant_series(df: pd.DataFrame, divs: dict):
    lb, vm, iv = rf.PUBLISHED
    net, cost, gross, res = rf.run_variant_full(df, divs, lb, vm, iv, fee_fn=mes_fee_fn())
    return net, cost, gross, res


# ── 1. Drawdown attribution ───────────────────────────────────────────────────

def drawdown_episodes(net: pd.Series, top: int = 5) -> list[dict]:
    """Peak-to-trough episodes from a daily net-return series, deepest first."""
    eq = (1 + net).cumprod()
    peak = eq.cummax()
    dd = eq / peak - 1
    episodes = []
    in_dd = False
    for i, (d, v) in enumerate(dd.items()):
        if v < 0 and not in_dd:
            start_idx, in_dd = i, True
        elif v >= 0 and in_dd:
            seg = dd.iloc[start_idx:i]
            trough = seg.idxmin()
            episodes.append({"start": _date_str(seg.index[0]), "trough": _date_str(trough),
                             "end": _date_str(d), "depth": float(seg.min()),
                             "n_days": len(seg)})
            in_dd = False
    if in_dd:
        seg = dd.iloc[start_idx:]
        trough = seg.idxmin()
        episodes.append({"start": _date_str(seg.index[0]), "trough": _date_str(trough),
                         "end": "ongoing", "depth": float(seg.min()), "n_days": len(seg)})
    return sorted(episodes, key=lambda e: e["depth"])[:top]


def _regime_of(d: str) -> str:
    dd = date.fromisoformat(d)
    for name, r in REGIMES.items():
        if r["start"] <= dd <= r["end"]:
            return name
    return "unclassified"


def counterfactual_without_episode(net: pd.Series, ep: dict) -> float:
    """Max drawdown of the SAME series with this episode's days zeroed out —
    answers 'how much of the excess drawdown is this one episode alone?'."""
    masked = net.copy()
    idx_str = pd.Index([_date_str(x) for x in masked.index])
    end = ep["end"] if ep["end"] != "ongoing" else idx_str[-1]
    mask = (idx_str >= ep["start"]) & (idx_str <= end)
    masked[np.asarray(mask)] = 0.0
    eq = (1 + masked).cumprod()
    return float((eq / eq.cummax() - 1).min())


def drawdown_attribution(market_db: str, out_dir: Path) -> dict:
    df = pr.load_spy(market_db)
    assert df.index.max().date() <= pr.WINDOW[1], "sealed window must not be read"
    divs = pr.load_dividends()
    net, cost, gross, res = _get_final_variant_series(df, divs)
    overall_dd = float(((1 + net).cumprod() / (1 + net).cumprod().cummax() - 1).min())
    episodes = drawdown_episodes(net, top=5)
    for ep in episodes:
        ep["regime"] = _regime_of(ep["trough"])
        ep["counterfactual_maxdd_without_it"] = counterfactual_without_episode(net, ep)
        window = net.loc[ep["start"]:ep["end"] if ep["end"] != "ongoing" else net.index[-1]]
        ep["window_gross_mean"] = float(gross.reindex(window.index).mean())
        ep["window_cost_mean"] = float(cost.reindex(window.index).mean())
    out = {"run_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "overall_max_dd": overall_dd,
          "episodes": episodes,
          "worst_episode_explains": (overall_dd - episodes[0]["counterfactual_maxdd_without_it"]) / overall_dd
          if episodes and overall_dd < 0 else None}
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"drawdown_attribution_{out['run_at'].replace(':', '')}.json"
    path.write_text(json.dumps(out, indent=2, default=str))
    out["path"] = str(path)
    return out


def show_drawdown(o: dict) -> None:
    print(f"\n=== Drawdown attribution: variant {rf.PUBLISHED}, MES costs, 2016-2024 ===")
    print(f"overall max drawdown: {o['overall_max_dd']:+.1%}")
    for e in o["episodes"]:
        print(f"  {e['start']} .. {e['trough']} .. {e['end']}  depth {e['depth']:+.1%}  "
             f"({e['n_days']} days, {e['regime']})  without it: {e['counterfactual_maxdd_without_it']:+.1%}  "
             f"gross/day {e['window_gross_mean']:+.3%}  cost/day {e['window_cost_mean']:+.3%}")
    if o["worst_episode_explains"] is not None:
        print(f"\nthe single worst episode explains {o['worst_episode_explains']:.0%} of the overall drawdown")
    print(f"Saved: {o['path']}")


# ── 2. Cost-sensitivity curve ─────────────────────────────────────────────────

COST_GRID = (0.50, 0.75, 1.00, 1.125, 1.50, 1.75, 2.00, 2.25, 2.62, 3.00, 3.50, 4.00, 5.00)


def cost_curve_from_series(gross: pd.Series, cost: pd.Series, base_rate: float,
                           grid=COST_GRID) -> list[dict]:
    out = []
    for rate in grid:
        net = gross - (rate / base_rate) * cost
        eq = (1 + net).cumprod()
        sh = float(net.mean() / net.std(ddof=1) * np.sqrt(252)) if net.std(ddof=1) > 0 else float("nan")
        out.append({"cost_per_contract_side": rate, "sharpe": sh,
                   "cagr": float(eq.iloc[-1] ** (252 / len(net)) - 1),
                   "max_dd": float((eq / eq.cummax() - 1).min())})
    return out


def find_cliff(curve: list[dict], rel_drop: float = 0.5) -> dict | None:
    """A step where Sharpe falls by >= rel_drop of its OWN scale between two
    adjacent grid points, vs the smooth average step elsewhere -- a genuine
    discontinuity, not just 'costs eventually hurt'."""
    sh = [c["sharpe"] for c in curve]
    steps = [sh[i] - sh[i + 1] for i in range(len(sh) - 1)]
    if not steps:
        return None
    med = np.median(np.abs(steps)) or 1e-9
    for i, s in enumerate(steps):
        if abs(s) > med * (1 + rel_drop) * 3:      # markedly bigger than the typical step
            return {"between": (curve[i]["cost_per_contract_side"], curve[i + 1]["cost_per_contract_side"]),
                    "sharpe_before": sh[i], "sharpe_after": sh[i + 1]}
    return None


def cost_sensitivity_curve(market_db: str, out_dir: Path) -> dict:
    df = pr.load_spy(market_db)
    assert df.index.max().date() <= pr.WINDOW[1], "sealed window must not be read"
    divs = pr.load_dividends()
    net, cost, gross, res = _get_final_variant_series(df, divs)
    curve = cost_curve_from_series(gross, cost, COST_PER_CONTRACT_SIDE)
    cliff = find_cliff(curve)
    out = {"run_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
          "base_rate": COST_PER_CONTRACT_SIDE, "curve": curve, "cliff": cliff}
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"cost_curve_{out['run_at'].replace(':', '')}.json"
    path.write_text(json.dumps(out, indent=2, default=str))
    out["path"] = str(path)
    return out


def show_cost_curve(o: dict) -> None:
    print(f"\n=== MES cost-sensitivity curve: variant {rf.PUBLISHED} (assumed rate ${o['base_rate']:.3f}) ===")
    print(f"{'$/contract/side':>16}{'Sharpe':>9}{'CAGR':>9}{'maxDD':>9}")
    for c in o["curve"]:
        print(f"{c['cost_per_contract_side']:>16.2f}{c['sharpe']:>9.2f}{c['cagr']:>+9.1%}{c['max_dd']:>+9.1%}")
    print(f"\ncliff detected: {o['cliff']}" if o["cliff"] else "\nno discontinuity detected: decline is smooth")
    print(f"Saved: {o['path']}")


# ── 3. Intraday block-shuffle placebo ─────────────────────────────────────────

def shuffle_day_blocks(day_bars: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Keep the first 30 minutes (09:30-09:59) fixed in place -- this anchors
    the day's open price and the pre-first-decision VWAP/band state. Shuffle
    the ORDER of the remaining half-hour blocks (10:00 onwards) among
    themselves; each block's own bars stay intact, only their position in the
    day changes. Bars are assumed already ordered 09:30..end for that day, so
    this is a plain positional split -- no timestamp arithmetic needed. The
    VALUES are reassigned onto the day's original, still-increasing index, so
    the runner sees a valid contiguous minute series (with deliberate
    discontinuities at the new block boundaries -- that is the point)."""
    idx = day_bars.index
    fixed, rest = day_bars.iloc[:30], day_bars.iloc[30:]
    blocks = [rest.iloc[i:i + 30] for i in range(0, len(rest), 30)]
    order = rng.permutation(len(blocks))
    shuffled = pd.concat([blocks[i] for i in order], ignore_index=False)
    values = pd.concat([fixed, shuffled])
    return pd.DataFrame(values.to_numpy(), columns=values.columns, index=idx)


def run_one_day_placebo(warmup_bars: pd.DataFrame, day_bars: pd.DataFrame, rng,
                        shuffle: bool) -> float:
    """warmup_bars: >= WARMUP_DAYS of REAL, unshuffled prior sessions -- box #3
    needs Params.lookback (14) days of history before its band/sizing produce
    any signal at all. The first version of this function fed only ONE prior
    day, so neither the true order nor any shuffle ever traded (observed and
    null Sharpe both came out exactly 0.00 on real data, 27 Sep 2026): that
    was an untraded strategy, not a null result about shuffling."""
    target = shuffle_day_blocks(day_bars, rng) if shuffle else day_bars
    two = pd.concat([warmup_bars, target])
    res = run_box(NoiseAreaMomentumBox(), Params(sizing="vol_target"), two, "SPY",
                  AccountConfig(cash=pr.AUM0, leverage=None), fee_fn=mes_fee_fn())
    eq = pd.Series(res.equity_by_day).sort_index()
    if len(eq) < 2:
        return 0.0
    return float(eq.iloc[-1] / eq.iloc[-2] - 1)


def intraday_shuffle_placebo(market_db: str, out_dir: Path, n_days: int = 60,
                             draws: int = 20, seed: int = 0,
                             warmup_days: int = WARMUP_DAYS) -> dict:
    df = pr.load_spy(market_db)
    assert df.index.max().date() <= pr.WINDOW[1], "sealed window must not be read"
    days = sorted(set(df.index.date))
    rng = np.random.default_rng(seed)
    eligible = days[warmup_days:]                        # need warmup_days of real history first
    sample = sorted(rng.choice(eligible, size=min(n_days, len(eligible)), replace=False))
    actual, shuffled = [], np.zeros((draws, len(sample)))
    for i, d in enumerate(sample):
        j = days.index(d)
        warm = df[np.isin(df.index.date, days[j - warmup_days:j])]
        db = df[df.index.date == d]
        actual.append(run_one_day_placebo(warm, db, rng, shuffle=False))
        for k in range(draws):
            shuffled[k, i] = run_one_day_placebo(warm, db, rng, shuffle=True)
        if (i + 1) % 10 == 0:
            log.info("placebo %d/%d sampled days", i + 1, len(sample))
    actual = np.array(actual)
    obs_sharpe = float(actual.mean() / actual.std(ddof=1) * np.sqrt(252)) if actual.std(ddof=1) > 0 else 0.0
    null_sharpes = np.array([float(shuffled[k].mean() / shuffled[k].std(ddof=1) * np.sqrt(252))
                             if shuffled[k].std(ddof=1) > 0 else 0.0 for k in range(draws)])
    p = (1 + int((null_sharpes >= obs_sharpe).sum())) / (1 + draws)
    out = {"run_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
          "n_days": len(sample), "draws": draws, "warmup_days": warmup_days, "observed_sharpe": obs_sharpe,
          "null_sharpe_mean": float(null_sharpes.mean()), "null_sharpe_p95": float(np.percentile(null_sharpes, 95)),
          "p_value": p, "edge_survives_shuffle": p >= 0.05}
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"placebo_{out['run_at'].replace(':', '')}.json"
    path.write_text(json.dumps(out, indent=2, default=str))
    out["path"] = str(path)
    return out


def show_placebo(o: dict) -> None:
    print(f"\n=== Intraday block-shuffle placebo: {o['n_days']} sampled days, {o['draws']} shuffles ===")
    print(f"observed Sharpe (true order): {o['observed_sharpe']:.2f}")
    print(f"shuffled Sharpe: mean {o['null_sharpe_mean']:.2f}, 95th pct {o['null_sharpe_p95']:.2f}")
    print(f"p-value (shuffled >= observed): {o['p_value']:.3f}")
    if o["edge_survives_shuffle"]:
        print("EDGE SURVIVES SHUFFLING -- concerning: performance may not depend on genuine intraday sequence")
    else:
        print("edge does NOT survive shuffling -- reassuring: the strategy needs real intraday sequence, not just volatility")
    print(f"Saved: {o['path']}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    for noisy in ("market_data_store", "execution_core"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    a = argparse.ArgumentParser()
    a.add_argument("--db", default="DATA/market_data.db")
    a.add_argument("--drawdown", action="store_true")
    a.add_argument("--cost-curve", action="store_true")
    a.add_argument("--placebo", action="store_true")
    a.add_argument("--days", type=int, default=60)
    a.add_argument("--draws", type=int, default=20)
    a.add_argument("--warmup-days", type=int, default=WARMUP_DAYS)
    args = a.parse_args()
    out_dir = Path("DATA/pr003")
    if args.drawdown:
        show_drawdown(drawdown_attribution(args.db, out_dir))
    if args.cost_curve:
        show_cost_curve(cost_sensitivity_curve(args.db, out_dir))
    if args.placebo:
        show_placebo(intraday_shuffle_placebo(args.db, out_dir, args.days, args.draws, warmup_days=args.warmup_days))
    if not any((args.drawdown, args.cost_curve, args.placebo)):
        show_drawdown(drawdown_attribution(args.db, out_dir))
        show_cost_curve(cost_sensitivity_curve(args.db, out_dir))
        show_placebo(intraday_shuffle_placebo(args.db, out_dir, args.days, args.draws, warmup_days=args.warmup_days))
