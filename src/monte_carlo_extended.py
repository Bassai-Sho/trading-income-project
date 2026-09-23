"""
monte_carlo_extended.py
=======================
Extended Monte Carlo simulation engine for the Trading Income Project.

THE GALTON BOARD CONNECTION
────────────────────────────
The Galton board (bean machine) is a physical device demonstrating the
central limit theorem. Each ball hits a row of pegs and bounces left
(loss) or right (win). After N rows, the distribution of balls at the
bottom forms the binomial distribution → approximates normal with
sufficient N.

In trading:
  Each peg row     = one trade
  P(bounce right)  = win probability (e.g., 0.44 from IDEXAONE benchmark)
  P(bounce left)   = 1 - win probability
  Ball's final bin = equity after N trades
  Distribution     = the Monte Carlo equity distribution

In options (Cox-Ross-Rubinstein binomial model):
  Each time step   = one period
  Up move / down   = price goes toward target / stop
  Option payoff    = P(ball lands in 'in the money' bin)

This module extends Group A monte_carlo_simulation with:
  1. galton_equity_simulation()       — forward equity paths
  2. path_dependent_target_prob()     — P(hit target before stop)
  3. regime_conditional_simulation()  — Markov regime-aware paths
  4. risk_of_ruin()                   — P(ruin below threshold)
  5. forecast_report()                — unified dashboard output

ACADEMIC BASIS
──────────────
• Galton (1889) — The bean machine, Pascal's triangle, binomial → normal
• Cox, Ross & Rubinstein (1979) — Binomial option pricing model
• Our Group A monte_carlo_simulation() in trading_quant_toolkit_v2_4.py
"""

from __future__ import annotations

import json
import math
import random
import sqlite3
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np


# ── 1. Forward equity simulation (Galton board analogy) ──────────────────────

def galton_equity_simulation(
    win_rate:     float,
    avg_win_r:    float,
    avg_loss_r:   float,
    n_trades:     int,
    n_sims:       int = 10_000,
    current_equity: float = 10_000.0,
    risk_pct:     float = 0.01,
    daily_stop_pct: float = 0.03,
    seed:         int | None = 42,
) -> dict:
    """
    Simulate N_SIMS equity curves of N_TRADES trades each.

    Each trade is a Galton board bead hitting a row of pegs:
      P(bounce right / win)  = win_rate
      P(bounce left  / loss) = 1 - win_rate
      Win step  = current_equity * risk_pct * avg_win_r
      Loss step = current_equity * risk_pct * avg_loss_r

    Returns
    -------
    {
        paths: list[list[float]] (sample of 100 paths for plotting),
        final_equity: {p5, p25, p50, p75, p95, mean, std},
        probability_reach_target: float,
        probability_ruin: float,
        expected_trades_to_target: float | None,
        galton_bins: dict — distribution of final equity (Galton board shape),
        ev_per_trade: float,
        n_sims: int,
        n_trades: int,
    }
    """
    rng = np.random.default_rng(seed)

    # Vectorised: shape (n_sims, n_trades + 1)
    equity = np.full((n_sims, n_trades + 1), current_equity)
    wins   = rng.random((n_sims, n_trades)) < win_rate
    risk_amounts = equity[:, :-1] * risk_pct
    changes = np.where(wins,
                        risk_amounts * avg_win_r,
                       -risk_amounts * avg_loss_r)
    equity[:, 1:] = current_equity + np.cumsum(changes, axis=1)

    final = equity[:, -1]
    target = current_equity * 1.20      # 20% account gain target
    ruin   = current_equity * (1 - daily_stop_pct * 10)  # 10 daily stops = ruin

    # Galton board bins (11 bins like a real Galton board)
    lo, hi   = final.min(), final.max()
    bin_edges = np.linspace(lo, hi, 12)
    bin_counts, _ = np.histogram(final, bins=bin_edges)
    galton_bins = {
        f"${(bin_edges[i]+bin_edges[i+1])/2:,.0f}": int(bin_counts[i])
        for i in range(len(bin_counts))
    }

    # Percentiles of final equity
    p5, p25, p50, p75, p95 = np.percentile(final, [5, 25, 50, 75, 95])

    # P(reach target) and P(ruin) at any point in the paths
    p_reach_target = float((equity.max(axis=1) >= target).mean())
    p_ruin         = float((equity.min(axis=1) <= ruin).mean())

    # Expected trades to reach target
    hits = equity >= target
    first_hit_idx = np.argmax(hits, axis=1)
    reached       = hits.any(axis=1)
    exp_trades_to_target = (float(first_hit_idx[reached].mean())
                             if reached.any() else None)

    ev_per_trade = (win_rate * avg_win_r) - ((1 - win_rate) * avg_loss_r)

    # Sample 100 paths for dashboard plotting
    sample_indices = np.linspace(0, n_sims - 1, min(100, n_sims), dtype=int)
    sample_paths   = equity[sample_indices, :].tolist()

    return {
        "paths":               sample_paths,
        "final_equity": {
            "p5":   round(float(p5),  2),
            "p25":  round(float(p25), 2),
            "p50":  round(float(p50), 2),
            "p75":  round(float(p75), 2),
            "p95":  round(float(p95), 2),
            "mean": round(float(final.mean()), 2),
            "std":  round(float(final.std()),  2),
        },
        "probability_reach_target":    round(p_reach_target, 4),
        "probability_ruin":            round(p_ruin, 4),
        "expected_trades_to_target":   (round(exp_trades_to_target, 1)
                                         if exp_trades_to_target else None),
        "galton_bins":         galton_bins,
        "ev_per_trade":        round(ev_per_trade, 4),
        "n_sims":              n_sims,
        "n_trades":            n_trades,
        "current_equity":      current_equity,
        "target_equity":       round(target, 2),
        "ruin_threshold":      round(ruin, 2),
        "interpretation": (
            f"After {n_trades} trades at {win_rate:.0%} WR / {avg_win_r:.1f}R avg win: "
            f"P(reach +20% target)={p_reach_target:.0%}, "
            f"P(ruin)={p_ruin:.0%}, "
            f"median equity=${p50:,.0f}. "
            f"EV/trade={ev_per_trade:+.4f}R."
        ),
    }


# ── 2. Path-dependent target probability (options-style) ─────────────────────

def path_dependent_target_prob(
    entry:          float,
    stop:           float,
    target:         float,
    atr:            float,
    n_bars_max:     int  = 30,
    n_sims:         int  = 100_000,
    direction:      str  = "long",
    seed:           int | None = 42,
) -> dict:
    """
    Monte Carlo over price paths to estimate P(hit target before stop).

    This is the Cox-Ross-Rubinstein binomial model applied to ORB trades.
    Each 5-minute bar the price moves up or down by ATR / sqrt(5) (scaling
    from daily ATR to 5-minute ATR). We count what fraction of paths:
      - Hit the target before hitting the stop (winners)
      - Hit the stop before hitting the target (losers)
      - Neither within n_bars_max (time exits)

    Unlike the simple win/loss Galton board, this models the actual
    PATH THROUGH WHICH price must travel — including the possibility
    of price moving away from both levels before eventually resolving.

    Parameters
    ----------
    entry     : entry price
    stop      : stop loss price
    target    : profit target price
    atr       : 14-bar ATR on the 5-minute chart
    n_bars_max: maximum number of bars before EOD exit
    direction : 'long' | 'short'

    Returns P(target_hit), P(stop_hit), P(eod_exit), expected_bars_to_exit
    """
    rng      = np.random.default_rng(seed)
    risk_r   = abs(entry - stop)
    reward_r = abs(entry - target)
    actual_rr = round(reward_r / max(risk_r, 1e-6), 2)

    # 5-minute bar sigma ≈ ATR / sqrt(13)  (13 5-min bars per ATR day proxy)
    bar_sigma = atr / math.sqrt(13)
    if bar_sigma < 1e-6:
        bar_sigma = risk_r * 0.05   # fallback

    # Simulate paths
    prices = np.full(n_sims, entry)
    target_hit  = np.zeros(n_sims, dtype=bool)
    stop_hit    = np.zeros(n_sims, dtype=bool)
    bars_to_exit = np.full(n_sims, n_bars_max, dtype=float)
    active      = np.ones(n_sims, dtype=bool)

    for bar in range(n_bars_max):
        if not active.any():
            break
        moves  = rng.normal(0, bar_sigma, n_sims)
        prices[active] += moves[active]
        if direction == "long":
            hit_target = active & (prices >= target)
            hit_stop   = active & (prices <= stop)
        else:
            hit_target = active & (prices <= target)
            hit_stop   = active & (prices >= stop)
        target_hit |= hit_target
        stop_hit   |= hit_stop
        bars_to_exit[hit_target | hit_stop] = bar + 1
        active[hit_target | hit_stop] = False

    p_target = float(target_hit.mean())
    p_stop   = float(stop_hit.mean())
    p_eod    = float((~target_hit & ~stop_hit).mean())
    exp_bars = float(bars_to_exit.mean())

    # EV from this path simulation
    ev_path = p_target * actual_rr - p_stop * 1.0

    return {
        "p_target_hit":       round(p_target, 4),
        "p_stop_hit":         round(p_stop, 4),
        "p_eod_exit":         round(p_eod, 4),
        "expected_bars_exit": round(exp_bars, 1),
        "actual_rr":          actual_rr,
        "ev_path_simulation": round(ev_path, 4),
        "n_sims":             n_sims,
        "interpretation": (
            f"Path simulation: P(target)={p_target:.0%}, "
            f"P(stop)={p_stop:.0%}, P(EOD)={p_eod:.0%}. "
            f"Path EV={ev_path:+.3f}R at {actual_rr:.1f}:1 R:R. "
            f"Avg exit: {exp_bars:.0f} bars."
        ),
    }


# ── 3. Regime-conditional simulation (Markov chain aware) ────────────────────

def regime_conditional_simulation(
    win_rates_by_regime: dict,       # {'NORMAL': 0.47, 'ELEVATED': 0.38, 'HIGH': 0.30}
    avg_win_r:           float,
    avg_loss_r:          float,
    n_trades:            int,
    n_sims:              int          = 10_000,
    current_equity:      float        = 10_000.0,
    risk_pct:            float        = 0.01,
    transition_matrix:   list | None  = None,
    starting_regime:     str          = "NORMAL",
    regime_states:       list         = None,
    seed:                int | None   = 42,
) -> dict:
    """
    Monte Carlo simulation that accounts for regime transitions
    between simulated trades using the Markov chain transition matrix.

    This links the Galton board simulation to the RegimeMarkovChain:
      - At each trade, the current regime determines win_rate
      - Regime transitions between trades follow the Markov matrix
      - Position size adjusts to the MarkovPositionSizer scale factor

    The result shows how the equity distribution CHANGES when you
    condition on regime uncertainty — the regime-conditional distribution
    is wider (fatter tails) than the single-regime simulation.
    """
    rng = np.random.default_rng(seed)
    if regime_states is None:
        regime_states = ["NORMAL", "ELEVATED", "HIGH"]
    if transition_matrix is None:
        # Default: high persistence regime matrix (from live VIX data)
        transition_matrix = [
            [0.898, 0.098, 0.004],
            [0.182, 0.759, 0.059],
            [0.003, 0.314, 0.683],
        ]
    P   = np.array(transition_matrix)
    idx = {s: i for i, s in enumerate(regime_states)}

    # Default win rates if not provided
    if not win_rates_by_regime:
        win_rates_by_regime = {
            "NORMAL":   0.46, "ELEVATED": 0.39, "HIGH": 0.28,
        }
    # Position size modifiers per regime (from MarkovPositionSizer)
    size_mods = {"NORMAL": 0.85, "ELEVATED": 0.64, "HIGH": 0.52}

    final_equities = np.zeros(n_sims)
    regime_exposure: dict[str, int] = {r: 0 for r in regime_states}

    for sim in range(n_sims):
        equity  = current_equity
        regime  = starting_regime
        for _ in range(n_trades):
            regime_exposure[regime] += 1
            wr    = win_rates_by_regime.get(regime, 0.40)
            scale = size_mods.get(regime, 0.85)
            r_amt = equity * risk_pct * scale
            if rng.random() < wr:
                equity += r_amt * avg_win_r
            else:
                equity -= r_amt * avg_loss_r
            # Regime transition
            probs  = P[idx.get(regime, 0)]
            regime = rng.choice(regime_states, p=probs)
        final_equities[sim] = equity

    p5, p25, p50, p75, p95 = np.percentile(final_equities, [5, 25, 50, 75, 95])
    # Total exposure
    total_exp = sum(regime_exposure.values())
    regime_pcts = {r: round(v/max(total_exp,1), 3) for r, v in regime_exposure.items()}

    return {
        "final_equity": {
            "p5":   round(float(p5), 2),
            "p25":  round(float(p25), 2),
            "p50":  round(float(p50), 2),
            "p75":  round(float(p75), 2),
            "p95":  round(float(p95), 2),
            "mean": round(float(final_equities.mean()), 2),
        },
        "regime_exposure_pct": regime_pcts,
        "n_sims":              n_sims,
        "n_trades":            n_trades,
        "interpretation": (
            f"Regime-conditional simulation ({n_sims:,} paths, {n_trades} trades). "
            f"Median equity ${p50:,.0f}. 90% interval: ${p5:,.0f}–${p95:,.0f}. "
            f"Regime exposure: {', '.join(f'{r}={pct:.0%}' for r, pct in regime_pcts.items())}."
        ),
    }


# ── 4. Risk of ruin ──────────────────────────────────────────────────────────

def risk_of_ruin(
    win_rate:      float,
    avg_win_r:     float,
    avg_loss_r:    float,
    risk_pct:      float,
    ruin_threshold_pct: float = 0.50,  # lose 50% of account
    n_trades_horizon: int = 1000,
    n_sims:        int  = 50_000,
    current_equity: float = 10_000.0,
    seed:          int | None = 42,
) -> dict:
    """
    Estimate P(account drawdown exceeds ruin_threshold_pct) over
    n_trades_horizon trades.

    Unlike the Group A Monte Carlo which targets a Sharpe gate,
    this directly answers: what is the probability of catastrophic
    drawdown at our current position sizing and win rate?

    Uses the Kelly criterion relationship: a strategy with EV > 0 and
    risk < Kelly fraction will have P(ruin) → 0 as N → ∞. A strategy
    with risk > Kelly fraction has P(ruin) → 1.
    """
    rng = np.random.default_rng(seed)
    ruin_floor = current_equity * (1 - ruin_threshold_pct)

    equity = np.full((n_sims, n_trades_horizon + 1), current_equity)
    wins   = rng.random((n_sims, n_trades_horizon)) < win_rate
    risk_dollars = equity[:, :-1] * risk_pct
    changes = np.where(wins,
                        risk_dollars * avg_win_r,
                       -risk_dollars * avg_loss_r)
    equity[:, 1:] = current_equity + np.cumsum(changes, axis=1)

    # P(ruin) = fraction of paths where min equity < ruin_floor
    p_ruin = float((equity.min(axis=1) < ruin_floor).mean())

    # Kelly fraction
    wr, al, aw = win_rate, avg_loss_r, avg_win_r
    kelly_full  = (wr / al) - ((1 - wr) / aw)
    kelly_half  = kelly_full / 2  # common half-Kelly recommendation
    at_kelly    = round(kelly_full * 100, 2)

    # Drawdown analysis
    running_max = np.maximum.accumulate(equity, axis=1)
    drawdowns   = (equity - running_max) / np.maximum(running_max, 1e-6)
    max_dd      = drawdowns.min(axis=1)
    p50_dd, p95_dd = np.percentile(max_dd * 100, [50, 95])

    return {
        "p_ruin":               round(p_ruin, 4),
        "ruin_threshold_pct":   ruin_threshold_pct,
        "ruin_floor":           round(ruin_floor, 2),
        "current_risk_pct":     risk_pct,
        "kelly_fraction":       round(kelly_full, 4),
        "kelly_half":           round(kelly_half, 4),
        "at_kelly_risk_pct":    at_kelly,
        "median_max_drawdown_pct": round(float(p50_dd), 2),
        "p95_max_drawdown_pct":    round(float(p95_dd), 2),
        "n_sims":               n_sims,
        "safe": p_ruin < 0.01,  # < 1% ruin probability
        "interpretation": (
            f"Risk of ruin (>{ruin_threshold_pct:.0%} drawdown) over {n_trades_horizon} trades: "
            f"P={p_ruin:.1%}. "
            f"{'✅ SAFE (<1%)' if p_ruin < 0.01 else '⚠️ ELEVATED' if p_ruin < 0.05 else '🛑 HIGH RISK'}. "
            f"Kelly fraction: {kelly_full:.1%} (current risk {risk_pct:.1%} = "
            f"{'below' if risk_pct < kelly_full else 'above'} Kelly). "
            f"Median max drawdown: {p50_dd:.1f}%."
        ),
    }


# ── 5. Load stats from DB ────────────────────────────────────────────────────

def load_trade_stats_from_db(db_path: str, n_recent: int = 50) -> dict:
    """Load win rate, avg win R, avg loss R from closed positions."""
    try:
        with sqlite3.connect(db_path) as conn:
            rows = conn.execute(
                "SELECT actual_r FROM positions WHERE status='closed' "
                "AND actual_r IS NOT NULL ORDER BY id DESC LIMIT ?",
                (n_recent,)
            ).fetchall()
    except Exception:
        return {}
    rs = [r[0] for r in rows]
    if not rs:
        return {}
    wins   = [r for r in rs if r > 0]
    losses = [r for r in rs if r < 0]
    return {
        "n":          len(rs),
        "win_rate":   len(wins) / len(rs),
        "avg_win_r":  statistics.mean(wins)           if wins   else 0.0,
        "avg_loss_r": abs(statistics.mean(losses))    if losses else 1.0,
    }


# ── 6. Unified forecast report ───────────────────────────────────────────────

def forecast_report(
    db_path:      str,
    current_equity: float  = 10_000.0,
    risk_pct:     float    = 0.01,
    current_vix:  float    = 15.0,
    n_sims:       int      = 5_000,
) -> dict:
    """
    Run all Monte Carlo analyses and return a unified report
    suitable for the dashboard Monte Carlo tab.

    Falls back to assumed parameters if fewer than 20 trades.
    """
    stats = load_trade_stats_from_db(db_path)
    assumed = len(stats) == 0 or stats.get("n", 0) < 20

    wr  = stats.get("win_rate",  0.44)     # IDEXAONE benchmark baseline
    aw  = stats.get("avg_win_r", 2.0)
    al  = stats.get("avg_loss_r",1.0)
    n   = stats.get("n", 0)

    # Galton board equity simulation
    galton = galton_equity_simulation(
        win_rate=wr, avg_win_r=aw, avg_loss_r=al,
        n_trades=50, n_sims=n_sims,
        current_equity=current_equity, risk_pct=risk_pct,
    )
    # Risk of ruin
    ror = risk_of_ruin(
        win_rate=wr, avg_win_r=aw, avg_loss_r=al,
        risk_pct=risk_pct, n_sims=n_sims,
        current_equity=current_equity,
    )
    # Regime-conditional
    reg = regime_conditional_simulation(
        win_rates_by_regime={},
        avg_win_r=aw, avg_loss_r=al,
        n_trades=50, n_sims=n_sims,
        current_equity=current_equity, risk_pct=risk_pct,
    )

    return {
        "generated_at":       datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds"),
        "n_trades_history":   n,
        "assumed_parameters": assumed,
        "win_rate":           round(wr, 4),
        "avg_win_r":          round(aw, 4),
        "avg_loss_r":         round(al, 4),
        "galton_simulation":  galton,
        "risk_of_ruin":       ror,
        "regime_conditional": reg,
        "headline": (
            f"{'⚠️ ASSUMED' if assumed else f'📊 {n} TRADES'} | "
            f"WR={wr:.0%} | EV={galton['ev_per_trade']:+.4f}R | "
            f"Median equity (50 trades)=${galton['final_equity']['p50']:,.0f} | "
            f"P(ruin)={ror['p_ruin']:.1%} | "
            f"{'✅ Risk OK' if ror['safe'] else '⚠️ Review risk sizing'}"
        ),
    }


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== monte_carlo_extended.py smoke test ===\n")

    # 1. Galton board simulation
    print("1. Galton Equity Simulation (Galton board analogy)")
    g = galton_equity_simulation(
        win_rate=0.44, avg_win_r=2.0, avg_loss_r=1.0,
        n_trades=50, n_sims=10_000,
        current_equity=10_000, risk_pct=0.01,
    )
    print(f"   EV/trade: {g['ev_per_trade']:+.4f}R")
    print(f"   Equity after 50 trades: p5=${g['final_equity']['p5']:,.0f} "
          f"| median=${g['final_equity']['p50']:,.0f} | p95=${g['final_equity']['p95']:,.0f}")
    print(f"   P(reach +20% target): {g['probability_reach_target']:.0%}")
    print(f"   P(ruin):              {g['probability_ruin']:.0%}")
    assert g["ev_per_trade"] > 0, "EV should be positive with 44% WR and 2:1 R:R"
    assert 0 < g["probability_reach_target"] < 1

    # 2. Path-dependent probability (options-style)
    print("\n2. Path-Dependent Target Probability (CRR binomial model)")
    p = path_dependent_target_prob(
        entry=770.0, stop=769.0, target=772.0,
        atr=0.8, n_bars_max=30, direction="long",
        n_sims=10_000,
    )
    print(f"   P(target): {p['p_target_hit']:.0%}  "
          f"P(stop): {p['p_stop_hit']:.0%}  "
          f"P(EOD exit): {p['p_eod_exit']:.0%}")
    print(f"   Path EV: {p['ev_path_simulation']:+.3f}R  "
          f"avg exit: {p['expected_bars_exit']:.0f} bars")

    # 3. Regime-conditional simulation
    print("\n3. Regime-Conditional Simulation (Markov chain aware)")
    r = regime_conditional_simulation(
        win_rates_by_regime={"NORMAL": 0.47, "ELEVATED": 0.38, "HIGH": 0.28},
        avg_win_r=2.0, avg_loss_r=1.0,
        n_trades=50, n_sims=5_000,
        current_equity=10_000, starting_regime="NORMAL",
    )
    print(f"   Median: ${r['final_equity']['p50']:,.0f}  "
          f"90% interval: ${r['final_equity']['p5']:,.0f}–${r['final_equity']['p95']:,.0f}")
    print(f"   Regime exposure: {r['regime_exposure_pct']}")

    # 4. Risk of ruin
    print("\n4. Risk of Ruin Analysis")
    ror = risk_of_ruin(
        win_rate=0.44, avg_win_r=2.0, avg_loss_r=1.0,
        risk_pct=0.01, n_sims=10_000,
    )
    print(f"   P(ruin > 50%): {ror['p_ruin']:.1%}  {'✅ SAFE' if ror['safe'] else '⚠️ ELEVATED'}")
    print(f"   Kelly fraction: {ror['kelly_fraction']:.1%}  (current risk 1.0%)")
    print(f"   Median max drawdown: {ror['median_max_drawdown_pct']:.1f}%")
    assert ror["p_ruin"] < 0.05, "1% risk should be safe"

    print("\n✅ monte_carlo_extended.py verified")
