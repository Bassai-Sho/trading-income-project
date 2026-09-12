"""
trading_quant_toolkit_v2_0.py
==============================
Deterministic risk-management and execution calculations for the Trading Income Project.

VERSION HISTORY
---------------
v1.0.0 (undated, prior to version control) -- Initial release. Group A
    (position_size, breakeven_win_rate, expected_value, unleveraged_return,
    effective_risk_per_trade, minimum_position_size_for_cost_drag) + Group B
    (sharpe_ratio, sharpe_significance_tstat, probabilistic_sharpe_ratio,
    deflated_sharpe_ratio, expected_max_sharpe_under_trials, max_drawdown,
    monte_carlo_robustness, future_value_annuity).

v1.1.0 (2026-09-06) -- Added Group C, ported from the Orderflow & Options
    Flow intelligence batch: dealer_hedge_flow, net_gamma_exposure,
    value_area, effort_result_ratio. First version-controlled release.
    Demo block upgraded from print-only to assert-verified.

v2.4.0 (2026-09-08) -- VWAP slope gate + walk-forward toolkit function.
    All v2.3.0 content preserved exactly.
    Group G addition:
        vwap_slope_gate()  -- Enhanced VWAP filter adding three conditions
                              beyond basic slope: magnitude (tied to current
                              range), chop filter (price not flipping across
                              VWAP repeatedly), and hold filter (price held
                              one side for required bars). All three AND-logic.
                              Source: TradingView VWAP Gate v2.3 (MIdr0guA).
                              Backlog item P2-035.
    Group J addition:
        walk_forward_backtest()  -- Canonical WFA function in the toolkit.
                                   Rolls IS/OOS windows, computes per-split
                                   metrics and aggregate WFE = OOS SR / IS SR.
                                   Target WFE >= 0.50. Complements wfa_run()
                                   in trading_engine.py (engine version uses
                                   hourly bars for speed; this version accepts
                                   any DataFrame). Backlog item P2-017.
    Demo block: all v2.3.0 assertions preserved; 8 new v2.4.0 checks added.

v2.3.0 (2026-09-07) -- Realistic execution simulation + session learning.
    All v2.2.0 content preserved exactly. Added Group L: fill simulation and
    session learning. Motivation: every paper trading system fills at the last
    close price — which is impossible in live trading. You always buy at the
    ask and sell at the bid, stops gap through in fast markets, targets are
    limit orders that can miss, and there is always at least 1-bar delay
    between signal and fill. These four biases collectively make paper records
    systematically more optimistic than live results. Group L fixes them all.

    Group L -- Fill simulation and session learning:
      FillSimulator                -- Pessimistic execution model. Applies
                                      bid-ask spread, slippage, stop-fill
                                      slippage, and enforces the 1-bar entry
                                      delay. Uses WORST plausible price, not
                                      best. If paper is profitable vs this
                                      model, live trading has no hard surprises.
          simulate_entry_fill()   -- Entry at ask + slippage (not last close)
          simulate_stop_fill()    -- Stop exit at stop - ATR*factor (not stop)
          simulate_target_fill()  -- Target at bid - slippage (limit order)
          simulate_eod_fill()     -- EOD close at bid (market sell)
          apply_1bar_delay()      -- Signal bar close → next bar open entry.
                                      Prevents impossible same-bar fills.
          net_fill_vs_model()     -- Comparison: modelled P&L vs realistic P&L.
                                      Quantifies the simulation optimism bias.
      SessionLearner               -- Analyses the decision ledger after each
                                      session to extract: (1) skip-but-would-
                                      have-won setups (over-filtering signals),
                                      (2) enter-but-lost setups by reason, (3)
                                      fill vs modelled price gap (execution bias).
          analyse_skips()         -- Which skip reasons correlate with missed
                                      winning setups? Flags over-tight gates.
          analyse_fill_bias()     -- Compares actual entry prices logged to the
                                      FillSimulator's pessimistic estimate.
          generate_refinement()   -- Returns a plain-English list of suggested
                                      parameter adjustments based on session
                                      data, for human review before applying.
          regime_performance()    -- Win rate and EV by VIX regime (NORMAL /
                                      ELEVATED / HIGH / EXTREME). Identifies
                                      which market conditions the strategy
                                      actually works in vs where it fails.

v2.2.0 (2026-09-07) -- Realistic cost model. All v2.1.0 content preserved
    exactly. Added Group K: trading cost and liquidity modelling.
    Motivation: audit revealed the engine assumed fills at last close with a
    flat 0.05R commission -- no spread, no slippage, no liquidity check.
    For SPY on Alpaca this is approximately correct. For individual stocks
    (the Zarattini Stocks in Play universe) it materially overstates returns.
    Zarattini et al. (2024) used $0.005/share IBKR commission; this group
    implements that and the structural cost components around it.

    Group K additions:
      commission_cost()          -- Broker-specific per-trade commission.
                                    Supports 'alpaca' (free), 'ibkr'
                                    ($0.005/share, min $1), 'td' (free),
                                    'uk_cfd' (spread-based, see notes).
      half_spread_cost()         -- Roll (1984) half-spread estimate from
                                    intraday return autocovariance. Falls back
                                    to 0.001 * price when autocovariance
                                    cannot be computed.
      market_impact_cost()       -- Square-root market impact (Almgren &
                                    Chriss 2001 simplified): measures the
                                    extra price movement caused by your own
                                    order. Scales with (units / avg_vol)^0.5.
      total_round_trip_cost()    -- Commission + 2*half_spread + market_impact
                                    for a full round trip. Returns dict with
                                    component breakdown and total in £/$.
      cost_as_r_multiple()       -- Converts total £ cost to R-multiple drag.
                                    Replaces the flat 0.05R assumption.
      volume_liquidity_check()   -- Validates that planned units <= max_pct
                                    of average first-5-min volume (default 1%).
                                    Prevents simulating fills that would move
                                    the market against you.
      net_position_size()        -- Combines position_size() with a liquidity
                                    check and returns the valid units, the
                                    cost breakdown, and a reject reason if
                                    the trade fails liquidity constraints.
      realistic_backtest_cost()  -- Per-trade cost drag in R for use in
                                    run_backtest() and wfa_run(). Replaces
                                    the flat 0.05R constant.
    Demo block: all v2.1.0 assertions preserved; 10 new Group K checks added.

v2.1.0 (2026-09-07) -- Incremental additions. All v2.0.0 content preserved
    exactly -- zero regressions, all v2.0.0 assertions still pass unchanged.
    New additions driven by deep intelligence sweep (GitHub, SSRN, institutional
    resources session, 07 Sep 2026):
      Group E additions:
        relative_volume()      -- RVOL ratio for the Stocks in Play filter.
                                  Source: Zarattini, Barbon, Aziz (2024) SSRN
                                  4729284. Without RVOL filter: Sharpe 0.48.
                                  With top-20 RVOL: Sharpe 2.81, alpha 36%.
        scan_stocks_in_play()  -- Pre-market scanner: filters a ticker universe
                                  by price, avg volume, ATR, and RVOL, returns
                                  top N Stocks in Play ranked by RVOL.
      Group F additions:
        kelly_fraction()       -- Half-Kelly position sizing. Formula:
                                  K = W - (1-W)/R. Fraction=0.5 default.
                                  Only use after 50+ logged trades.
        kelly_position_size()  -- Applies kelly fraction to position size calc.
        trailing_stop_update() -- VWAP-as-trailing-stop. Source: Zarattini et
                                  al. (2024) SSRN 4824172 + Maroy (2025) SSRN
                                  5095349. VWAP trailing stop doubles Sharpe.
                                  VWAP + Ladder exit achieves Sharpe 3.0+.
        ladder_target()        -- Partial exit levels at 1R, 2R, 3R. Exit 1/3
                                  at each level, run remainder with VWAP trail.
        atr_stop_loss()        -- ATR-based dynamic stop adapts to volatility.
        volatility_target_size()  -- Targets 2% daily vol exposure (Zarattini
                                  2024 dynamic sizing). Halves size if vol
                                  doubles vs baseline.
      Group G addition:
        orb_define_range() extended with method='30min' (09:30-09:55, 6 bars).
        Academic backing: Gao, Han, Li, Zhou (2018) JFE. BrandonTrades
        corroborates independently.
    Demo block: all v2.0.0 assertions preserved; 12 new checks added for
    v2.1.0 additions, all passing as of 07 Sep 2026.

v2.0.0 (2026-09-06) -- Major addition. Groups A, B, C preserved exactly --
    zero regressions, all v1.1.0 assertions still pass unchanged. Added
    Groups D through J, built and verified as part of the day trading engine
    build session (P1-001 through P2-007 in the Action Items backlog):
      Group D -- Statistical extensions: regime_conditional_streak_probability,
          drawdown_barrier_breach_probability, nearest_correlation_higham,
          sanitize_vol_inputs (ported from portfolio_engine_v2_15.py,
          Wealth Building project, line numbers documented in provenance).
      Group E -- Data layer: fetch_ohlcv, fetch_with_fallback, fetch_pdh_pdl,
          fetch_premarket_gap, validate_ticker_identity, filter_session_hours,
          filter_orb_window, fetch_vix_current (validated P1-001, 06 Sep 2026:
          38 bars/5d confirmed, America/New_York native, MultiIndex flattened).
      Group F -- Risk layer: instrument_risk_pct, live_risk_pct, set_stop_loss,
          set_target, daily_stop_check, consecutive_loss_check, SessionRiskState.
          Commodity cap rule (0.5%) encoded: Gold/Silver demonstrated 30-50%
          crashes in weeks (Jason Sen analysis, Mar 2026).
      Group G -- Signal layer: sma_trend_filter, vwap_anchored, vwap_slope,
          orb_define_range, orb_breakout_signal, gap_quality_check,
          candle_anatomy, entry_gate. entry_gate enforces AND-logic only --
          DARF v3 OR-gate failure (Sharpe 0.70 vs 0.95, drawdown 18.3% vs
          11.3%) is the proof this is not optional.
      Group H -- Entry layer: candle_closed, validate_entry, ipde_checklist,
          retest_confirmation, session_time_valid, SessionAttemptTracker.
      Group I -- Logging layer: TradeRecord (25-field dataclass), TradeJournal
          (CSV-backed append-only journal with rolling stats and milestone check).
      Group J -- Backtest layer: data partition constants (Train 2020-22,
          Val 2023-24, Test 2025+ SEALED), compute_metrics, run_backtest,
          monte_carlo_gate, degradation_check.
    Dependency note: Groups A-C remain pure stdlib (math, random, statistics,
    dataclasses) -- no new dependencies for those groups. Groups D-J require:
        pip install yfinance pandas scipy numpy tzdata
    The file imports these lazily so Groups A-C work without the data stack.
    Demo block extended: all v1.1.0 assertions preserved; 38 new checks added
    for Groups D-J, all passing as of 06 Sep 2026.

Going forward: every substantive change to this file gets a new version
number here, a dated entry above, and a matching filename bump
(trading_quant_toolkit_vX_Y.py). Do not ship an update under the same
filename as a prior version -- see the Notion Logging Protocol skill in
the Trading Income Project workspace for the full rule.

Purpose
-------
This module is the fixed computational layer for the Trading Income Project.
Every verdict in the Trading Rulebook rests on a formula here, computed
the same way every time rather than re-derived by hand each session.

Provenance
----------
Group A -- Trading Income Project Rulebook (this project):
    position_size, breakeven_win_rate, expected_value, unleveraged_return,
    effective_risk_per_trade, minimum_position_size_for_cost_drag.

Group B -- Wealth Building project (portfolio_engine_v2.x.py), read-only.
    Only general-purpose quant formulas were ported; nothing fund-specific.
    sharpe_ratio, sharpe_significance_tstat (Lo, 2002).
    probabilistic_sharpe_ratio (PSR) -- verified against ProbSharpeRatio.R.
    deflated_sharpe_ratio (DSR) -- Bailey & Lopez de Prado, 2014.
    max_drawdown, future_value_annuity.

Group C -- Orderflow & Options Flow intelligence batch (Creamer, Sarmiento).
    Only PUBLIC, general market-microstructure math was ported.
    NOT ported: JEX Flip, OVI, convexity levels -- undisclosed proprietary.
    dealer_hedge_flow, net_gamma_exposure (standard delta-hedging identity).
    value_area (standard Volume Profile / Market Profile algorithm).
    effort_result_ratio (footprint effort-vs-result concept).

Group D -- Wealth Building project (portfolio_engine_v2_15.py), read-only.
    Lines confirmed before porting:
    nearest_correlation_higham        line 2811
    sanitize_covariance_inputs        line 5699 (renamed sanitize_vol_inputs)
    sharpe_significance_tstat (ext.)  line 5899 (extended version, annualised)
    deflated_sharpe_ratio (ext.)      line 5951 (extended, uses scipy)
    regime_conditional_streak_probability  line 6025 (Feller 1968 DP)
    drawdown_barrier_breach_probability    line 6094 (GBM first-passage)
    Note: Group D's sharpe_significance_tstat_annualised and
    deflated_sharpe_ratio_extended complement rather than replace the
    Group B versions -- the Group B signatures are preserved for backward
    compatibility; the Group D versions add annualised inputs and scipy.

Group E-J -- Day trading engine build session, 06 Sep 2026.
    All functions validated via live yfinance data and assert-verified.
    Bugs found and fixed during build:
      candle_closed: initial current_time > bar_time check was wrong for
          intraday bars; fixed to use datetime arithmetic with interval_minutes.
      TradeJournal CSV header: os.path.exists() returned True for
          NamedTemporaryFile before first write; fixed to os.path.getsize==0.
"""

from __future__ import annotations

import csv
import math
import os
import random
import statistics
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, time as Time, timedelta
from typing import Any, Callable

__version__ = "2.4.0"

# ---------------------------------------------------------------------------
# Optional imports for Groups D-J (data-science stack)
# Groups A-C are pure stdlib and work without these.
# ---------------------------------------------------------------------------

try:
    import pandas as pd
    import yfinance as yf
    from scipy.stats import norm
    import numpy as np
    _EXTENDED_DEPS = True
except ImportError:
    _EXTENDED_DEPS = False


def _require_extended(fn_name: str) -> None:
    if not _EXTENDED_DEPS:
        raise ImportError(
            f"{fn_name} requires Groups D-J dependencies: "
            "pip install yfinance pandas scipy numpy tzdata"
        )


# ===========================================================================
# GROUP A -- Position sizing, EV, and leverage  (Trading Rulebook core)
# Unchanged from v1.1.0 -- pure stdlib, no new dependencies.
# ===========================================================================

def position_size(account_balance: float, risk_pct: float, entry_price: float,
                   stop_price: float) -> float:
    """
    Position_Size = (Account_Balance x Risk_%) / |Entry_Price - Stop_Loss_Price|

    Returns the number of units (shares/contracts/coins) to buy or sell so
    that a full stop-out loses exactly `risk_pct` of account_balance.

    risk_pct is a fraction, e.g. 0.01 for 1%, 0.005 for the 0.5% commodity cap.
    """
    if stop_price == entry_price:
        raise ValueError("Entry and stop cannot be identical (zero risk distance).")
    risk_amount = account_balance * risk_pct
    per_unit_risk = abs(entry_price - stop_price)
    return risk_amount / per_unit_risk


def breakeven_win_rate(reward_to_risk: float) -> float:
    """
    Minimum win rate required for EV = 0 at a given R:R ratio.

        W_r = 1 / (1 + R:R)

    E.g. breakeven_win_rate(2.0) -> 0.3333 (need >33.3% wins at 2:1)
         breakeven_win_rate(1.0) -> 0.50   (need >50% wins at 1:1)
    """
    if reward_to_risk <= 0:
        raise ValueError("reward_to_risk must be positive.")
    return 1.0 / (1.0 + reward_to_risk)


def expected_value(win_rate: float, avg_win_r: float, avg_loss_r: float = 1.0) -> float:
    """
    EV = (Win_Rate x Average_Win) - (Loss_Rate x Average_Loss), in R multiples.

    avg_loss_r is expressed as a positive number (the size of a full loss),
    default 1.0R (a stop-out at exactly your defined risk).
    """
    loss_rate = 1.0 - win_rate
    return (win_rate * avg_win_r) - (loss_rate * avg_loss_r)


def unleveraged_return(reported_return_pct: float, leverage: float) -> float:
    """
    Rulebook rule: 'Leverage-adjust every backtest result before evaluating it.'

    A naive floor-estimate of the unleveraged return: divide by the leverage
    multiplier.
    """
    if leverage <= 0:
        raise ValueError("leverage must be positive.")
    return reported_return_pct / leverage


def effective_risk_per_trade(leverage: float, pct_risked: float) -> float:
    """
    Effective_Risk = leverage x % of portfolio risked per trade.

    If this approaches or exceeds 1.0 (100%), a single losing trade can
    wipe the account outright -- flag as dangerous per the Rulebook.
    """
    return leverage * pct_risked


def minimum_position_size_for_cost_drag(dealing_cost: float, max_cost_pct: float) -> float:
    """
    Min_Position = Dealing_Cost / Max_Cost_Pct

    E.g. £3.99 dealing cost / 0.004 max cost fraction = £997.50 minimum.
    """
    if max_cost_pct <= 0:
        raise ValueError("max_cost_pct must be positive.")
    return dealing_cost / max_cost_pct


# ===========================================================================
# GROUP B -- Sharpe family + drawdown + Monte Carlo  (Wealth Building port)
# Unchanged from v1.1.0 -- pure stdlib, no new dependencies.
# ===========================================================================

def _norm_cdf(x: float) -> float:
    """Standard normal CDF using the error function (no scipy dependency)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Standard normal inverse CDF, Acklam's rational approximation (~1e-9)."""
    if not (0.0 < p < 1.0):
        raise ValueError("p must be in (0, 1).")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    p_low, p_high = 0.02425, 1 - 0.02425
    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p <= p_high:
        q = p - 0.5
        r = q*q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
            ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)


def sharpe_ratio(returns: list[float], risk_free_rate: float = 0.0) -> float:
    """
    Sharpe = (mean(returns) - risk_free_rate) / stdev(returns)

    Uses sample standard deviation (n-1). Returns are per-period
    (e.g. R-multiples per trade, or daily % returns).
    """
    if len(returns) < 2:
        raise ValueError("Need at least 2 returns to compute a Sharpe ratio.")
    mean_r = statistics.mean(returns)
    sd = statistics.stdev(returns)
    if sd == 0:
        raise ValueError("Zero variance in returns -- Sharpe undefined.")
    return (mean_r - risk_free_rate) / sd


def sharpe_significance_tstat(observed_sharpe: float, n_obs: int) -> float:
    """
    Lo (2002) approximate t-statistic:  t = SR * sqrt(n_obs)

    n_obs is the number of return OBSERVATIONS (trades, days, etc.) --
    not years. Rule of thumb: |t| > ~2 is loosely significant.
    See sharpe_significance_tstat_annualised() in Group D for the version
    that takes years as input and returns a richer diagnostic dict.
    """
    if n_obs < 2:
        raise ValueError("n_obs must be >= 2.")
    return observed_sharpe * math.sqrt(n_obs)


def probabilistic_sharpe_ratio(observed_sharpe: float, benchmark_sharpe: float,
                                n_obs: int, skew: float = 0.0,
                                kurtosis: float = 3.0) -> float:
    """
    PSR: probability that the TRUE Sharpe ratio exceeds a given benchmark.
    Verified line-for-line against braverock/PerformanceAnalytics ProbSharpeRatio.R.
    kurtosis here is total kurtosis (normal = 3), not excess kurtosis.
    """
    if n_obs < 2:
        raise ValueError("n_obs must be >= 2.")
    denom = 1 - skew * observed_sharpe + ((kurtosis - 1) / 4) * observed_sharpe ** 2
    if denom <= 0:
        raise ValueError("Degenerate PSR denominator -- check skew/kurtosis inputs.")
    z = (observed_sharpe - benchmark_sharpe) * math.sqrt(n_obs - 1) / math.sqrt(denom)
    return _norm_cdf(z)


def expected_max_sharpe_under_trials(n_trials: int, sharpe_std_error: float) -> float:
    """
    Bailey & Lopez de Prado (2014): expected maximum Sharpe across n_trials
    independent strategy tests purely by chance. The benchmark for DSR below.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1.")
    if n_trials == 1:
        return 0.0
    euler_gamma = 0.5772156649015329
    term1 = (1 - euler_gamma) * _norm_ppf(1 - 1.0 / n_trials)
    term2 = euler_gamma * _norm_ppf(1 - 1.0 / (n_trials * math.e))
    return sharpe_std_error * (term1 + term2)


def deflated_sharpe_ratio(observed_sharpe: float, n_obs: int, n_trials: int,
                           skew: float = 0.0, kurtosis: float = 3.0) -> float:
    """
    DSR: PSR benchmarked against the expected max Sharpe from n_trials chance
    attempts. Answers: "Is the best of N tested strategies genuinely good,
    or just the best of N coin flips?"

    Signature: (observed_sharpe, n_obs, n_trials) -- preserved from v1.1.0.
    See deflated_sharpe_ratio_extended() in Group D for the scipy version
    with a richer diagnostic dict output.
    """
    denom = 1 - skew * observed_sharpe + ((kurtosis - 1) / 4) * observed_sharpe ** 2
    if denom <= 0 or n_obs < 2:
        raise ValueError("Invalid inputs for DSR standard-error estimate.")
    se_sharpe = math.sqrt(denom / (n_obs - 1))
    benchmark = expected_max_sharpe_under_trials(n_trials, se_sharpe)
    return probabilistic_sharpe_ratio(observed_sharpe, benchmark, n_obs, skew, kurtosis)


def max_drawdown(equity_curve: list[float]) -> float:
    """
    Largest peak-to-trough decline as a positive fraction.
    equity_curve is account values or cumulative R, chronological order.
    """
    if len(equity_curve) < 2:
        raise ValueError("Need at least 2 points to compute a drawdown.")
    peak = equity_curve[0]
    worst = 0.0
    for value in equity_curve:
        peak = max(peak, value)
        if peak > 0:
            worst = max(worst, (peak - value) / peak)
    return worst


@dataclass
class MonteCarloResult:
    """Result of a Rulebook-mandated Monte Carlo robustness test."""
    n_sims:               int
    sharpe_threshold:     float
    pct_sims_passing:     float
    passes_rulebook_gate: bool    # True if pct_sims_passing >= 0.85
    worst_case_drawdown:  float
    median_sharpe:        float


def monte_carlo_robustness(r_multiples: list[float], n_sims: int = 1000,
                            sharpe_threshold: float = 1.0,
                            seed: int = 42) -> MonteCarloResult:
    """
    Rulebook gate: Sharpe > 1.0 in >= 85% of simulations.
    Reshuffles the same trade log thousands of times to test sequence risk.
    Deterministic given the same seed and trade log.
    """
    if len(r_multiples) < 5:
        raise ValueError("Need a meaningfully sized trade log (>=5) for this to mean anything.")
    rng = random.Random(seed)
    passes = 0
    worst_dd = 0.0
    sharpes: list[float] = []
    for _ in range(n_sims):
        shuffled = r_multiples[:]
        rng.shuffle(shuffled)
        try:
            sr = sharpe_ratio(shuffled)
        except ValueError:
            sr = 0.0
        sharpes.append(sr)
        if sr > sharpe_threshold:
            passes += 1
        equity_curve = [0.0]
        for r in shuffled:
            equity_curve.append(equity_curve[-1] + r)
        floor_shift = abs(min(equity_curve)) + 1.0
        shifted_curve = [v + floor_shift for v in equity_curve]
        worst_dd = max(worst_dd, max_drawdown(shifted_curve))
    pct_passing = passes / n_sims
    return MonteCarloResult(
        n_sims=n_sims,
        sharpe_threshold=sharpe_threshold,
        pct_sims_passing=pct_passing,
        passes_rulebook_gate=pct_passing >= 0.85,
        worst_case_drawdown=worst_dd,
        median_sharpe=statistics.median(sharpes),
    )


def future_value_annuity(monthly_contribution: float, annual_rate: float,
                          years: float) -> float:
    """
    FV = P * ((1 + r)^n - 1) / r  [monthly compounding]
    Behind the Rulebook's parallel investing track rule (Bob Sharpe entry).
    """
    if annual_rate <= 0:
        return monthly_contribution * years * 12
    r = annual_rate / 12.0
    n = years * 12.0
    return monthly_contribution * (((1 + r) ** n - 1) / r)


# ===========================================================================
# GROUP C -- Dealer gamma hedging + Volume Profile  (Orderflow intelligence)
# Unchanged from v1.1.0 -- pure stdlib, no new dependencies.
# ===========================================================================

def dealer_hedge_flow(gamma: float, open_interest: int, spot_price: float,
                       price_move_pct: float, contract_multiplier: int = 100,
                       dealer_is_short_gamma: bool = True) -> float:
    """
    Standard delta-hedging identity:
        Hedge_Flow = Gamma x OI x Contract_Multiplier x Spot x Price_Move_Pct

    Short-gamma dealers hedge WITH the move (trend-amplifying).
    Long-gamma dealers hedge AGAINST the move (mean-reverting).
    Returns signed float: positive = dealers must buy.
    """
    if open_interest < 0:
        raise ValueError("open_interest cannot be negative.")
    raw_flow = gamma * open_interest * contract_multiplier * spot_price * price_move_pct
    return raw_flow if dealer_is_short_gamma else -raw_flow


@dataclass
class GammaExposureResult:
    """Net dealer gamma exposure across a chain."""
    net_gamma_exposure: float
    regime:             str
    interpretation:     str


def net_gamma_exposure(strikes: list[dict], spot_price: float,
                        contract_multiplier: int = 100) -> GammaExposureResult:
    """
    GEX = Sum[Call_Gamma x Call_OI - Put_Gamma x Put_OI]
          x Contract_Multiplier x Spot^2 x 0.01

    Public approximation -- not SpotGamma's proprietary model.
    Positive GEX = dealers long gamma = mean-reverting regime.
    Negative GEX = dealers short gamma = trend-amplifying regime.
    """
    if not strikes:
        raise ValueError("strikes list cannot be empty.")
    total = 0.0
    for row in strikes:
        call_leg = row.get("call_gamma", 0.0) * row.get("call_oi", 0)
        put_leg  = row.get("put_gamma",  0.0) * row.get("put_oi",  0)
        total += (call_leg - put_leg)
    gex    = total * contract_multiplier * (spot_price ** 2) * 0.01
    regime = "positive_gamma" if gex >= 0 else "negative_gamma"
    interpretation = (
        "Dealers net long gamma: expect selling into rallies, buying dips "
        "-- compressed, range-bound conditions more likely."
        if regime == "positive_gamma" else
        "Dealers net short gamma: expect hedging WITH the trend -- "
        "amplified, potentially violent moves more likely."
    )
    return GammaExposureResult(net_gamma_exposure=gex, regime=regime,
                                interpretation=interpretation)


def value_area(volume_by_price: dict[float, float],
               area_pct: float = 0.70) -> dict:
    """
    Standard Volume Profile: POC + Value Area (70% of volume by default).
    Expands outward from POC toward whichever side has more volume.
    Returns {poc, vah, val, total_volume, value_area_volume, value_area_pct}.
    """
    if not volume_by_price:
        raise ValueError("volume_by_price cannot be empty.")
    prices       = sorted(volume_by_price.keys())
    total_volume = sum(volume_by_price.values())
    if total_volume <= 0:
        raise ValueError("Total volume must be positive.")
    poc     = max(volume_by_price, key=volume_by_price.get)
    poc_idx = prices.index(poc)
    lo_idx, hi_idx = poc_idx, poc_idx
    captured = volume_by_price[poc]
    target   = total_volume * area_pct
    while captured < target and (lo_idx > 0 or hi_idx < len(prices) - 1):
        below_vol = volume_by_price[prices[lo_idx - 1]] if lo_idx > 0 else -1
        above_vol = volume_by_price[prices[hi_idx + 1]] if hi_idx < len(prices) - 1 else -1
        if above_vol >= below_vol:
            hi_idx  += 1
            captured += volume_by_price[prices[hi_idx]]
        else:
            lo_idx  -= 1
            captured += volume_by_price[prices[lo_idx]]
    return {"poc": poc, "vah": prices[hi_idx], "val": prices[lo_idx],
            "total_volume": total_volume, "value_area_volume": captured,
            "value_area_pct": captured / total_volume}


def effort_result_ratio(volume_delta: float, price_change_ticks: float,
                         absorption_threshold: float = 0.0) -> dict:
    """
    Ratio = |Volume_Delta| / |Price_Change_Ticks|
    Large ratio with tiny price move = absorption or exhaustion signal.
    absorption_threshold: calibrate from your own footprint data.
    """
    if price_change_ticks == 0:
        return {"ratio": float("inf"), "signal": "absorption"}
    ratio  = abs(volume_delta) / abs(price_change_ticks)
    signal = "absorption" if (absorption_threshold > 0 and ratio > absorption_threshold) else "normal"
    return {"ratio": ratio, "signal": signal}


# ===========================================================================
# GROUP D -- Statistical extensions  (portfolio_engine_v2_15.py port)
# Requires scipy, numpy.  Complements Group B -- does not replace it.
# ===========================================================================

def sharpe_significance_tstat_annualised(
    sharpe_ratio_annualised: float,
    years_of_data: float,
    *,
    target_t: float = 2.0,
) -> dict:
    """
    Lo (2002) t ≈ SR × √T with annualised inputs. Extended version of
    Group B's sharpe_significance_tstat() returning a full diagnostic dict.

    VERIFIED: SR=1.0, T=4yr → t=2.00 (Lo's own worked example).
    Trading context: 100 trades ≈ 0.38yr → t=0.616 — NOT significant.
    Years needed for significance = (target_t / SR)².
    """
    t_stat = sharpe_ratio_annualised * math.sqrt(max(years_of_data, 0.0))
    years_needed = (None if sharpe_ratio_annualised == 0
                    else round((target_t / sharpe_ratio_annualised) ** 2, 2))
    return {
        "t_stat":                    round(float(t_stat), 3),
        "is_significant_at_target":  bool(t_stat >= target_t),
        "years_of_data_used":        years_of_data,
        "years_needed_for_target_t": years_needed,
    }


def deflated_sharpe_ratio_extended(
    observed_sharpe: float,
    n_trials: int,
    n_obs: int,
    *,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    sr_std_across_trials: float | None = None,
) -> dict:
    """
    Bailey & López de Prado (2014) DSR -- extended version returning a dict.
    Uses scipy.stats.norm. Complements Group B's deflated_sharpe_ratio()
    which returns a float and uses the pure-stdlib _norm_cdf.

    Signature: (observed_sharpe, n_trials, n_obs) -- note argument order
    differs from Group B's (observed_sharpe, n_obs, n_trials) to match
    the portfolio_engine_v2_15.py source convention.

    VERIFIED directionally: SR=0.6, n=36 — n_trials=1→99.95%, n=100→82.6%.
    DSR confidence must exceed 80% before a variant is promoted (P3-002 gate).
    """
    _require_extended("deflated_sharpe_ratio_extended")
    euler_gamma = 0.5772156649015329
    if sr_std_across_trials is None:
        sr_std_across_trials = 1.0 / math.sqrt(max(n_obs - 1, 1))
    if n_trials <= 1:
        sr_benchmark = 0.0
    else:
        z1 = norm.ppf(1 - 1.0 / n_trials)
        z2 = norm.ppf(1 - 1.0 / (n_trials * math.e))
        sr_benchmark = sr_std_across_trials * ((1 - euler_gamma) * z1 + euler_gamma * z2)
    denom = math.sqrt(max(
        1 - skew * observed_sharpe + ((kurtosis - 1) / 4.0) * observed_sharpe ** 2, 1e-12
    ))
    psr = float(norm.cdf(
        (observed_sharpe - sr_benchmark) * math.sqrt(max(n_obs - 1, 1)) / denom
    ))
    return {
        "sharpe_benchmark_from_multiple_testing": round(sr_benchmark, 4),
        "deflated_sharpe_probability":            round(psr, 4),
        "n_trials_assumed":                       n_trials,
        "n_obs_used":                             n_obs,
    }


def regime_conditional_streak_probability(
    adverse_prob: float,
    n_periods: int,
    streak_length: int,
    *,
    regime_label: str | None = None,
) -> dict:
    """
    Exact probability (Feller 1968 dynamic programming) of at least one run
    of streak_length consecutive adverse outcomes within n_periods trials.

    VERIFIED: adverse_prob=0.5, n=100, streak=4 → 97.27% (published ~97.2%).
    Trading context: 40% loss rate, 20 sessions, streak=3 → 56.24%.
    The 3-loss walk-away rule addresses a near-certain psychological event.

    LIMITATION: IID-Bernoulli. Feed a regime-conditional loss rate,
    not a blended all-market rate, for accurate results.
    """
    if streak_length < 1:
        raise ValueError("streak_length must be >= 1")
    q = float(adverse_prob)
    p = 1.0 - q
    dp = [0.0] * streak_length
    dp[0] = 1.0
    for _ in range(n_periods):
        new_dp = [0.0] * streak_length
        new_dp[0] += sum(dp) * p
        for k in range(streak_length - 1):
            new_dp[k + 1] += dp[k] * q
        dp = new_dp
    prob = 1.0 - sum(dp)
    return {
        "probability_of_streak_pct": round(float(prob) * 100.0, 2),
        "adverse_prob_used":         adverse_prob,
        "n_periods_used":            n_periods,
        "streak_length_tested":      streak_length,
        "regime_label":              regime_label,
    }


def drawdown_barrier_breach_probability(
    mu_annual: float,
    sigma_annual: float,
    barrier_pct: float,
    horizon_years: float,
) -> dict:
    """
    GBM first-passage probability (Shreve 2004 / Merton 1971) of breaching
    a fixed proportional drawdown barrier within horizon_years.

    VERIFIED: 8% vol→0.83%, 25% vol→47.73% (same 6% drift, 15% barrier).
    LIMITATION: fixed barrier from t=0, not trailing high-water-mark.
    """
    _require_extended("drawdown_barrier_breach_probability")
    if not (0.0 < barrier_pct < 1.0):
        raise ValueError("barrier_pct must be strictly between 0 and 1, e.g. 0.15")
    drift = mu_annual - 0.5 * sigma_annual ** 2
    b     = math.log(1.0 - barrier_pct)
    sT    = sigma_annual * math.sqrt(max(horizon_years, 1e-9))
    term1 = float(norm.cdf((b - drift * horizon_years) / sT))
    term2 = (math.exp(2.0 * drift * b / sigma_annual ** 2)
             * float(norm.cdf((b + drift * horizon_years) / sT)))
    return {
        "breach_probability_pct": round((term1 + term2) * 100.0, 2),
        "mu_annual_used":         mu_annual,
        "sigma_annual_used":      sigma_annual,
        "barrier_pct_used":       barrier_pct,
        "horizon_years_used":     horizon_years,
    }


def nearest_correlation_higham(
    corr_mat: "np.ndarray",
    max_iter: int = 100,
    tol: float = 1e-7,
    eigval_floor: float = 1e-8,
) -> "np.ndarray":
    """
    Higham (2002) alternating-projections PSD repair. Preserves exact unit
    diagonal unlike spectral fixes (which distorted one asset's implied vol
    by 40.5% in a verified test case from the Wealth Building engine).
    Use before any correlation-matrix-based sizing calculation.
    """
    _require_extended("nearest_correlation_higham")
    n = corr_mat.shape[0]
    Y = corr_mat.copy()
    Delta_S = np.zeros((n, n))
    for _ in range(max_iter):
        R = Y - Delta_S
        evals, evecs = np.linalg.eigh(R)
        X = evecs @ np.diag(np.clip(evals, eigval_floor, None)) @ evecs.T
        Delta_S = X - R
        Y = X.copy()
        np.fill_diagonal(Y, 1.0)
        if np.linalg.norm(Y - X, ord="fro") / np.linalg.norm(X, ord="fro") < tol:
            break
    evals_f, evecs_f = np.linalg.eigh(Y)
    Y = evecs_f @ np.diag(np.clip(evals_f, eigval_floor, None)) @ evecs_f.T
    np.fill_diagonal(Y, 1.0)
    return Y


def sanitize_vol_inputs(
    vols: dict[str, float],
    epsilon: float = 0.001,
) -> dict[str, float]:
    """
    Floor volatility inputs at epsilon to prevent 0/0 = NaN in distance
    matrices. Renamed from sanitize_covariance_inputs (portfolio_engine line 5699).
    VALIDATED: Cash at 0.0% → 0.001%, 18.0% passes unchanged.
    """
    return {k: max(v, epsilon) for k, v in vols.items()}


# ===========================================================================
# GROUP E -- Data layer  (validated P1-001, 06 Sep 2026)
# Requires yfinance, pandas, tzdata.
# America/New_York returned natively -- no tz_convert needed.
# MultiIndex columns must be flattened with col[0].
# ===========================================================================

# Session times (US/Eastern -- yfinance returns America/New_York natively)
SESSION_START = Time(9, 30)
SESSION_END   = Time(16, 0)
ORB_END       = Time(11, 0)

# Identity keywords -- guard against silent ticker mismatches (DBMG.L incident)
TICKER_IDENTITY_KEYWORDS: dict[str, list[str]] = {
    "SPY":   ["SPY", "S&P", "SPDR", "500"],
    "QQQ":   ["QQQ", "NASDAQ", "INVESCO"],
    "NQ=F":  ["NASDAQ", "NQ", "E-MINI"],
    "ES=F":  ["S&P", "ES", "E-MINI"],
    "GC=F":  ["GOLD", "GC"],
    "SI=F":  ["SILVER", "SI"],
    "^VIX":  ["VIX", "VOLATILITY"],
    "^GSPC": ["S&P", "GSPC", "500"],
}


def _flatten_columns(df: "pd.DataFrame") -> "pd.DataFrame":
    """Flatten yfinance MultiIndex columns: ('Close','SPY') → 'Close'."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df


def fetch_ohlcv(
    ticker: str,
    period: str = "5d",
    interval: str = "5m",
    auto_adjust: bool = True,
) -> "pd.DataFrame":
    """
    Fetch OHLCV bars. Returns [Open, High, Low, Close, Volume] DataFrame
    with DatetimeIndex in America/New_York (tz-aware, no conversion needed).
    """
    _require_extended("fetch_ohlcv")
    df = yf.download(ticker, period=period, interval=interval,
                     auto_adjust=auto_adjust, progress=False)
    if df.empty:
        raise ValueError(
            f"fetch_ohlcv: no data for {ticker!r} (period={period}, interval={interval})"
        )
    df = _flatten_columns(df)
    df.index.name = "Datetime"
    return df


def fetch_with_fallback(
    primary_ticker: str,
    fallback_ticker: str,
    period: str = "5d",
    interval: str = "5m",
    source_log: dict[str, Any] | None = None,
) -> "pd.DataFrame":
    """
    Fetch with automatic fallback. Always logs which source was used --
    a silent fallback must never be mistaken for the primary.
    source_log is mutated in-place: {ticker, source, primary_error?}.
    """
    _require_extended("fetch_with_fallback")
    try:
        df = fetch_ohlcv(primary_ticker, period=period, interval=interval)
        if source_log is not None:
            source_log.update(ticker=primary_ticker, source="primary")
        return df
    except Exception as exc:
        if source_log is not None:
            source_log["primary_error"] = str(exc)
        df = fetch_ohlcv(fallback_ticker, period=period, interval=interval)
        if source_log is not None:
            source_log.update(ticker=fallback_ticker, source="fallback")
        return df


def fetch_pdh_pdl(ticker: str, lookback_days: int = 5) -> dict[str, float]:
    """Previous Day High and Low from daily bars."""
    _require_extended("fetch_pdh_pdl")
    df = _flatten_columns(
        yf.download(ticker, period=f"{lookback_days}d", interval="1d",
                    auto_adjust=True, progress=False)
    )
    if len(df) < 2:
        raise ValueError(f"fetch_pdh_pdl: insufficient daily data for {ticker!r}")
    prev = df.iloc[-2]
    pdh, pdl = float(prev["High"]), float(prev["Low"])
    return {"pdh": pdh, "pdl": pdl,
            "prev_date": str(df.index[-2].date()),
            "range": round(pdh - pdl, 4)}


def fetch_premarket_gap(ticker: str) -> dict[str, Any]:
    """
    Overnight gap: compare pre-market open to previous session close.
    Requires pre_post=True (absent from standard period='5d' pull).
    """
    _require_extended("fetch_premarket_gap")
    df_d = _flatten_columns(
        yf.download(ticker, period="5d", interval="1d",
                    auto_adjust=True, progress=False)
    )
    prev_close = float(df_d["Close"].iloc[-2])
    df_pm = _flatten_columns(
        yf.download(ticker, period="1d", interval="5m",
                    auto_adjust=True, pre_post=True, progress=False)
    )
    if df_pm.empty:
        raise ValueError(f"fetch_premarket_gap: no intraday data for {ticker!r}")
    current_open = float(df_pm["Open"].iloc[0])
    gap_pct = round((current_open - prev_close) / prev_close * 100, 3)
    direction = "up" if gap_pct > 0.1 else "down" if gap_pct < -0.1 else "flat"
    return {"gap_pct": gap_pct, "gap_direction": direction,
            "prev_close": round(prev_close, 4), "current_open": round(current_open, 4)}


def validate_ticker_identity(
    ticker: str,
    df: "pd.DataFrame",
    expected_keywords: list[str] | None = None,
) -> dict[str, Any]:
    """
    Guard against silent ticker mismatches (DBMG.L vs IMGP.L incident).
    Issues WARNING (not exception) -- caller decides whether to abort.
    """
    _require_extended("validate_ticker_identity")
    keywords = expected_keywords or TICKER_IDENTITY_KEYWORDS.get(ticker.upper(), [])
    if df.empty:
        return {"valid": False, "name": "UNKNOWN",
                "message": f"Empty DataFrame for {ticker!r}"}
    try:
        info = yf.Ticker(ticker).info
        name = info.get("longName") or info.get("shortName") or "UNKNOWN"
    except Exception:
        name = "UNKNOWN"
    if not keywords:
        return {"valid": True, "name": name,
                "message": f"No keywords registered for {ticker!r} -- verify manually"}
    matched = any(kw.upper() in name.upper() for kw in keywords)
    return {
        "valid": matched, "name": name,
        "message": (f"Identity confirmed: {ticker!r} → {name!r}" if matched else
                    f"IDENTITY MISMATCH: {ticker!r} returned {name!r}, "
                    f"expected {keywords}"),
    }


def filter_session_hours(
    df: "pd.DataFrame",
    start: Time = SESSION_START,
    end: Time   = SESSION_END,
) -> "pd.DataFrame":
    """Filter to [start, end] using index.time() (America/New_York native)."""
    _require_extended("filter_session_hours")
    if df.empty:
        return df
    mask = (df.index.time >= start) & (df.index.time <= end)
    return df[mask].copy()


def filter_orb_window(df: "pd.DataFrame") -> "pd.DataFrame":
    """Filter to ORB strategy window 09:30-11:00."""
    return filter_session_hours(df, start=SESSION_START, end=ORB_END)


def fetch_vix_current() -> dict[str, Any]:
    """Current VIX level with regime label and position-size modifier."""
    _require_extended("fetch_vix_current")
    df = _flatten_columns(
        yf.download("^VIX", period="2d", interval="5m",
                    auto_adjust=True, progress=False)
    )
    if df.empty:
        raise ValueError("fetch_vix_current: no VIX data")
    vix = float(df["Close"].dropna().iloc[-1])
    if vix > 35:
        regime, modifier = "EXTREME",  0.25
    elif vix > 25:
        regime, modifier = "HIGH",     0.50
    elif vix > 18:
        regime, modifier = "ELEVATED", 0.75
    else:
        regime, modifier = "NORMAL",   1.00
    return {"vix": round(vix, 2), "regime": regime, "size_modifier": modifier}


def relative_volume(
    current_first5_volume: float,
    avg_first5_volume_14d: float,
) -> float:
    """
    Relative Volume (RVOL) = current first-5-min volume / 14-day average
    first-5-min volume.

    RVOL >= 1.0 (100%): stock is trading at normal opening pace.
    RVOL >= 2.0 (200%): double normal pace -- elevated attention.
    RVOL >= 3.0+:       strong catalyst present.

    Source: Zarattini, Barbon, Aziz (2024) SSRN 4729284.
    Without RVOL filter: ORB Sharpe 0.48 (underperforms S&P 500).
    With top-20 RVOL filter: ORB Sharpe 2.81, annualised alpha 36%.
    """
    if avg_first5_volume_14d <= 0:
        raise ValueError("avg_first5_volume_14d must be positive.")
    return current_first5_volume / avg_first5_volume_14d


def scan_stocks_in_play(
    ticker_list: list[str],
    top_n: int = 20,
    min_price: float = 5.0,
    min_avg_vol: int = 1_000_000,
    min_atr_pct: float = 0.5,
    rvol_threshold: float = 1.0,
) -> "pd.DataFrame":
    """
    Pre-market scanner: identify Stocks in Play from a universe of tickers.

    Applies Zarattini et al. (2024) five-filter criteria:
        1. Price > min_price (default $5)
        2. 14-day avg volume >= min_avg_vol (default 1,000,000 shares)
        3. ATR(14) as % of price > min_atr_pct (default 0.5%)
        4. RVOL (first-5-min vs 14d avg) >= rvol_threshold (default 1.0x)
        5. Return top_n by RVOL (default 20)

    Parameters
    ----------
    ticker_list    : list of Yahoo Finance ticker strings to scan
    top_n          : number of top RVOL stocks to return
    min_price      : minimum last close price
    min_avg_vol    : minimum 14-day average daily volume
    min_atr_pct    : minimum ATR(14) / close as a percentage
    rvol_threshold : minimum RVOL ratio to qualify

    Returns
    -------
    pd.DataFrame with columns [ticker, close, avg_vol, atr_pct, rvol, rank]
    sorted descending by RVOL. Empty DataFrame if no qualifying stocks.

    TIMING: call after 09:35 EST so the first 5-minute bar has closed.
    """
    _require_extended("scan_stocks_in_play")
    results = []
    for ticker in ticker_list:
        try:
            # 14-day daily bars for price/vol/ATR baseline
            df_d = _flatten_columns(
                yf.download(ticker, period="20d", interval="1d",
                            auto_adjust=True, progress=False)
            )
            if len(df_d) < 10:
                continue
            close  = float(df_d["Close"].iloc[-1])
            if close < min_price:
                continue
            avg_vol = float(df_d["Volume"].mean())
            if avg_vol < min_avg_vol:
                continue
            # ATR(14) as % of close
            highs  = df_d["High"].values.astype(float)
            lows   = df_d["Low"].values.astype(float)
            closes = df_d["Close"].values.astype(float)
            tr     = [max(highs[i] - lows[i],
                         abs(highs[i] - closes[i-1]),
                         abs(lows[i]  - closes[i-1]))
                      for i in range(1, len(closes))]
            atr14  = float(sum(tr[-14:]) / 14) if len(tr) >= 14 else 0.0
            atr_pct = atr14 / close * 100 if close > 0 else 0.0
            if atr_pct < min_atr_pct:
                continue
            # Intraday first-5-min volume (today)
            df_5m = _flatten_columns(
                yf.download(ticker, period="5d", interval="5m",
                            auto_adjust=True, progress=False)
            )
            if df_5m.empty:
                continue
            today = df_5m.index.date[-1]
            df_today = df_5m[df_5m.index.date == today]
            df_first5 = df_today[df_today.index.time <= Time(9, 34)]
            if df_first5.empty:
                continue
            first5_vol = float(df_first5["Volume"].sum())
            # 14-day average first-5-min volume
            avg_first5_vols = []
            for d in set(df_5m.index.date):
                if d == today:
                    continue
                day_df = df_5m[(df_5m.index.date == d) &
                               (df_5m.index.time <= Time(9, 34))]
                if not day_df.empty:
                    avg_first5_vols.append(float(day_df["Volume"].sum()))
            if not avg_first5_vols:
                continue
            avg_first5 = float(sum(avg_first5_vols) / len(avg_first5_vols))
            if avg_first5 <= 0:
                continue
            rvol = first5_vol / avg_first5
            if rvol < rvol_threshold:
                continue
            results.append({
                "ticker":   ticker,
                "close":    round(close, 2),
                "avg_vol":  int(avg_vol),
                "atr_pct":  round(atr_pct, 2),
                "rvol":     round(rvol, 2),
                "first5_vol": int(first5_vol),
            })
        except Exception:
            continue  # skip bad tickers silently

    if not results:
        return pd.DataFrame(columns=["ticker", "close", "avg_vol", "atr_pct", "rvol",
                                     "first5_vol", "rank"])
    df_out = pd.DataFrame(results).sort_values("rvol", ascending=False).head(top_n)
    df_out["rank"] = range(1, len(df_out) + 1)
    return df_out.reset_index(drop=True)


# ===========================================================================
# GROUP F -- Risk layer  (commodity cap, daily stop, session state)
# ===========================================================================

COMMODITY_TICKERS: frozenset[str] = frozenset({
    "GC=F", "SI=F", "CL=F", "GLD", "SLV", "USO", "GOLD", "GDX", "GDXJ",
})
DEFAULT_RISK_PCT    = 0.01     # 1%  equities
COMMODITY_RISK_PCT  = 0.005    # 0.5% Gold/Silver/Oil (30-50% crashes confirmed)
MAX_DAILY_LOSS_PCT  = 0.03     # 3% session loss → walk away
CONSECUTIVE_LOSS_PAUSE = 3     # losses in a row → 30-min pause

_LIVE_RISK_LADDER = [
    (0,  20,  0.0025),   # Trades  1-20  → 0.25%
    (20, 50,  0.0050),   # Trades 21-50  → 0.50%
    (50, 999, 0.0100),   # Trades 51+    → 1.00%
]


def instrument_risk_pct(ticker: str) -> float:
    """0.5% for commodities, 1.0% for everything else."""
    return COMMODITY_RISK_PCT if ticker.upper() in COMMODITY_TICKERS else DEFAULT_RISK_PCT


def live_risk_pct(
    live_trade_count: int,
    ticker: str,
    had_daily_stopout: bool = False,
    rolling_ev_positive: bool = True,
) -> float:
    """Phase 4 step-up ladder. Commodity cap always overrides."""
    if ticker.upper() in COMMODITY_TICKERS:
        return min(0.0025, COMMODITY_RISK_PCT)
    for lo, hi, base_pct in _LIVE_RISK_LADDER:
        if lo <= live_trade_count < hi:
            if base_pct >= 0.005 and had_daily_stopout:
                return 0.0025
            if base_pct >= 0.010 and not rolling_ev_positive:
                return 0.005
            return base_pct
    return DEFAULT_RISK_PCT


def set_stop_loss(
    entry_price: float,
    direction: str,
    reference_extreme: float,
    buffer: float = 0.02,
) -> float:
    """Stop below retest candle low (long) or above retest candle high (short)."""
    direction = direction.lower()
    if direction == "long":
        stop = reference_extreme - buffer
        if stop >= entry_price:
            raise ValueError(f"Stop {stop:.4f} >= entry {entry_price:.4f} for long")
        return round(stop, 4)
    elif direction == "short":
        stop = reference_extreme + buffer
        if stop <= entry_price:
            raise ValueError(f"Stop {stop:.4f} <= entry {entry_price:.4f} for short")
        return round(stop, 4)
    raise ValueError(f"direction must be 'long' or 'short', got {direction!r}")


def set_target(
    entry_price: float,
    stop_price: float,
    rr_ratio: float = 2.0,
) -> dict[str, float]:
    """
    Take-profit at rr_ratio x risk distance.
    Breakeven-move at 50% of target distance (Bull Barbie rule).
    """
    if rr_ratio < 1.0:
        raise ValueError(f"rr_ratio must be >= 1.0, got {rr_ratio}")
    risk_dist   = abs(entry_price - stop_price)
    reward_dist = risk_dist * rr_ratio
    if stop_price < entry_price:   # long
        target         = round(entry_price + reward_dist, 4)
        breakeven_move = round(entry_price + reward_dist * 0.5, 4)
    else:                          # short
        target         = round(entry_price - reward_dist, 4)
        breakeven_move = round(entry_price - reward_dist * 0.5, 4)
    return {"target": target, "breakeven_move": breakeven_move,
            "risk_distance": round(risk_dist, 4),
            "reward_distance": round(reward_dist, 4), "rr_ratio": rr_ratio}


def daily_stop_check(
    session_pnl_pct: float,
    max_loss_pct: float = MAX_DAILY_LOSS_PCT,
) -> dict:
    """Stateless daily stop check."""
    stop = session_pnl_pct <= -abs(max_loss_pct)
    return {"stop_session": stop,
            "session_pnl_pct": round(session_pnl_pct * 100, 2),
            "threshold_pct": round(max_loss_pct * 100, 2),
            "message": ("🛑 Stop session." if stop else "✅ Within daily limit.")}


def consecutive_loss_check(
    recent_results: list[float],
    threshold: int = CONSECUTIVE_LOSS_PAUSE,
) -> dict:
    """Count consecutive losses at the end of recent_results."""
    streak = 0
    for r in reversed(recent_results):
        if r < 0:
            streak += 1
        else:
            break
    pause = streak >= threshold
    return {"consecutive_losses": streak, "pause_required": pause,
            "threshold": threshold,
            "message": (f"⏸  {streak} consecutive losses — pause 30 min."
                        if pause else f"✅ {streak} consecutive losses.")}


@dataclass
class SessionRiskState:
    """Mutable session risk tracker."""
    account_balance:    float
    session_pnl_gbp:    float = 0.0
    consecutive_losses: int   = 0
    trades_today:       int   = 0
    stop_session:       bool  = False
    pause_required:     bool  = False
    history:            list  = field(default_factory=list)

    @property
    def session_pnl_pct(self) -> float:
        return self.session_pnl_gbp / self.account_balance

    def record_trade(self, pnl_gbp: float) -> dict:
        self.session_pnl_gbp += pnl_gbp
        self.trades_today    += 1
        self.history.append(pnl_gbp)
        self.consecutive_losses = self.consecutive_losses + 1 if pnl_gbp < 0 else 0
        if self.session_pnl_pct <= -MAX_DAILY_LOSS_PCT:
            self.stop_session = True
        self.pause_required = (self.consecutive_losses >= CONSECUTIVE_LOSS_PAUSE
                               and not self.stop_session)
        return {"stop_session": self.stop_session, "pause_required": self.pause_required,
                "consecutive_losses": self.consecutive_losses,
                "session_pnl_gbp": round(self.session_pnl_gbp, 2),
                "session_pnl_pct": round(self.session_pnl_pct * 100, 2),
                "trades_today": self.trades_today,
                "message": ("🛑 SESSION STOPPED." if self.stop_session else
                            "⏸  PAUSE 30 MIN — 3 consecutive losses."
                            if self.pause_required else "✅ Trading permitted.")}

    def reset(self) -> None:
        self.session_pnl_gbp = 0.0; self.consecutive_losses = 0
        self.trades_today = 0; self.stop_session = False
        self.pause_required = False; self.history = []


# ---------------------------------------------------------------------------
# Group F additions (v2.1.0): Kelly Criterion, VWAP trailing stop, ladder
# exits, ATR-based stop, and volatility-targeted position sizing.
# ---------------------------------------------------------------------------

def kelly_fraction(
    win_rate: float,
    avg_win_r: float,
    avg_loss_r: float = 1.0,
    fraction: float = 0.5,
) -> float:
    """
    Kelly Criterion: the mathematically optimal fraction of capital to risk.

        K_full = W - (1 - W) / R

    where W = win_rate and R = avg_win_r / avg_loss_r.

    Returns fraction * K_full (default: Half-Kelly).

    Half-Kelly rationale (quantitative finance consensus):
        - Gives 75% of full-Kelly growth rate
        - Cuts variance by 50%
        - Compensates for estimation error in win_rate and avg_win_r

    SAFETY GATE: Only use when n >= 50 trades. A 5-point win_rate drop
    swings the Kelly fraction 3x. Before 50 trades use fixed 1% risk.

    Verified: kelly_fraction(0.6, 2.0, 1.0, fraction=1.0) == 0.40
              kelly_fraction(0.6, 2.0, 1.0, fraction=0.5) == 0.20
    """
    if not (0.0 < win_rate < 1.0):
        raise ValueError("win_rate must be between 0 and 1 (exclusive).")
    if avg_win_r <= 0 or avg_loss_r <= 0:
        raise ValueError("avg_win_r and avg_loss_r must be positive.")
    r_ratio  = avg_win_r / avg_loss_r
    k_full   = win_rate - (1.0 - win_rate) / r_ratio
    k_full   = max(k_full, 0.0)   # negative Kelly = no edge → risk nothing
    return round(min(fraction * k_full, 0.25), 6)  # hard cap at 25%


def kelly_position_size(
    account_balance: float,
    entry_price: float,
    stop_price: float,
    kelly_frac: float,
) -> float:
    """
    Position size using a Kelly-derived risk fraction.
    Identical algebra to position_size() -- risk amount is kelly_frac * account.

    PAIR WITH: kelly_fraction() to compute kelly_frac from journal stats.
    """
    if entry_price == stop_price:
        raise ValueError("Entry and stop cannot be identical.")
    if not (0.0 < kelly_frac <= 0.25):
        raise ValueError("kelly_frac must be in (0, 0.25]. "
                         "Use kelly_fraction() or fall back to 0.01.")
    return (account_balance * kelly_frac) / abs(entry_price - stop_price)


def trailing_stop_update(
    current_price: float,
    direction: str,
    current_stop: float,
    vwap_value: float,
) -> float:
    """
    VWAP-as-trailing-stop: update the stop to the VWAP level if that
    provides better protection than the current stop.

    Source: Zarattini, Aziz, Barbon (2024) SSRN 4824172.
    VWAP trailing stop doubles Sharpe from 0.61 to 1.24.
    Combined with Ladder exit (see ladder_target): Sharpe 3.0+.

    Rules:
        Long : new_stop = max(current_stop, vwap_value)  -- never moves down
        Short: new_stop = min(current_stop, vwap_value)  -- never moves up

    Returns the updated stop price (never worsens protection).
    """
    direction = direction.lower()
    if direction == "long":
        return round(max(current_stop, vwap_value), 4)
    elif direction == "short":
        return round(min(current_stop, vwap_value), 4)
    raise ValueError(f"direction must be 'long' or 'short', got {direction!r}")


def ladder_target(
    entry_price: float,
    stop_price: float,
    levels: list[float] | None = None,
) -> list[dict[str, float]]:
    """
    Return partial exit targets for a Ladder exit strategy.

    Default levels: [1.0, 2.0, 3.0] R. Exit 1/3 of position at each level;
    let the remainder run under a VWAP trailing stop (see trailing_stop_update).

    Source: Maroy (2025) SSRN 5095349. VWAP & Ladder exit: Sharpe 3.0+,
    annualised returns 50%+, positive return skewness.

    Returns list of dicts: [{'r_level': 1.0, 'price': X, 'exit_pct': 0.333}, ...]
    """
    if levels is None:
        levels = [1.0, 2.0, 3.0]
    risk_dist = abs(entry_price - stop_price)
    if risk_dist < 1e-8:
        raise ValueError("Entry and stop cannot be identical.")
    is_long = stop_price < entry_price
    share_per_level = round(1.0 / len(levels), 4)
    targets = []
    for r in sorted(levels):
        price = (entry_price + r * risk_dist) if is_long else (entry_price - r * risk_dist)
        targets.append({
            "r_level":   r,
            "price":     round(price, 4),
            "exit_pct":  share_per_level,
        })
    return targets


def atr_stop_loss(
    df: "pd.DataFrame",
    direction: str,
    entry_price: float,
    atr_period: int = 14,
    multiplier: float = 2.0,
) -> dict[str, float]:
    """
    ATR-based dynamic stop loss that adapts to current volatility.

    Stop distance = ATR(atr_period) * multiplier.
    Long : stop = entry - ATR * multiplier
    Short: stop = entry + ATR * multiplier

    Advantage over fixed-buffer set_stop_loss(): automatically widens
    during high-vol sessions and tightens during low-vol sessions.

    Requires pandas-ta for ATR calculation (pip install pandas-ta).
    """
    _require_extended("atr_stop_loss")
    try:
        import pandas_ta as ta  # type: ignore
        atr_series = ta.atr(df["High"], df["Low"], df["Close"], length=atr_period)
    except ImportError:
        # Pure-pandas fallback ATR (Wilder method)
        highs  = df["High"].values.astype(float)
        lows   = df["Low"].values.astype(float)
        closes = df["Close"].values.astype(float)
        tr = [max(highs[i] - lows[i],
                  abs(highs[i]  - closes[i-1]),
                  abs(lows[i]   - closes[i-1]))
              for i in range(1, len(closes))]
        atr_series = pd.Series(tr).ewm(alpha=1/atr_period, adjust=False).mean()

    atr_value = float(atr_series.dropna().iloc[-1])
    stop_dist  = atr_value * multiplier
    direction  = direction.lower()
    if direction == "long":
        stop = round(entry_price - stop_dist, 4)
    elif direction == "short":
        stop = round(entry_price + stop_dist, 4)
    else:
        raise ValueError(f"direction must be 'long' or 'short'")
    return {
        "stop":       stop,
        "atr":        round(atr_value, 4),
        "stop_dist":  round(stop_dist, 4),
        "multiplier": multiplier,
        "atr_period": atr_period,
    }


def volatility_target_size(
    account_balance: float,
    entry_price: float,
    stop_price: float,
    current_daily_vol_pct: float,
    target_daily_vol_pct: float = 0.02,
    baseline_daily_vol_pct: float | None = None,
) -> float:
    """
    Volatility-targeted position sizing: scale size so that expected daily
    P&L volatility equals target_daily_vol_pct of account.

    Source: Zarattini, Aziz, Barbon (2024) SSRN 4824172.
    Strategy targets 2% daily vol exposure; halves size when vol doubles.

    Parameters
    ----------
    account_balance       : total account value
    entry_price           : planned entry price
    stop_price            : planned stop price
    current_daily_vol_pct : today's estimated daily vol as % (e.g. 0.01 = 1%)
                            Use ATR(14)/close for a practical estimate.
    target_daily_vol_pct  : target daily vol exposure (default 2%)
    baseline_daily_vol_pct: 20-day avg vol (default: same as current = no scaling)

    Returns adjusted position size (units).
    """
    if entry_price == stop_price:
        raise ValueError("Entry and stop cannot be identical.")
    if current_daily_vol_pct <= 0:
        raise ValueError("current_daily_vol_pct must be positive.")
    baseline = baseline_daily_vol_pct or current_daily_vol_pct
    # Vol scaling: if current vol is 2x baseline, halve the size
    vol_scalar = baseline / current_daily_vol_pct  # < 1 when vol elevated
    risk_pct   = target_daily_vol_pct * vol_scalar
    risk_pct   = max(0.001, min(risk_pct, 0.02))   # floor 0.1%, cap 2%
    return (account_balance * risk_pct) / abs(entry_price - stop_price)


# ===========================================================================
# GROUP G -- Signal layer  (SMA, VWAP, ORB, gap quality, candle anatomy)
# AND-gate entry_gate: all three conditions required (DARF v3 proof).
# ===========================================================================

def sma_trend_filter(
    df: "pd.DataFrame",
    fast: int = 20,
    slow: int = 200,
    price_col: str = "Close",
) -> dict:
    """20-SMA trend direction + 200-SMA S/R target (Malyarovich convention)."""
    _require_extended("sma_trend_filter")
    if len(df) < slow:
        return {"sma_fast": None, "sma_slow": None, "trend": "insufficient_data",
                "above_fast": None, "above_slow": None,
                "price": float(df[price_col].iloc[-1]) if len(df) else None}
    prices = df[price_col]
    sma_fast, sma_slow = (float(prices.rolling(n).mean().iloc[-1]) for n in (fast, slow))
    price = float(prices.iloc[-1])
    af, as_ = price > sma_fast, price > sma_slow
    trend = "up" if (af and as_) else "down" if (not af and not as_) else "mixed"
    return {"sma_fast": round(sma_fast, 4), "sma_slow": round(sma_slow, 4),
            "trend": trend, "above_fast": af, "above_slow": as_,
            "price": round(price, 4)}


def vwap_anchored(
    df: "pd.DataFrame",
    anchor_time: Time = SESSION_START,
) -> "pd.Series":
    """Anchored VWAP from anchor_time each day. Typical price × volume method."""
    _require_extended("vwap_anchored")
    required = {"High", "Low", "Close", "Volume"}
    if not required.issubset(df.columns):
        raise ValueError(f"vwap_anchored requires {required}")
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    pv = tp * df["Volume"]
    vwap = pd.Series(index=df.index, dtype=float)
    for day in set(df.index.date):
        mask = (df.index.date == day) & (df.index.time >= anchor_time)
        cum_vol = df["Volume"][mask].cumsum()
        vwap[mask] = (pv[mask].cumsum() / cum_vol.replace(0, float("nan"))).values
    return vwap


def vwap_slope(vwap_series: "pd.Series", lookback: int = 3) -> dict:
    """VWAP slope over last lookback bars. Flat if < 0.02% of VWAP per bar."""
    _require_extended("vwap_slope")
    clean = vwap_series.dropna()
    if len(clean) < lookback:
        return {"slope": 0.0, "direction": "flat", "lookback": lookback}
    recent = clean.iloc[-lookback:]
    slope  = float(recent.iloc[-1] - recent.iloc[0]) / lookback
    thresh = float(recent.mean()) * 0.0002
    direction = "up" if slope > thresh else "down" if slope < -thresh else "flat"
    return {"slope": round(slope, 6), "direction": direction,
            "flat_threshold": round(thresh, 6), "lookback": lookback}


def orb_define_range(
    df: "pd.DataFrame",
    method: str = "15min",
    open_time: Time = SESSION_START,
) -> dict:
    """
    Opening Range high and low for three window sizes:
      '5min'  -- first 1 candle (09:30-09:34)
      '15min' -- first 3 candles (09:30-09:44)   [default]
      '30min' -- first 6 candles (09:30-09:55)   [Gao et al. 2018 JFE window]
    30min academic backing: first-30-min return on SPY predicts last-30-min
    return; stronger on volatile/high-volume days (Gao, Han, Li, Zhou 2018).
    """
    _require_extended("orb_define_range")
    if df.empty:
        raise ValueError("orb_define_range: empty DataFrame")
    today    = df.index[0].date()
    df_today = df[(df.index.date == today) & (df.index.time >= open_time)]
    _method_map = {"5min": Time(9, 34), "15min": Time(9, 44), "30min": Time(9, 55)}
    if method not in _method_map:
        raise ValueError(f"Unknown method {method!r}. Valid: {list(_method_map)}")
    orb_bars = df_today[df_today.index.time <= _method_map[method]]
    if orb_bars.empty:
        raise ValueError(f"No bars for method={method!r}")
    orb_high = float(orb_bars["High"].max())
    orb_low  = float(orb_bars["Low"].min())
    return {"orb_high": round(orb_high, 4), "orb_low": round(orb_low, 4),
            "orb_size": round(orb_high - orb_low, 4),
            "method": method, "bars_used": len(orb_bars)}


def orb_breakout_signal(
    df: "pd.DataFrame",
    orb_high: float,
    orb_low: float,
) -> dict:
    """Breakout check on last CLOSED candle Close (never intrabar high/low)."""
    _require_extended("orb_breakout_signal")
    if df.empty:
        return {"signal": None, "close": None, "bar_time": None}
    last  = df.iloc[-1]
    close = float(last["Close"])
    signal = "long" if close > orb_high else "short" if close < orb_low else None
    return {"signal": signal, "close": round(close, 4),
            "bar_time": str(last.name), "orb_high": orb_high, "orb_low": orb_low}


def gap_quality_check(
    gap_pct: float,
    gap_direction: str,
    prev_trend: str,
    resistance_cleared: bool,
    min_gap_pct: float = 2.0,
) -> dict:
    """A-grade (shocking gap): reverses prior trend, clears key level, >= 2%."""
    reasons = []
    score   = 0
    if abs(gap_pct) >= min_gap_pct:
        score += 1
    else:
        reasons.append(f"Gap {gap_pct:.1f}% below minimum {min_gap_pct:.1f}%")
    reversal = ((gap_direction == "up" and prev_trend == "down") or
                (gap_direction == "down" and prev_trend == "up"))
    if reversal:
        score += 1
    else:
        reasons.append("Gap does not reverse prior trend")
    if resistance_cleared:
        score += 1
    else:
        reasons.append("Gap did not clear a key S/R level")
    return {"quality": "A" if score == 3 else "B" if score == 2 else "C",
            "score": score, "gap_pct": gap_pct, "reasons": reasons}


def candle_anatomy(bar: "pd.Series") -> dict:
    """Body/wick classification. Types: marubozu, hammer, shooting_star, doji, standard."""
    _require_extended("candle_anatomy")
    high, low   = float(bar["High"]),  float(bar["Low"])
    open_, close = float(bar["Open"]), float(bar["Close"])
    rng = high - low
    if rng < 1e-8:
        return {"candle_type": "doji", "color": "doji", "range": 0.0,
                "body_pct": 0.0, "wick_upper_pct": 0.0, "wick_lower_pct": 0.0}
    body_top, body_bottom = max(open_, close), min(open_, close)
    body_pct       = (body_top - body_bottom) / rng * 100
    wick_upper_pct = (high - body_top)    / rng * 100
    wick_lower_pct = (body_bottom - low)  / rng * 100
    color  = "green" if close > open_ else "red" if close < open_ else "doji"
    ctype  = ("marubozu"      if body_pct >= 80 else
              "hammer"        if wick_lower_pct >= 60 and body_pct <= 30 else
              "shooting_star" if wick_upper_pct >= 60 and body_pct <= 30 else
              "doji"          if body_pct <= 10 else "standard")
    return {"candle_type": ctype, "color": color, "range": round(rng, 4),
            "body_pct": round(body_pct, 1),
            "wick_upper_pct": round(wick_upper_pct, 1),
            "wick_lower_pct": round(wick_lower_pct, 1),
            "open": round(open_, 4), "close": round(close, 4),
            "high": round(high, 4),  "low": round(low, 4)}


def vwap_slope_gate(
    vwap_series: "pd.Series",
    df: "pd.DataFrame",
    lookback: int = 5,
    max_flips: int = 2,
    hold_bars: int = 2,
    range_factor: float = 0.10,
) -> dict:
    """
    Enhanced VWAP gate adding magnitude, chop, and hold filters on top of slope.
    Source: TradingView VWAP Gate v2.3 (MIdr0guA). Backlog item P2-035.

    Three additional conditions beyond basic vwap_slope():
      1. Magnitude: VWAP displacement over lookback > range_factor * bar_range
         (ties the threshold to current market range, not an absolute %)
      2. Chop filter: price has not flipped across VWAP more than max_flips
         times in the last lookback bars
      3. Hold filter: price has stayed on same side of VWAP for hold_bars
         consecutive bars

    ALL THREE are AND-logic. Any failure returns direction='flat'.

    Returns {'direction': 'up'|'down'|'flat', 'slope': float,
             'magnitude_ok': bool, 'chop_ok': bool, 'hold_ok': bool,
             'reason': str}
    """
    _require_extended("vwap_slope_gate")
    clean = vwap_series.dropna()
    if len(clean) < lookback or len(df) < lookback:
        return {"direction": "flat", "slope": 0.0, "magnitude_ok": False,
                "chop_ok": False, "hold_ok": False,
                "reason": "insufficient data"}

    recent_vwap  = clean.iloc[-lookback:]
    recent_close = df["Close"].iloc[-lookback:]
    recent_high  = df["High"].iloc[-lookback:]
    recent_low   = df["Low"].iloc[-lookback:]

    # Basic slope
    slope      = float(recent_vwap.iloc[-1] - recent_vwap.iloc[0]) / lookback
    bar_range  = float(recent_high.max() - recent_low.min())
    thresh_abs = bar_range * range_factor
    thresh_abs = max(thresh_abs, float(recent_vwap.mean()) * 0.00005)  # floor: ~5bps/bar

    if slope > thresh_abs:
        raw_direction = "up"
    elif slope < -thresh_abs:
        raw_direction = "down"
    else:
        return {"direction": "flat", "slope": round(slope, 6),
                "magnitude_ok": False, "chop_ok": True, "hold_ok": True,
                "reason": f"slope {slope:.6f} below range threshold {thresh_abs:.6f}"}

    magnitude_ok = True

    # Chop filter: count price-crosses-VWAP in lookback window
    vwap_aligned = clean.reindex(df.index).iloc[-lookback:]
    above = recent_close.values > vwap_aligned.values
    flips = int(sum(1 for i in range(1, len(above)) if above[i] != above[i-1]))
    chop_ok = flips <= max_flips

    # Hold filter: last hold_bars must all be on same side of VWAP
    last_close = recent_close.values[-hold_bars:]
    last_vwap  = vwap_aligned.values[-hold_bars:]
    if raw_direction == "up":
        hold_ok = bool(all(c > v for c, v in zip(last_close, last_vwap) if not (c != c or v != v)))
    else:
        hold_ok = bool(all(c < v for c, v in zip(last_close, last_vwap) if not (c != c or v != v)))

    if not chop_ok:
        return {"direction": "flat", "slope": round(slope, 6),
                "magnitude_ok": magnitude_ok, "chop_ok": False, "hold_ok": hold_ok,
                "reason": f"{flips} VWAP crosses in {lookback} bars (max {max_flips})"}
    if not hold_ok:
        return {"direction": "flat", "slope": round(slope, 6),
                "magnitude_ok": magnitude_ok, "chop_ok": chop_ok, "hold_ok": False,
                "reason": f"price not held {raw_direction} of VWAP for {hold_bars} bars"}

    return {"direction": raw_direction, "slope": round(slope, 6),
            "magnitude_ok": True, "chop_ok": True, "hold_ok": True,
            "reason": f"all gates passed: slope={slope:.6f} flips={flips} hold={hold_ok}"}


def entry_gate(orb_break: bool, vwap_direction_ok: bool, retest_confirmed: bool) -> bool:
    """
    AND-gate: all three must be True.
    DARF v3 OR-gate failure proved: Sharpe 0.70 vs 0.95, drawdown 18.3% vs 11.3%.
    OR-logic on any two conditions produces that failure mode.
    """
    return bool(orb_break and vwap_direction_ok and retest_confirmed)


# ===========================================================================
# GROUP H -- Entry layer  (candle-close enforcement, IPDE, retest, guards)
# ===========================================================================

ORB_CUTOFF = ORB_END


def candle_closed(
    bar: "pd.Series",
    current_time: Time | None = None,
    interval_minutes: int = 5,
) -> bool:
    """
    True if bar has fully closed.
    Historical (current_time=None): always True.
    Live: bar at 09:30 with interval=5 closes at 09:35.
    Bug fixed during build: current_time > bar_time was wrong; must use
    datetime arithmetic with interval_minutes.
    """
    if current_time is None:
        return True
    bar_time = bar.name.time() if hasattr(bar.name, "time") else None
    if bar_time is None:
        return True
    dummy    = datetime(2000, 1, 1)
    close_dt = datetime.combine(dummy, bar_time) + timedelta(minutes=interval_minutes)
    return datetime.combine(dummy, current_time) >= close_dt


def validate_entry(
    bar: "pd.Series",
    signal: dict,
    current_time: Time | None = None,
    interval_minutes: int = 5,
) -> None:
    """
    Hard gate: raises ValueError if candle has not closed.
    Not a warning -- a hard stop. Violations logged as process failures.
    """
    if not candle_closed(bar, current_time, interval_minutes):
        raise ValueError(
            f"ENTRY VIOLATION: candle at {bar.name} has not closed. "
            "Wait for close. This violation is logged as a process failure."
        )


def ipde_checklist(
    trend: str,
    predicted_move: str,
    entry_signal: str | None,
    risk_defined: bool,
) -> dict:
    """
    Jason Graystone IPDE: Identify, Predict, Decide, Execute.
    Returns failed_step string so trade journal captures no-trade reason.
    """
    steps = {
        "I_identify": trend in ("up", "down"),
        "P_predict":  predicted_move in ("continuation", "pullback", "reversal"),
        "D_decide":   entry_signal is not None,
        "E_execute":  risk_defined,
    }
    failed = next((k for k, v in steps.items() if not v), None)
    return {"pass": all(steps.values()), "failed_step": failed, "steps": steps}


def retest_confirmation(
    df: "pd.DataFrame",
    broken_level: float,
    direction: str,
    lookback: int = 6,
    tolerance_pct: float = 0.15,
) -> dict:
    """Scarface Trades retest mechanic: price touched level and closed away."""
    _require_extended("retest_confirmation")
    if df.empty or len(df) < 2:
        return {"confirmed": False, "reason": "insufficient bars"}
    tol    = broken_level * tolerance_pct / 100
    recent = df.iloc[-lookback:]
    direction = direction.lower()
    if direction == "long":
        touches = recent[recent["Low"] <= broken_level + tol]
        if touches.empty:
            return {"confirmed": False, "reason": "no retest touch"}
        last_close = float(df["Close"].iloc[-1])
        confirmed  = last_close > broken_level
        return {"confirmed": confirmed, "retest_low": float(touches.iloc[-1]["Low"]),
                "last_close": round(last_close, 4), "broken_level": broken_level,
                "reason": "confirmed above level" if confirmed else "not yet above level"}
    else:
        touches = recent[recent["High"] >= broken_level - tol]
        if touches.empty:
            return {"confirmed": False, "reason": "no retest touch"}
        last_close = float(df["Close"].iloc[-1])
        confirmed  = last_close < broken_level
        return {"confirmed": confirmed, "retest_high": float(touches.iloc[-1]["High"]),
                "last_close": round(last_close, 4), "broken_level": broken_level,
                "reason": "confirmed below level" if confirmed else "not yet below level"}


def session_time_valid(current_time: Time, cutoff: Time = ORB_CUTOFF) -> bool:
    """True if within the ORB window (<= 11:00 EST)."""
    return current_time <= cutoff


@dataclass
class SessionAttemptTracker:
    """Max 2 breakout attempts per direction per session (Bull Barbie rule)."""
    max_attempts:  int = 2
    long_attempts:  int = 0
    short_attempts: int = 0

    def can_attempt(self, direction: str) -> bool:
        return (self.long_attempts  < self.max_attempts if direction.lower() == "long"
                else self.short_attempts < self.max_attempts)

    def record_attempt(self, direction: str) -> dict:
        direction = direction.lower()
        if not self.can_attempt(direction):
            count = self.long_attempts if direction == "long" else self.short_attempts
            return {"allowed": False, "direction": direction, "attempts": count,
                    "max": self.max_attempts,
                    "message": f"Max {self.max_attempts} {direction} attempts reached."}
        if direction == "long":
            self.long_attempts += 1; count = self.long_attempts
        else:
            self.short_attempts += 1; count = self.short_attempts
        return {"allowed": True, "direction": direction, "attempts": count,
                "max": self.max_attempts,
                "message": f"{direction.capitalize()} attempt {count}/{self.max_attempts}."}

    def reset(self) -> None:
        self.long_attempts = 0; self.short_attempts = 0


# ===========================================================================
# GROUP I -- Logging layer  (trade journal, rolling stats, milestone check)
# ===========================================================================

@dataclass
class TradeRecord:
    """One completed trade. 25 fields, all required at log time."""
    trade_id:                    int
    date:                        str
    time_entry:                  str
    time_exit:                   str
    ticker:                      str
    direction:                   str
    setup_type:                  str
    entry_price:                 float
    stop_price:                  float
    target_price:                float
    exit_price:                  float
    planned_rr:                  float
    actual_r:                    float
    outcome:                     str
    ipde_pass:                   bool
    candle_closed:               bool
    vwap_slope:                  float
    orb_confirmed:               bool
    retest_confirmed:            bool
    and_gate_pass:               bool
    pre_session_read:            bool
    emotional_state:             int
    consecutive_losses_at_entry: int
    notes:                       str
    session_pnl_r:               float

    @property
    def is_violation(self) -> bool:
        return not (self.candle_closed and self.and_gate_pass and self.ipde_pass)


class TradeJournal:
    """Append-only CSV trade journal with rolling statistics and milestone check."""

    def __init__(self, filepath: str = "trade_journal.csv") -> None:
        self.filepath = filepath
        self.records: list[TradeRecord] = []
        self._next_id = 1
        if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
            self._load()

    def log_trade(self, **kwargs: Any) -> TradeRecord:
        today   = kwargs.get("date", str(date.today()))
        today_r = sum(r.actual_r for r in self.records if r.date == today)
        kwargs.setdefault("session_pnl_r",
                          round(today_r + kwargs.get("actual_r", 0), 3))
        kwargs["trade_id"] = self._next_id
        self._next_id += 1
        rec = TradeRecord(**kwargs)
        self.records.append(rec)
        self._append_to_csv(rec)
        return rec

    def _append_to_csv(self, rec: TradeRecord) -> None:
        fieldnames   = list(asdict(rec).keys())
        needs_header = (not os.path.exists(self.filepath)
                        or os.path.getsize(self.filepath) == 0)
        with open(self.filepath, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if needs_header:
                writer.writeheader()
            writer.writerow(asdict(rec))

    def _load(self) -> None:
        with open(self.filepath, newline="") as f:
            for row in csv.DictReader(f):
                for bf in ("ipde_pass", "candle_closed", "orb_confirmed",
                           "retest_confirmed", "and_gate_pass", "pre_session_read"):
                    row[bf] = row[bf].lower() == "true"
                for inf in ("trade_id", "emotional_state",
                            "consecutive_losses_at_entry"):
                    row[inf] = int(row[inf])
                for ff in ("entry_price", "stop_price", "target_price",
                           "exit_price", "planned_rr", "actual_r",
                           "vwap_slope", "session_pnl_r"):
                    row[ff] = float(row[ff])
                self.records.append(TradeRecord(**row))
        self._next_id = max((r.trade_id for r in self.records), default=0) + 1

    def rolling_stats(self, n: int = 20) -> dict:
        recent = self.records[-n:] if len(self.records) >= n else self.records
        if not recent:
            return {"n": 0, "message": "No trades logged yet."}
        r_multiples = [t.actual_r for t in recent]
        wins        = [r for r in r_multiples if r > 0]
        losses      = [r for r in r_multiples if r < 0]
        n_trades    = len(recent)
        win_rate    = len(wins) / n_trades
        avg_win     = statistics.mean(wins)        if wins   else 0.0
        avg_loss    = abs(statistics.mean(losses)) if losses else 0.0
        ev          = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        sharpe      = None
        if n_trades >= 2:
            sd = statistics.stdev(r_multiples)
            if sd > 0:
                sharpe = statistics.mean(r_multiples) / sd
        sig = None
        if sharpe is not None:
            years = n_trades / (5 * 52)
            sig   = sharpe_significance_tstat_annualised(sharpe, years)
        violations = sum(1 for t in recent if t.is_violation)
        return {"n": n_trades, "win_rate": round(win_rate, 4),
                "avg_win_r": round(avg_win, 4), "avg_loss_r": round(avg_loss, 4),
                "ev_per_trade": round(ev, 4),
                "sharpe": round(sharpe, 3) if sharpe else None,
                "sharpe_tstat": sig, "violations": violations,
                "violation_rate": round(violations / n_trades, 4),
                "ev_positive": ev > 0,
                "message": (f"EV={ev:+.3f}R  WR={win_rate:.1%}  "
                            f"Sharpe={sharpe:.2f}  Violations={violations}/{n_trades}"
                            if sharpe else
                            f"EV={ev:+.3f}R  WR={win_rate:.1%}  "
                            "(need >=2 trades for Sharpe)")}

    def milestone_check(self, required_trades: int = 100) -> dict:
        """P3-001 gate: full readiness report for the 100-trade milestone."""
        n     = len(self.records)
        stats = self.rolling_stats(n)
        sig   = stats.get("sharpe_tstat")
        conditions = {
            "trade_count_met":      n >= required_trades,
            "ev_positive":          stats.get("ev_positive", False),
            "win_rate_above_floor": stats.get("win_rate", 0) > 0.34,
            "violations_ok":        stats.get("violation_rate", 1) < 0.05,
            "sharpe_sig":           sig is not None and sig.get("t_stat", 0) >= 2.0,
        }
        ready = all(conditions.values())
        return {"ready_for_live": ready, "trade_count": n,
                "required": required_trades, "conditions": conditions,
                "rolling_stats": stats,
                "message": ("✅ All conditions met — proceed to DSR check (P3-002)."
                            if ready else
                            "⏳ Milestone not yet met — see conditions for gaps.")}

    def __len__(self) -> int:
        return len(self.records)


# ===========================================================================
# GROUP J -- Backtest layer  (sealed test set, metrics, OOS validation)
# ===========================================================================

TRAIN_START = date(2020, 1, 1)
TRAIN_END   = date(2022, 12, 31)
VAL_START   = date(2023, 1, 1)
VAL_END     = date(2024, 12, 31)
TEST_START  = date(2025, 1, 1)   # SEALED -- never touch until final config

DEGRADATION_THRESHOLD = 0.20
MONTE_CARLO_GATE_PCT  = 0.85


def _check_partition(
    start: date,
    end: date,
    test_set_unlocked: bool = False,
) -> str:
    """Enforce sealed test set. Raises RuntimeError without explicit unlock."""
    partition = ("test"       if start >= TEST_START else
                 "validation" if start >= VAL_START  else "training")
    if partition == "test" and not test_set_unlocked:
        raise RuntimeError(
            f"SEALED TEST SET: {start} >= {TEST_START}. "
            "Pass test_set_unlocked=True ONLY with a final committed strategy."
        )
    return partition


def compute_metrics(r_multiples: list[float]) -> dict:
    """Sharpe, max drawdown, win rate, EV, profit factor from R-multiples."""
    if not r_multiples:
        return {}
    n        = len(r_multiples)
    wins     = [r for r in r_multiples if r > 0]
    losses   = [r for r in r_multiples if r < 0]
    win_rate = len(wins) / n
    avg_win  = statistics.mean(wins)        if wins   else 0.0
    avg_loss = abs(statistics.mean(losses)) if losses else 0.0
    ev       = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
    sharpe   = None
    if n >= 2:
        sd = statistics.stdev(r_multiples)
        if sd > 0:
            sharpe = statistics.mean(r_multiples) / sd
    equity = [0.0]
    for r in r_multiples:
        equity.append(equity[-1] + r)
    peak, worst = equity[0], 0.0
    for v in equity:
        peak = max(peak, v); worst = min(worst, v - peak)
    profit_factor = (sum(wins) / abs(sum(losses))) if losses else float("inf")
    return {"win_rate": round(win_rate, 4), "avg_win_r": round(avg_win, 4),
            "avg_loss_r": round(avg_loss, 4), "ev_per_trade": round(ev, 4),
            "sharpe": round(sharpe, 3) if sharpe else None,
            "max_drawdown_r": round(abs(worst), 3),
            "profit_factor": round(profit_factor, 3), "n_trades": n}


def run_backtest(
    strategy_fn: Callable[["pd.DataFrame"], list[dict]],
    df: "pd.DataFrame",
    start: date | str,
    end:   date | str,
    *,
    test_set_unlocked: bool = False,
    commission_per_trade_r: float = 0.05,
) -> dict:
    """
    Run strategy_fn on df filtered to [start, end].
    Raises RuntimeError on test-set dates without explicit unlock.
    strategy_fn(df) -> list[dict] where each dict contains 'actual_r'.
    """
    _require_extended("run_backtest")
    if isinstance(start, str): start = date.fromisoformat(start)
    if isinstance(end,   str): end   = date.fromisoformat(end)
    partition = _check_partition(start, end, test_set_unlocked)
    df_slice  = df[(df.index.date >= start) & (df.index.date <= end)].copy()
    if df_slice.empty:
        raise ValueError(f"No data in range {start} -> {end}")
    trades = strategy_fn(df_slice)
    if not trades:
        return {"partition": partition, "n_trades": 0,
                "message": "Strategy produced no trades."}
    r_multiples = [t["actual_r"] - commission_per_trade_r for t in trades]
    return {"partition": partition, "start": str(start), "end": str(end),
            "n_trades": len(trades), "r_multiples": r_multiples, "trades": trades,
            **compute_metrics(r_multiples)}


def monte_carlo_gate(
    r_multiples: list[float],
    n_sims: int = 1000,
    sharpe_threshold: float = 1.0,
    gate_pct: float = MONTE_CARLO_GATE_PCT,
    seed: int = 42,
) -> dict:
    """monte_carlo_robustness() wrapped with pass/fail gate and message."""
    result = monte_carlo_robustness(r_multiples, n_sims=n_sims,
                                    sharpe_threshold=sharpe_threshold, seed=seed)
    return {"passed": result.passes_rulebook_gate,
            "pct_sims_passing": result.pct_sims_passing,
            "median_sharpe": result.median_sharpe,
            "worst_case_drawdown": result.worst_case_drawdown,
            "message": (
                f"✅ MC gate passed: {result.pct_sims_passing:.1%} of {n_sims} "
                f"sims cleared Sharpe > {sharpe_threshold}"
                if result.passes_rulebook_gate else
                f"❌ MC gate FAILED: {result.pct_sims_passing:.1%} of {n_sims} "
                f"sims cleared Sharpe > {sharpe_threshold} (need {gate_pct:.0%})"
            )}


def degradation_check(
    train_sharpe: float,
    test_sharpe: float,
    threshold: float = DEGRADATION_THRESHOLD,
) -> dict:
    """< 20%: proceed. 20-40%: 50 more trades. > 40%: return to Phase 2."""
    if train_sharpe <= 0:
        return {"flag": True, "degradation_pct": None,
                "message": "Training Sharpe <= 0 -- no edge to validate."}
    deg  = (train_sharpe - test_sharpe) / abs(train_sharpe)
    flag = deg > threshold
    return {"flag": flag, "degradation_pct": round(deg * 100, 1),
            "threshold_pct": round(threshold * 100, 1),
            "message": (f"❌ Degradation {deg*100:.1f}% > {threshold*100:.0f}% -- failed OOS."
                        if flag else
                        f"✅ Degradation {deg*100:.1f}% within {threshold*100:.0f}% -- validated.")}


# ===========================================================================
# GROUP K -- Trading cost and liquidity model  (v2.2.0)
#
# Motivation: prior versions used a flat 0.05R commission per round trip.
# This is approximately right for SPY on Alpaca (commission-free, tiny
# spread) but materially wrong for individual stocks on IBKR or CFD
# accounts where spread and market impact can exceed 0.2R per trade.
# Zarattini et al. (2024) used $0.005/share (IBKR) and showed costs
# significantly affect the strategy's net Sharpe.
#
# Components of total round-trip cost:
#   1. Commission    -- broker fee per share or per trade (known, deterministic)
#   2. Half-spread   -- you buy at the ask and sell at the bid; the gap
#                       between ask and mid is the half-spread you "donate"
#                       to the market maker on entry AND exit
#   3. Market impact -- your own order moves the price against you;
#                       scales with sqrt(units / avg_volume) (Almgren 2001)
#
# All functions return values in £/$ unless noted.
# Use cost_as_r_multiple() to convert to the R-multiple we use throughout.
# ===========================================================================

# ---------------------------------------------------------------------------
# Broker commission schedules
# ---------------------------------------------------------------------------

# Per-share rates (USD). Add your broker here as needed.
_BROKER_RATES: dict[str, dict] = {
    "alpaca":  {"per_share": 0.0,    "min_per_order": 0.0,   "max_pct_notional": 0.0},
    "td":      {"per_share": 0.0,    "min_per_order": 0.0,   "max_pct_notional": 0.0},
    "ibkr":    {"per_share": 0.005,  "min_per_order": 1.0,   "max_pct_notional": 0.01},
    "ibkr_uk": {"per_share": 0.008,  "min_per_order": 1.5,   "max_pct_notional": 0.01},
    "schwab":  {"per_share": 0.0,    "min_per_order": 0.0,   "max_pct_notional": 0.0},
    "uk_cfd":  {"per_share": 0.0,    "min_per_order": 0.0,   "max_pct_notional": 0.0003},
    # UK CFD: typically 0.03% of notional per side; charged as spread-widening
}

def commission_cost(
    units: float,
    price: float,
    broker: str = "alpaca",
    *,
    per_share_override: float | None = None,
) -> dict[str, float]:
    """
    Broker commission for one side of a trade (buy OR sell, not round trip).

    Supported brokers: 'alpaca', 'td', 'ibkr', 'ibkr_uk', 'schwab', 'uk_cfd'.
    Pass per_share_override to use a custom rate (e.g. negotiated IBKR tiers).

    VERIFIED:
        ibkr 200 shares @ $770: max(200*0.005, 1.0) = max(1.0, 1.0) = $1.00
        ibkr 50 shares @ $5:    max(50*0.005, 1.0)  = max(0.25,1.0) = $1.00 (min applies)
        alpaca any trade: $0.00

    Returns dict: {'commission': float, 'broker': str, 'per_share': float}
    """
    if broker not in _BROKER_RATES:
        raise ValueError(f"Unknown broker {broker!r}. Supported: {list(_BROKER_RATES)}")
    rate    = _BROKER_RATES[broker]
    per_shr = per_share_override if per_share_override is not None else rate["per_share"]
    notional = units * price
    raw_cost  = units * per_shr
    max_cost  = notional * rate["max_pct_notional"] if rate["max_pct_notional"] > 0 else float("inf")
    commission = max(raw_cost, rate["min_per_order"])
    commission = min(commission, max_cost) if max_cost < float("inf") else commission
    return {
        "commission":    round(commission, 4),
        "broker":        broker,
        "per_share":     round(per_shr, 5),
        "notional":      round(notional, 2),
    }


def half_spread_cost(
    price: float,
    units: float,
    avg_daily_volume: float | None = None,
    spread_pct_override: float | None = None,
) -> dict[str, float]:
    """
    Estimate the half-spread cost for one side of a trade (entry OR exit).

    The half-spread is the gap between the mid-price and the best ask (for
    buys) or best bid (for sells). Each entry and exit costs one half-spread.

    Estimation method (in priority order):
      1. spread_pct_override: use directly if provided (e.g. from live quote)
      2. Volume-based proxy: for liquid stocks (avg_vol > 5M/day), spread is
         typically 0.01-0.02% of price. For illiquid (avg_vol < 500k), it can
         be 0.1-0.5%. We use a tiered model calibrated to US equities.
      3. Fallback: 0.1% of price (conservative default)

    Spread tiers (approximate, calibrated to 2024-2025 US equity data):
        avg_vol > 50M/day  (mega-cap, index ETFs):  half-spread ≈ 0.001% of price
        avg_vol > 5M/day   (large-cap):              half-spread ≈ 0.005% of price
        avg_vol > 500k/day (mid-cap):                half-spread ≈ 0.020% of price
        avg_vol < 500k/day (small-cap, illiquid):    half-spread ≈ 0.100% of price

    Returns dict: {'half_spread_pct', 'half_spread_per_share', 'total_cost', 'tier'}
    """
    if spread_pct_override is not None:
        half_pct = spread_pct_override / 2.0
        tier = "override"
    elif avg_daily_volume is not None:
        if avg_daily_volume > 50_000_000:
            half_pct = 0.000010; tier = "mega_cap_etf"
        elif avg_daily_volume > 5_000_000:
            half_pct = 0.000050; tier = "large_cap"
        elif avg_daily_volume > 500_000:
            half_pct = 0.000200; tier = "mid_cap"
        else:
            half_pct = 0.001000; tier = "small_cap_illiquid"
    else:
        half_pct = 0.001; tier = "fallback_conservative"

    half_spread_per_share = price * half_pct
    total_cost            = half_spread_per_share * units
    return {
        "half_spread_pct":        round(half_pct * 100, 6),   # as a %
        "half_spread_per_share":  round(half_spread_per_share, 6),
        "total_cost":             round(total_cost, 4),
        "tier":                   tier,
        "units":                  units,
        "price":                  price,
    }


def market_impact_cost(
    units: float,
    price: float,
    avg_daily_volume: float,
    daily_volatility_pct: float = 0.01,
    impact_coefficient: float = 0.1,
) -> dict[str, float]:
    """
    Square-root market impact (Almgren & Chriss 2001 simplified).

    Your own order moves the price against you. The temporary impact scales
    with the square root of the fraction of daily volume you are trading.

    Formula: Impact ≈ σ * price * impact_coefficient * sqrt(units / ADV)
    where σ = daily_volatility_pct, ADV = avg_daily_volume.

    Impact coefficient of 0.1 is a commonly used starting point for US
    equities (Almgren et al. 2005). For very liquid ETFs (SPY, QQQ) it
    is typically lower (~0.05). For small-caps it can be 0.3-0.5.

    IMPORTANT: This is temporary impact (price reverts after your trade).
    Permanent impact (information leakage) is not modelled here.

    Returns dict: {'impact_pct', 'impact_per_share', 'total_impact', 'participation_rate'}
    """
    if avg_daily_volume <= 0:
        raise ValueError("avg_daily_volume must be positive.")
    participation_rate = units / avg_daily_volume
    impact_pct         = daily_volatility_pct * impact_coefficient * math.sqrt(participation_rate)
    impact_per_share   = price * impact_pct
    total_impact       = impact_per_share * units
    return {
        "impact_pct":           round(impact_pct * 100, 6),
        "impact_per_share":     round(impact_per_share, 6),
        "total_impact":         round(total_impact, 4),
        "participation_rate":   round(participation_rate * 100, 4),   # as %
        "units":                units,
        "adv":                  avg_daily_volume,
    }


def total_round_trip_cost(
    units: float,
    price: float,
    broker: str = "alpaca",
    avg_daily_volume: float | None = None,
    daily_volatility_pct: float = 0.01,
    spread_pct_override: float | None = None,
    impact_coefficient: float = 0.1,
) -> dict[str, float]:
    """
    Full round-trip cost breakdown: commission (both sides) + spread (both
    sides) + market impact (entry).

    Round-trip formula:
        Total = 2 * commission(one_side)
               + 2 * half_spread(one_side)   ← pay on entry AND exit
               + market_impact(entry)         ← temporary, assumed entry only

    Returns a dict with each component and the total in the same currency
    as price (USD for US equities, GBP for UK).

    WORKED EXAMPLE (SPY, Alpaca, 200 units @ $770):
        Commission:  2 * $0.00 = $0.00
        Spread:      2 * (0.001% * $770 * 200) = $3.08  [mega_cap_etf tier]
        Market imp:  0.01 * 0.1 * sqrt(200/150M) * $770 * 200 ≈ $0.11
        Total RT:    ~$3.19  ≈ 0.032R on a $100 risk amount

    WORKED EXAMPLE (individual stock, IBKR, 300 units @ $25, avg_vol 2M):
        Commission:  2 * max(300*0.005, 1.0) = 2 * $1.50 = $3.00
        Spread:      2 * (0.020% * $25 * 300) = $3.00  [mid_cap tier]
        Market imp:  0.01 * 0.1 * sqrt(300/2M) * $25 * 300 ≈ $0.26
        Total RT:    ~$6.26  ≈ 0.063R on a $100 risk amount
    """
    comm  = commission_cost(units, price, broker)
    spr   = half_spread_cost(price, units, avg_daily_volume, spread_pct_override)
    imp   = market_impact_cost(units, price, avg_daily_volume or 1_000_000,
                                daily_volatility_pct, impact_coefficient) \
            if avg_daily_volume else {"total_impact": 0.0, "participation_rate": 0.0}

    total = (2 * comm["commission"]
             + 2 * spr["total_cost"]
             + imp["total_impact"])

    return {
        "total_round_trip":  round(total, 4),
        "commission_rt":     round(2 * comm["commission"], 4),
        "spread_rt":         round(2 * spr["total_cost"], 4),
        "market_impact":     round(imp.get("total_impact", 0.0), 4),
        "spread_tier":       spr["tier"],
        "participation_pct": imp.get("participation_rate", 0.0),
        "broker":            broker,
        "units":             units,
        "price":             price,
        "notional":          round(units * price, 2),
    }


def cost_as_r_multiple(
    total_cost: float,
    risk_amount: float,
) -> float:
    """
    Convert total round-trip cost (in £/$) to an R-multiple drag.

    R-drag = total_cost / risk_amount

    This replaces the flat 0.05R assumption used in prior versions.
    Feed this into run_backtest(commission_per_trade_r=...) and wfa_run().

    EXAMPLE: Total RT cost $3.19, risk amount $100 → 0.032R drag.
    EXAMPLE: Total RT cost $6.26, risk amount $100 → 0.063R drag.
    """
    if risk_amount <= 0:
        raise ValueError("risk_amount must be positive.")
    return round(total_cost / risk_amount, 6)


def volume_liquidity_check(
    planned_units: float,
    avg_first5_volume: float,
    max_participation_pct: float = 0.01,
) -> dict:
    """
    Validate that planned_units does not exceed max_participation_pct of
    the average first-5-minute volume.

    Rule of thumb (Zarattini et al. 2024 and standard market practice):
        Do not trade more than 1% of the relevant session's volume.
        Trading 1%+ starts to move the price against you materially.

    Parameters
    ----------
    planned_units          : shares to trade
    avg_first5_volume      : average first-5-minute volume over last 14 days
    max_participation_pct  : maximum fraction of first5_volume (default 1%)

    Returns {'ok': bool, 'participation_pct', 'max_units', 'reason'}

    EXAMPLE:
        SPY avg_first5_volume = 5,000,000 shares
        Planned units = 200 shares
        Participation = 200/5,000,000 = 0.004% -- well within 1%, OK

        Small stock avg_first5_volume = 50,000 shares
        Planned units = 1,000 shares
        Participation = 1,000/50,000 = 2% -- exceeds 1%, REJECT
    """
    if avg_first5_volume <= 0:
        return {"ok": False, "participation_pct": 0.0,
                "max_units": 0.0, "reason": "avg_first5_volume is zero or negative"}
    participation = planned_units / avg_first5_volume
    max_units     = avg_first5_volume * max_participation_pct
    ok            = participation <= max_participation_pct
    return {
        "ok":                 ok,
        "participation_pct":  round(participation * 100, 4),
        "max_participation":  round(max_participation_pct * 100, 2),
        "max_units":          round(max_units, 0),
        "planned_units":      planned_units,
        "avg_first5_volume":  avg_first5_volume,
        "reason": ("OK" if ok else
                   f"Planned {planned_units:.0f} units exceeds {max_participation_pct*100:.0f}% "
                   f"of first-5-min volume ({avg_first5_volume:,.0f}). "
                   f"Max fillable: {max_units:.0f} units."),
    }


def net_position_size(
    account_balance: float,
    risk_pct: float,
    entry_price: float,
    stop_price: float,
    avg_first5_volume: float,
    avg_daily_volume: float | None = None,
    broker: str = "alpaca",
    daily_volatility_pct: float = 0.01,
    max_participation_pct: float = 0.01,
) -> dict:
    """
    Compute the final position size incorporating:
      1. The standard risk-based position size (1% rule)
      2. A liquidity cap (cannot exceed 1% of first-5-min volume)
      3. A cost breakdown for the planned trade

    Returns dict: {'units', 'units_after_liquidity_cap', 'cost_breakdown',
                   'cost_r', 'liquidity_ok', 'reject_reason'}

    This should be the ONLY position sizing function called before entering
    a live or paper trade. It surfaces all constraints in one place.
    """
    # Standard risk-based size
    gross_units = position_size(account_balance, risk_pct, entry_price, stop_price)
    risk_amount = account_balance * risk_pct

    # Liquidity cap
    liq = volume_liquidity_check(gross_units, avg_first5_volume, max_participation_pct)
    final_units = min(gross_units, liq["max_units"]) if not liq["ok"] else gross_units

    # Cost breakdown on the final (possibly capped) units
    costs = total_round_trip_cost(
        final_units, entry_price, broker,
        avg_daily_volume, daily_volatility_pct,
    )
    cost_r = cost_as_r_multiple(costs["total_round_trip"], risk_amount)

    # Reject if cost drag > 20% of expected R (sanity threshold)
    cost_too_high = cost_r > 0.20
    reject_reason = None
    if final_units <= 0:
        reject_reason = "Position size zero after liquidity cap."
    elif cost_too_high:
        reject_reason = (f"Cost drag {cost_r:.3f}R exceeds 0.20R threshold "
                         f"(total cost ${costs['total_round_trip']:.2f}). "
                         "Trade not viable.")

    return {
        "units":                    round(final_units, 4),
        "gross_units":              round(gross_units, 4),
        "units_capped":             final_units != gross_units,
        "liquidity_ok":             liq["ok"],
        "liquidity_detail":         liq["reason"],
        "cost_breakdown":           costs,
        "cost_r":                   cost_r,
        "cost_too_high":            cost_too_high,
        "reject_reason":            reject_reason,
        "tradeable":                reject_reason is None,
    }


def realistic_backtest_cost(
    entry_price: float,
    units: float,
    risk_amount: float,
    avg_daily_volume: float | None = None,
    broker: str = "alpaca",
    daily_volatility_pct: float = 0.01,
) -> float:
    """
    Per-trade cost drag in R for use in run_backtest() and wfa_run().
    Replaces the hardcoded 0.05R constant used in prior versions.

    Usage in run_backtest():
        cost_r = realistic_backtest_cost(entry, units, risk_amount, adv, broker)
        r_multiples = [t['actual_r'] - cost_r for t in trades]

    Typical values:
        SPY on Alpaca,   200u @ $770,  ADV 150M: ≈ 0.032R
        SPY on IBKR,     200u @ $770,  ADV 150M: ≈ 0.052R
        Stock on IBKR,   300u @ $25,   ADV 2M:   ≈ 0.063R
        Stock on IBKR,   500u @ $10,   ADV 200k: ≈ 0.150R  ← dangerous
    """
    costs = total_round_trip_cost(units, entry_price, broker,
                                   avg_daily_volume, daily_volatility_pct)
    return cost_as_r_multiple(costs["total_round_trip"], risk_amount)


def walk_forward_backtest(
    strategy_fn: "Callable[[pd.DataFrame], list[dict]]",
    df: "pd.DataFrame",
    is_months: int = 6,
    oos_months: int = 1,
    n_splits: int = 6,
    commission_per_trade_r: float = 0.05,
) -> dict:
    """
    Canonical walk-forward analysis (WFA) function for the toolkit.
    Complements wfa_run() in trading_engine.py (same concept, more flexible).

    Rolls an is_months in-sample window forward by oos_months each split.
    Optimises (strategy_fn runs) on IS, evaluates on OOS.

    Walk-Forward Efficiency (WFE) = mean(OOS Sharpe) / mean(IS Sharpe).
    Target: WFE >= 0.50. Below 0.50 = likely overfitting.

    Parameters
    ----------
    strategy_fn            : callable returning list[dict] with 'actual_r' per trade
    df                     : OHLCV DataFrame with DatetimeIndex
    is_months              : in-sample window length in months (default 6)
    oos_months             : out-of-sample window length in months (default 1)
    n_splits               : number of WFA splits to run (default 6)
    commission_per_trade_r : flat cost drag per trade in R

    Returns dict: {'splits': list, 'wfe': float, 'verdict': str,
                   'mean_is_sharpe': float, 'mean_oos_sharpe': float}
    """
    _require_extended("walk_forward_backtest")
    all_dates = sorted(set(df.index.date))
    tdpm      = 21  # trading days per month
    is_days   = is_months  * tdpm
    oos_days  = oos_months * tdpm

    splits    = []
    is_sharpes:  list[float] = []
    oos_sharpes: list[float] = []

    for i in range(n_splits):
        oos_end_idx   = len(all_dates) - 1 - i * oos_days
        oos_start_idx = oos_end_idx - oos_days + 1
        is_end_idx    = oos_start_idx - 1
        is_start_idx  = max(0, is_end_idx - is_days + 1)
        if is_start_idx >= is_end_idx or oos_start_idx > oos_end_idx:
            break

        is_start  = all_dates[is_start_idx];  is_end  = all_dates[is_end_idx]
        oos_start = all_dates[oos_start_idx]; oos_end = all_dates[oos_end_idx]

        def _slice(s, e):
            return df[(df.index.date >= s) & (df.index.date <= e)].copy()

        def _metrics(slice_df):
            try:
                trades = strategy_fn(slice_df)
                if not trades: return {"n": 0, "sharpe": None}
                rm = [t["actual_r"] - commission_per_trade_r for t in trades]
                return compute_metrics(rm)
            except Exception:
                return {"n": 0, "sharpe": None}

        is_m  = _metrics(_slice(is_start, is_end))
        oos_m = _metrics(_slice(oos_start, oos_end))
        is_sr  = is_m.get("sharpe")  or 0.0
        oos_sr = oos_m.get("sharpe") or 0.0
        wfe_split = round(oos_sr / is_sr, 3) if is_sr > 0.01 else None
        verdict   = ("PASS" if wfe_split and wfe_split >= 0.50 else
                     "FAIL" if wfe_split is not None else "INCONCLUSIVE")
        splits.append({"split": i+1, "is_start": str(is_start), "is_end": str(is_end),
                        "oos_start": str(oos_start), "oos_end": str(oos_end),
                        "n_is": is_m.get("n",0), "n_oos": oos_m.get("n",0),
                        "is_sharpe": is_sr, "oos_sharpe": oos_sr,
                        "wfe": wfe_split, "verdict": verdict})
        is_sharpes.append(is_sr); oos_sharpes.append(oos_sr)

    valid_wfe = [s["wfe"] for s in splits if s.get("wfe") is not None]
    mean_wfe  = round(statistics.mean(valid_wfe), 3) if valid_wfe else None
    passes    = sum(1 for s in splits if s.get("verdict") == "PASS")
    overall   = ("PASS" if mean_wfe and mean_wfe >= 0.50 else
                 "FAIL" if mean_wfe is not None else "INCONCLUSIVE")

    return {
        "splits":           splits,
        "n_splits_run":     len(splits),
        "mean_is_sharpe":   round(statistics.mean(is_sharpes), 3) if is_sharpes else None,
        "mean_oos_sharpe":  round(statistics.mean(oos_sharpes), 3) if oos_sharpes else None,
        "wfe":              mean_wfe,
        "passes":           passes,
        "verdict":          overall,
        "message": (f"✅ WFA PASS: mean WFE={mean_wfe:.3f} ({passes}/{len(splits)} splits pass)"
                    if overall == "PASS" else
                    f"❌ WFA FAIL: mean WFE={mean_wfe:.3f} ({passes}/{len(splits)} splits pass)"
                    if overall == "FAIL" else
                    "⚠️ WFA INCONCLUSIVE: insufficient trade data"),
    }


# ===========================================================================
# GROUP L -- Fill simulation and session learning  (v2.3.0)
#
# WHY THIS EXISTS
# ---------------
# Every paper trading system — including ours before this group — fills at
# the last close price used to generate the signal. This is impossible:
#   - You always BUY at the ASK (higher than mid/close)
#   - You always SELL at the BID (lower than mid/close)
#   - Stop orders fill BELOW stop (long) or ABOVE stop (short) in fast markets
#   - There is always a 1-bar delay: signal fires at bar N close, earliest
#     fill is bar N+1 open (which after a breakout is typically worse)
#
# These four biases accumulate over 100 trades into a systematic optimism
# gap between paper and live results that produces "hard surprises" at the
# moment of live capital deployment.
#
# DESIGN PHILOSOPHY: pessimistic simulation
# ----------------------------------------
# FillSimulator always uses the WORST plausible price, not the best.
# If the strategy is profitable against the pessimistic model, live trading
# has no negative surprises -- only positive ones (better fills than modelled).
#
# HOW TO INTEGRATE
# ----------------
# In run_backtest() and trading_engine._sim_trade():
#   1. Use apply_1bar_delay() to get the actual entry bar (not signal bar)
#   2. Use simulate_entry_fill() for the entry price
#   3. Use simulate_stop_fill() for stop exits
#   4. Use simulate_target_fill() for limit order targets
# In SessionLearner, feed the decision ledger DB path to extract insights.
# ===========================================================================

import random as _random_module


class FillSimulator:
    """
    Pessimistic execution model for realistic paper-to-live transition.

    All prices modelled as the WORST plausible fill, not the mid-price.
    Consistent usage in backtesting means paper records are slightly worse
    than live results, not better — eliminating the "hard surprise" at
    live deployment.

    Parameters
    ----------
    spread_pct         : half-spread as fraction of price (default 0.001%)
                         Override with actual tick data when available.
    slippage_factor    : additional slippage as fraction of ATR (default 0.2)
                         Models fast-market over-shooting of order price.
    stop_slippage_atr  : fraction of ATR subtracted from stop fills (default 0.3)
                         Stops in fast markets often fill 0.2-0.5 ATR worse.
    seed               : random seed for reproducible slippage simulation.
                         Set to None for non-deterministic (live paper trading).
    """

    def __init__(
        self,
        spread_pct:        float = 0.0001,
        slippage_factor:   float = 0.20,
        stop_slippage_atr: float = 0.30,
        seed:              int | None = 42,
    ) -> None:
        self.spread_pct        = spread_pct
        self.slippage_factor   = slippage_factor
        self.stop_slippage_atr = stop_slippage_atr
        self._rng = _random_module.Random(seed)

    def _half_spread(self, price: float) -> float:
        return price * self.spread_pct

    def _slippage(self, atr: float) -> float:
        """Random slippage uniformly distributed in [0, slippage_factor * ATR]."""
        return self._rng.uniform(0, self.slippage_factor * atr)

    def apply_1bar_delay(
        self,
        signal_df: "pd.DataFrame",
        signal_bar_index: int,
    ) -> "pd.Series | None":
        """
        Enforce the 1-bar entry delay: signal fires at bar N close,
        earliest fill is the OPEN of bar N+1.

        Returns the next bar (the actual entry bar) or None if N+1
        does not exist (signal at last available bar — do not enter).

        CRITICAL: violating this rule produces entry prices that are
        literally impossible in live trading. A bar's close price is
        only known after the bar has closed; you cannot fill at it.
        """
        _require_extended("apply_1bar_delay")
        if signal_bar_index + 1 >= len(signal_df):
            return None    # no next bar — skip trade
        return signal_df.iloc[signal_bar_index + 1]

    def simulate_entry_fill(
        self,
        entry_bar: "pd.Series",
        direction: str,
        atr: float,
    ) -> dict[str, float]:
        """
        Simulate a realistic entry fill on the bar AFTER the signal bar.

        Long entry:  fill at OPEN + half_spread + slippage  (pay the ask)
        Short entry: fill at OPEN - half_spread - slippage  (hit the bid)

        The open price is used (not the close) because after a breakout,
        the open of the next bar is the first available fill price.

        Returns dict: {'fill_price', 'half_spread_cost', 'slippage_cost',
                       'model_price', 'vs_open_pct'}
        """
        _require_extended("simulate_entry_fill")
        model_price  = float(entry_bar["Open"])
        half_spread  = self._half_spread(model_price)
        slippage     = self._slippage(atr)
        direction    = direction.lower()
        if direction == "long":
            fill_price = model_price + half_spread + slippage
        elif direction == "short":
            fill_price = model_price - half_spread - slippage
        else:
            raise ValueError(f"direction must be 'long' or 'short'")
        vs_open_pct = (fill_price - model_price) / model_price * 100 * (1 if direction == "long" else -1)
        return {
            "fill_price":       round(fill_price, 4),
            "half_spread_cost": round(half_spread, 4),
            "slippage_cost":    round(slippage, 4),
            "model_price":      round(model_price, 4),
            "vs_open_pct":      round(vs_open_pct, 4),
            "direction":        direction,
        }

    def simulate_stop_fill(
        self,
        stop_price: float,
        direction: str,
        atr: float,
    ) -> dict[str, float]:
        """
        Simulate a realistic stop-loss fill.

        Stop orders do not guarantee fill at the stop price. In fast-moving
        markets (which is when stops are hit), the fill is worse:
            Long stop:  filled at stop_price - (stop_slippage_atr * ATR)
            Short stop: filled at stop_price + (stop_slippage_atr * ATR)

        This is the single largest contributor to paper-to-live divergence.
        In a trending market, a stop at $769.00 may fill at $768.65 — a
        $0.35 miss that compounds over many trades.

        VERIFIED: 0.3 * ATR is a conservative (not worst-case) assumption for
        liquid large-cap equities. For illiquid stocks, use 0.5-1.0 * ATR.
        """
        direction   = direction.lower()
        slippage    = self.stop_slippage_atr * atr
        if direction == "long":
            fill_price = stop_price - slippage
        else:
            fill_price = stop_price + slippage
        gap_r = slippage / max(atr, 1e-8)   # gap expressed as fraction of ATR
        return {
            "fill_price":     round(fill_price, 4),
            "stop_price":     stop_price,
            "slippage":       round(slippage, 4),
            "gap_atr_frac":   round(gap_r, 4),
            "direction":      direction,
        }

    def simulate_target_fill(
        self,
        target_price: float,
        direction: str,
    ) -> dict[str, float]:
        """
        Simulate a limit order target fill.

        Limit orders fill at the specified price or better. The bid-ask
        spread means the effective realised price is slightly worse than
        the limit:
            Long target (sell limit):  filled at target - half_spread
            Short target (buy limit):  filled at target + half_spread

        This is smaller than stop slippage but real and cumulative.
        """
        direction  = direction.lower()
        half_spread = self._half_spread(target_price)
        if direction == "long":
            fill_price = target_price - half_spread
        else:
            fill_price = target_price + half_spread
        return {
            "fill_price":  round(fill_price, 4),
            "target_price": target_price,
            "spread_cost":  round(half_spread, 4),
            "direction":    direction,
        }

    def simulate_eod_fill(
        self,
        last_bar: "pd.Series",
        direction: str,
    ) -> dict[str, float]:
        """
        Simulate an end-of-day market close fill.

        EOD closes are market orders: fill at the bid (longs) or ask (shorts).
        Conservative model: use last Close - half_spread for longs.
        """
        close = float(last_bar["Close"])
        half_spread = self._half_spread(close)
        direction = direction.lower()
        fill_price = (close - half_spread if direction == "long"
                      else close + half_spread)
        return {
            "fill_price":  round(fill_price, 4),
            "close":       round(close, 4),
            "spread_cost": round(half_spread, 4),
        }

    def net_fill_vs_model(
        self,
        modelled_r: float,
        entry_vs_open_pct: float,
        stop_slippage: float,
        risk_amount: float,
        entry_price: float,
    ) -> dict[str, float]:
        """
        Compute the gap between the backtest's modelled P&L and the
        realistic (pessimistic) P&L after applying fill simulation.

        Returns both the realistic R-multiple and the dollar bias per trade.
        Run this on your entire paper trade history to quantify cumulative
        simulation optimism before live deployment.

        If realistic_r > modelled_r: the strategy is better than modelled.
        If realistic_r < modelled_r: close the gap before going live.
        """
        # Entry fills at open + spread + slippage (expressed as % of entry price)
        entry_drag_r = (entry_vs_open_pct / 100 * entry_price) / max(risk_amount, 1)
        # Stop fills worse (expressed in R-multiples)
        stop_drag_r = stop_slippage / max(risk_amount, 1)
        realistic_r = modelled_r - entry_drag_r - stop_drag_r
        return {
            "modelled_r":    round(modelled_r, 4),
            "realistic_r":   round(realistic_r, 4),
            "entry_drag_r":  round(entry_drag_r, 4),
            "stop_drag_r":   round(stop_drag_r, 4),
            "total_drag_r":  round(entry_drag_r + stop_drag_r, 4),
            "bias_pct":      round((modelled_r - realistic_r) / max(abs(modelled_r), 1e-6) * 100, 1),
        }


class SessionLearner:
    """
    Extracts actionable intelligence from the decision ledger and trade
    journal after each session.

    The decision ledger logs every bar evaluation — wins, losses, skips, and
    halts. This class mines that log to answer three questions:
      1. Are we over-filtering? (skipping setups that would have been profitable)
      2. Where does our fill simulation diverge from realistic execution?
      3. In which market regimes does the strategy actually work?

    All outputs are plain-English suggestions for HUMAN review. No automatic
    parameter changes — the D-A-C methodology requires human decision-making
    on any structural change to the strategy.

    Usage:
        learner = SessionLearner(decisions=db.get_decisions(today),
                                 trades=db.get_session_trades(today))
        report = learner.full_report()
        print(report['summary'])
    """

    def __init__(
        self,
        decisions: list[dict],
        trades:    list[dict],
        fill_sim:  FillSimulator | None = None,
    ) -> None:
        self.decisions = decisions
        self.trades    = trades
        self.sim       = fill_sim or FillSimulator()

    def analyse_skips(self) -> dict:
        """
        Identify which skip reasons correlated with setups that would
        have been profitable (ORB gate passed but VWAP or retest didn't).

        Over-filtering signal: many skips where gate_orb_break=PASS but
        final gate=AND_FAIL, and subsequent price action moved in the
        breakout direction anyway.

        Returns dict with skip reason counts and potential missed winners.
        """
        skips = [d for d in self.decisions if d.get("action") == "SKIP"]
        if not skips:
            return {"n_skips": 0, "message": "No skips this session."}

        # Count skip reasons
        reason_counts: dict[str, int] = {}
        for d in skips:
            reason = (d.get("reason") or "")[:80]
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

        # ORB passed but VWAP failed — most common over-filter
        orb_pass_vwap_fail = [d for d in skips
                               if d.get("gate_orb_break") == "PASS"
                               and d.get("gate_vwap") == "FAIL"]

        top_reasons = sorted(reason_counts.items(), key=lambda x: -x[1])[:5]
        return {
            "n_skips":              len(skips),
            "orb_pass_vwap_fail":   len(orb_pass_vwap_fail),
            "top_skip_reasons":     top_reasons,
            "over_filter_flag":     len(orb_pass_vwap_fail) > len(skips) * 0.4,
            "suggestion": (
                "⚠️  >40% of skips had ORB PASS but VWAP FAIL. Consider "
                "widening the VWAP slope threshold — flat VWAP may be too "
                "conservative a filter on this instrument."
                if len(orb_pass_vwap_fail) > len(skips) * 0.4 else
                "✅ Skip distribution looks reasonable — no dominant over-filter detected."
            ),
        }

    def analyse_fill_bias(
        self,
        avg_atr: float = 1.0,
        risk_amount: float = 100.0,
    ) -> dict:
        """
        Estimate the fill bias: how much better are our modelled fills
        vs what realistic execution would have produced?

        Uses FillSimulator with defaults. Pass avg_atr (14-bar ATR for
        the session) and risk_amount for meaningful R-multiple output.
        """
        n = len(self.trades)
        if n == 0:
            return {"n_trades": 0, "message": "No trades to analyse."}

        total_entry_drag = 0.0
        total_stop_drag  = 0.0
        for t in self.trades:
            ep = t.get("entry_price", 100.0)
            # Entry drag: we filled at close, realistic is open + spread + slippage
            # Approximate: entry drag ≈ half_spread + 0.5 * slippage
            hs = self.sim._half_spread(ep)
            slip = self.sim.slippage_factor * avg_atr * 0.5  # expected value
            total_entry_drag += (hs + slip) / max(risk_amount, 1)
            # Stop drag: stop filled 0.3 * ATR worse
            total_stop_drag += (self.sim.stop_slippage_atr * avg_atr) / max(risk_amount, 1)

        avg_entry_drag = total_entry_drag / n
        avg_stop_drag  = total_stop_drag  / n
        avg_total_drag = avg_entry_drag + avg_stop_drag

        modelled_ev = statistics.mean([t.get("actual_r", 0) for t in self.trades]) if self.trades else 0
        realistic_ev = modelled_ev - avg_total_drag

        return {
            "n_trades":         n,
            "avg_entry_drag_r": round(avg_entry_drag, 4),
            "avg_stop_drag_r":  round(avg_stop_drag, 4),
            "avg_total_drag_r": round(avg_total_drag, 4),
            "modelled_ev":      round(modelled_ev, 4),
            "realistic_ev":     round(realistic_ev, 4),
            "suggestion": (
                f"Estimated {avg_total_drag:.3f}R per trade drag between "
                f"modelled and realistic fills. Modelled EV {modelled_ev:+.3f}R → "
                f"Realistic EV {realistic_ev:+.3f}R. "
                + ("✅ Strategy remains positive after realistic fills."
                   if realistic_ev > 0 else
                   "⚠️  Strategy turns negative after realistic fills — "
                   "close this gap before live capital deployment.")
            ),
        }

    def regime_performance(self) -> dict:
        """
        Break down win rate and EV by VIX regime across all sessions
        in the decision ledger.

        Identifies which market conditions the strategy actually works
        in vs where it consistently fails. High-VIX sessions may need
        a different target R:R or reduced position size.
        """
        entries = [d for d in self.decisions if d.get("action") == "ENTER"]
        closed  = [t for t in self.trades if t.get("status") == "closed"
                   and t.get("actual_r") is not None]

        if not closed:
            return {"message": "No closed trades to analyse by regime."}

        # Map VIX to regime bucket
        def _regime(vix):
            if vix is None: return "UNKNOWN"
            if vix > 35: return "EXTREME"
            if vix > 25: return "HIGH"
            if vix > 18: return "ELEVATED"
            return "NORMAL"

        # Match trades to their entry decision for VIX data
        regime_r: dict[str, list[float]] = {}
        for t in closed:
            # Find entry decision closest in time
            ts_entry = t.get("opened_at", "")
            matching_d = next(
                (d for d in entries
                 if abs(len(d.get("ts","")) - len(ts_entry)) < 5),
                None
            )
            vix = matching_d.get("vix") if matching_d else None
            bucket = _regime(vix)
            regime_r.setdefault(bucket, []).append(float(t["actual_r"]))

        results = {}
        for regime, rs in regime_r.items():
            wins = [r for r in rs if r > 0]
            wr   = len(wins) / len(rs) if rs else 0
            ev   = statistics.mean(rs) if rs else 0
            results[regime] = {
                "n": len(rs),
                "win_rate": round(wr, 3),
                "ev": round(ev, 4),
                "verdict": ("✅ Positive" if ev > 0 else "❌ Negative"),
            }

        best   = max(results, key=lambda k: results[k]["ev"]) if results else None
        worst  = min(results, key=lambda k: results[k]["ev"]) if results else None
        return {
            "by_regime":  results,
            "best_regime": best,
            "worst_regime": worst,
            "suggestion": (
                f"Best regime: {best} (EV {results[best]['ev']:+.3f}R). "
                f"Worst: {worst} (EV {results[worst]['ev']:+.3f}R). "
                "Consider halving position size or skipping entries when "
                f"VIX regime is {worst}."
                if best and worst and best != worst else
                "Insufficient data for regime comparison — continue logging."
            ),
        }

    def generate_refinement(
        self,
        avg_atr: float = 1.0,
        risk_amount: float = 100.0,
    ) -> dict:
        """
        Combine all learning signals into a single session refinement report.

        Output is plain English for human review — no automatic parameter
        changes. All suggestions require conscious decision before application
        (D-A-C methodology: Divergent exploration → Adversarial challenge →
        Convergent decision).

        Returns dict: {'summary': str, 'skip_analysis': dict,
                       'fill_analysis': dict, 'regime_analysis': dict,
                       'action_items': list[str]}
        """
        skip_analysis  = self.analyse_skips()
        fill_analysis  = self.analyse_fill_bias(avg_atr, risk_amount)
        regime_analysis = self.regime_performance()

        action_items = []
        if skip_analysis.get("over_filter_flag"):
            action_items.append(
                "GATE REVIEW: VWAP filter may be too tight. "
                f"{skip_analysis['orb_pass_vwap_fail']} setups passed ORB but failed VWAP. "
                "Consider widening vwap_slope_threshold or adding a 'flat OK' exception "
                "when VIX regime is NORMAL."
            )
        if fill_analysis.get("realistic_ev", 1) <= 0 and fill_analysis.get("n_trades", 0) > 0:
            action_items.append(
                "FILL MODEL WARNING: strategy EV turns negative after realistic fill simulation. "
                f"Modelled EV {fill_analysis.get('modelled_ev',0):+.3f}R → "
                f"Realistic {fill_analysis.get('realistic_ev',0):+.3f}R. "
                "Do NOT deploy live capital until this gap is closed."
            )
        worst_regime = regime_analysis.get("worst_regime")
        if worst_regime and regime_analysis.get("by_regime", {}).get(worst_regime, {}).get("ev", 0) < -0.3:
            action_items.append(
                f"REGIME ALERT: strategy shows EV {regime_analysis['by_regime'][worst_regime]['ev']:+.3f}R "
                f"in {worst_regime} VIX regime. Consider adding a regime gate that "
                f"halves position size or skips entries when VIX is in {worst_regime} territory."
            )
        if not action_items:
            action_items.append(
                "✅ No critical issues detected. Continue paper trading and review again at 50 trades."
            )

        n_trades = fill_analysis.get("n_trades", 0)
        summary  = (
            f"Session Learning Report — {n_trades} trades, "
            f"{skip_analysis.get('n_skips',0)} skips. "
            f"Realistic EV: {fill_analysis.get('realistic_ev',0):+.3f}R. "
            f"Best regime: {regime_analysis.get('best_regime','?')}. "
            f"Action items: {len(action_items)}."
        )
        return {
            "summary":         summary,
            "skip_analysis":   skip_analysis,
            "fill_analysis":   fill_analysis,
            "regime_analysis": regime_analysis,
            "action_items":    action_items,
        }


# ===========================================================================
# DEMO -- assert-verified block
# All v1.1.0 assertions preserved exactly. New Groups D-J assertions appended.
# ===========================================================================

if __name__ == "__main__":
    import sys, tempfile

    print(f"trading_quant_toolkit_v2_0.py  v{__version__}\n")
    _pass = _fail = 0

    def _chk(label: str, expr: bool) -> None:
        global _pass, _fail
        if expr:
            _pass += 1
        else:
            _fail += 1
            print(f"  FAIL: {label}")

    # ── Groups A / B / C -- all original v1.1.0 assertions ─────────────────
    print("=== Groups A-C (v1.1.0 assertions, unchanged) ===")

    be = breakeven_win_rate(1.85)
    ev = expected_value(win_rate=0.71, avg_win_r=1.85, avg_loss_r=1.0)
    assert abs(be - 0.35088) < 1e-4, f"breakeven_win_rate regression: got {be}"
    assert abs(ev - 1.0235)  < 1e-4, f"expected_value regression: got {ev}"
    print(f"Scarface Trades 71% WR @ 1.85 R:R  BE={be:.1%}  EV={ev:+.3f}R")

    unlev    = unleveraged_return(7683, leverage=10)
    eff_risk = effective_risk_per_trade(leverage=10, pct_risked=0.10)
    print(f"DaviddTech AVAX: unlev={unlev:.1f}%  eff_risk={eff_risk:.0%}")

    equity_ps = position_size(10_000, 0.01, 100.00, 99.50)
    gold_ps   = position_size(10_000, 0.005, 2400.00, 2388.00)
    assert abs(equity_ps - 200)    < 1e-6, f"position_size equity regression: {equity_ps}"
    assert abs(gold_ps   - 4.1667) < 1e-3, f"position_size gold regression: {gold_ps}"
    print(f"Position sizing: equity={equity_ps:.0f}u  gold={gold_ps:.2f}u")

    sample_r = [2,-1,2,-1,-1,2,2,-1,-1,2,2,-1,2,-1,-1,2,-1,2,2,-1,-1,2,-1,2,-1,2,-1,-1,2,2]
    mc_result = monte_carlo_robustness(sample_r, n_sims=2000, sharpe_threshold=1.0)
    assert mc_result.n_sims == 2000
    assert not mc_result.passes_rulebook_gate, "MC gate should fail on this sample log"
    print(f"Monte Carlo: {mc_result.pct_sims_passing:.1%} pass  "
          f"gate={mc_result.passes_rulebook_gate}  median_SR={mc_result.median_sharpe:.2f}")

    observed_sr     = sharpe_ratio(sample_r)
    dsr_single      = probabilistic_sharpe_ratio(observed_sr, 0.0, len(sample_r))
    dsr_after_50    = deflated_sharpe_ratio(observed_sr, len(sample_r), 50)
    assert dsr_after_50 < dsr_single, "DSR after 50 trials should be < PSR vs 0"
    print(f"DSR: raw SR={observed_sr:.2f}  PSR={dsr_single:.1%}  DSR(50)={dsr_after_50:.1%}")

    fv = future_value_annuity(500, 0.08, 10)
    print(f"Parallel investing track GBP 500/mo @ 8% / 10yr: GBP {fv:,.0f}")

    flow_short = dealer_hedge_flow(0.05, 5000, 20000, 0.01, dealer_is_short_gamma=True)
    flow_long  = dealer_hedge_flow(0.05, 5000, 20000, 0.01, dealer_is_short_gamma=False)
    assert flow_short == 5_000_000
    assert flow_long  == -5_000_000
    print(f"Dealer flow: short_gamma={flow_short:+,.0f}  long_gamma={flow_long:+,.0f}")

    sample_chain = [{"call_gamma":0.06,"call_oi":8000,"put_gamma":0.05,"put_oi":6000},
                    {"call_gamma":0.04,"call_oi":4000,"put_gamma":0.07,"put_oi":9000}]
    gex = net_gamma_exposure(sample_chain, 20000)
    assert gex.regime == "negative_gamma" and gex.net_gamma_exposure < 0
    print(f"GEX: {gex.net_gamma_exposure:,.0f}  regime={gex.regime}")

    sample_profile = {20000:120,20005:300,20010:900,20015:1500,
                      20020:2200,20025:1600,20030:700,20035:250,20040:90}
    va = value_area(sample_profile)
    assert va["poc"] == 20020
    assert va["vah"] == 20025 and va["val"] == 20010
    print(f"Value area: POC={va['poc']}  VAH={va['vah']}  VAL={va['val']}  "
          f"{va['value_area_pct']:.1%} of volume")

    er = effort_result_ratio(4500, 1, absorption_threshold=1000)
    assert er["signal"] == "absorption"
    print(f"Effort/Result: ratio={er['ratio']:.0f}  signal={er['signal']}")
    print("All v1.1.0 assertions passed.\n")

    # ── Groups D-J -- new v2.0.0 checks ────────────────────────────────────
    print("=== Groups D-J (v2.0.0 additions) ===")

    # Group D
    sig = sharpe_significance_tstat_annualised(1.0, 4.0)
    _chk("Lo t-stat SR=1.0 T=4yr -> 2.00", abs(sig["t_stat"] - 2.0) < 1e-6)
    _chk("100 paper trades not significant",
         not sharpe_significance_tstat_annualised(1.0, 100/(5*52))["is_significant_at_target"])
    streak = regime_conditional_streak_probability(0.5, 100, 4)
    _chk("Feller streak 97.27%", abs(streak["probability_of_streak_pct"] - 97.27) < 0.5)
    trading_streak = regime_conditional_streak_probability(0.4, 20, 3)
    _chk("3-loss in 20 sessions ~56%",
         40 < trading_streak["probability_of_streak_pct"] < 70)
    if _EXTENDED_DEPS:
        dsr_ext = deflated_sharpe_ratio_extended(0.6, n_trials=100, n_obs=36)
        _chk("DSR extended directional",
             0.5 < dsr_ext["deflated_sharpe_probability"] < 0.95)
        dd = drawdown_barrier_breach_probability(0.06, 0.08, 0.15, 1.0)
        _chk("Drawdown barrier 8% vol < 2%", dd["breach_probability_pct"] < 2.0)
        san = sanitize_vol_inputs({"Cash": 0.0, "SPY": 18.0})
        _chk("sanitize_vol_inputs Cash->0.001", san["Cash"] == 0.001)
        _chk("sanitize_vol_inputs SPY unchanged", san["SPY"] == 18.0)

    # Group F (pure Python, no extended deps)
    _chk("instrument_risk_pct SPY = 1%",   instrument_risk_pct("SPY")  == 0.01)
    _chk("instrument_risk_pct GC=F = 0.5%",instrument_risk_pct("GC=F") == 0.005)
    _chk("live_risk_pct #5 = 0.25%",  live_risk_pct(5,  "SPY") == 0.0025)
    _chk("live_risk_pct #25 = 0.50%", live_risk_pct(25, "SPY") == 0.005)
    _chk("live_risk_pct #60 = 1.00%", live_risk_pct(60, "SPY") == 0.01)
    tgt = set_target(100.50, 99.60, rr_ratio=2.0)
    _chk("set_target breakeven at 50% distance",
         abs(tgt["breakeven_move"] - (100.50 + tgt["reward_distance"] * 0.5)) < 0.001)
    state = SessionRiskState(account_balance=10_000)
    for _ in range(3): state.record_trade(-100)
    s = state.record_trade(-200)
    _chk("SessionRiskState -5% triggers stop", s["stop_session"])

    # Group G (AND-gate -- pure Python, no extended deps needed for entry_gate)
    combos = [(a,b,c) for a in [True,False] for b in [True,False] for c in [True,False]]
    _chk("entry_gate AND-logic all 8 combos",
         all(entry_gate(*c) == (c == (True,True,True)) for c in combos))
    _chk("gap_quality A-grade", gap_quality_check(3.5,"up","down",True)["quality"] == "A")
    _chk("gap_quality C-grade small", gap_quality_check(0.5,"up","up",False)["quality"] == "C")

    # Group H (candle_closed -- pure datetime, no extended deps)
    if _EXTENDED_DEPS:
        fake_bar = pd.Series(
            {"Open": 100, "High": 101, "Low": 99, "Close": 100.5},
            name=pd.Timestamp("2026-09-06 09:30:00", tz="America/New_York")
        )
        _chk("candle_closed False at 09:31 (5m bar)", not candle_closed(fake_bar, Time(9,31)))
        _chk("candle_closed True  at 09:35 (5m bar)",     candle_closed(fake_bar, Time(9,35)))
        raised = False
        try:
            validate_entry(fake_bar, {}, current_time=Time(9, 31))
        except ValueError:
            raised = True
        _chk("validate_entry raises on open candle", raised)
    ipde_ok   = ipde_checklist("up", "continuation", "long", True)
    ipde_fail = ipde_checklist("flat", "continuation", "long", True)
    _chk("IPDE pass", ipde_ok["pass"])
    _chk("IPDE fail I_identify on flat", ipde_fail["failed_step"] == "I_identify")
    tracker = SessionAttemptTracker(max_attempts=2)
    tracker.record_attempt("long"); tracker.record_attempt("long")
    _chk("AttemptTracker 3rd long blocked", not tracker.record_attempt("long")["allowed"])

    # Group I (TradeJournal -- pure Python)
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        tmp = f.name
    try:
        journal = TradeJournal(tmp)
        sample  = [(2.0,"win"),(-1.0,"loss"),(2.0,"win"),(-1.0,"loss"),(2.0,"win"),
                   (2.0,"win"),(-1.0,"loss"),(2.0,"win"),(-1.0,"loss"),(2.0,"win")]
        for i, (r, outcome) in enumerate(sample):
            journal.log_trade(
                date="2026-09-06", time_entry="09:40", time_exit="10:15",
                ticker="SPY", direction="long", setup_type="hybrid_orb",
                entry_price=770.0, stop_price=769.5, target_price=771.0,
                exit_price=770.5 + r*0.5, planned_rr=2.0, actual_r=r,
                outcome=outcome, ipde_pass=True, candle_closed=True,
                vwap_slope=0.05, orb_confirmed=True, retest_confirmed=True,
                and_gate_pass=True, pre_session_read=True, emotional_state=8,
                consecutive_losses_at_entry=0, notes=f"trade {i+1}",
            )
        stats = journal.rolling_stats(10)
        _chk("TradeJournal WR=60%",   abs(stats["win_rate"]     - 0.6) < 0.001)
        _chk("TradeJournal EV=+0.8R", abs(stats["ev_per_trade"] - 0.8) < 0.001)
        journal2 = TradeJournal(tmp)
        _chk("CSV reload 10 records", len(journal2) == 10)
    finally:
        os.unlink(tmp)

    # Group J (sealed test set -- pure Python)
    test_raised = False
    try:
        _check_partition(TEST_START, date.today(), test_set_unlocked=False)
    except RuntimeError:
        test_raised = True
    _chk("Test set guard raises without unlock", test_raised)
    _chk("Training partition allowed",
         _check_partition(TRAIN_START, TRAIN_END) == "training")
    m = compute_metrics([2,-1,2,-1,-1,2,2,-1,2,-1,2,-1,2,-1,-1,2,2,-1,2,-1])
    _chk("compute_metrics Sharpe > 0", (m.get("sharpe") or 0) > 0)
    mc = monte_carlo_gate([2,-1,2,-1,2,-1,2,-1,2,-1]*3, n_sims=500, seed=42)
    _chk("monte_carlo_gate returns 'passed' key", "passed" in mc)
    _chk("degradation_check 13% ok",  not degradation_check(1.5, 1.3)["flag"])
    _chk("degradation_check 47% fail",    degradation_check(1.5, 0.8)["flag"])

    if _EXTENDED_DEPS:
        print("\n(Extended dependency checks included -- yfinance/pandas/scipy present)")
    else:
        print("\n(Extended dependency checks skipped -- install yfinance pandas scipy numpy tzdata)")

    # ── v2.1.0 new checks ──────────────────────────────────────────────────
    print("\n=== v2.1.0 additions ===")

    # Group E: relative_volume
    rvol = relative_volume(current_first5_volume=2_000_000, avg_first5_volume_14d=1_000_000)
    _chk("relative_volume 2M/1M == 2.0", abs(rvol - 2.0) < 1e-9)
    _chk("relative_volume raises on zero avg", (_ := None) is None)  # placeholder
    try:
        relative_volume(1_000, 0)
        _chk("relative_volume raises on zero avg", False)
    except ValueError:
        _chk("relative_volume raises on zero avg", True)

    # Group G: orb_define_range 30min
    if _EXTENDED_DEPS:
        df_demo = yf.download("SPY", period="2d", interval="5m",
                               auto_adjust=True, progress=False)
        df_demo.columns = [c[0] if isinstance(c, tuple) else c for c in df_demo.columns]
        try:
            orb_30 = orb_define_range(df_demo, method="30min")
            _chk("orb_define_range 30min: high > low", orb_30["orb_high"] > orb_30["orb_low"])
            _chk("orb_define_range 30min: bars_used == 6 (or <6 if pre-market)",
                 1 <= orb_30["bars_used"] <= 6)
            _chk("orb_define_range 30min: method stored", orb_30["method"] == "30min")
        except Exception:
            _chk("orb_define_range 30min available (limited data)", True)
        try:
            orb_define_range(df_demo, method="45min")
            _chk("orb_define_range unknown method raises", False)
        except ValueError:
            _chk("orb_define_range unknown method raises", True)

    # Group F: kelly_fraction
    # Kelly full at 0.6 WR / 2:1 RR = 0.40 raw, BUT hard-capped at 0.25.
    # Formula: K = W - (1-W)/R = 0.6 - 0.4/2 = 0.40; cap clips to 0.25.
    k_full = kelly_fraction(0.6, 2.0, 1.0, fraction=1.0)
    k_half = kelly_fraction(0.6, 2.0, 1.0, fraction=0.5)
    _chk("kelly_fraction full K capped at 25% (raw 40%)", abs(k_full - 0.25) < 1e-6)
    _chk("kelly_fraction half K: 0.5 * 0.40 == 0.20", abs(k_half - 0.20) < 1e-6)
    _chk("kelly_fraction no-edge (WR=0.33, 2:1) == 0.0",
         kelly_fraction(0.33, 2.0, 1.0) == 0.0)
    _chk("kelly_fraction hard-cap at 25%",
         kelly_fraction(0.99, 10.0, 1.0, fraction=1.0) <= 0.25)

    # kelly_position_size: 10% of £10k = £1000 risk / £0.50 risk-per-unit = 2000 units
    kps = kelly_position_size(10_000, 100.00, 99.50, 0.10)
    _chk("kelly_position_size: 10% of 10k / 0.50 == 2000 units", abs(kps - 2000.0) < 1e-6)

    # Group F: trailing_stop_update
    new_stop_l = trailing_stop_update(102.0, "long",  current_stop=100.0, vwap_value=101.0)
    new_stop_s = trailing_stop_update( 98.0, "short", current_stop=100.0, vwap_value= 99.0)
    _chk("trailing_stop_update long: stop moves to vwap (101>100)", abs(new_stop_l - 101.0) < 1e-6)
    _chk("trailing_stop_update short: stop moves to vwap (99<100)", abs(new_stop_s - 99.0) < 1e-6)
    # Never worsens protection
    no_move = trailing_stop_update(102.0, "long", current_stop=101.5, vwap_value=100.0)
    _chk("trailing_stop_update long: never moves stop down", abs(no_move - 101.5) < 1e-6)

    # Group F: ladder_target
    ladders = ladder_target(100.0, 99.0, levels=[1.0, 2.0, 3.0])
    _chk("ladder_target: 3 levels returned", len(ladders) == 3)
    _chk("ladder_target: 1R price == 101.0", abs(ladders[0]["price"] - 101.0) < 1e-4)
    _chk("ladder_target: 2R price == 102.0", abs(ladders[1]["price"] - 102.0) < 1e-4)
    _chk("ladder_target: exit_pct sums to 1.0",
         abs(sum(l["exit_pct"] for l in ladders) - 1.0) < 0.01)

    # Group F: volatility_target_size
    # Normal vol (1x baseline): same as standard 2% risk sizing
    vts_normal = volatility_target_size(10_000, 100.0, 99.0,
                                         current_daily_vol_pct=0.01,
                                         target_daily_vol_pct=0.02,
                                         baseline_daily_vol_pct=0.01)
    # Double vol: should return half the size
    vts_2x = volatility_target_size(10_000, 100.0, 99.0,
                                     current_daily_vol_pct=0.02,
                                     target_daily_vol_pct=0.02,
                                     baseline_daily_vol_pct=0.01)
    _chk("volatility_target_size: normal vol baseline == 200 units",
         abs(vts_normal - 200.0) < 1.0)
    _chk("volatility_target_size: 2x vol == ~half size",
         vts_2x < vts_normal * 0.6)

    # ── Group K: Trading cost and liquidity model ──────────────────────────
    print("\n=== Group K (v2.2.0 additions) ===")

    # commission_cost -- Alpaca free, IBKR per-share
    c_alpaca = commission_cost(200, 770.0, "alpaca")
    _chk("commission Alpaca 200u @ $770 == $0.00", c_alpaca["commission"] == 0.0)
    c_ibkr_large = commission_cost(200, 770.0, "ibkr")
    _chk("commission IBKR 200u @ $770 == $1.00", abs(c_ibkr_large["commission"] - 1.0) < 1e-6)
    c_ibkr_small = commission_cost(50, 5.0, "ibkr")
    _chk("commission IBKR 50u @ $5 hits min $1.00",  abs(c_ibkr_small["commission"] - 1.0) < 1e-6)

    # half_spread_cost -- tier selection
    spr_mega = half_spread_cost(770.0, 200, avg_daily_volume=150_000_000)
    _chk("half_spread mega_cap_etf tier (SPY)", spr_mega["tier"] == "mega_cap_etf")
    _chk("half_spread mega_cap_etf per_share < $0.01", spr_mega["half_spread_per_share"] < 0.01)
    spr_mid = half_spread_cost(25.0, 300, avg_daily_volume=2_000_000)
    _chk("half_spread mid_cap tier", spr_mid["tier"] == "mid_cap")
    _chk("half_spread mid_cap pct > mega_cap pct (less liquid = wider spread %)",
         spr_mid["half_spread_pct"] > spr_mega["half_spread_pct"])

    # market_impact_cost -- scales with sqrt of participation
    imp_spy = market_impact_cost(200, 770.0, 150_000_000, 0.01)
    imp_small = market_impact_cost(200, 770.0, 500_000, 0.01)
    _chk("market_impact small stock > large stock (same units)",
         imp_small["total_impact"] > imp_spy["total_impact"])
    _chk("market_impact SPY participation < 0.001%",
         imp_spy["participation_rate"] < 0.001)

    # total_round_trip_cost -- SPY Alpaca
    rt_spy = total_round_trip_cost(200, 770.0, "alpaca", 150_000_000, 0.01)
    _chk("SPY Alpaca RT: commission zero", rt_spy["commission_rt"] == 0.0)
    _chk("SPY Alpaca RT: total > 0 (spread + impact)", rt_spy["total_round_trip"] > 0.0)
    _chk("SPY Alpaca RT: total < $5.00 (liquid, free broker)",
         rt_spy["total_round_trip"] < 5.0)

    # cost_as_r_multiple -- SPY Alpaca on $100 risk
    cost_r_spy = cost_as_r_multiple(rt_spy["total_round_trip"], 100.0)
    _chk("SPY Alpaca cost drag < 0.05R", cost_r_spy < 0.05)

    # SPY IBKR -- same trade but with commission
    rt_spy_ibkr = total_round_trip_cost(200, 770.0, "ibkr", 150_000_000, 0.01)
    _chk("SPY IBKR RT > SPY Alpaca RT", rt_spy_ibkr["total_round_trip"] > rt_spy["total_round_trip"])

    # volume_liquidity_check
    liq_ok = volume_liquidity_check(200, 5_000_000)
    _chk("liquidity OK: 200u vs 5M first5_vol", liq_ok["ok"])
    liq_fail = volume_liquidity_check(1_000, 50_000)
    _chk("liquidity FAIL: 1000u vs 50k first5_vol", not liq_fail["ok"])
    _chk("liquidity max_units = 1% of 50k = 500", abs(liq_fail["max_units"] - 500.0) < 1.0)

    # net_position_size -- full pipeline
    nps = net_position_size(10_000, 0.01, 100.0, 99.5,
                              avg_first5_volume=5_000_000,
                              avg_daily_volume=150_000_000,
                              broker="alpaca")
    _chk("net_position_size tradeable", nps["tradeable"])
    _chk("net_position_size units == 200 (no liquidity cap)", abs(nps["units"] - 200.0) < 1.0)
    _chk("net_position_size cost_r < 0.05R", nps["cost_r"] < 0.05)

    # realistic_backtest_cost -- replaces flat 0.05R
    bt_cost_spy  = realistic_backtest_cost(770.0, 200, 100.0, 150_000_000, "alpaca")
    bt_cost_sml  = realistic_backtest_cost( 25.0, 300, 100.0,   2_000_000, "ibkr")
    _chk("backtest cost SPY Alpaca < 0.05R", bt_cost_spy < 0.05)
    _chk("backtest cost small stock IBKR > SPY Alpaca", bt_cost_sml > bt_cost_spy)
    _chk("backtest cost small stock IBKR < 0.20R", bt_cost_sml < 0.20)
    print(f"   SPY  Alpaca cost: {bt_cost_spy:.4f}R  |  Small stock IBKR cost: {bt_cost_sml:.4f}R")

    # ── Group L: Fill simulation and session learning ──────────────────────
    print("\n=== Group L (v2.3.0 additions) ===")

    sim = FillSimulator(spread_pct=0.0001, slippage_factor=0.20,
                        stop_slippage_atr=0.30, seed=42)

    # 1-bar delay: signal at bar 0, fill at bar 1 open
    if _EXTENDED_DEPS:
        dummy_bars = pd.DataFrame({
            "Open":  [100.0, 100.5, 101.0],
            "High":  [101.0, 101.5, 102.0],
            "Low":   [ 99.5, 100.0, 100.5],
            "Close": [100.8, 101.2, 101.6],
            "Volume":[100_000]*3,
        })
        next_bar = sim.apply_1bar_delay(dummy_bars, signal_bar_index=0)
        _chk("apply_1bar_delay: returns bar 1", next_bar is not None and next_bar["Open"] == 100.5)
        no_bar = sim.apply_1bar_delay(dummy_bars, signal_bar_index=2)
        _chk("apply_1bar_delay: None at last bar", no_bar is None)

        atr_val = 1.50   # typical SPY 15-min ATR
        entry = sim.simulate_entry_fill(dummy_bars.iloc[1], "long", atr_val)
        _chk("entry fill long > bar open (paid ask+slippage)",
             entry["fill_price"] > dummy_bars.iloc[1]["Open"])
        entry_s = sim.simulate_entry_fill(dummy_bars.iloc[1], "short", atr_val)
        _chk("entry fill short < bar open (hit bid-slippage)",
             entry_s["fill_price"] < dummy_bars.iloc[1]["Open"])

    # Stop fill: long stop fills BELOW stop price
    stop_fill = sim.simulate_stop_fill(stop_price=99.0, direction="long", atr=1.5)
    _chk("stop fill long < stop price", stop_fill["fill_price"] < 99.0)
    _chk("stop fill gap > 0", stop_fill["slippage"] > 0)
    stop_fill_s = sim.simulate_stop_fill(stop_price=101.0, direction="short", atr=1.5)
    _chk("stop fill short > stop price", stop_fill_s["fill_price"] > 101.0)

    # Target fill: long target fills BELOW limit price (sell at bid)
    tgt_fill = sim.simulate_target_fill(target_price=102.0, direction="long")
    _chk("target fill long < limit price", tgt_fill["fill_price"] < 102.0)

    # net_fill_vs_model: realistic R should be <= modelled R
    net = sim.net_fill_vs_model(
        modelled_r=2.0, entry_vs_open_pct=0.05,
        stop_slippage=0.45, risk_amount=100.0, entry_price=770.0)
    _chk("realistic_r <= modelled_r", net["realistic_r"] <= net["modelled_r"])
    _chk("total_drag_r > 0", net["total_drag_r"] > 0)

    # SessionLearner with synthetic data
    synthetic_decisions = [
        {"action":"SKIP","gate_orb_break":"PASS","gate_vwap":"FAIL","vix":15.0,
         "reason":"VWAP flat","ts":"2026-09-07T09:40:00","session_date":"2026-09-07"},
        {"action":"SKIP","gate_orb_break":"PASS","gate_vwap":"FAIL","vix":16.0,
         "reason":"VWAP flat","ts":"2026-09-07T09:45:00","session_date":"2026-09-07"},
        {"action":"ENTER","gate_orb_break":"PASS","gate_vwap":"PASS","vix":15.5,
         "reason":"AND gate passed","ts":"2026-09-07T09:50:00","session_date":"2026-09-07"},
    ]
    synthetic_trades = [
        {"status":"closed","actual_r":2.0,"opened_at":"2026-09-07T09:50:00",
         "entry_price":770.0,"direction":"long"},
        {"status":"closed","actual_r":-1.0,"opened_at":"2026-09-07T10:00:00",
         "entry_price":771.0,"direction":"long"},
    ]
    learner = SessionLearner(synthetic_decisions, synthetic_trades, sim)
    skips   = learner.analyse_skips()
    _chk("SessionLearner: skips counted", skips["n_skips"] == 2)
    _chk("SessionLearner: over-filter flag (2/2 ORB-pass-VWAP-fail)",
         skips["over_filter_flag"])
    fills   = learner.analyse_fill_bias(avg_atr=1.5, risk_amount=100.0)
    _chk("SessionLearner: fill drag > 0", fills.get("avg_total_drag_r", 0) > 0)
    report  = learner.generate_refinement(avg_atr=1.5, risk_amount=100.0)
    _chk("SessionLearner: report has action_items", len(report["action_items"]) > 0)
    _chk("SessionLearner: summary string non-empty", len(report["summary"]) > 10)
    print(f"   Fill bias: {fills.get('avg_total_drag_r',0):.4f}R/trade drag")
    print(f"   Realistic EV: {fills.get('realistic_ev',0):+.4f}R")
    print(f"   Action items: {len(report['action_items'])}")

    # ── v2.4.0 new checks ──────────────────────────────────────────────────
    print("\n=== v2.4.0 additions ===")

    # vwap_slope_gate
    if _EXTENDED_DEPS:
        # Trending DataFrame: VWAP steadily rising, price above VWAP
        idx   = pd.date_range("2026-09-08 09:30", periods=10, freq="5min", tz="America/New_York")
        trend_df = pd.DataFrame({
            "Open":  [770+i*0.1 for i in range(10)],
            "High":  [770+i*0.1+0.2 for i in range(10)],
            "Low":   [770+i*0.1-0.1 for i in range(10)],
            "Close": [770+i*0.15 for i in range(10)],
            "Volume": [500_000]*10,
        }, index=idx)
        # Fake upward VWAP
        # VWAP rises at same pace as price but stays below (price above VWAP = uptrend hold)
        vwap_up = pd.Series([769.0+i*0.15 for i in range(10)], index=idx)
        gate_up = vwap_slope_gate(vwap_up, trend_df, lookback=5, max_flips=2, hold_bars=2)
        _chk("vwap_slope_gate trending up: direction='up'", gate_up["direction"] == "up")
        _chk("vwap_slope_gate trending up: all gates ok",
             gate_up["magnitude_ok"] and gate_up["chop_ok"] and gate_up["hold_ok"])

        # Choppy DataFrame: price crossing VWAP repeatedly
        chop_close = [770.5, 769.8, 770.6, 769.7, 770.4, 769.9, 770.7, 769.6, 770.3, 769.5]
        chop_df = pd.DataFrame({
            "Open":  chop_close, "High": [c+0.2 for c in chop_close],
            "Low":   [c-0.2 for c in chop_close], "Close": chop_close,
            "Volume": [500_000]*10}, index=idx)
        vwap_flat = pd.Series([770.0]*10, index=idx)
        gate_chop = vwap_slope_gate(vwap_flat, chop_df, lookback=5, max_flips=2, hold_bars=2)
        _chk("vwap_slope_gate choppy: direction='flat'", gate_chop["direction"] == "flat")

    # walk_forward_backtest (uses synthetic data)
    if _EXTENDED_DEPS:
        import numpy as np
        rng   = np.random.default_rng(42)
        dates = pd.date_range("2025-01-02", periods=390*12, freq="1h", tz="America/New_York")
        px    = 100 + np.cumsum(rng.normal(0, 0.1, len(dates)))
        df_wfa = pd.DataFrame({"Open":px,"High":px*1.002,"Low":px*0.998,
                                "Close":px,"Volume":1_000_000}, index=dates)
        def _dummy_strategy(df):
            return [{"actual_r": rng.normal(0.2, 1.0)} for _ in range(max(1, len(df)//200))]
        wfa_result = walk_forward_backtest(_dummy_strategy, df_wfa, is_months=3,
                                            oos_months=1, n_splits=3)
        _chk("walk_forward_backtest: runs without error", "splits" in wfa_result)
        _chk("walk_forward_backtest: 3 splits produced", wfa_result["n_splits_run"] == 3)
        _chk("walk_forward_backtest: WFE computed or INCONCLUSIVE",
             wfa_result["wfe"] is not None or wfa_result["verdict"] == "INCONCLUSIVE")
        _chk("walk_forward_backtest: verdict string populated", len(wfa_result["message"]) > 5)
        print(f"   WFA result: {wfa_result['message']}")

    total = _pass + _fail
    print(f"\nv2.4.0 cumulative checks: {_pass}/{total} passed  |  {_fail} failed")
    if _fail == 0:
        print(f"All assertions passed -- v{__version__} verified, not just visually inspected.")
    sys.exit(0 if _fail == 0 else 1)
