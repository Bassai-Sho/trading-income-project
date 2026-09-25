"""
evaluation/null_baseline.py  (Phase A, P2-123 — slice 3, gate 1)
=================================================================
The null-alpha gate: does a box's ENTRY DECISION beat chance? Box-agnostic —
it reads only a trade log and the bars, never a box. Every counterfactual trade
is simulated through the real ExecutionCore (no copied fill logic).

Common exit (the harness cannot know box-specific exits such as a VWAP trail):
    market entry at the open of the minute after the signal bar closes; resting
    stop at the trade's own stop distance (as a fraction of price) and a 1R
    target, one OCO bracket live from the fill minute; market exit at the open
    of the session_end minute. The box's trades are re-run under this SAME exit,
    so the comparison is like for like. The box's real expectancy is reported
    alongside, not used by the gate.

Test A — DIRECTION (does the chosen side beat a shuffled side?)
    Each trade is simulated both ways at its own signal time. Nulls shuffle the
    long/short LABELS across trades (1,000 draws), preserving the box's
    long/short mix so market drift helps box and null alike.
Test B — TIMING (does the chosen moment beat a random moment the same day?)
    Each trade keeps its date, side and stop distance; nulls move its signal
    to a uniformly random 5-min bar close in the entry window (1,000 draws).

p = (1 + #null means >= observed) / (1 + draws). Gate: p < 0.05 on BOTH.
A box failing this gate is not evaluated further (no DSR / PBO / walk-forward).
Design record: handover doc section 10 + review round 4 (25 Sep 2026).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.execution_core import AccountConfig, ExecutionCore   # noqa: E402
from core.interfaces import Bar, OrderAction                  # noqa: E402

FIVE = timedelta(minutes=5)
CostFn = Callable[[float, float, str], float]     # (entry, stop, session_date) -> cost in R


@dataclass(frozen=True)
class TradeIn:
    """The minimum the gate needs from ANY box's trade log."""
    session_date: str
    direction: str          # "long" | "short"
    signal_ts: datetime     # START of the bar whose close produced the signal
    entry_price: float
    stop_price: float       # initial stop (defines 1R)
    net_r: float | None = None   # the box's own realised net R, for reporting only


@dataclass
class NullReport:
    n_trades: int
    box_expectancy: float | None
    common_exit_expectancy: float
    direction_p: float
    direction_null_mean: float
    direction_null_p95: float
    timing_p: float
    timing_null_mean: float
    timing_null_p95: float
    alpha: float
    passed: bool
    notes: list[str] = field(default_factory=list)


# ── One bracket trade through the real core ───────────────────────────────────

class _Day:
    """A session's 1-min bars as arrays + an index by timestamp."""
    def __init__(self, symbol: str, m1: pd.DataFrame):
        self.symbol = symbol
        self.ts = m1.index
        self.o, self.h, self.l, self.c, self.v = (
            m1[k].to_numpy(float) for k in ("Open", "High", "Low", "Close", "Volume"))

    def bar(self, j: int) -> Bar:
        return Bar(self.symbol, self.ts[j].to_pydatetime(), self.o[j], self.h[j],
                   self.l[j], self.c[j], self.v[j])


def simulate_bracket(day: _Day, signal_ts: pd.Timestamp, side: str, stop_frac: float,
                     session_end: time, target_r: float = 1.0) -> tuple[float, float, float] | None:
    """Gross R of one bracket trade, plus (entry, stop). None if it cannot enter."""
    start = signal_ts + FIVE
    j0 = day.ts.searchsorted(start)
    if j0 >= len(day.ts) or day.ts[j0].time() >= session_end:
        return None
    core = ExecutionCore(AccountConfig(leverage=None), session_close=lambda d: None)
    long = side == "long"
    core.submit([OrderAction("SUBMIT", "e", day.symbol, "BUY" if long else "SELL",
                             "MARKET", 1.0, tag="entry")], start.to_pydatetime())
    entry = stop = None
    for j in range(j0, len(day.ts)):
        b = day.bar(j)
        if entry is not None and day.ts[j].time() >= session_end:
            core.submit([OrderAction("SUBMIT", "x", day.symbol, "SELL" if long else "BUY",
                                     "MARKET", 1.0, oco_group="b", tag="eod")], b.timestamp)
        pending = core.process_bar(b)
        while pending:
            r = pending.pop(0)
            if r.status != "FILLED":
                continue
            if r.tag == "entry":
                entry = r.fill_price
                stop = entry * (1 - stop_frac) if long else entry * (1 + stop_frac)
                tgt = entry + target_r * (entry - stop) if long else entry - target_r * (stop - entry)
                core.submit([OrderAction("SUBMIT", "s", day.symbol, "SELL" if long else "BUY",
                                         "STOP", 1.0, stop_price=stop, oco_group="b", tag="stop"),
                             OrderAction("SUBMIT", "t", day.symbol, "SELL" if long else "BUY",
                                         "LIMIT", 1.0, price=tgt, oco_group="b", tag="target")],
                            b.timestamp, after_price=entry)
                pending += core.process_bar(b)
            else:
                return ((r.fill_price - entry) / abs(entry - stop) * (1 if long else -1),
                        entry, stop)
    if entry is None:
        return None
    return ((day.c[-1] - entry) / abs(entry - stop) * (1 if long else -1), entry, stop)


# ── The gate ────────────────────────────────────────────────────────────────

def run_null_gate(trades: Iterable[TradeIn], df_1m: pd.DataFrame, symbol: str,
                  session_end: time, cost_fn: CostFn | None = None,
                  window_start: time | None = None, draws: int = 1000,
                  alpha: float = 0.05, seed: int = 0) -> NullReport:
    """df_1m: capitalised OHLCV, tz-aware, the same sessions the box traded."""
    trades = list(trades)
    cost_fn = cost_fn or (lambda e, s, d: 0.0)
    days = {str(d): _Day(symbol, g) for d, g in df_1m.groupby(df_1m.index.date)}
    tz = df_1m.index.tz
    if window_start is None:
        window_start = min(pd.Timestamp(t.signal_ts).time() for t in trades)
    notes: list[str] = []

    def net(res, date_):
        return None if res is None else res[0] - cost_fn(res[1], res[2], date_)

    obs, both, cands = [], [], []
    for t in trades:
        day = days.get(t.session_date)
        sig = pd.Timestamp(t.signal_ts)
        sig = sig.tz_localize(tz) if sig.tzinfo is None else sig.tz_convert(tz)
        frac = abs(t.entry_price - t.stop_price) / t.entry_price
        if day is None or frac <= 0:
            continue
        mine = net(simulate_bracket(day, sig, t.direction, frac, session_end), t.session_date)
        other = net(simulate_bracket(day, sig, "short" if t.direction == "long" else "long",
                                     frac, session_end), t.session_date)
        if mine is None or other is None:
            continue
        # every 5-min bar close in the window, same side and stop distance
        grid = pd.date_range(pd.Timestamp.combine(sig.date(), window_start).tz_localize(tz),
                             pd.Timestamp.combine(sig.date(), session_end).tz_localize(tz) - FIVE * 2,
                             freq="5min")
        alts = [x for x in (net(simulate_bracket(day, g, t.direction, frac, session_end),
                                t.session_date) for g in grid) if x is not None]
        if not alts:
            continue
        obs.append(mine)
        both.append((mine, other, t.direction == "long"))
        cands.append(np.array(alts))
    if not obs:
        raise ValueError("no trades could be simulated")
    if len(obs) < len(trades):
        notes.append(f"{len(trades) - len(obs)} trade(s) skipped (no bars / no entry possible)")

    rng = np.random.default_rng(seed)
    observed = float(np.mean(obs))
    L = np.array([b[0] if b[2] else b[1] for b in both])    # outcome if long
    S = np.array([b[1] if b[2] else b[0] for b in both])    # outcome if short
    is_long = np.array([b[2] for b in both])
    dir_null = np.array([np.where(rng.permutation(is_long), L, S).mean() for _ in range(draws)])
    tim_null = np.array([np.mean([c[rng.integers(len(c))] for c in cands]) for _ in range(draws)])

    p_dir = (1 + int((dir_null >= observed).sum())) / (1 + draws)
    p_tim = (1 + int((tim_null >= observed).sum())) / (1 + draws)
    box_exp = [t.net_r for t in trades if t.net_r is not None]
    return NullReport(
        n_trades=len(obs), box_expectancy=float(np.mean(box_exp)) if box_exp else None,
        common_exit_expectancy=observed,
        direction_p=p_dir, direction_null_mean=float(dir_null.mean()),
        direction_null_p95=float(np.percentile(dir_null, 95)),
        timing_p=p_tim, timing_null_mean=float(tim_null.mean()),
        timing_null_p95=float(np.percentile(tim_null, 95)),
        alpha=alpha, passed=(p_dir < alpha and p_tim < alpha), notes=notes)
