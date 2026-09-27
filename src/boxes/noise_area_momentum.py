"""
boxes/noise_area_momentum.py — box #3  (PR-003, Notion P2-130)
==============================================================
"Beat the Market: An Effective Intraday Momentum Strategy for S&P500 ETF (SPY)",
Zarattini, Aziz & Barbon (SSRN 4824172, version 3 Feb 2025), section 3, as
published — cross-checked against the authors' own Python reference code.

On day t, for each time-of-day HH:MM:
  move(t-i, HH:MM) = |Close(t-i, HH:MM) / Open(t-i, 09:30) - 1|,  i = 1..lookback
  sigma(t, HH:MM)  = mean of those moves
  Upper = max(Open_t, PrevClose_adj) x (1 + vm x sigma)
  Lower = min(Open_t, PrevClose_adj) x (1 - vm x sigma)
  PrevClose_adj = yesterday's close minus any dividend going ex today.
At each half-hour (HH:00 / HH:30; first 10:00) the price is the close of the
minute ending then (the bar labelled HH:MM-1):
  long  if price > Upper and price > VWAP
  short if price < Lower and price < VWAP
  flat  otherwise
Exposure is set to the signal at each check (a zero exits; a flip reverses),
executed as a market order (the core fills it at the next minute's open).
Anything still open is closed at the session close (market-on-close).

Sizing, fixed for the day at the open:
  "vol_target" (headline): shares = AUM(t-1) x min(max_lev, target / sigma_SPY) / Open_t,
      sigma_SPY = sample st. dev. of the last `lookback` daily close-to-close returns
  "full"      (1x variant): shares = AUM(t-1) / Open_t
Pure: new state + orders only. Daily history lives in frozen tuples.
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field, replace
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Mapping

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.interfaces import Bar, CoreView, ExecutionReport, OrderAction   # noqa: E402

OPEN = time(9, 30)
ONE = timedelta(minutes=1)


@dataclass(frozen=True)
class Params:
    lookback: int = 14
    vm: float = 1.0                      # volatility multiplier (paper: 1)
    trade_every_min: int = 30            # HH:00 / HH:30
    sizing: str = "vol_target"           # "vol_target" | "full"
    target_vol: float = 0.02             # daily
    max_leverage: float = 4.0
    dividends: Mapping[str, float] = field(default_factory=dict)   # ex-date -> $/share


@dataclass(frozen=True)
class State:
    day: date | None = None
    open_px: float | None = None
    prev_close: float | None = None      # last close of the previous session (raw)
    last_close: float | None = None      # latest close seen today
    ref_price: float | None = None       # max/min base: set at the open
    lo_base: float | None = None
    shares: int = 0
    cum_pv: float = 0.0
    cum_v: float = 0.0
    exposure: int = 0                    # -1 / 0 / +1 (target most recently ordered)
    today_moves: tuple = ()              # ((minute_index, |move|), ...) at check minutes
    hist_moves: tuple = ()               # last `lookback` days of dict-like tuples
    closes: tuple = ()                   # recent daily closes (for sigma_SPY)
    n: int = 0                           # order counter
    moc_id: str | None = None


def _minute_index(ts: datetime) -> int:
    """Minutes from 09:30 counting the bar that ENDS at ts+1 (bar 09:59 -> 30)."""
    return (ts.hour * 60 + ts.minute) - (9 * 60 + 30) + 1


class NoiseAreaMomentumBox:
    name = "noise_area_momentum"
    version = "1.0.0"
    timeframe_minutes = 1

    def init_state(self, params: Params) -> State:
        return State()

    # ── helpers ───────────────────────────────────────────────────────────────

    def _sigma(self, s: State, p: Params, idx: int) -> float | None:
        vals = [dict(h).get(idx) for h in s.hist_moves[-p.lookback:]]
        vals = [v for v in vals if v is not None]
        if len(s.hist_moves) < p.lookback or len(vals) < p.lookback - 1:
            return None
        return sum(vals) / len(vals)

    def _sigma_spy(self, s: State, p: Params) -> float | None:
        c = s.closes
        if len(c) < p.lookback + 1:
            return None
        rets = [c[i] / c[i - 1] - 1 for i in range(len(c) - p.lookback, len(c))]
        m = sum(rets) / len(rets)
        return math.sqrt(sum((r - m) ** 2 for r in rets) / (len(rets) - 1))

    def _roll_day(self, s: State, p: Params, d: date) -> State:
        """Archive yesterday (moves + close) and reset the intraday fields."""
        hist, closes = s.hist_moves, s.closes
        if s.day is not None and s.last_close is not None:
            hist = (hist + (s.today_moves,))[-p.lookback:]
            closes = (closes + (s.last_close,))[-(p.lookback + 1):]
        return State(day=d, prev_close=s.last_close if s.day is not None else s.prev_close,
                     hist_moves=hist, closes=closes, n=s.n)

    def _orders_to(self, s: State, target: int, sym: str, now_day: date) -> tuple[State, list]:
        acts: list[OrderAction] = []
        delta = (target - s.exposure) * s.shares
        n = s.n
        if s.moc_id:
            acts.append(OrderAction("CANCEL", s.moc_id, sym))
        if delta:
            n += 1
            acts.append(OrderAction("SUBMIT", f"{now_day}-{n}-mkt", sym, "BUY" if delta > 0 else "SELL",
                                    "MARKET", abs(delta), tag="trade"))
        moc = None
        if target:
            n += 1
            moc = f"{now_day}-{n}-moc"
            acts.append(OrderAction("SUBMIT", moc, sym, "SELL" if target > 0 else "BUY", "MOC",
                                    s.shares, tag="close"))
        return replace(s, exposure=target, n=n, moc_id=moc), acts

    # ── contract ──────────────────────────────────────────────────────────────

    def on_bar(self, bar: Bar, s: State, p: Params, view: CoreView) -> tuple[State, list[OrderAction]]:
        d = bar.timestamp.date()
        if s.day != d:
            s = self._roll_day(s, p, d)
        t = bar.timestamp.time()
        if s.open_px is None:
            if t < OPEN:
                return s, []
            prev = s.prev_close
            div = p.dividends.get(str(d), 0.0)
            prev_adj = prev - div if prev is not None else None
            hi = max(bar.open, prev_adj) if prev_adj is not None else bar.open
            lo = min(bar.open, prev_adj) if prev_adj is not None else bar.open
            shares = 0
            sig_spy = self._sigma_spy(s, p)
            if p.sizing == "full":
                shares = int(round(view.equity / bar.open))
            elif sig_spy and sig_spy > 0:
                shares = int(round(view.equity * min(p.max_leverage, p.target_vol / sig_spy) / bar.open))
            s = replace(s, open_px=bar.open, ref_price=hi, lo_base=lo, shares=shares)
        # running VWAP (market hours, HLC3 x volume)
        s = replace(s, cum_pv=s.cum_pv + (bar.high + bar.low + bar.close) / 3 * bar.volume,
                    cum_v=s.cum_v + bar.volume, last_close=bar.close)
        idx = _minute_index(bar.timestamp)
        if idx % p.trade_every_min != 0:
            return s, []
        # a check minute: record today's move for future sigma, then decide
        s = replace(s, today_moves=s.today_moves + ((idx, abs(bar.close / s.open_px - 1)),))
        sigma = self._sigma(s, p, idx)
        check_time = (bar.timestamp + ONE).time()
        if sigma is None or s.shares <= 0 or s.cum_v <= 0 or check_time >= time(16, 0):
            return s, []
        upper = s.ref_price * (1 + p.vm * sigma)
        lower = s.lo_base * (1 - p.vm * sigma)
        vwap = s.cum_pv / s.cum_v
        px = bar.close
        target = 1 if (px > upper and px > vwap) else -1 if (px < lower and px < vwap) else 0
        if target == s.exposure:
            return s, []
        return self._orders_to(s, target, bar.symbol, d)

    def on_execution(self, r: ExecutionReport, s: State, p: Params,
                     view: CoreView) -> tuple[State, list[OrderAction]]:
        if r.tag == "close" and r.status == "FILLED":
            return replace(s, exposure=0, moc_id=None), []
        return s, []
