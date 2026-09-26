"""
boxes/orb_stocks_in_play.py — box #2  (PR-002, Notion P2-127)
=============================================================
Zarattini, Barbon & Aziz (2024), "A Profitable Day Trading Strategy For The
U.S. Equity Market", sections 2.1 and 4 — verified against the paper itself.
The stock selection (top 20 by opening relative volume) is done upstream by
universe_select; this box trades ONE selected stock-day:

  * direction = colour of the 09:30-09:35 candle: close > open -> long only,
    close < open -> short only, doji -> no trade
  * entry: STOP order at that candle's high (long) / low (short), live from
    09:35 to the close (the execution core fills a gap-through at the open)
  * stop loss: stop_atr_frac x ATR14 from the EXECUTED entry price, live from
    the entry minute (same-minute follow-on; conservative)
  * no target; exit at the session close (market-on-close order, OCO with the
    stop, so exactly one of them fills)
  * one trade per stock per day

ATR14 comes from the universe (prior 14 sessions, known at the open) through
Params.atr14 {session_date: ATR}. Pure: returns new state + orders; completed
trades go to the state's journal.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field, replace
from datetime import date, datetime, time
from pathlib import Path
from typing import Mapping

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.interfaces import Bar, CoreView, ExecutionReport, OrderAction   # noqa: E402


@dataclass(frozen=True)
class Params:
    atr14: Mapping[str, float] = field(default_factory=dict)   # session_date -> ATR14
    stop_atr_frac: float = 0.10
    opening_range_start: time = time(9, 30)
    account: float = 25_000.0
    risk_pct: float = 0.01
    # DIAGNOSTIC ONLY (PR-002 'optimistic bound'): 1-min bars cannot show whether
    # the entry or the stop came first inside the entry minute. False = assume the
    # worst (the frozen rule). True = stop first live on the NEXT minute.
    stop_from_next_minute: bool = False


@dataclass(frozen=True)
class TradeRecord:
    session_date: str
    symbol: str
    direction: str
    or_open: float
    or_high: float
    or_low: float
    or_close: float
    atr14: float
    entry_order_price: float
    entry_fill_ts: datetime
    entry_price: float
    stop_price: float
    exit_ts: datetime
    exit_price: float
    exit_reason: str          # stop | close


@dataclass(frozen=True)
class State:
    day: date | None = None
    phase: str = "idle"       # idle | pending | in_position | done
    symbol: str = ""
    direction: str | None = None
    or_bar: tuple | None = None
    atr: float | None = None
    entry_order_price: float | None = None
    entry_price: float | None = None
    entry_fill_ts: datetime | None = None
    stop: float | None = None
    qty: float = 0.0
    journal: tuple = ()


class OrbStocksInPlayBox:
    name = "orb_stocks_in_play"
    version = "1.0.0"
    timeframe_minutes = 5

    def init_state(self, params: Params) -> State:
        return State()

    def on_bar(self, bar: Bar, s: State, p: Params, view: CoreView) -> tuple[State, list[OrderAction]]:
        d = bar.timestamp.date()
        if s.day != d:
            s = State(day=d, journal=s.journal)
        if s.phase != "idle" or bar.timestamp.time() != p.opening_range_start:
            return s, []
        s = replace(s, phase="done", symbol=bar.symbol,
                    or_bar=(bar.open, bar.high, bar.low, bar.close))
        atr = p.atr14.get(str(d))
        if not atr or atr <= 0 or bar.close == bar.open:          # no ATR, or a doji
            return s, []
        long = bar.close > bar.open
        level = bar.high if long else bar.low
        risk_per_share = p.stop_atr_frac * atr
        qty = p.account * p.risk_pct / risk_per_share
        order = OrderAction("SUBMIT", f"{d}-{bar.symbol}-entry", bar.symbol,
                            "BUY" if long else "SELL", "STOP", qty, stop_price=level, tag="entry")
        return replace(s, phase="pending", direction="long" if long else "short", atr=atr,
                       entry_order_price=level), [order]

    def on_execution(self, r: ExecutionReport, s: State, p: Params,
                     view: CoreView) -> tuple[State, list[OrderAction]]:
        if r.tag == "entry":
            if r.status == "FILLED":
                long = s.direction == "long"
                stop = r.fill_price - p.stop_atr_frac * s.atr if long else \
                    r.fill_price + p.stop_atr_frac * s.atr
                side = "SELL" if long else "BUY"
                grp = f"{s.day}-{r.symbol}"
                acts = [OrderAction("SUBMIT", f"{grp}-stop", r.symbol, side, "STOP", r.fill_qty,
                                    stop_price=stop, oco_group=grp, tag="stop",
                                    live_from_next_bar=p.stop_from_next_minute),
                        OrderAction("SUBMIT", f"{grp}-moc", r.symbol, side, "MOC", r.fill_qty,
                                    oco_group=grp, tag="close")]
                return replace(s, phase="in_position", entry_price=r.fill_price,
                               entry_fill_ts=r.timestamp, stop=stop, qty=r.fill_qty), acts
            if r.status in ("EXPIRED", "CANCELLED", "REJECTED"):
                return replace(s, phase="done"), []
            return s, []
        if r.status == "FILLED" and r.tag in ("stop", "close") and s.phase == "in_position":
            o, h, l, c = s.or_bar
            rec = TradeRecord(str(s.day), r.symbol, s.direction, o, h, l, c, s.atr,
                              s.entry_order_price, s.entry_fill_ts, s.entry_price, s.stop,
                              r.timestamp, r.fill_price, r.tag)
            return replace(s, phase="done", journal=s.journal + (rec,)), []
        return s, []
