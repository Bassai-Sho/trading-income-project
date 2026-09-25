"""
boxes/orb_vwap_legacy.py — box #1  (Phase A, P2-123 — slice 2)
===============================================================
The PR-001 strategy (shelved: no edge under honest fills) rebuilt as a
strategy box. Its only job is to prove the modular path reproduces the legacy
resting-fill backtest trade for trade (tests/fixtures/golden_orb_resting_trades).

Strategy (unchanged from trading_engine._backtest_orb_full_gate):
  * opening range: engine's _orb_range(method) from session_start
  * entry signal on a 5-min close beyond the range with the VWAP slope agreeing
    (engine's _vwap / _vwap_slope); market order, filled at the next minute's open
  * stop = range low - buffer (long) / range high + buffer (short),
    buffer = max(range x buffer_frac, buffer_floor)
  * once filled: resting stop + target (entry +/- target_mult x risk, measured
    from the FILL) as one OCO bracket, live from the fill minute
  * VWAP trailing (engine's _vwap_trailing) on each 5-min close
    (trail_after_1r: only after a 5-min close beyond 1R; fixed_1_5r: no trail)
  * flat at session_end: market exit at the open of that minute
  * one position at a time; no new signal on a 5-min bar in which a trade exited;
    no entry from a signal bar at/after session_end

Pure: on_bar / on_execution return a NEW state and a list of OrderActions. The
box never fills orders or computes costs; completed trades are appended to the
state's journal for the harness to read.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field, replace
from datetime import date, datetime, time, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.interfaces import Bar, CoreView, ExecutionReport, OrderAction   # noqa: E402
from trading_engine import _orb_range, _vwap, _vwap_slope, _vwap_trailing  # noqa: E402

FIVE = timedelta(minutes=5)


@dataclass(frozen=True)
class Params:
    orb_method: str = "15min"
    vwap_lookback: int = 3
    session_start: time = time(9, 30)
    session_end: time = time(11, 0)
    exit_mode: str = "baseline"          # baseline | fixed_1_5r | trail_after_1r
    buffer_frac: float = 0.1
    buffer_floor: float = 0.02           # absolute $ (legacy); a PR-002 box should not do this
    account: float = 10_000.0
    risk_pct: float = 0.01

    @property
    def target_mult(self) -> float:
        return 1.5 if self.exit_mode == "fixed_1_5r" else 1.0


@dataclass(frozen=True)
class TradeRecord:
    session_date: str
    direction: str
    signal_ts: datetime
    signal_close: float
    entry_fill_ts: datetime
    entry_price: float
    stop_price: float                    # initial stop (defines 1R)
    exit_ts: datetime
    exit_price: float
    exit_reason: str                     # trailing_stop | target_hit | eod | invalid_entry


@dataclass(frozen=True)
class State:
    day: date | None = None
    bars: tuple = ()                     # today's 5-min Bars
    phase: str = "flat"                  # flat | entry_pending | in_position
    n: int = 0                           # trade counter (order ids)
    direction: str | None = None
    signal_ts: datetime | None = None
    signal_close: float | None = None
    stop: float | None = None
    initial_stop: float | None = None
    target: float | None = None
    entry_price: float | None = None
    entry_fill_ts: datetime | None = None
    qty: float = 0.0                     # filled shares (from the entry fill report)
    stop_id: str | None = None
    stop_seq: int = 0
    trail_armed: bool = False
    last_exit_ts: datetime | None = None
    journal: tuple = ()


class OrbVwapLegacyBox:
    name = "orb_vwap_legacy"
    version = "1.0.0"
    timeframe_minutes = 5

    # ── contract ──────────────────────────────────────────────────────────────

    def init_state(self, params: Params) -> State:
        return State()

    def on_bar(self, bar: Bar, s: State, p: Params,
               view: CoreView) -> tuple[State, list[OrderAction]]:
        d = bar.timestamp.date()
        if s.day != d:
            s = replace(State(n=s.n, journal=s.journal), day=d)
        s = replace(s, bars=s.bars + (bar,))
        t = bar.timestamp.time()

        if s.phase == "in_position":
            if (bar.timestamp + FIVE).time() >= p.session_end:
                return s, [self._market_exit(s, "eod")]
            return self._trail(s, bar, p)

        if s.phase != "flat" or t < p.session_start or t >= p.session_end:
            return s, []
        if s.last_exit_ts is not None and bar.timestamp <= s.last_exit_ts < bar.timestamp + FIVE:
            return s, []                                  # legacy: this bar was an in-position bar
        return self._signal(s, bar, p)

    def on_execution(self, r: ExecutionReport, s: State, p: Params,
                     view: CoreView) -> tuple[State, list[OrderAction]]:
        kind = r.tag
        if kind == "entry":
            if r.status == "FILLED":
                return self._on_entry_fill(r, s, p)
            if r.status in ("REJECTED", "EXPIRED", "CANCELLED"):
                return replace(s, phase="flat", direction=None), []
            return s, []
        if r.status == "FILLED" and kind in ("stop", "target", "eod", "invalid_entry"):
            reason = {"stop": "trailing_stop", "target": "target_hit"}.get(kind, kind)
            rec = TradeRecord(str(s.day), s.direction, s.signal_ts, s.signal_close,
                              s.entry_fill_ts, s.entry_price, s.initial_stop,
                              r.timestamp, r.fill_price, reason)
            return replace(s, phase="flat", direction=None, stop=None, target=None,
                           stop_id=None, qty=0.0, last_exit_ts=r.timestamp,
                           journal=s.journal + (rec,)), []
        return s, []

    # ── internals ─────────────────────────────────────────────────────────────

    def _frame(self, s: State) -> pd.DataFrame:
        b = s.bars
        return pd.DataFrame({"Open": [x.open for x in b], "High": [x.high for x in b],
                             "Low": [x.low for x in b], "Close": [x.close for x in b],
                             "Volume": [x.volume for x in b]},
                            index=pd.DatetimeIndex([x.timestamp for x in b]))

    def _signal(self, s: State, bar: Bar, p: Params) -> tuple[State, list[OrderAction]]:
        df = self._frame(s)
        orb = _orb_range(df, p.orb_method, p.session_start)
        if orb is None:
            return s, []
        close = bar.close
        if close > orb["orb_high"]:
            direction = "long"
        elif close < orb["orb_low"]:
            direction = "short"
        else:
            return s, []
        vs = _vwap_slope(_vwap(df, p.session_start), lookback=p.vwap_lookback)
        if not ((direction == "long" and vs["direction"] == "up") or
                (direction == "short" and vs["direction"] == "down")):
            return s, []
        buffer = max((orb["orb_high"] - orb["orb_low"]) * p.buffer_frac, p.buffer_floor)
        stop = (round(orb["orb_low"] - buffer, 4) if direction == "long"
                else round(orb["orb_high"] + buffer, 4))
        risk = abs(close - stop)
        if risk <= 1e-6:
            return s, []
        qty = p.account * p.risk_pct / risk               # sized on the signal close
        n = s.n + 1
        act = OrderAction("SUBMIT", f"{s.day}-{n}-entry", bar.symbol,
                          "BUY" if direction == "long" else "SELL", "MARKET", qty,
                          oco_group="", tag="entry")
        return replace(s, phase="entry_pending", n=n, direction=direction,
                       signal_ts=bar.timestamp, signal_close=close, stop=stop,
                       initial_stop=stop, stop_seq=0), [act]

    def _on_entry_fill(self, r: ExecutionReport, s: State,
                       p: Params) -> tuple[State, list[OrderAction]]:
        fill, long = r.fill_price, s.direction == "long"
        s = replace(s, phase="in_position", entry_price=fill, entry_fill_ts=r.timestamp,
                    qty=r.fill_qty)
        if (long and fill <= s.stop) or (not long and fill >= s.stop):
            return s, [self._market_exit(s, "invalid_entry", qty=r.fill_qty)]
        if r.timestamp.time() >= p.session_end:
            return s, [self._market_exit(s, "eod", qty=r.fill_qty)]
        risk = abs(fill - s.stop)
        target = fill + risk * p.target_mult if long else fill - risk * p.target_mult
        armed = p.exit_mode != "trail_after_1r"
        s = replace(s, target=target, trail_armed=armed)
        acts = [self._stop_order(s, s.stop, r.fill_qty)]
        s = replace(s, stop_id=acts[0].client_order_id, stop_seq=1)
        if p.exit_mode in ("baseline", "fixed_1_5r"):
            acts.append(OrderAction("SUBMIT", f"{s.day}-{s.n}-target", r.symbol,
                                    "SELL" if long else "BUY", "LIMIT", r.fill_qty,
                                    price=target, oco_group=self._oco(s), tag="target"))
        return s, acts

    def _trail(self, s: State, bar: Bar, p: Params) -> tuple[State, list[OrderAction]]:
        long = s.direction == "long"
        if p.exit_mode == "trail_after_1r" and not s.trail_armed:
            if (long and bar.close >= s.target) or (not long and bar.close <= s.target):
                s = replace(s, trail_armed=True)
        if p.exit_mode == "fixed_1_5r" or not s.trail_armed:
            return s, []
        v = _vwap(self._frame(s), p.session_start).dropna()
        if v.empty:
            return s, []
        new = _vwap_trailing(s.stop, float(v.iloc[-1]), s.direction)
        if new == s.stop:
            return s, []
        qty = s.qty
        cancel = OrderAction("CANCEL", s.stop_id, bar.symbol)
        s = replace(s, stop=new, stop_seq=s.stop_seq + 1)
        stop = self._stop_order(s, new, qty)
        return replace(s, stop_id=stop.client_order_id), [cancel, stop]

    def _stop_order(self, s: State, price: float, qty: float) -> OrderAction:
        long = s.direction == "long"
        return OrderAction("SUBMIT", f"{s.day}-{s.n}-stop{s.stop_seq}", self._sym(s),
                           "SELL" if long else "BUY", "STOP", qty, stop_price=price,
                           oco_group=self._oco(s), tag="stop")

    def _market_exit(self, s: State, tag: str, qty: float | None = None) -> OrderAction:
        long = s.direction == "long"
        return OrderAction("SUBMIT", f"{s.day}-{s.n}-{tag}", self._sym(s),
                           "SELL" if long else "BUY", "MARKET",
                           qty if qty is not None else s.qty,
                           oco_group=self._oco(s), tag=tag)

    def _oco(self, s: State) -> str:
        return f"{s.day}-{s.n}"

    def _sym(self, s: State) -> str:
        return s.bars[-1].symbol if s.bars else ""
