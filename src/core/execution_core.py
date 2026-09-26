"""
core/execution_core.py  (Phase A, P2-123 — slice 1)
===================================================
The one place orders are matched. Backtests feed it 1-minute bars; the paper
adapter (later) feeds it broker events. Strategy boxes only emit OrderActions.

FILL RULES (1-minute bars; each rule pinned by tests/test_execution_core.py)
  MARKET        fills at the open of the first 1-min bar at/after it became active.
  STOP  (sell)  bar.open <= stop -> fill at OPEN (gapped through: never the stop)
                bar.low  <= stop -> fill at the STOP
  STOP  (buy)   mirror: open >= stop -> OPEN; high >= stop -> STOP
  LIMIT (sell)  bar.open >= limit -> fill at OPEN (price improvement); high >= limit -> LIMIT
  LIMIT (buy)   mirror: open <= limit -> OPEN; low <= limit -> LIMIT
  MOC           never matched intrabar; filled by end_session() at the symbol's
                last traded price (the close of its last bar that session), before
                DAY orders expire. OCO siblings are cancelled as for any fill.
  Within a bar, per symbol: market orders, then stops, then limits. With an
  OCO group this makes a minute that touches both stop and target a STOP
  (conservative). When an order fills, its OCO siblings are cancelled at once.
  An order is only eligible on bars starting at/after the time it became active
  (no filling on the bar that produced the signal).

SAME-MINUTE FOLLOW-ON ORDERS (bracket behaviour)
  Orders a box submits in response to a fill during minute m can be passed
  with after_price = that fill price. They are matched against the REST of
  minute m (call process_bar(m) again — already-evaluated orders cannot change
  outcome on the same bar), with after_price standing in for the open:
  a stop placed after a long entry fills at the stop if the minute's low
  reaches it (at after_price if the entry itself was at/through the stop); a
  target fills at the limit if the minute's high reaches it; a follow-on MARKET
  order fills at after_price. Stops still precede targets. This mirrors a
  broker bracket order whose legs go live the moment the entry fills.

LIFECYCLE
  SUBMIT  -> ACCEPTED, or REJECTED (INVALID_ORDER, DUPLICATE_CLIENT_ORDER_ID,
             INSUFFICIENT_BUYING_POWER). A client_order_id can never be reused.
  CANCEL  -> CANCELLED, or REJECTED (UNKNOWN_ORDER).
  end_session() at the exchange-calendar close -> every open DAY order EXPIRED
  (SESSION_CLOSE). Fail-safe: a DAY order still open when a later day's first
  bar arrives is EXPIRED (SESSION_CLOSE_MISSED) before that bar is matched —
  it can never fill on the next morning.

FAIL CLOSED: a malformed bar (NaN, high < low, open/close outside the range)
raises ValueError rather than being matched.

NOT MODELLED (documented, deliberate): partial fills (bars have no depth),
borrow / hard-to-borrow, per-order buying-power reservation for resting orders
(the check is made at submission against current exposure).
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date, datetime, time
from typing import Callable

from core.interfaces import (Bar, CoreView, ExecutionReport, OrderAction, OrderView,
                             PositionView)

log = logging.getLogger("execution_core")

FeeFn = Callable[[str, str, float, float, datetime], float]   # (symbol, side, qty, price, ts) -> $


@dataclass
class AccountConfig:
    cash: float = 10_000.0
    leverage: float | None = 1.0     # None = unlimited (legacy-parity mode only)


@dataclass
class _Order:
    a: OrderAction
    active_from: datetime
    seq: int
    after_price: float | None = None   # same-minute follow-on (see module doc)


@dataclass
class _Position:
    qty: float = 0.0
    avg: float = 0.0
    realized: float = 0.0


def _default_session_close(d: date) -> time | None:
    from market_data_store import session_close
    return session_close(d)


class ExecutionCore:
    def __init__(self, account: AccountConfig | None = None, fee_fn: FeeFn | None = None,
                 session_close: Callable[[date], time | None] | None = None):
        self.account = account or AccountConfig()
        self._fee_fn = fee_fn or (lambda *_: 0.0)
        self._session_close = session_close or _default_session_close
        self._cash = float(self.account.cash)
        self._orders: dict[str, _Order] = {}
        self._seen: set[str] = set()
        self._pos: dict[str, _Position] = {}
        self._last: dict[str, float] = {}
        self._seq = 0
        self._day: date | None = None
        self._now: datetime | None = None

    # ── Views ────────────────────────────────────────────────────────────────

    def equity(self) -> float:
        return self._cash + sum(p.qty * self._last.get(s, p.avg) for s, p in self._pos.items())

    def gross_exposure(self) -> float:
        return sum(abs(p.qty) * self._last.get(s, p.avg) for s, p in self._pos.items())

    def buying_power(self) -> float:
        if self.account.leverage is None:
            return math.inf
        return max(0.0, self.equity() * self.account.leverage - self.gross_exposure())

    def view(self) -> CoreView:
        return CoreView(
            timestamp=self._now or datetime.min, cash=self._cash, equity=self.equity(),
            buying_power=self.buying_power(),
            positions={s: PositionView(s, p.qty, p.avg, p.realized)
                       for s, p in self._pos.items() if p.qty or p.realized},
            open_orders=tuple(OrderView(o.a.client_order_id, o.a.symbol, o.a.side,
                                        o.a.order_type, o.a.qty, o.a.price, o.a.stop_price,
                                        o.a.oco_group, o.a.tag)
                              for o in sorted(self._orders.values(), key=lambda o: o.seq)))

    def position(self, symbol: str) -> float:
        return self._pos.get(symbol, _Position()).qty

    # ── Commands ─────────────────────────────────────────────────────────────

    def submit(self, actions: list[OrderAction], now: datetime,
               after_price: float | None = None) -> list[ExecutionReport]:
        """Accept/reject actions. `now` = when they become active (e.g. the END
        of the 5-min bar the box just saw); they can fill on 1-min bars
        starting at or after it. after_price: see SAME-MINUTE FOLLOW-ON ORDERS
        (then `now` is the start of the minute in which the triggering fill
        happened)."""
        self._now = now
        out: list[ExecutionReport] = []
        for a in actions:
            if a.action_type == "CANCEL":
                o = self._orders.pop(a.client_order_id, None)
                out.append(self._report(a, "CANCELLED" if o else "REJECTED", now,
                                        reason="USER" if o else "UNKNOWN_ORDER",
                                        leaves=o.a.qty if o else 0.0))
                continue
            why = self._invalid(a)
            if why is None and a.client_order_id in self._seen:
                why = "DUPLICATE_CLIENT_ORDER_ID"
            if why is None and self._exceeds_buying_power(a):
                why = "INSUFFICIENT_BUYING_POWER"
            if why:
                if why != "DUPLICATE_CLIENT_ORDER_ID":
                    self._seen.add(a.client_order_id)
                out.append(self._report(a, "REJECTED", now, reason=why))
                continue
            self._seen.add(a.client_order_id)
            self._seq += 1
            self._orders[a.client_order_id] = _Order(a, now, self._seq, after_price)
            out.append(self._report(a, "ACCEPTED", now, leaves=a.qty))
        return out

    def process_bar(self, bar: Bar) -> list[ExecutionReport]:
        """Match open orders for bar.symbol against one 1-minute bar."""
        self._check_bar(bar)
        out: list[ExecutionReport] = []
        d = bar.timestamp.date()
        if self._day is not None and d > self._day:
            out += self._expire_day_orders(bar.timestamp, "SESSION_CLOSE_MISSED")
        self._day = d
        self._now = bar.timestamp

        mine = sorted((o for o in self._orders.values()
                       if o.a.symbol == bar.symbol and bar.timestamp >= o.active_from
                       and o.a.order_type != "MOC"),
                      key=lambda o: ({"MARKET": 0, "STOP": 1, "LIMIT": 2}[o.a.order_type], o.seq))
        for o in mine:
            if o.a.client_order_id not in self._orders:      # cancelled by an OCO sibling
                continue
            eff_open = (o.after_price if o.after_price is not None
                        and bar.timestamp == o.active_from else bar.open)
            px = self._fill_price(o.a, bar, eff_open)
            if px is None:
                continue
            out += self._fill(o, px, bar.timestamp)
        self._last[bar.symbol] = bar.close
        return out

    def end_session(self, now: datetime) -> list[ExecutionReport]:
        """Call at the exchange-calendar close: fill market-on-close orders at
        each symbol's last traded price, then expire every open DAY order."""
        self._now = now
        out: list[ExecutionReport] = []
        for o in sorted([o for o in self._orders.values() if o.a.order_type == "MOC"],
                        key=lambda o: o.seq):
            if o.a.client_order_id not in self._orders:      # cancelled by an OCO sibling
                continue
            px = self._last.get(o.a.symbol)
            if px is None:
                continue                                      # never traded: expires below
            out += self._fill(o, px, now)
        return out + self._expire_day_orders(now, "SESSION_CLOSE")

    def session_close_for(self, d: date) -> time | None:
        return self._session_close(d)

    # ── Internals ────────────────────────────────────────────────────────────

    @staticmethod
    def _invalid(a: OrderAction) -> str | None:
        if a.side not in ("BUY", "SELL") or a.order_type not in ("MARKET", "LIMIT", "STOP", "MOC"):
            return "INVALID_ORDER"
        if not (a.qty > 0 and math.isfinite(a.qty)):
            return "INVALID_ORDER"
        if a.order_type == "LIMIT" and not (a.price and a.price > 0):
            return "INVALID_ORDER"
        if a.order_type == "STOP" and not (a.stop_price and a.stop_price > 0):
            return "INVALID_ORDER"
        return None

    def _exceeds_buying_power(self, a: OrderAction) -> bool:
        if self.account.leverage is None:
            return False
        cur = self.position(a.symbol)
        signed = a.qty if a.side == "BUY" else -a.qty
        if abs(cur + signed) <= abs(cur) + 1e-12:            # reduces or closes: always allowed
            return False
        ref = (a.price if a.order_type == "LIMIT" else
               a.stop_price if a.order_type == "STOP" else self._last.get(a.symbol))   # MARKET / MOC
        if ref is None:
            return False                                      # no price yet: nothing to check against
        added = (abs(cur + signed) - abs(cur)) * ref
        return added > self.buying_power() + 1e-9

    @staticmethod
    def _fill_price(a: OrderAction, b: Bar, o: float) -> float | None:
        """o = the effective open: the bar's open, or after_price for a
        same-minute follow-on order."""
        if a.order_type == "MARKET":
            return o
        if a.order_type == "STOP":
            s = a.stop_price
            if a.side == "SELL":
                return o if o <= s else (s if b.low <= s else None)
            return o if o >= s else (s if b.high >= s else None)
        lim = a.price
        if a.side == "SELL":
            return o if o >= lim else (lim if b.high >= lim else None)
        return o if o <= lim else (lim if b.low <= lim else None)

    def _fill(self, o: _Order, px: float, ts: datetime) -> list[ExecutionReport]:
        a = o.a
        del self._orders[a.client_order_id]
        fee = float(self._fee_fn(a.symbol, a.side, a.qty, px, ts))
        self._apply(a.symbol, a.qty if a.side == "BUY" else -a.qty, px, fee)
        out = [self._report(a, "FILLED", ts, px=px, qty=a.qty, fee=fee)]
        if a.oco_group:
            for sib in [x for x in self._orders.values()
                        if x.a.oco_group == a.oco_group and x.a.symbol == a.symbol]:
                del self._orders[sib.a.client_order_id]
                out.append(self._report(sib.a, "CANCELLED", ts, reason="OCO", leaves=sib.a.qty))
        return out

    def _apply(self, sym: str, signed: float, px: float, fee: float) -> None:
        p = self._pos.setdefault(sym, _Position())
        if p.qty == 0 or (p.qty > 0) == (signed > 0):
            new = p.qty + signed
            p.avg = (abs(p.qty) * p.avg + abs(signed) * px) / abs(new)
            p.qty = new
        else:
            closed = min(abs(p.qty), abs(signed))
            p.realized += closed * (px - p.avg) * (1 if p.qty > 0 else -1)
            new = p.qty + signed
            if abs(new) < 1e-12:
                p.qty, p.avg = 0.0, 0.0
            elif (new > 0) != (p.qty > 0):                    # flipped through zero
                p.qty, p.avg = new, px
            else:
                p.qty = new
        p.realized -= fee
        self._cash -= signed * px + fee
        self._last.setdefault(sym, px)

    def _expire_day_orders(self, ts: datetime, reason: str) -> list[ExecutionReport]:
        out = []
        for o in sorted([o for o in self._orders.values() if o.a.time_in_force == "DAY"],
                        key=lambda o: o.seq):
            del self._orders[o.a.client_order_id]
            out.append(self._report(o.a, "EXPIRED", ts, reason=reason, leaves=o.a.qty))
        if out and reason == "SESSION_CLOSE_MISSED":
            log.warning("%d DAY order(s) expired late (end_session not called)", len(out))
        return out

    @staticmethod
    def _check_bar(b: Bar) -> None:
        vals = (b.open, b.high, b.low, b.close)
        if not all(math.isfinite(v) for v in vals) or b.high < b.low or \
                not (b.low <= b.open <= b.high) or not (b.low <= b.close <= b.high):
            raise ValueError(f"Malformed bar refused (fail closed): {b}")

    @staticmethod
    def _report(a: OrderAction, status: str, ts: datetime, *, px: float = 0.0, qty: float = 0.0,
                fee: float = 0.0, reason: str = "", leaves: float = 0.0) -> ExecutionReport:
        return ExecutionReport(client_order_id=a.client_order_id, symbol=a.symbol, status=status,
                               timestamp=ts, fill_price=px, fill_qty=qty, leaves_qty=leaves,
                               fee=fee, reason=reason, tag=a.tag)
