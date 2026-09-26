"""
core/interfaces.py  (Phase A, P2-123 — slice 1)
===============================================
Contracts between strategy boxes and the execution core.

Design record: Claude Doc "Modular Strategy Architecture — Review Handover",
section 10 (three adversarial review rounds, 25 Sep 2026).

  * A box is a pure function: (state, bar, view) -> (new state, [OrderAction]),
    plus on_execution(report) for fills / cancels / rejects / expiries.
  * A box never fills orders, computes costs, or sees future bars.
  * The execution core is the only place orders are matched, and the SAME core
    runs backtests and paper trading, so they cannot drift apart.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Protocol, TypeVar

OrderStatus = Literal[
    "ACCEPTED",          # resting / queued in the core (or acknowledged by the broker)
    "PENDING_SUBMIT",    # live adapter only: sent to the broker, not yet acknowledged
    "PARTIALLY_FILLED",  # live adapter only: bars carry no depth, so the backtest fills in full
    "FILLED",
    "CANCELLED",
    "EXPIRED",           # DAY order still open at the session close
    "REJECTED",
]
Side = Literal["BUY", "SELL"]
OrderType = Literal["MARKET", "LIMIT", "STOP", "MOC"]   # MOC = market on close
TimeInForce = Literal["DAY", "GTC"]


@dataclass(frozen=True, slots=True)
class Bar:
    """One closed bar. `timestamp` is the bar's START (09:30 covers 09:30-09:30:59
    for 1-min bars), matching the data store and the resampler."""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True, slots=True)
class OrderAction:
    """A command from a box. SUBMIT needs side/order_type/qty (+ price for LIMIT,
    stop_price for STOP). CANCEL needs only client_order_id and symbol.

    oco_group: orders sharing a non-empty group are one-cancels-other — when one
    fills, the core cancels the rest in the same minute (a stop and a target can
    never both fill). Mirrors Alpaca's native OCO / bracket legs.
    """
    action_type: Literal["SUBMIT", "CANCEL"]
    client_order_id: str
    symbol: str
    side: Side | None = None
    order_type: OrderType | None = None
    qty: float = 0.0                 # fractional allowed (legacy parity); live boxes use whole shares
    price: float | None = None       # LIMIT price
    stop_price: float | None = None  # STOP trigger
    time_in_force: TimeInForce = "DAY"
    oco_group: str = ""
    tag: str = ""                    # attribution, e.g. "entry", "stop", "target", "eod"


@dataclass(frozen=True, slots=True)
class ExecutionReport:
    client_order_id: str
    symbol: str
    status: OrderStatus
    timestamp: datetime
    fill_price: float = 0.0
    fill_qty: float = 0.0
    leaves_qty: float = 0.0
    fee: float = 0.0
    reason: str = ""                 # e.g. INSUFFICIENT_BUYING_POWER, SESSION_CLOSE, OCO
    tag: str = ""


@dataclass(frozen=True, slots=True)
class PositionView:
    symbol: str
    qty: float                       # + long, - short
    avg_price: float
    realized_pnl: float


@dataclass(frozen=True, slots=True)
class OrderView:
    """A live (accepted, unfilled) order, as the box sees it."""
    client_order_id: str
    symbol: str
    side: Side
    order_type: OrderType
    qty: float
    price: float | None
    stop_price: float | None
    oco_group: str
    tag: str


@dataclass(frozen=True, slots=True)
class CoreView:
    """Read-only snapshot a box receives with each bar."""
    timestamp: datetime
    cash: float
    equity: float
    buying_power: float
    positions: dict[str, PositionView] = field(default_factory=dict)
    open_orders: tuple[OrderView, ...] = ()


# ── Strategy box contract (implemented from slice 2) ───────────────────────────

S = TypeVar("S")
P = TypeVar("P")


class StrategyBox(Protocol[S, P]):
    name: str
    version: str
    timeframe_minutes: int

    def init_state(self, params: P) -> S: ...

    def on_bar(self, bar: Bar, state: S, params: P,
               view: CoreView) -> tuple[S, list[OrderAction]]: ...

    def on_execution(self, report: ExecutionReport, state: S, params: P,
                     view: CoreView) -> tuple[S, list[OrderAction]]: ...
