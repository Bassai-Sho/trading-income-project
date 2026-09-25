"""P2-123 slice 1: the execution core. The reviewer's six acceptance fixtures
(round 3, 25 Sep 2026) plus OCO, activation timing, calendar expiry, the
next-day fail-safe, accounting and fail-closed checks."""
import sys
from datetime import datetime, time, date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from core.interfaces import Bar, OrderAction
from core.execution_core import AccountConfig, ExecutionCore

T0 = datetime(2019, 3, 12, 10, 0)


def core(cash=10_000.0, leverage=None, close=time(16, 0), fee=None):
    return ExecutionCore(AccountConfig(cash, leverage), fee_fn=fee,
                         session_close=lambda d: close)


def bar(o, h, l, c, ts=T0, sym="SPY"):
    return Bar(sym, ts, o, h, l, c, 1000)


def sub(cid, side, typ, qty=10, price=None, stop=None, oco="", tif="DAY", sym="SPY", tag=""):
    return OrderAction("SUBMIT", cid, sym, side, typ, qty, price, stop, tif, oco, tag)


def long_position(c, px=100.0, qty=10, ts=T0):
    c.submit([sub("e", "BUY", "MARKET", qty)], ts)
    c.process_bar(bar(px, px, px, px, ts))
    assert c.position("SPY") == qty


def fills(reports):
    return [(r.client_order_id, r.fill_price) for r in reports if r.status == "FILLED"]


# ── Reviewer acceptance fixtures (round 3) ───────────────────────────────────

def test_1_resting_stop_normal_cross():
    c = core(); long_position(c)
    c.submit([sub("s", "SELL", "STOP", stop=99.0)], T0)
    assert fills(c.process_bar(bar(100, 101, 98, 99, T0.replace(minute=1)))) == [("s", 99.0)]


def test_2_resting_stop_gap_down_fills_at_open_never_the_stop():
    c = core(); long_position(c)
    c.submit([sub("s", "SELL", "STOP", stop=99.0)], T0)
    assert fills(c.process_bar(bar(97.0, 97.5, 96.0, 96.5, T0.replace(minute=1)))) == [("s", 97.0)]


def test_3_minute_touching_stop_and_target_is_a_stop():
    c = core(); long_position(c)
    c.submit([sub("s", "SELL", "STOP", stop=98.0, oco="b1"),
              sub("t", "SELL", "LIMIT", price=102.0, oco="b1")], T0)
    r = c.process_bar(bar(100.0, 103.0, 97.0, 101.0, T0.replace(minute=1)))
    assert fills(r) == [("s", 98.0)]
    assert [(x.client_order_id, x.status, x.reason) for x in r if x.status == "CANCELLED"] == \
        [("t", "CANCELLED", "OCO")]
    assert c.position("SPY") == 0


def test_4_day_order_expires_at_session_close():
    c = core(); long_position(c)
    c.submit([sub("s", "SELL", "STOP", stop=90.0)], T0.replace(hour=15, minute=30))
    r = c.end_session(T0.replace(hour=16))
    assert [(x.client_order_id, x.status, x.reason) for x in r] == [("s", "EXPIRED", "SESSION_CLOSE")]
    assert c.view().open_orders == () and c.position("SPY") == 10     # position survives


def test_5_duplicate_client_order_id_is_rejected():
    c = core(); long_position(c)
    c.submit([sub("s", "SELL", "STOP", stop=99.0)], T0)
    r = c.submit([sub("s", "SELL", "STOP", stop=99.0)], T0)
    assert (r[0].status, r[0].reason) == ("REJECTED", "DUPLICATE_CLIENT_ORDER_ID")
    assert len(c.view().open_orders) == 1


def test_6_buying_power_rejection():
    c = core(cash=10_000, leverage=1.0)
    c.process_bar(bar(100, 100, 100, 100))                             # establishes a price
    r = c.submit([sub("big", "BUY", "MARKET", qty=1_000)], T0)
    assert (r[0].status, r[0].reason) == ("REJECTED", "INSUFFICIENT_BUYING_POWER")


# ── Additional fill rules ────────────────────────────────────────────────────

def test_market_order_fills_at_next_open_not_on_the_signal_bar():
    c = core()
    end_of_signal_bar = T0.replace(minute=5)                           # 5-min bar 10:00-10:04 closed
    c.submit([sub("e", "BUY", "MARKET")], end_of_signal_bar)
    assert fills(c.process_bar(bar(100, 101, 99, 100.5, T0.replace(minute=4)))) == []
    assert fills(c.process_bar(bar(100.7, 101, 100, 100.9, T0.replace(minute=5)))) == [("e", 100.7)]


def test_limit_gap_gives_price_improvement_and_short_stop_gaps_up():
    c = core(); long_position(c)
    c.submit([sub("t", "SELL", "LIMIT", price=102.0)], T0)
    assert fills(c.process_bar(bar(103.0, 103.5, 102.5, 103.0, T0.replace(minute=1)))) == [("t", 103.0)]
    s = core()
    s.submit([sub("e", "SELL", "MARKET")], T0); s.process_bar(bar(100, 100, 100, 100))
    s.submit([sub("s", "BUY", "STOP", stop=101.0)], T0)
    assert fills(s.process_bar(bar(102.0, 102.5, 101.5, 102.0, T0.replace(minute=1)))) == [("s", 102.0)]


def test_buy_limit_fills_at_limit_when_crossed_intrabar():
    c = core()
    c.submit([sub("b", "BUY", "LIMIT", price=99.0)], T0)
    assert fills(c.process_bar(bar(100, 100.5, 98.5, 99.5))) == [("b", 99.0)]


# ── Lifecycle ────────────────────────────────────────────────────────────────

def test_next_day_fail_safe_expires_before_matching():
    """end_session forgotten: a DAY stop must NOT fill on the next morning's gap."""
    c = core(); long_position(c)
    c.submit([sub("s", "SELL", "STOP", stop=99.0)], T0)
    r = c.process_bar(bar(90, 91, 89, 90, datetime(2019, 3, 13, 9, 30)))
    assert [(x.client_order_id, x.status, x.reason) for x in r] == [("s", "EXPIRED", "SESSION_CLOSE_MISSED")]
    assert c.position("SPY") == 10


def test_gtc_orders_survive_the_close():
    c = core(); long_position(c)
    c.submit([sub("s", "SELL", "STOP", stop=99.0, tif="GTC")], T0)
    assert c.end_session(T0.replace(hour=16)) == []
    assert len(c.view().open_orders) == 1


def test_early_close_comes_from_the_calendar():
    real = ExecutionCore()                                              # default: exchange calendar
    assert real.session_close_for(date(2024, 11, 29)) == time(13, 0)
    assert real.session_close_for(date(2024, 11, 27)) == time(16, 0)
    assert real.session_close_for(date(2024, 11, 28)) is None           # Thanksgiving


def test_cancel_and_unknown_cancel():
    c = core(); long_position(c)
    c.submit([sub("s", "SELL", "STOP", stop=99.0)], T0)
    r = c.submit([OrderAction("CANCEL", "s", "SPY"), OrderAction("CANCEL", "nope", "SPY")], T0)
    assert [(x.status, x.reason) for x in r] == [("CANCELLED", "USER"), ("REJECTED", "UNKNOWN_ORDER")]


def test_ids_can_never_be_reused_even_after_completion():
    c = core(); long_position(c)                                        # "e" filled
    r = c.submit([sub("e", "BUY", "MARKET")], T0)
    assert (r[0].status, r[0].reason) == ("REJECTED", "DUPLICATE_CLIENT_ORDER_ID")


def test_invalid_orders_rejected():
    c = core()
    bad = [sub("a", "BUY", "LIMIT"), sub("b", "SELL", "STOP"), sub("c", "BUY", "MARKET", qty=0)]
    assert {r.reason for r in c.submit(bad, T0)} == {"INVALID_ORDER"}


def test_reducing_orders_never_blocked_by_buying_power():
    c = core(cash=10_000, leverage=1.0)
    c.process_bar(bar(100, 100, 100, 100))
    c.submit([sub("e", "BUY", "MARKET", qty=100)], T0)                  # $10k: exactly 1x
    c.process_bar(bar(100, 100, 100, 100, T0.replace(minute=1)))
    r = c.submit([sub("x", "SELL", "MARKET", qty=100)], T0.replace(minute=1))
    assert r[0].status == "ACCEPTED"


def test_unlimited_leverage_mode_for_legacy_parity():
    c = core(leverage=None)
    c.process_bar(bar(100, 100, 100, 100))
    assert c.submit([sub("big", "BUY", "MARKET", qty=10_000)], T0)[0].status == "ACCEPTED"


# ── Accounting and fail-closed ───────────────────────────────────────────────

def test_cash_position_and_realized_pnl_with_fees():
    c = core(cash=10_000, fee=lambda *a: 1.0)
    long_position(c, px=100.0, qty=10)                                  # buy 10 @ 100, fee 1
    c.submit([sub("x", "SELL", "LIMIT", qty=10, price=105.0)], T0)
    c.process_bar(bar(104, 106, 104, 105.5, T0.replace(minute=1)))     # sell 10 @ 105, fee 1
    v = c.view()
    assert v.cash == pytest.approx(10_000 - 1_000 - 1 + 1_050 - 1)
    assert v.positions["SPY"].qty == 0
    assert v.positions["SPY"].realized_pnl == pytest.approx(50 - 2)


def test_flip_through_zero_resets_average():
    c = core(); long_position(c, qty=10)
    c.submit([sub("x", "SELL", "MARKET", qty=15)], T0)
    c.process_bar(bar(110, 110, 110, 110, T0.replace(minute=1)))
    p = c.view().positions["SPY"]
    assert (p.qty, p.avg_price, p.realized_pnl) == (-5, 110.0, 100.0)


@pytest.mark.parametrize("o,h,l,cl", [(100, 99, 101, 100), (float("nan"), 1, 1, 1), (100, 101, 99, 102)])
def test_malformed_bar_fails_closed(o, h, l, cl):
    with pytest.raises(ValueError, match="fail closed"):
        core().process_bar(bar(o, h, l, cl))


# ── Same-minute follow-on orders (bracket legs; slice 2) ────────────────────

def test_follow_on_stop_uses_fill_price_not_bar_open():
    """Entry filled mid-minute at 100.40; a stop at 100.50 placed after it is
    already through: fills at the entry price, never at the stop or the open."""
    c = core()
    c.submit([sub("e", "BUY", "LIMIT", price=100.40)], T0)
    b = bar(101.0, 101.2, 100.0, 100.8)
    assert fills(c.process_bar(b)) == [("e", 100.40)]
    c.submit([sub("s", "SELL", "STOP", stop=100.50)], T0, after_price=100.40)
    assert fills(c.process_bar(b)) == [("s", 100.40)]


def test_follow_on_bracket_triggers_in_rest_of_minute_stop_first():
    c = core()
    c.submit([sub("e", "BUY", "MARKET")], T0)
    b = bar(100.0, 101.5, 98.5, 100.2)
    assert fills(c.process_bar(b)) == [("e", 100.0)]
    c.submit([sub("s", "SELL", "STOP", stop=99.0, oco="b"),
              sub("t", "SELL", "LIMIT", price=101.0, oco="b")], T0, after_price=100.0)
    r = c.process_bar(b)
    assert fills(r) == [("s", 99.0)] and c.position("SPY") == 0


def test_follow_on_market_fills_at_the_triggering_fill_price():
    c = core()
    c.submit([sub("e", "BUY", "MARKET")], T0)
    b = bar(100.0, 100.5, 99.5, 100.2)
    c.process_bar(b)
    c.submit([sub("x", "SELL", "MARKET")], T0, after_price=100.0)
    assert fills(c.process_bar(b)) == [("x", 100.0)]


def test_follow_on_price_applies_only_to_its_own_minute():
    c = core(); long_position(c)
    c.submit([sub("s", "SELL", "STOP", stop=99.0)], T0, after_price=100.0)
    nxt = bar(98.0, 98.5, 97.5, 98.0, T0.replace(minute=1))       # next minute gaps below the stop
    assert fills(c.process_bar(nxt)) == [("s", 98.0)]              # its own open, not 100.0


def test_reprocessing_a_bar_is_idempotent_for_existing_orders():
    c = core(); long_position(c)
    c.submit([sub("s", "SELL", "STOP", stop=95.0)], T0)
    b = bar(100, 101, 99, 100, T0.replace(minute=1))
    assert c.process_bar(b) == [] and c.process_bar(b) == []
    assert len(c.view().open_orders) == 1
