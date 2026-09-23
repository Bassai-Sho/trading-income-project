"""
market_calendar.py
==================
NYSE market calendar check for the Trading Income Project.

Used by trading_engine.py and morning_brief.py to:
  - Detect US market holidays (engine exits cleanly; morning brief shows NO-GO)
  - Detect early-close days and return the correct session_end time
  - Detect weekend days before any data fetch is attempted

Requires:  pip install exchange_calendars

Fallback:  if exchange_calendars is unavailable, falls back to a hardcoded
           2026 holiday list so the engine never silently breaks.

Usage:
    from market_calendar import check_market_session

    result = check_market_session()          # uses today's date (EST), NYSE equities
    result = check_market_session("2026-11-26")   # Thanksgiving
    result = check_market_session(market_type="forex")   # P2-072: simplified FX gate

    result.is_open         → bool
    result.session_end     → datetime.time  (EST) — 11:00 for strategy, but
                                             13:00 on early-close days caps it
    result.reason          → str  e.g. "NYSE closed — Thanksgiving Day"
    result.is_early_close  → bool
    result.normal_close    → datetime.time  (EST) e.g. 15:00 normal, 13:00 early

market_type (P2-072):
    'nyse_equities' (default) — full NYSE holiday/early-close calendar via
        exchange_calendars, or the hardcoded fallback below. Unchanged from
        the original implementation.
    'forex'         — simplified date-level gate: closed Saturday/Sunday and
        a short major-holiday list only. No early-close or single-session-
        window concept — forex trades roughly 24/5, so is_early_close is
        always False and normal_close is unused for this market_type.
        Intentionally minimal for a first cut (see P2-072 notes); revisit
        if/when a forex account actually goes live.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, time as Time, timedelta, timezone
from typing import Optional

log = logging.getLogger(__name__)


def _utcnow() -> datetime:
    """Naive UTC 'now' — same value as the removed datetime.utcnow(), via
    the non-deprecated timezone-aware path."""
    return datetime.now(timezone.utc).replace(tzinfo=None)

# ---------------------------------------------------------------------------
# Hardcoded 2026 holiday fallback (used if exchange_calendars unavailable)
# ---------------------------------------------------------------------------
_HOLIDAYS_2026: set[str] = {
    "2026-01-01",   # New Year's Day
    "2026-01-19",   # Martin Luther King Jr. Day
    "2026-02-16",   # Presidents' Day
    "2026-04-03",   # Good Friday
    "2026-05-25",   # Memorial Day
    "2026-07-03",   # Independence Day (observed)
    "2026-09-07",   # Labor Day
    "2026-11-26",   # Thanksgiving Day
    "2026-12-25",   # Christmas Day
}

_EARLY_CLOSES_2026: dict[str, Time] = {
    "2026-07-02":  Time(13, 0),   # Day before Independence Day
    "2026-11-27":  Time(13, 0),   # Day after Thanksgiving
    "2026-12-24":  Time(13, 0),   # Christmas Eve
}

# Normal NYSE close (EST).  Strategy session_end is 11:00 regardless, but
# this cap prevents the engine attempting to fetch data after market close.
_NORMAL_CLOSE_EST = Time(15, 0)   # 16:00 EST actually, but 15:00 is safe cap

# ---------------------------------------------------------------------------
# Forex (P2-072): deliberately minimal. Major global forex liquidity centres
# effectively close only for Christmas and New Year's Day; every other
# weekday is treated as open. No early-close concept for this market_type.
# ---------------------------------------------------------------------------
_FOREX_HOLIDAYS_2026: dict[str, str] = {
    "2026-01-01": "New Year's Day",
    "2026-12-25": "Christmas Day",
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class MarketSessionResult:
    is_open:        bool
    reason:         str
    is_early_close: bool    = False
    normal_close:   Time    = _NORMAL_CLOSE_EST   # market close time (EST)
    session_date:   str     = ""                  # ISO date string

    @property
    def session_end(self) -> Time:
        """
        Returns the correct market close time in EST.
        This is NOT the strategy session_end (always 11:00) — it is the
        hard cap beyond which no data fetch should be attempted.
        """
        return self.normal_close


# ---------------------------------------------------------------------------
# Core check
# ---------------------------------------------------------------------------
def check_market_session(
    date_str: Optional[str] = None,
    tz_offset_hours: int = -5,
    market_type: str = "nyse_equities",
) -> MarketSessionResult:
    """
    Check whether the given market is open on the given date.

    Args:
        date_str:         ISO date string 'YYYY-MM-DD'.  If None, uses today
                          in EST (UTC + tz_offset_hours).
        tz_offset_hours:  UTC offset for EST.  -5 standard, -4 during DST.
                          The engine CONFIG['tz_offset_hours'] should be
                          passed here for consistency.
        market_type:      'nyse_equities' (default, unchanged behaviour) or
                          'forex' (P2-072 — see module docstring).

    Returns:
        MarketSessionResult
    """
    if market_type not in ("nyse_equities", "forex"):
        raise ValueError(
            f"Unknown market_type: {market_type!r} — expected "
            f"'nyse_equities' or 'forex'"
        )

    # Resolve target date
    if date_str is None:
        now_est = _utcnow() + timedelta(hours=tz_offset_hours)
        target_date = now_est.date()
    else:
        target_date = date.fromisoformat(date_str)

    date_iso = str(target_date)

    # ── Weekend check (fast path, no library needed) — shared by both
    #    market types; wording keeps "NYSE"/"FX" so log lines stay accurate ──
    if target_date.weekday() >= 5:
        day_name = "Saturday" if target_date.weekday() == 5 else "Sunday"
        label = "FX" if market_type == "forex" else "NYSE"
        return MarketSessionResult(
            is_open=False,
            reason=f"{label} closed — {day_name}",
            session_date=date_iso,
        )

    if market_type == "forex":
        return _check_forex_session(target_date, date_iso)

    # ── Try exchange_calendars (preferred) ──
    try:
        import exchange_calendars as xcals
        import pandas as pd

        nyse = xcals.get_calendar("XNYS")

        if not nyse.is_session(date_iso):
            # Holiday — get a human-readable name if possible
            reason = _holiday_name(target_date)
            return MarketSessionResult(
                is_open=False,
                reason=f"NYSE closed — {reason}",
                session_date=date_iso,
            )

        # Open day — check for early close
        sched = nyse.schedule
        row = sched[sched.index == date_iso]
        if not row.empty:
            close_utc = pd.Timestamp(row["close"].values[0])
            close_est = (close_utc + pd.Timedelta(hours=tz_offset_hours)).time()
            is_early = close_est < Time(15, 0)
            reason = (f"NYSE early close at {close_est.strftime('%H:%M')} EST"
                      if is_early else "NYSE open — normal session")
            return MarketSessionResult(
                is_open=True,
                reason=reason,
                is_early_close=is_early,
                normal_close=close_est,
                session_date=date_iso,
            )

        # Session confirmed open but no schedule row (shouldn't happen)
        return MarketSessionResult(
            is_open=True,
            reason="NYSE open — normal session",
            session_date=date_iso,
        )

    except ImportError:
        log.warning("exchange_calendars not installed — using hardcoded 2026 fallback")

    # ── Fallback: hardcoded 2026 lists ──
    if date_iso in _HOLIDAYS_2026:
        reason = _holiday_name(target_date)
        return MarketSessionResult(
            is_open=False,
            reason=f"NYSE closed — {reason} (hardcoded fallback)",
            session_date=date_iso,
        )

    if date_iso in _EARLY_CLOSES_2026:
        early_time = _EARLY_CLOSES_2026[date_iso]
        return MarketSessionResult(
            is_open=True,
            reason=f"NYSE early close at {early_time.strftime('%H:%M')} EST (hardcoded fallback)",
            is_early_close=True,
            normal_close=early_time,
            session_date=date_iso,
        )

    return MarketSessionResult(
        is_open=True,
        reason="NYSE open — normal session (hardcoded fallback)",
        session_date=date_iso,
    )


# ---------------------------------------------------------------------------
# Forex (P2-072): simplified date-level gate
# ---------------------------------------------------------------------------
def _check_forex_session(target_date: date, date_iso: str) -> MarketSessionResult:
    """Weekend already ruled out by the caller. Closed only on the short
    major-holiday list above; every other weekday is open, no early-close
    concept (forex trades roughly 24/5)."""
    if date_iso in _FOREX_HOLIDAYS_2026:
        reason = _FOREX_HOLIDAYS_2026[date_iso]
        return MarketSessionResult(
            is_open=False,
            reason=f"FX closed — {reason}",
            session_date=date_iso,
        )
    return MarketSessionResult(
        is_open=True,
        reason="FX open — 24/5 session",
        session_date=date_iso,
    )


# ---------------------------------------------------------------------------
# Convenience: holiday name lookup
# ---------------------------------------------------------------------------
_NAMED_HOLIDAYS: dict[str, str] = {
    "2026-01-01": "New Year's Day",
    "2026-01-19": "Martin Luther King Jr. Day",
    "2026-02-16": "Presidents' Day",
    "2026-04-03": "Good Friday",
    "2026-05-25": "Memorial Day",
    "2026-07-03": "Independence Day (observed)",
    "2026-09-07": "Labor Day",
    "2026-11-26": "Thanksgiving Day",
    "2026-12-25": "Christmas Day",
}


def _holiday_name(d: date) -> str:
    return _NAMED_HOLIDAYS.get(str(d), "Market Holiday")


# ---------------------------------------------------------------------------
# CLI self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    test_dates = [
        ("2026-09-15", "Normal trading day (today)"),
        ("2026-11-26", "Thanksgiving — holiday"),
        ("2026-11-27", "Day after Thanksgiving — early close"),
        ("2026-12-25", "Christmas Day — holiday"),
        ("2026-07-02", "Early close before July 4th"),
        ("2026-09-19", "Saturday — weekend"),
        ("2026-09-20", "Sunday — weekend"),
    ]

    print("\n  MARKET CALENDAR SELF-TEST — NYSE EQUITIES (unchanged path)")
    print("  " + "─" * 56)
    for date_s, label in test_dates:
        r = check_market_session(date_s)
        status = "✅ OPEN " if r.is_open else "🚫 CLOSED"
        early  = f"  [early close {r.normal_close.strftime('%H:%M')} EST]" if r.is_early_close else ""
        print(f"  {date_s}  {status}  {r.reason}{early}")
        print(f"             ({label})")

    forex_test_dates = [
        ("2026-09-15", "Normal weekday (today)"),
        ("2026-11-26", "Thanksgiving — NYSE holiday, FX stays open"),
        ("2026-12-25", "Christmas Day — closed for both"),
        ("2026-01-01", "New Year's Day — closed for both"),
        ("2026-09-19", "Saturday — weekend"),
        ("2026-09-20", "Sunday — weekend"),
    ]
    print("\n  MARKET CALENDAR SELF-TEST — FOREX (P2-072)")
    print("  " + "─" * 56)
    for date_s, label in forex_test_dates:
        r = check_market_session(date_s, market_type="forex")
        status = "✅ OPEN " if r.is_open else "🚫 CLOSED"
        print(f"  {date_s}  {status}  {r.reason}")
        print(f"             ({label})")
    print()
