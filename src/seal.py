"""
seal.py — the research seal on data from 2025-01-01 onward  (28 Sep 2026)
=========================================================================
WHY THIS EXISTS. Until now the seal was a convention: a date constant that each
research script asserted for itself. A new script, a notebook or a data-refresh
job could ignore it -- and one did, by design: runner.py's daily job appended
SPY bars up to 2026-06-30 into the SAME table every PR-003 script reads (found
28 Sep 2026; PR-003 itself never read them: it loads through a fixed window and
asserts the last bar is <= 2024-12-31).

WHAT IS PROTECTED. Not the bytes (a vendor can re-deliver them and the window
grows every day). The asset is the UNTOUCHED PROTOCOL: a pre-registered final
analysis, run once. So the control is at the data layer:
  * a research store (sealed=True, the default) refuses to READ or WRITE any
    range that ends on/after SEAL_START;
  * live/forward data (paper trading) goes to a separate LIVE store
    (sealed=False), so the renewable evidence keeps accumulating unsealed;
  * unlocking is a deliberate act (see UNLOCK_* below and docs/SEAL_PROTOCOL.md).

HONEST LIMITS. The unlock phrase is a SPEED BUMP against accidents, not
cryptography: anyone with a shell can set an environment variable, edit this
file, or open the SQLite file directly. It defends against a script or session
casually reading the window; it does not defend against a determined person.
"""
from __future__ import annotations

import os
from datetime import date, datetime

SEAL_START = date(2025, 1, 1)                 # first sealed date
UNLOCK_ENV = "TIC_SEAL_UNLOCK"
UNLOCK_PHRASE = "I-HAVE-COMMITTED-THE-FINAL-ANALYSIS-SCRIPT"


class SealedDataError(RuntimeError):
    """Raised when research code would read or write sealed (2025+) data."""


def is_unlocked() -> bool:
    return os.environ.get(UNLOCK_ENV, "") == UNLOCK_PHRASE


def as_date(x) -> date:
    """date | datetime | pandas.Timestamp | 'YYYY-MM-DD[...]' string -> date."""
    if isinstance(x, datetime):               # pandas.Timestamp is a datetime subclass
        return x.date()
    if isinstance(x, date):
        return x
    if isinstance(x, str):
        return date.fromisoformat(x[:10])
    raise TypeError(f"cannot interpret {x!r} as a date")


def check(end, what: str) -> None:
    """Allow the operation only if the LAST date it touches is before the seal,
    or the seal has been deliberately unlocked."""
    if is_unlocked():
        return
    d = as_date(end)
    if d >= SEAL_START:
        raise SealedDataError(
            f"{what}: touches {d}, on/after the seal date {SEAL_START}. Research stores are "
            f"sealed from that date until the pre-registered final analysis is run "
            f"(docs/SEAL_PROTOCOL.md). Live/forward data belongs in a live store: "
            f"MarketDataStore(path, sealed=False).")
