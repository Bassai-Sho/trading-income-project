# Research Seal Protocol

*Written 28 Sep 2026. Code: `src/seal.py`, `src/market_data_store.py`, `src/seal_migrate.py`.*

## What is sealed, and why

Market data **on or after 2025-01-01** is sealed for *research*. The point is not the bytes
(a vendor can re-deliver them, and the window grows every day). The point is the **untouched
protocol**: a final analysis that was written and committed *before* anyone looked at the
window, run **once**. Looking early — even "just to sanity-check a threshold" — spends it,
silently and irreversibly.

## How it is enforced

* A **research store** (`MarketDataStore(path)`, the default, `sealed=True`) refuses to read
  or write any range that ends on/after 2025-01-01, and refuses to be reopened unsealed.
* **Forward / paper-trading data** goes to a separate **live store**
  (`DATA/live_market_data.db`, `MarketDataStore(path, sealed=False)`). It is deliberately not
  sealed: forward time is the renewable evidence source. The runner's daily update writes there.
* The window the PR-003 scripts read (`pr003_replication.WINDOW`, ending 2024-12-31) is the day
  before the seal. A test asserts the two agree.
* History: until 28 Sep 2026 this was only a convention, and the runner had appended SPY bars
  to 2026-06-30 into the research table. No PR-003 result ever read them (fixed window +
  assertion on the last bar). `src/seal_migrate.py` moves those rows to the live store.

## How it is unlocked (all steps, in order)

1. The risk-policy decisions are made (sizing option, vehicle, halt threshold).
2. The **final analysis script is written and committed first**: frozen parameters (14, 1.0, 30),
   the cost model, the metrics and pass criteria, and the pre-registered VIX bucket contrast.
   The commit hash is recorded.
3. A cold-fork review of the *decision to unlock*.
4. **Only the investor** sets `TIC_SEAL_UNLOCK` to the phrase defined in `src/seal.py`. No
   assistant sets it, in any session, for any reason.
5. The script runs **once**. Its raw output is saved verbatim and committed.
6. The window is then spent for this strategy. A new strategy needs new data (forward data
   from the live store), not a second look.

## Rules that outlive any one session

* **The 2025+ result cannot by itself reopen the volatility-gate question.** That question
  (did low volatility cause the 2016-2020 drawdown?) was closed on non-persistence grounds.
  Reopening needs a *new multi-year episode* with its own full decomposition, not ~21 months
  of agreement.
* A request to "peek at 2025 just to check X" is a refusal case. Cite this file.
* Never edit `SEAL_START` or `pr003_replication.WINDOW` without the investor's explicit
  approval and a cold-fork review.

## Honest limits

The unlock phrase is a **speed bump against accidents, not security**. Anyone with a shell can
set an environment variable, edit `seal.py`, or open the SQLite file directly. It stops a
script or a session from casually reading the window; it does not stop a determined person.
The window also cannot un-know what anyone already knows about 2025-26 market history.

## Commands

```bash
# forward data into the LIVE store (what the runner now does)
python src/market_data_store.py --update --tickers SPY --db DATA/live_market_data.db --live

# move rows dated 2025-01-01 or later out of the research store
python src/seal_migrate.py                                    # dry run
python src/seal_migrate.py --execute                          # copy + verify (deletes nothing)
python src/seal_migrate.py --execute --delete-from-research --confirm DELETE-SEALED-ROWS
```
