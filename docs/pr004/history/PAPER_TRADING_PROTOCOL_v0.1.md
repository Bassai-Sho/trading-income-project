# PR-004 Paper Trading Protocol — DRAFT v0.1 (NOT ACTIVE)

*29 Sep 2026. Drafted per the owner's request, in advance of PR-004's own freeze — matching the discipline already used for PR-005 (pre-register the protocol before results, not after). This protocol only activates if PR-004 reaches its own value-CONFIRMED-equivalent verdict; see §0.*

## 0. What must be resolved before this protocol can start

1. **PR-004 itself is not frozen.** Your sign-off on the corrected decision-rule wording (spec §8, condition 1) is still awaited, and three further freeze conditions remain open (PR-005 hash-sequencing, source archiving already done, the owner-executed hash step). This protocol is drafted now so it exists the moment PR-004 clears its gates — it is not a signal that PR-004 has passed anything.
2. **PR-004's decision rule, once signed off, determines whether this protocol ever activates at all.** Per the spec: proceed to forward paper trading iff G0–G2 pass and the H1 verdict is not CONTRADICTED. If PR-004 is CONTRADICTED or fails replication, this protocol is moot and should not be started.

## 1. Purpose

At roughly 3.8 round trips a year (49 trades over E1's 13 years, 30 over E2's 8 — both eras agree closely), this is **more frequent than PR-005's 0.67/yr, but still nowhere near enough to power a performance judgment within any reasonable paper window.** Computed directly, not estimated: even a modest target of 20 trades has under a 2% chance of being reached within 3 years at this rate; 10 trades has roughly a 70% chance within 3 years but under 1% within a single year. **Purpose is operational validation only, exactly matching PR-005's framing** — fill mechanics, signal-computation correctness against the frozen spec, realized slippage versus the assumed cost model. Performance during the paper window is explicitly non-evidential and must not be read as confirming or contradicting anything about H1.

## 2. Duration

12 months minimum, or 10 completed round trips, whichever is later. At the real rate (~3.8/yr), 10 round trips is expected to take roughly 2.5–3 years (the same order as PR-005's multi-year expectation, computed the same way). **State this plainly to yourself before starting:** this is a multi-year operational commitment if run to the trade-count floor, and the risk of quiet abandonment over that span is a real failure mode, not a hypothetical one — matching the caution already logged for PR-005's own protocol.

## 3. Kill criteria (operational only, matching PR-005's discipline)

- Realized slippage exceeds 3× the spec's assumed cost (2 bps/side, 3× after a >2σ prior-day move) on 2 or more fills.
- Any day where the paper-traded signal diverges from what the frozen rule's own script computes for that day (a direct check against the frozen spec, not a judgment call).
- A fill-mechanism failure: an order unfilled within a stated window, or filled at the wrong size or side.

**Explicitly prohibited, as with PR-005:** any performance-based continuation or kill decision during the window. A quiet, uneventful year of few or no trades is the expected outcome at this rate and must not be read as either good or bad.

## 4. Data source

The live store, once it exists and is populated (it already is, as of the seal work — `DATA/live_market_data.db`, continuous from 2025-01-02). Same hard rule as PR-003's protocol: never the sealed research store.

## 5. Sizing and vehicle

Unlike PR-003, PR-004 has no risk-policy memo yet — sizing and vehicle for this strategy specifically haven't been discussed at all. This is a gap to fill once PR-004 is closer to freezing, not something to backfill now on a strategy that hasn't passed its own gates.

## 6. Relationship to the stop rule

PR-004 is named inside the same 8-week stop-rule box as PR-003, with no scope ambiguity (unlike PR-005). No drawdown-wording tension of the kind found for PR-003 — PR-004 has not run its gates yet, so there's no result to check against the stop bar.

## Sources

PR-004 spec v0.8 (`docs/pr004/PR004_SPEC.md`), the same duration/kill-criteria discipline established for PR-005's protocol (§8a of its spec) and reused here rather than re-derived.
