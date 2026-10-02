# PR-004 Paper Trading Protocol — v0.3, CONSOLIDATED (NOT ACTIVE)

*2 Oct 2026. This is the full standalone text, consolidated from v0.1 and v0.2 (29 Sep). The previously committed v0.2 carried §1–4 "unchanged from v0.1" by reference; both are preserved in `docs/pr004/history/`. It is aligned with PR-004 spec v0.7 §8, which now resolves the stop-rule question v0.2 left open, and it adds the pipeline-only window (state S2). Nothing here overrides the spec. Where they differ, the spec governs.*

## 0. Activation

This protocol activates only in states **S1** (full window) or **S2** (pipeline-only window) of the spec's consequence table (§8.2). Both states execute at joint-verdict determination (§8.6). In states S3 and S4, and whenever the project's joint stop has fired, no window starts. If the spec's b̂12 ≥ 0 fallback applies, the owner's documented review comes first. Clearing PR-004's gates makes this protocol *relevant*, not immediately *runnable*: sizing and vehicle are still undecided (§5).

## 1. Purpose

At ~3.8 round trips a year (49 trades over 1995–2007, 30 over 2008–2015), paper performance cannot support a judgment in any reasonable window. Computed directly: reaching 20 trades has under a 2% chance within 3 years; reaching 10 has under 1% within one year and roughly 70% within three. **The purpose is operational validation only:** fill mechanics, signal-computation correctness against the frozen spec, and realized slippage against the cost model. Performance is non-evidential and must not be read as confirming or contradicting H1.

## 2. Duration

- **S1 (full window):** 12 months minimum, or 10 completed round trips, whichever is later. The 10-round-trip floor exists to give the slippage check a meaningful sample. At ~3.8 a year it implies roughly 2.5–3 years, a multi-year commitment with real risk of quiet abandonment, stated here so it's approved knowingly.
- **S2 (pipeline-only):** a fixed 12-month hard stop. No round-trip leg, and no early termination for "success" (spec §8.5).
- **Both:** a running window halts if the project's joint stop fires (spec §8.6 d).

## 3. Kill criteria (operational only)

- Realized slippage exceeds **3× the model-implied cost for that specific fill** (2 bps per side on normal days; 3× that after a prior-day move above 2σ) on 2 or more fills. This is not a flat threshold.
- Any day on which the paper-traded signal diverges from what the frozen rule's own script computes for that day. This check accrues daily and does not wait on trade count.
- A fill-mechanism failure: an order unfilled within a stated window, or filled at the wrong size or side.
- A data-feed gap or an unexpected fetch failure, matching PR-003's set.

**Prohibited:** any performance-based continuation, kill or admission decision during the window. **Firewall (spec §8.5):** paper P&L is recorded as data and never narrated in any decision log, summary, tracker note or report. A quiet year with few or no trades is the expected outcome and must not be read as good or bad.

## 4. Data source

The live store, `DATA/live_market_data.db`, which already exists and is populated continuously from 2025-01-02 (through at least 2026-09-25 at last check). Never the sealed research store.

## 5. Sizing, vehicle and cost model

No risk-policy memo exists for PR-004. Sizing and vehicle are undecided and must be decided before a window starts, not switched mid-window. The paper engine logs costs under the spec's Mode F, the mode pinned for the stop bar, so slippage checks compare like with like.

## 6. Shared components (state S2's reason to exist)

The S2 window exists to validate machinery that PR-005 will reuse: daily SPY bar ingestion from the live store, the XNYS calendar, close-of-day signals, next-open order generation and fill recording, ex-dividend handling, and per-fill cost logging. These must be the same imported modules PR-005 will use, not copies (spec §8.5). The one-way guard applies: PR-005's admission may not cite this window.

## 7. Relationship to the stop rule

Resolved by spec v0.7 §8.4–8.6. PR-004's per-strategy stop bar is computed at G4 with pinned windows and Mode F. Its result selects S1 or S2. The joint stop overrides both.

## Sources

PR-004 spec v0.7 (`docs/pr004/PR004_SPEC.md`); this protocol's v0.1 and v0.2 (in `docs/pr004/history/`); PR-005 spec §8a, the original source of the operational-only discipline.
