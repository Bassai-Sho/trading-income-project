# PR-004 Paper Trading Protocol — DRAFT v0.2 (NOT ACTIVE)

*29 Sep 2026. Supersedes v0.1. Small fixes; the design itself was already sound.*

## 0. What changed from v0.1

| # | Change |
|---|---|
| 1 | **§5's self-contradicting sentence fixed.** "Once it exists and is populated (it already is…)" read oddly in an operational document — restated plainly as: the live store already exists and is populated, 2025-01-02 through 2026-09-25 (matching PR-003's protocol's description of the same store exactly, after that one was corrected too). |
| 2 | **Slippage kill criterion pinned to the model, not a flat number.** "3× the assumed cost" was ambiguous between a flat 6bp bar and the spec's own regime-conditional model (2bps normal days, 3× after a >2σ prior-day move). Now explicit: the trigger is 3× whatever the model-implied cost was for that specific fill, not a single flat threshold — a normal-day fill at 4bps should not silently sit below a flat 6bp bar that was never the intended comparison. |
| 3 | **Feed-gap kill criterion added**, matching PR-003's set — PR-004's v0.1 kill list didn't have one, and there's no reason it should be more permissive than PR-003's on basic data integrity. |
| 4 | **The 10-round-trip floor's purpose stated explicitly.** It's the sample size the slippage check needs to be meaningful — the signal-vs-frozen-script divergence check (item 2 above notwithstanding) accrues daily regardless of trade count and doesn't wait on it. |
| 5 | **§6's framing softened.** "Exists the moment PR-004 clears its gates" oversold readiness — §5 (unchanged in substance) still leaves sizing and vehicle undecided for this strategy specifically, so clearing PR-004's own gates makes this protocol relevant, not immediately runnable. |
| 6 | **The activation-bar asymmetry against PR-005 named, not silently carried.** PR-004 proceeds on "not CONTRADICTED" (which includes INCONCLUSIVE); PR-005 was deliberately tightened to require CONFIRMED specifically. I'm not resolving which is right here — PR-004's decision rule is a property of its own frozen spec, not something this protocol document should override — but leaving the difference unstated risked it looking like an accident rather than a choice made (or not yet properly examined) in the spec itself. Worth your attention at PR-004's own sign-off, not here. |
| 7 | **Noted, not resolved here:** the broader question of whether the stop rule's own Sharpe/corr/drawdown bar applies on top of PR-004's internal decision rule, or whether passing PR-004's own gates *is* what counts toward the stop rule — raised in PR-003's protocol (§9 there), and it affects this protocol's own §0 item 2 (which currently just says "PR-004 clearing G0–G2 and not being CONTRADICTED" without addressing the stop rule's separate numeric bar). Needs resolving once, for both strategies together, not twice. |

## 1–4. [Unchanged from v0.1]

Purpose, duration, kill criteria (with item 3's addition above), data source.

## 5. Sizing and vehicle

Unchanged: no risk-policy memo exists yet for this strategy, and building one on a strategy that hasn't passed its own gates would be premature.

## 6. Relationship to the stop rule

Unchanged in substance from v0.1 — no scope ambiguity like PR-003's (PR-004 is unambiguously named in the box) — but see item 7 above: the *numeric bar itself*, not PR-004's presence in the box, is the open question, and it's shared with PR-003.

## Sources

Unchanged from v0.1, plus PR-003 protocol v0.2 (the store-description and kill-criteria harmonization, and §9's cross-cutting stop-rule question, both authored there and referenced here rather than duplicated).
