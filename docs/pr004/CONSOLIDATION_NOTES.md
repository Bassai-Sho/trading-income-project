# PR-004 — consolidation notes (2 Oct 2026)

## What was done

The committed `PR004_SPEC.md` (commit 3ba087c) was the v0.6 change-document. Its §1–5 and §9 said "unchanged from v0.4… carried forward verbatim", but no earlier version had ever been committed, so the hashed file would not have contained the rules it pre-registers. The full chain was recovered verbatim from the drafting chat and is preserved in `docs/pr004/history/` (v0.1, v0.2, v0.3, v0.3.1, v0.4, v0.5; v0.6 is the previously committed file).

**Fidelity check:** applying v0.5.1's recorded edits to the recovered v0.5 reproduces the committed v0.6 text exactly, apart from v0.6's own recorded changes. That confirms the recovered chain matches what was committed.

`PR004_SPEC.md` now states every section in full. Base texts are v0.2 (full rules, data, costs) and v0.3 (full G2 equation, eras, gates, effort), with every later change-log item applied. A keyword check confirms every pinned item from every change log appears in the consolidated text.

## Points for the owner to confirm before freeze

1. **RSI(2) seeding was silently dropped.** v0.1 pinned it: "Wilder smoothing on daily closes, seeded with the simple average of the first two changes." From v0.2 onward the spec says only "Wilder smoothing." No version records a decision to drop it. Two independent G1 implementations need a seeding convention, and an unpinned one is exactly what makes them disagree (or agree on the same unvalidated guess). **Not restored, because restoring it changes the spec. The owner decides whether to re-pin it.**
2. **Misquote corrected.** v0.3's G1b quote said E1's annualised return was "not given". The archived source gives 3.91% and max drawdown −6.48% (CAR/MDD 0.60). The consolidated quote uses the archived figures. Both are descriptive only, so no gate changes.
3. **Stale disclosure replaced.** v0.5.1 added to §9: "G1b replicates Marwood's S&P 500 Index test using SPY … as the nearest available proxy." v0.6 §0.1 then established the source test *was* on SPY, but left that sentence in place, so the committed v0.6 contradicted itself. The consolidated §10 item 7 uses v0.6's reading.
4. **Sign-off scope merged.** v0.3/v0.3.1's condition 1 was the Sharpe-vs-buy-and-hold hurdle becoming a report. v0.4 rewrote condition 1 around the mechanism-gate wording and stopped mentioning the hurdle explicitly. The consolidated §8 condition 1 states that the sign-off covers both. Confirm that matches your intent.
5. **No owner-executed hash step in PR-004's spec.** The handover says you must personally perform PR-004's final hash. PR-005 v0.8 wrote that into its own spec (§8 item 8), but PR-004's spec only says "hash recorded at commit time". **Not added; the owner decides whether to mirror PR-005's wording.**
6. **A placeholder would freeze as a placeholder.** Mode S's "2.0% is a placeholder until the provider's published terms are sourced". Mode S is report-only, so this doesn't affect any gate, but freezing a placeholder is worth a deliberate yes.
7. **Condition 1 is still "awaiting answer"**, as in v0.6.
