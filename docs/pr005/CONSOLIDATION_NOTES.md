# PR-005 — consolidation notes (2 Oct 2026)

## What was done

The committed `PR005_SPEC.md` (commit 3ba087c) was the v0.8 change-document plus the real-data continuation (§0.2). Sections 1–7 and 9–12 lived only in drafts outside the repository. The full chain v0.1–v0.7 was recovered verbatim from the drafting chat and is preserved in `docs/pr005/history/`; v0.8 is the previously committed file.

`PR005_SPEC.md` now states every section in full. Base text is v0.1 (full), with every later change-log item applied in order, and later supersessions winning. A keyword check confirms every pinned item from every change log appears in the consolidated text.

## Points for the owner to confirm before freeze

1. **H0 was never rewritten.** v0.3 item 7 says "H0 is now restated to match the corrected, verified prior", but no version supplies the restated text; §2 kept saying "corrected per §0 item 7". The consolidated H0 is new wording, marked **[CONSOLIDATION WORDING — owner to confirm]**.
2. **H1 was never rewritten either.** v0.1's H1 ("improves Sharpe and/or reduces max drawdown") was superseded by the dominance test (v0.3–v0.5), but its text was never updated. The consolidated H1 is new wording that restates G3, also marked.
3. **The tie tolerance value was never pinned.** T1 says "to a fixed tolerance", and the round-1 reviewer flagged the value as a free parameter. v0.8 replaced the price-based tolerance in the inverted-control identity with a tie-month count assert (= 0), but the rule's own tie tolerance still has no number. **Not invented here.**
4. **Inconsistent change-log remark about the 0.05 leniency.** v0.3 item 9 said the "0.05 leniency" was "superseded by item 1's dominance test anyway". That is wrong: the dominance test replaced G3, while the 0.05 leniency lives in G2. v0.4's negative control explicitly uses 0.65 (= 0.60 + 0.05), so the consolidated G2 keeps 0.65 and tags the leniency [ASSUMED].
5. **Split handling: v0.3 items 5 and 6 pulled in different directions.** Item 5 says "explicit adjustment only, never exclusion"; item 6 says the data is already split-adjusted and rescoped the check to dividend completeness and scale. The consolidated G0 does both: it asserts against vendor split records, checks for any unadjusted discontinuity, and requires explicit adjustment if one is found.
6. **Stop-rule interaction follows the owner's ruling.** v0.7 relayed a recommendation to count PR-005 *inside* the stop-rule box; the owner then ruled it *outside*, in both directions (v0.8 §0.2). The consolidated §8 uses the ruling.
7. **The total-return comparison series changed in practice.** v0.8 item 6 pinned "a published S&P 500 total-return index series"; the check actually run used Shiller's dataset on a monthly-average basis, under the protocol's one permitted fix. The consolidated §4 records what was actually done.
8. **A stale weak point was dropped.** v0.3 §9 said "the dominance test … has not itself been red-teamed beyond this round". Rounds 3–6 then red-teamed it, so the line is omitted.
9. **Owner actions still outstanding**, as in v0.8: review the paper-trading protocol (§8a), do the Table 2 read-back, and perform the hash step yourself.
