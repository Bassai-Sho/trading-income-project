# PR-004 Pre-registration — DRAFT v0.6 (NOT FROZEN)

*28 Sep 2026. Supersedes v0.5.1. Document review is closed (see v0.5.1 §0); this revision comes from actually archiving the source — condition 3 of freeze — which surfaced one further correction no red-team round found, because neither reviewer could browse. **Nothing has been run.***

## 0.1 Correction found while archiving (new, not from red-team review)

Archiving the source meant reading the full page, including its comment thread, for the first time — and it changes the symbol determination from v0.5.1:

> **Jay Eiser** (comment, 5 May 2019): *"You seem to ignore that Connors did this testing on a fixed set of ETF's. Stocks were not involved in the Connors testing. As you can see your testing on the SPY matched his results exactly."*
> **Joe Marwood** (reply, 14 May 2019): *"I have tested on ETFs various strategies by Connors. Some of them are quite good such as RSI4."*

A reader states plainly that Test One ran on SPY; the author's reply does not correct this and confirms testing "on ETFs" generally. **Weight of evidence: Marwood's Test One was run on SPY, via Norgate — not the raw S&P 500 index**, reversing v0.5.1's item 2. This actually simplifies G1b: it is a direct SPY-to-SPY comparison, not a proxy for an index. It does **not** resolve the two things that were always the real open questions — whether the Norgate SPY series was dividend-adjusted, and whether sizing was fixed-notional or compounded — so **return and max drawdown stay demoted to descriptive, unchanged from v0.5.1**, now for the accurate reason (an ETF's dividend/adjustment treatment is a real, unresolved choice; an index's would not have been). The source and its full relevant text are archived at `docs/sources/marwood_rsi2_2016.md` (local copy, git-tracked) and, pending the owner's one click at `web.archive.org/save/<url>`, on the Wayback Machine (condition 3, partially complete — see §8).

## 0. What changed, and why

| # | Change | Verification |
|---|---|---|
| 1 | **§8 and §6's fallback contradicted each other — fixed.** §8 said "proceed iff H0 holds and the verdict is not CONTRADICTED"; the b̂12≥0 fallback in §6 separately required "the owner's direct review" in that same case. Since that fallback's verdict is INCONCLUSIVE (not CONTRADICTED), §8's plain iff would have auto-proceeded on H0 alone, skipping the review §6 demanded. §8 now states the exception explicitly. | Reread both sentences side by side in my own document — the contradiction was real, not a matter of interpretation. |
| 2 | **G1b's instrument, superseded by §0.1 above:** believed at this point to be SPY directly (via Norgate), not the raw index. **Position sizing (fixed-notional vs. compounded) and dividend/adjustment treatment are genuinely not stated** anywhere in the archived text. Annualised return and max drawdown both depend on these; trade count, win rate and average hold time do not (they're just counts of episodes). Fix, unchanged in effect from the earlier reasoning: **return and max drawdown are demoted to descriptive (reported, not gated)**; trades/win-rate/hold remain the actual gated bands; "sizing or adjustment mismatch" is in the failure-protocol's investigation checklist. | Superseded by the comment-thread evidence in §0.1 — see there for the correction and its source. |
| 3 | **The sign-off overclaimed.** It said a dead effect "is blocked" — true only if E3 is measured precisely enough. Verified: with b̂12=−8, se12=2, a dead effect (b̂3=0) triggers CONTRADICTED only if se3 < 4.43; measured less precisely, it reads INCONCLUSIVE and does not block. Fixed with a precision-conditional clause, pointing to the printed MDE as the thing that tells you in advance whether this will bind. | Recomputed the threshold directly (below). |
| 4 | **A dropped sentence, restored.** v0.3.1 added an interpretation sentence for the INCONCLUSIVE branch but never added the symmetric one for CONTRADICTED — whether the decayed effect is still individually significant (U3 < 0) was left unstated, risking "decayed" collapsing into "dead" in the prose. Added. | Reread v0.3.1 §6 directly — confirmed only one of the two symmetric sentences existed. |
| 5 | **A dangling reference removed.** v0.4's change-log pointed to "the tool output above this document" — true only inside this conversation; a frozen, standalone file has no such thing above it. The actual numbers are already in the text; the pointer is deleted. | Reread my own v0.4 table cell. (One related claim — that the version header said "Supersedes v0.3" — was checked and found **false**: it already correctly said v0.3.1. Not everything flagged this round was a real defect.) |

**Precision check on item 3:** with b̂12=−8, se12=2 (illustrative), CONTRADICTED on a dead effect requires `8/√(se3²+4) > 1.645`, i.e. `se3 < 4.43`. Above that, a genuinely dead mechanism reads INCONCLUSIVE, not CONTRADICTED — the printed MDE is what tells the reader in advance whether this threshold will bind.

**On stopping here:** four rounds have each found a real, independently-verified defect — a wrong primary rule, a wrong statistical test, then increasingly small spec gaps. This round's findings are five one-line edits, and one of the five claims made didn't hold up under checking. That pattern — real findings shrinking in size each round, alongside a first false positive — is itself evidence the document is close to its practical limit. No further red-team round is planned; the next check on this spec should be the owner's own read of §8, not another model pass.

## 1–5. [Unchanged from v0.4]

Carried forward verbatim.

## 6. Costs and the G2 equation — one added sentence

Unchanged from v0.4, plus: **for the CONTRADICTED branch, the report states whether U3 < 0** — i.e. whether E3's effect, though significantly weaker than published, is still individually significant — so "decayed" is never silently read as "dead."

## 7–8. Gates, freeze conditions, and the decision rule — reconciled

Unchanged from v0.4 except:

**Decision rule (§8, reconciled):** proceed to forward paper trading iff H0 holds and the H1 verdict is not CONTRADICTED — **except** in §6's b̂12≥0 fallback (undefined published-era comparison), where H0 alone is not sufficient: that branch additionally requires the owner's documented review of G2's descriptive output before proceeding.

**Owner sign-off text (§8, condition 1), final wording:** *"The mechanism gate blocks proceeding to paper trading if E3's effect is significantly weaker than the published era's. This includes a completely dead effect, but only if it's measured precisely enough — the printed MDE tells you in advance whether that will hold. It also includes a real, still-statistically-significant effect that has merely halved or more — and whether that halved-but-real case is blocked also depends on how precisely it is measured, not only on its size; the MDE is computed from 1995–2015 data as a preview of E3's expected precision, so treat it as a good-faith estimate, not a guarantee. If you would rather proceed on a decayed-but-real effect regardless of this gate, that is a legitimate choice, and it should be recorded here as a stated exception, not left as a silent gap. Separately: if the published-era comparison itself turns out not to have the expected sign, the gate cannot judge decay at all, and the decision falls to replication (H0) plus your own direct review of the descriptive results — not an automatic proceed."* — **awaiting answer.**

**G1b's bands (§7), corrected scope:** trades and win-rate bands are gated as before (E1: 45–53 trades, 78–89% win rate; E2: 26–34 trades, 72–88% win rate; both eras' average hold 3–5 days). **Annualised return and max drawdown are reported, not gated**, given the unverified sizing convention (§0.1). The failure-protocol checklist now includes "sizing or adjustment mismatch" alongside data and implementation causes.

**Condition 2** (PR-005 hashed before the G2 session runs): **enforced as a hard precondition, not a reminder.** Before this G2 session runs, its setup step must verify `docs/PR005_HASH.txt` exists, is non-empty, and — case-insensitively — is exactly 64 hexadecimal characters, **or** that a logged, non-empty, dated closure record exists at `docs/PR005_CLOSURE.txt`. If neither holds, this session halts and does not run. Implementation and a self-test live in `docs/PR004_hash_precondition_addendum.md`; the check itself is not decorative — it was run against both a valid and an invalid case before being relied on. A human checklist line duplicates this as a second enforcement layer: *before running G2, confirm `docs/PR005_HASH.txt` exists and is well-formed, or that `docs/PR005_CLOSURE.txt` exists and is dated.*
**Condition 3** (source archived before G1b is coded): **done, both halves.** Local copy at `docs/sources/marwood_rsi2_2016.md`, containing the verbatim load-bearing quotes and the comment-thread correction in §0.1, git-tracked. Wayback snapshot confirmed live at `web.archive.org/web/20260928211054/https://stocksoftresearch.com/rsi-2-trading-strategy/` — the owner triggered the save directly, and the resulting capture was cross-checked line-by-line against the local copy with no discrepancy (28 Sep 2026). G1b may be coded now.

## 9. [Unchanged from v0.4, plus one disclosure]

Added to the known-weak-points list: G1b replicates Marwood's S&P 500 Index test using SPY unadjusted price as the nearest available proxy (§0 item 2) — a stated, accepted approximation, not a hidden one. **Clarification on G1b's scope:** only the gated bands (trades, win rate, hold time) trigger the failure protocol on a miss. A descriptive statistic (return, max drawdown) falling outside its quoted range is disclosed, with an attribution attempt noted in the report, but does not by itself close the item.

## Sources

Unchanged from v0.4.
