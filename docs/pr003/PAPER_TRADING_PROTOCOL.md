# PR-003 Paper Trading Protocol — DRAFT v0.2 (NOT ACTIVE)

*29 Sep 2026. Supersedes v0.1. One critical design fix (the paper window would have partially spent the sealed window's value) and a full reframe of the stop-rule analysis. Nothing here has been checked with the owner yet.*

## 0. What changed from v0.1, and why

| # | Change | Verification |
|---|---|---|
| 1 | **CRITICAL — the paper window is now performance-blind on 2025+ dates.** v0.1's purpose ("early performance signal, complementing the sealed one-shot test") and duration trigger ("run until enough paper trades accumulated") both computed and viewed P&L on dates inside the sealed window. I worked through why this is a real problem, not a technicality: the seal protects *one thing* — an untouched evaluation of the frozen strategy on 2025+ data, run once. If the paper engine runs those same frozen rules over 2025+ prices already sitting in the live store, and a person looks at the resulting P&L, that person has seen exactly what the one-shot test exists to produce — just generated through a table labelled "live" instead of "research." **Moving the data to an unsealed store never protected this on its own.** The seal's real boundary is evaluation-blindness, not data location — that was true from the start of this project's seal work; v0.1 just didn't apply it here. Fixed in §1 and §6: P&L on the pre-start segment is computed and hashed, never viewed, until it joins the one-shot evaluation at one pre-registered point; the duration trigger reads trade counts and calendar dates only, never P&L; and the paper engine is explicitly barred from backfilling over the live store's pre-go-live segment — it starts generating signals from go-live forward, nothing earlier. |
| 2 | **§1's motivating claim corrected.** v0.1 argued "13 checks a day means weeks of paper trading can accumulate real evidence." That conflates *check count* with *evidence count* — half-hourly checks within one intraday session are heavily autocorrelated, and the real unit of independent information is closer to trades or days, not checks. The 21 Nov backstop caps any window at roughly 7.5 weeks regardless. Restated in §1: the likely evidential gain from a short window is modest, while the seal cost of getting it wrong is fixed and permanent — that asymmetry is the actual argument for item 1's blind design, not a reason to relax it. |
| 3 | **§0's stop-rule item rewritten** to remove my own stated inclination, add a third option that needs no rule change at all, and include the timeline evidence I could actually check. See §0 item 1 below. |
| 4 | **Store description disambiguated.** v0.1 said "verified 2026-09-25 onward, 61 real sessions" in a way that could read as the store's start date. That was describing the most recent *backfill verification event*, not the store's range. The store's actual combined range (sealed-window migration plus the backfill) is 2025-01-02 through 2026-09-25, matching PR-004's protocol description of the same store — both protocols now describe it the same way. |
| 5 | **HWM ratchet: a bounded, mechanics-only pre-check permitted, distinct from a strategy-verdict backtest.** v0.1 left the ratchet's own drawdown behaviour unestimated. The refusal discipline that blocked the leverage-cap re-run doesn't apply here in the same way: that was a question about the strategy's *edge*; this is a question about whether a *safety overlay* behaves sanely — engineering, not evidence. One read-only mechanics run is permitted: confirm the ratchet scales monotonically, that its halt threshold is actually reachable, and that it produces no blow-ups on historical paths — with an explicit ban on any "would have made X%" narrative entering the record. Separately, the ratchet must simulate **integer MES contracts**, not continuous proportional scaling — a continuous ratchet is partly fictional at this capital size (already flagged: one contract ≈ 3× leverage), and a ratchet action smaller than one contract increment is a no-op the paper engine needs to represent honestly. |
| 6 | **Kill criteria harmonized with PR-004's set**, adding the one PR-003 was missing (a signal-vs-frozen-script divergence check) and fixing the self-contradicting store sentence. |
| 7 | **The connection-string check upgraded** from a one-time test to a startup assert *every session*, plus a store-identity check — a one-time test at go-live would miss a configuration drift discovered three months in. |
| 8 | **Trade-frequency status corrected.** v0.1 said the real count was unknown; P2-130 (already fetched and read this round) does not contain it either — I checked before re-asserting unavailability, rather than repeating the earlier claim on faith. The command in §2 to pull it from Stage V's own JSON stands. |

## 1. Purpose

1. **Operational validation** — does the live pipeline, fill logic, and half-hourly signal check behave as the backtest assumed, on genuinely new data.
2. **A performance readout — but not an early one.** The strategy's realized P&L on the paper window's pre-start (2025+) segment is real evidence, and it will inform the eventual one-shot evaluation. It is computed and cryptographically hashed as it accrues, and **not viewed by anyone until the pre-registered one-shot evaluation point**, at which time it becomes part of that evaluation rather than a separate, earlier look at it. The likely evidential gain from a short window is modest (§0 item 2) — the reason for the blind design is that the seal's cost of getting this wrong is fixed and irreversible, regardless of how much or little the short window would actually have told us.

## 2. Trade frequency — still needs your number

**Action needed**, unchanged from v0.1:
```bash
.venv/bin/python -c "
import json
d = json.load(open('DATA/pr003/stage_v_2026-09-27T122802Z.json'))
print({k: v for k, v in d.items() if 'trade' in k.lower() or 'fill' in k.lower() or 'n_' in k.lower()})
"
```
I checked P2-130's own content before re-stating this as unknown — it isn't there either. Until the real figure exists, every frequency claim here stays tagged **[EST]**.

## 3. Data source

`DATA/live_market_data.db`, combined range **2025-01-02 through 2026-09-25** (seal-migration rows plus the manual backfill, verified against the exchange calendar for the backfilled segment specifically — that calendar check confirms presence, not content, and doesn't extend to the migrated segment). The paper engine must never read the research store. Per item 1, it must also never backfill a strategy run over the live store's pre-go-live segment — it only generates new signals from go-live forward. **P&L on the pre-start segment**, if computed for the eventual one-shot evaluation, is a separate, explicitly blind computation — not something the paper engine does as part of normal operation.

## 4. Sizing and vehicle — parameterized, pending §0 item 2

Unchanged from v0.1: written to run under either sizing column or vehicle, since neither is decided. One addition, directly relevant to §0 item 1 below: **the sizing column choice changes the stop-rule drawdown arithmetic**, not just position sizes — at the behaviourally-adjusted default (~25% of full sizing), Stage V's −39.5% scales to about −9.9%, comfortably inside the stop rule's 15% bar; at the stated-tolerance column (~51%), it scales to about −20.1%, which does not clear it. The sizing decision and the stop-rule question aren't just related — for this specific tension, the sizing choice can determine the answer.

## 5. HWM ratchet

Implemented from day one (unchanged from v0.1), now with the mechanics-only pre-check and integer-contract discretization from §0 item 5. The ratchet's spec still needs: update frequency, floor, recovery behaviour, and exactly what the equity curve it reacts to includes (fees, interest, or price only).

## 6. Duration

**Trade-count and calendar only — never performance.** Run until either (a) enough paper trades have accumulated on the go-live-forward segment to say something about operational stability (the real threshold depends on §2's still-unknown rate), or (b) the 21 Nov 2026 project review, whichever comes first. The pre-start (2025+) segment's P&L is tracked separately, blindly, per §1, and plays no role in this trigger.

## 7. Kill criteria

**Operational (halt immediately, investigate, do not resume without a fix) — harmonized with PR-004's set:**
- A seal violation of any kind, or the paper engine backfilling over the pre-go-live segment (§0 item 1) — both should be structurally impossible; a startup assert plus a store-identity check confirms this every session, not just once.
- A stray/non-flat position detected at any session close (the half-day-close bug class, commit 5555c5e).
- A data feed gap or unexpected fetch failure.
- Any day where the paper-traded signal diverges from what the frozen rule's own script computes for that day — a direct check against the frozen spec.
- A fill materially (3× the assumed cost model) inconsistent with the gap-tail work's execution-haircut figures.

**Risk-based, per §0 item 1 of v0.1 (still unreconciled into one rule, unchanged pending your decision):**
- The HWM ratchet, continuously live.
- A halt at some multiple of expected max drawdown — the exact multiple and what "expected" means here still need pinning.
- A discrete review trigger at N months without a new equity high — N still unset.

## 8. Relationship to the stop rule — reframed, no assistant recommendation

**Timeline, checked rather than asserted:**
- The stop rule (P2-129/DAC-WORTH-01) is a formally adopted, dated Rulebook entry: 26 Sep 2026.
- Stage V ran 27 Sep 2026, 12:28 UTC — a full calendar day later.
- Stage V, as actually run, has exactly three gates: null-alpha, correlation, expectancy. No max-drawdown gate exists in the run that produced −39.5%. A gate cannot be removed from a script after it has already run and produced a number — so whatever excluded max-DD from Stage V's own gates had to be decided before 12:28 UTC, before anyone could have known what number it would produce. **This is solid evidence that Stage V's own gate design was pre-committed, not a reaction to seeing a bad number.**
- That finding is about Stage V's *own* internal gates. It does not by itself resolve the *separate* stop rule, which predates Stage V by a day and is a different rule object. I searched for a standalone, dated Rulebook entry for the "risk expressed relative to volatility" amendment that Stage V's own logging references — there isn't one. It exists only as prose inside P2-130's running log, never promoted to its own governed Rulebook row with its own adoption date, unlike the stop rule itself. That's worth knowing when weighing how much authority it should carry against the stop rule's plain text.

**Three options, not two — the third needs no rule change:**
1. **Clarify, don't amend:** if the 15% bar was always intended as a sized criterion (i.e., "15% at whatever allocation is actually deployed," not "the strategy's own full-leverage historical max drawdown"), then nothing needs changing — only stating what the bar always meant. The sizing arithmetic in §4 makes this plausible at the behaviourally-adjusted column and implausible at the stated-tolerance column, so this option itself depends on the sizing decision.
2. **Amend prospectively**, matching the process already used to scope PR-005 out of the stop rule — a dated, logged change to the stop rule's own text.
3. **Treat the 15% bar as literal and currently unmet** — PR-003 would not count as having passed the stop rule until sizing brings realized drawdown under it, regardless of what Stage V's own three gates concluded.

I am not recommending between these. Options 1 and 3 both require no amendment; only option 2 does, and option 2 is the one I'd be voicing an inclination toward if I picked — which is exactly why I'm not picking.

**One more honest sentence, replacing v0.1's framing:** Stage V's gates, as they were actually run, did not include drawdown — the gates that were applied were satisfied, and separately, the criterion that later turned out to be the hard one was already excluded before the run. Both things are true; stating only the first is a slant.

## 9. A structural question this raised, bigger than PR-003 alone

The stop rule's pass definition ("Sharpe ≥ 0.5, corr ≤ 0.3, max DD ≤ 15%") is written as its own, separate test — not explicitly tied to whatever each strategy's own internal research gates decide. Faber's own published Sharpe (PR-005's success criterion) is 0.43, under 0.5, meaning a *perfect replication of the published result* would fail this specific bar if it applied. PR-005 is already settled as outside the stop rule's scope, so this doesn't bind there — but PR-004 is inside the box, and its own decision rule (the dominance test, not a Sharpe threshold) may not obviously map onto "Sharpe ≥ 0.5" either. Whether the stop rule's own numeric bar is meant to apply on top of each strategy's own internal gates, or whether passing a strategy's own gates *is* what counts as passing the stop rule, isn't stated anywhere I can find. This needs resolving before PR-004 paper trading could start, independent of anything specific to PR-003.

## Sources

Unchanged from v0.1, plus: P2-130's full content (checked directly this round for the amendment's status and the trade-count question), the Rulebook database's schema (checked for a standalone amendment entry — none found).
