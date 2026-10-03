# PR-003 — Per-strategy stop-bar mapping (v0.1, NOT FROZEN)

*3 Oct 2026. Decisions D1–D3 and the pre-write package were taken by the owner after a D-A-C pass and two external cold-fork rounds; this file records them. It is the PR-003 counterpart of `docs/pr004/PR004_SPEC.md` §8.4 and §8.6, and it must be consistent with them. **No computation on the sealed window (2025+) has been performed or viewed.** Nothing here has been hashed. The owner performs the freeze step (§12); the authoring session does not.*

## 1. The bar and the readings pinned

Stop rule (Rulebook, adopted 26 Sep 2026, text unedited): *Pass = daily Sharpe ≥ 0.5 after costs in-sample AND out-of-sample (OOS ≥ half of IS), corr(SPY) ≤ 0.3, max DD ≤ 15%.* If both PR-003 and PR-004 fail, active-strategy research stops.

| Phrase | Reading used for PR-003 | Basis |
|---|---|---|
| "daily Sharpe" | Mean/std of daily net returns × √252, over **all sessions** (flat days included), no risk-free subtraction | The project's refine report uses "daily Sharpe" for the unannualized per-day figure, which no strategy could reach (0.5/day ≈ 7.9 annualized), so annualized is forced. Stage V's 0.80 is on this basis (verified in `src/pr003_stage_v.py`: daily returns from equity by day, ×√252, rf = 0). |
| "OOS ≥ half of IS" | Sharpe ratio: Sharpe_OOS ≥ 0.5 × Sharpe_IS (walk-forward efficiency, as in `src/evaluation/robustness.py`) | With Sharpe_IS = 0.80 the ratio leg needs 0.40, so the absolute 0.5 leg binds. |
| "corr(SPY) ≤ 0.3" | The rule's literal one-sided ≤ 0.3, daily net returns against SPY daily buy-and-hold returns, pooled IS + OOS | Stage V's own gate was two-sided; −0.08 passes both. Disclosure: PR-003 is flat overnight, so its measured correlation is diluted by zero overnight exposure and is not the account's beta. |
| "after costs" | The Stage V MES-equivalent cost model, **frozen**: $1.12 per contract per side [ASSUMED: $0.50 commission + exchange, $0.625 slippage]. The OOS break-even cost is reported (IS break-even was $2.62). | Same model in IS and OOS keeps the legs comparable. |
| "max DD ≤ 15%" | See §5 (owner override). | |

**Risk-free pin.** rf = 0 is binding: PR-003's P&L is a flat-overnight overlay, which is the same futures-style convention PR-004 uses (Mode F), and it is the documented convention in Stage V and the SPY benchmark comparison. Both sensitivities are printed in the verdict report, not in a footnote: mean DFF was 1.97%/yr over 2016–2024 and 3.97%/yr over 2025-01..2026-09, which would cost 0.14 and 0.28 of Sharpe at 14.4% vol if subtracted. Subtracting rf would also make the Sharpe leg depend on sizing (the penalty is rf/(s·σ), ×4 at 25% sizing), whereas the other legs are scale-free.

**Ambiguity rule.** Where the record supports two readings, the reading with the strongest pre-run documentation is used and outcome-informedness is disclosed. Neither a default toward pass nor a default toward fail is used.

## 2. Windows (decision D1)

- **In-sample (IS):** 2016-01-04 to 2024-12-31, the Stage V window. Recorded: Sharpe 0.80, vol 14.4%, CAGR +11.0%, max DD −39.5% at full sizing, MDD/vol 2.75, corr −0.08. Those are the only IS inputs, and they are already in the record.
- **Out-of-sample (OOS), binding:** the sealed window, 2025-01-02 to the cutoff in §3, read **once**. It is the only data the paper's authors never saw.
- **2023–24** may be reported only as labelled non-evidentiary history. It cannot be the stop-bar OOS leg: that window was superseded as a hold-out on 27 Sep, has been analysed repeatedly, lies partly inside the paper's own sample, and its returns are on record.
- **Approval clause.** Approval to open the sealed window is conditional on the mechanical checks in Appendix B only. It must be signed and dated before any stop-bar arithmetic on 2023–24 is produced (§12).

## 3. Cutoff (decision D3)

The OOS window ends at the **last complete session on or before 2026-09-30** (21 months). It is fixed now, with no extension clause. 26 Sep + 8 weeks = 21 Nov, the review date, and a 24-month window would reduce the Sharpe standard error by only ~6.5%.

**Data rule inside the window (frozen before hashing).** A complete session has every bar the XNYS calendar expects, with early closes honoured. Before any P&L is computed, incomplete sessions are re-fetched once from the same vendor feed; this step looks at bar counts only. A session still incomplete after repair is treated as flat (zero return) and listed. If more than 2 sessions remain incomplete **[ASSUMED threshold — owner to review]**, the read is INVALID (§6).

## 4. Pre-read pins that do not depend on the outcome

- Code frozen at a tagged commit; cost model frozen (§1); the paper engine's connection string points at the live store only, never the research store.
- The sizing decision (§5) is hashed before the read and cannot be changed afterwards.
- Store continuity through the cutoff is verified, metadata only (session counts against the exchange calendar, no P&L).

## 5. Max-drawdown leg (decision D2): a logged deliberate override

**What is adopted.** For PR-003 the 15% leg is evaluated at the **deployed sizing**: the sizing fraction and vehicle fixed in `docs/pr003/SIZING_DECISION.md` (template in Appendix A) and hashed before the sealed read. The measured quantity is the maximum drawdown, over the pooled window 2016-01-04 to the cutoff, of the strategy's daily net returns re-run through the deployed-sizing model, with integer-contract granularity if the vehicle is MES. The linear scaling f × DD_full is printed as a cross-check. PASS requires ≤ 15%.

**This is an override, not a clarification.** The stop rule's text was not edited. The pre-run record does not support calling this a clarification. It is a deliberate choice to evaluate the leg at a sizing set after Stage V.

**Counterfactual, stated plainly.** Under every referent fixed before the artifact's own drawdown ratio was known, PR-003's drawdown leg FAILS:

| Referent | Realized max DD |
|---|---|
| Unsized / full (Stage V) | −39.5% |
| The 10% vol target of the 26–27 Sep record (Stage V's own rescale, `max_dd_at_10pct_vol`) | −27.5% |
| Freeze mapping, 15% tolerance → 8.6% vol using the paper's 1.75 ratio, realized at the artifact's 2.75 | −23.6% |
| Freeze mapping, 10% tolerance → 5.7% vol | −15.7% |

Even the paper's own ratio gives 17.5% at a 10% target. The pre-run record already contains the warning that the test "will almost certainly fail on drawdown even if we reproduce the paper exactly."

**Outcome-informedness, disclosed.** The choice is made with Stage V known (−39.5%, MDD/vol 2.75). The memo's arithmetic is also known: 25% of full sizing ≈ −9.9%, 51% ≈ −20.1%, break-even 37.9% of full sizing (vol 5.45%). The owner's drawdown tolerance was first received on 28 Sep, after Stage V's 27 Sep 12:28 UTC run, so no pre-run tolerance declaration exists. External review (two rounds) found the literal reading to be invariant across the pre-run-fixed referents and judged this override acceptable only as a logged deliberate override with an outcome-independent rationale.

**Vehicle feasibility (a fact bearing on whether this leg can pass).** At the paper account size in the owner's private risk memo, one MES contract is ≈ 3× effective leverage against ≈ 4× for full sizing (index level and FX assumed in the memo). That is ≈ 74% of full sizing, i.e. a scaled drawdown of ≈ −29%. The pass region (≤ 37.9% of full sizing) is reachable only with a vehicle that allows finer sizing. The vehicle decision therefore determines this leg's verdict, and it must be made on its own merits and hashed before the read.

**Owner rationale (outcome-independent grounds)** — **[OWNER: confirm or replace, in your own words, before hashing]**. Candidate grounds drawn from pre-run text: (a) the 27 Sep freeze separates research gates from deployment, with sizing decided last from account-level tolerance; (b) the stop rule's own live-trading clause ties 15% to "account drawdown".

**Safeguards.**
1. The sizing fraction and vehicle are decided and hashed before the seal opens; they are not tunable after the read.
2. The override and the sizing decision are committed together and are dated.
3. Prospective symmetric rule: for any future strategy evaluated against this stop bar, the drawdown leg is evaluated at a sizing fixed and hashed before its OOS read.
4. The counterfactual above (FAIL under pre-run-fixed referents) is carried into the verdict report.
5. **Cost to the device, stated:** using this override spends the commitment device's credibility. A later loosening is harder to defend after this one.

## 6. The read

One read serves three evaluations: the stop-bar legs, PR-003's own sealed test, and the pre-registered bucket-contrast check. **The stop-bar legs decide.** The other two are recorded non-decisionally for the stop bar. The read runs once, at the frozen commit, on the frozen cost model and data, and its output is hashed.

**Procedural-failure path.** If a hash does not reconcile or a data gap exceeds §3, the **frozen repair procedure** runs once: (1) re-run the same code at the same tagged commit on the same input snapshot and compare the output hash; (2) if inputs differ from the pre-read manifest, restore them from the pre-read backup; (3) no code, parameter or cost edits are allowed. If the result still cannot be reconciled, the read is **INVALID: a definitional stop-bar FAIL, with the cause recorded** (implementation, data, indeterminate). The item can then only be re-attempted as a new hashed item carrying this history.

## 7. Error rates, acknowledged in writing before the read

A stop rule built on ~21 months of OOS data and a 15-year-old Sharpe bar is noisy. These figures are standard-error arithmetic, with no strategy data involved.

- **PR-003's OOS leg (Sharpe ≥ 0.5, 21 months).** A true Sharpe of 0.8 passes ~64% of the time (so it is stopped ~36% of the time); true 1.0 ~71%; true 0.5 ~50%; true 0 ~25%.
- **PR-004's legs (IS 21 years, OOS 7 years, all three Sharpe legs).** In-market Sharpe 1.0 passes ~3%; 2.0 ~24%; 2.2 ~30%; 3.0 ~58%. PR-004's own legs, not PR-003's drawdown leg, dominate the joint device's false-stop channel.
- The owner accepts that the device may stop a sound program and continue an unsound one.

## 8. Ordering and consequences

PR-003's stop-bar verdict is recorded as PASS, FAIL or INVALID (= definitional FAIL) at the read. Per PR-004 spec §8.6, per-strategy consequences execute at joint-verdict determination, the joint stop fires only when both verdicts exist and both are FAIL, and "research stops" has the six clauses (a)–(f) written there.

**OPEN before the read:** the consequence of a PR-003-only stop-bar FAIL (PR-004 not failing). Under the stop rule's text it triggers nothing. PR-004's version of this gap was closed in spec §8.2. PR-003's has not been decided.

## 9. Wording issues logged (not amended)

- "daily Sharpe" in the stop rule collides with the project's per-day usage; the annualized reading is forced (§1).
- The correlation leg is structurally satisfied by any low-exposure rule (corr ≈ √f) and is diluted for PR-003 by flat nights; it carries little evidential weight.

## 10. Successor-device clause

After any stop, research restarts only through a new pre-registration with a new device, adopted when no verdict is pending. No resumption by reinterpretation.

## 11. Provenance log (facts only)

- **26 Sep 2026:** the stop rule is adopted in the same session that later read the paper; the record states "The rule was set before either of us had read the paper." On reading the paper, the 15% guardrail was found to fail the paper's own drawdown at the project's 10% target (17.5% scaled; 18.7% for 2016–22).
- **26–27 Sep:** a structured decision cycle named the 15% a "deployment rule bundled into a research gate", replaced PR-003's fixed 15% research gate with a volatility-relative guardrail (no number frozen), and stated that the time-box and stop-if-both-fail rule "stand unchanged". Deployment: tolerance sets the vol target; live halt at 1.5× expected max DD.
- **27 Sep:** the first PR-003 code commit is `f1a4fd6` at 07:22 UTC; Stage R's result file is 07:45 UTC; Stage V ran at 12:28 UTC with max DD −39.5%. The amendment prose lives only in the tracker, with no dated Rulebook row. The stop rule's own text was never edited.
- **28 Sep:** the owner's drawdown tolerance and capital were first received (held privately).
- **3 Oct 2026:** D1–D3 and the package were decided. The record shows the pre-run-fixed referents fail, and the owner chose the override in §5.

## 12. Owner actions before the read

1. Complete the override rationale in §5 (owner's words).
2. Decide sizing fraction and vehicle; commit `docs/pr003/SIZING_DECISION.md` (Appendix A).
3. Decide the PR-003-only FAIL consequence (§8).
4. Locate or pin PR-003's own sealed-test pass criteria; they are not restated here and the stop-bar legs do not depend on them.
5. Review the [ASSUMED] threshold in §3.
6. Assemble `docs/pr003/PRE_READ_MANIFEST.txt`: SHA-256 of this file, the sizing decision, the frozen cost-model config, the tagged commit id, and the live-store segment's bar hash.
7. Sign and date the acknowledgement (§7) and the approval clause (§2) **before any 2023–24 stop-bar arithmetic**:
   - Acknowledged: ______ date: ______
   - Approval (mechanical checks only): ______ date: ______

## Appendix A — `docs/pr003/SIZING_DECISION.md` template (no personal figures)

```
Sizing fraction of full system sizing: [OWNER]
Vehicle: [OWNER: MES | spread bet | other]
Granularity rule: [integer MES contracts | fractional]
Decided on: [date]   Hash recorded in PRE_READ_MANIFEST.txt: [yes]
Capital used for granularity: held in the owner's private memo (not committed)
```

## Appendix B — mechanical checks (approval is gated on these only)

1. Store continuity through the cutoff: complete sessions match the exchange calendar (metadata only).
2. The tagged commit, cost-model config, sizing decision and this file are all hashed in the manifest.
3. The paper engine's connection string is the live store; the research store is untouched.
4. No stop-bar arithmetic on 2023–24 exists yet.
