# PR-005 Pre-registration — v0.8, CONSOLIDATED STANDALONE TEXT (NOT FROZEN)

*Consolidated 2 Oct 2026 from the full draft chain v0.1 → v0.8 (28 Sep 2026) plus the v0.8 real-data continuation (§0.2 of the committed v0.8). Earlier committed text (commit 3ba087c) was the v0.8 change-document only; sections 1–7 and 9–11 existed only in drafts outside the repository. This file states every section in full so the hash, when taken, covers the actual rules. **No rule, threshold, gate or decision rule has been changed in consolidating.** Two sentences had to be written because a version announced a change without supplying the new wording (H0 and H1, marked **[CONSOLIDATION WORDING — owner to confirm]**); every such point, and every stale or contradictory sentence resolved by a later version's explicit supersession, is logged in `docs/pr005/CONSOLIDATION_NOTES.md`. The full recovered drafts are preserved verbatim in `docs/pr005/history/`. **No strategy signal or P&L has been computed; the strategy-blind benchmark computations performed pre-freeze are enumerated in Appendix A.***

## 0. Why this item, and what it is not

PR-004 tests a ~3–4 day mean-reversion rule. The owner's original description was "weeks or months" — trend-following territory, not PR-004's family. Whatever PR-004 shows is evidence about one days-scale rule, not about "swing trading" as a family. This item tests the weeks-to-months instinct on its own terms, **decided and hashed before PR-004's mechanism test runs**, so it cannot become a fallback chosen after seeing PR-004's result.

**This is not a full replication of the published evidence base.** The strongest published results for this idea (Moskowitz, Ooi & Pedersen, "Time Series Momentum," *Journal of Financial Economics*, 2012; Faber's multi-asset extensions) get their diversification benefit from **combining many low-correlation instruments**. This project holds data for US-equity-family tickers only — no bonds, commodities or currencies. **A genuine cross-asset test is out of scope** and would need new data acquisition, a separate explicit decision, never a "descriptive extension" of this item. What is testable is a **single-asset trend-timing overlay**: does a slow, mechanical rule reduce the drawdown of the SPY exposure the owner already holds, without abandoning the upside?

## 1. Question, role and stated prior

Owner goals (recorded 28 Sep 2026): (1) income/wealth, (3) learn/build capability, with (2) diversification/drawdown-reduction secondary. This item is aimed at (2): can a slow, published, unoptimised timing rule reduce the drawdown the owner is already exposed to, at low turnover and low cost-sensitivity? Time-box: within the 21 Nov 2026 review.

**Stated prior, from Faber (2006 working paper), Table 2 — S&P 500 alone, 1900–2005:** CAGR 10.66% (timed) vs 9.75% (buy-and-hold); Sharpe 0.43 vs 0.29; max drawdown 49.98% vs 83.66% (ratio 0.597); time in market 69.77%; 0.67 round-trip trades per year; 63% winning trades. (Table 5, the five-asset-class portfolio, gives different figures — a 54.8% average win rate, winners about 7× the size and 6× the duration of losers — and is **not** the anchor for this single-asset test.) If this replicates on modern data, the honest expectation is **similar-or-better return with materially smaller drawdown** — not an income source. The century-sample drawdown ratio rests partly on pre-1993 crashes this data cannot contain; this test addresses the modern-era pattern only.

## 2. Hypotheses

**H0 — replication.** **[CONSOLIDATION WORDING — owner to confirm]** The rule, as specified in §3, reproduces the pattern of Faber's Table 2 on the data this project holds (from 1993, not 1900): similar-or-better CAGR than buy-and-hold with a materially smaller maximum drawdown, at monthly turnover — tested by the G2 bands in §7.

**H1 — value over free de-risking.** **[CONSOLIDATION WORDING — owner to confirm]** Net of Stage V costs, the timed exposure earns more than any static SPY/cash mix with the same realized drawdown ratio on E3 — the dominance test in §7 (G3).

**H2 — the blend test (the owner's real decision).** Does using this rule as a *partial replacement* for existing SPY exposure improve the risk profile of a portfolio that already holds SPY? Reported, not gated: the answer depends on allocation choices that are the owner's.

## 3. Rules

**Primary, T1 — Faber's 10-month SMA timing rule, on SPY:**
- Monthly signal, computed on the last trading day of each month (month-ends from the `exchange_calendars` XNYS calendar).
- The 10-month SMA at month-end *t* is the simple average of the ten month-end closes up to and **including** month *t* (contemporaneous convention; no lookahead, since signal and SMA use the same close).
- **Long (100% SPY)** when month-end close > the 10-month SMA; **flat (100% cash)** when month-end close < the SMA. Never short, no leverage, no stops, single instrument.
- **Ties** (close equal to the SMA, to a fixed tolerance): hold the current position. (A convention, not in Faber's text.)

**Execution stages.** **Stage R (fidelity):** trade at the signal day's close, no costs, as Faber does — **G2 runs on Stage R.** **Stage V (viability):** trade at the next trading day's open, with costs — **G3 runs on Stage V.**

**Transition-day convention.** On an execution day, the old position is held from the prior close to the open and the new position from the open to the close, in either direction. A dividend whose ex-date falls on an execution day belongs to the holder **at the prior close**. Cash interest accrues on the DFF series with an **act/360** day count.

**Declared secondary variants** (each counts as a trial):
- **V2** — 200-day SMA on *daily* data: daily signal, execution at the next day's open, the same hold-current tie policy, its own Stage R/V split. A genuinely different cadence from T1, disclosed as such (§10), kept as a declared trial rather than its own pre-registration.
- **V3** — T1 applied to QQQ, from QQQ's inception (1999-03-10); for V3 only, E1 is 1999-03 to 2007-12. Every V3 statistic is computed against **QQQ's own** buy-and-hold, never SPY's.

**Excluded, to control multiplicity:** any multi-asset or cross-sectional-momentum variant; any SMA length other than 10-month/200-day; any leverage or short extension; any volatility-targeting overlay; any combination with PR-003 or PR-004.

## 4. Data and integrity (gate G0)

**Primary:** SPY daily open/high/low/close, unadjusted (`Close`, never `Adj Close`), plus dividends, yfinance, 1993-01-29 to 2024-12-31. **Loader:** yfinance's `end` is exclusive; request `end = intended_end + 1 day` and assert the last row matches.

**Total return:** Faber's series are total-return; SPY's price series is not. The primary test uses a total-return reconstruction (price return plus reinvested dividends), stated explicitly; a price-only sensitivity is reported, not gated.

**Asserts:** OHLC consistency (`high ≥ max(open, close, low)`, `low ≤ min(open, close, high)`), `volume ≥ 0`; no NaN in price or dividend series; no forward-filled rows; every XNYS month-end row present; no duplicate dates; ≥ 10 months of data, including the current month, before every evaluation month; |daily return| ≤ 15% sanity bound; dividend rows ≈ 4 per year, positive amounts, no duplicate ex-dates, yield within a plausible annual band.

**Splits:** yfinance's `Close` with `auto_adjust=False` is already split-adjusted (verified across QQQ's 2:1 split on 2000-03-20, which shows a normal daily move). G0 asserts the vendor's split records against each series under test and checks there is no unadjusted split discontinuity; any unadjusted split found is handled by **explicit adjustment only — exclusion is not an option.** The dividend-yield band check is load-bearing for V3, since "QQQ paid no dividends before its split" (checked on one fetched window) is the only evidence against a pre-split scale mismatch.

**DFF plausibility:** the cash-rate series must have an annualized rate between 0% and 5% over E3 (catches percent-vs-fraction errors).

**Independent total-return cross-check — done and passed (§8).** Comparison series: Robert Shiller's Yale dataset (monthly S&P 500 price and dividends, extended past 2023-06 with FRED's SP500), independent of both Yahoo and Alpaca. Because Shiller's price series is a **monthly average** of daily closes (confirmed: Shiller ÷ SPY monthly-average price is a near-constant 10.01, std 0.0115), the comparison is made on a matched monthly-average basis. Tolerance: annual anchor-date differences within 0.5 percentage points **[ASSUMED]**. Failure path: investigate (data first, then reconstruction convention) → at most one fix-and-rerun → an unattributable mismatch halts the item.

## 5. Eras

| Era | Dates | Status |
|---|---|---|
| E1 | 1993-01 to 2007-12 (V3: 1999-03 to 2007-12) | this project's data cannot reach Faber's 1900 start, so Faber's exact figures are not replicated; checked against the Table 2 pattern via G2's bands |
| E2 | 2008-01 to 2015-12 | contains the 2008–09 crisis, the sharpest test of the drawdown claim available |
| E3 | 2016-01 to 2022-12 | value test (G3) — a regime-consistency check, not the sole primary test |
| E4 | 2023-01 to 2024-12 | one look, single hashed script |
| E5 | 2025-01 onward | sealed (`src/seal.py`) |

**Evidential weight:** G2 (E1+E2, containing 2008–09) carries most of the evidence for the drawdown claim. E3 is historically the harder regime for a monthly rule (fast V-shaped crashes), and the **2020 COVID crash sets E3's buy-and-hold maximum drawdown** — so both a typical rule's realized drawdown ratio and the frontier's crash structure in G3 are anchored on that single episode. Era boundaries are kept identical to PR-004's for comparability.

**Disclosure:** the analysts already know 2008–2022 market history in outline (2008–09, 2020, 2022). What is genuinely new evidence is E4 and, later, forward paper trading.

## 6. Costs

**[ASSUMED — for review]** Execution: 5 bps of notional per side. Cash while flat: the daily effective fed funds rate (DFF), not zero — a deliberate, disclosed deviation from Faber's 90-day commercial-paper proxy. The cash leg of the rule and of the G3 frontier must use the **same** DFF series under the **same** act/360 convention. Dividends accrue via the total-return reconstruction while long, never while flat. Costs are modelled in a USD cash account; a UK investor's FX, vehicle and tax frictions on each cash↔equity switch are owner-decision overlays, not modelled here.

## 7. Gates (in order)

- **G0 Data integrity** (§4). Failure stops the item.
- **G1 Implementation.** Two independently coded versions (one vectorised, one an explicit month-by-month loop) must agree exactly on the position series, and on monthly P&L within 1 bp relative tolerance; any mismatch prints the month and both values. **Runtime invariants (binding on both G1 implementations, the G5/E4 script, and the live paper-trading pipeline):** every executable path emits machine-checked action counters (signal flips, fills, rebalance events, with first and last indices), and three identities are hard-asserted: a *w* = 1 mix equals buy-and-hold exactly; a *w* = 0 mix equals all-cash exactly; the inverted-signal control's time in market equals (1 − the primary rule's time in market), checked by asserting the **tie-month count equals 0**. **Soft, investigated-not-auto-failed:** the primary rule's round-trip count over the pooled E1+E2 horizon (23 years; point estimate 0.67 × 23 ≈ 15) should fall within **7–30** **[ASSUMED, wide on purpose]**. Plus the owner's independent manual spot-check of several transition months.
- **G2 (H0), replication with negative control — Stage R, E1+E2 pooled (one concatenated sample).** Bands: maximum-drawdown ratio (rule ÷ buy-and-hold) ≤ **0.65** (Table 2's 0.597 plus a 0.05 leniency for a 31-year modern sample against a 105-year one **[ASSUMED leniency]**); CAGR within **−2 to +4 percentage points** of buy-and-hold's **[ASSUMED]**; time in market **55–85%**, centred on Table 2's 69.77% **[ASSUMED]**. **Negative control:** the inverted-signal rule (long below the SMA, flat above; same tie policy and stage) must have a drawdown ratio on Stage R, E1+E2 pooled, **exceeding 0.65**; if the inverted rule's ratio is ≤ 0.65 (i.e. it clears the bar the primary rule must clear), the verdict is REPLICATION-FAILED. **Failure protocol:** one documented investigation (data first, then implementation), at most one fix-and-rerun; an unattributable miss closes the item.
- **G3 (H1), dominance test — Stage V, E3.** Let *F*(ρ) be the maximum CAGR among a grid of at least 201 **daily-rebalanced** static SPY/cash mixes (cash on the real DFF series) whose maximum-drawdown ratio is ≤ ρ, interpolated linearly between grid points. Let ρ* be the rule's own realized Stage-V-net drawdown ratio on E3. **The rule passes iff its realized Stage-V-net CAGR exceeds *F*(ρ*) by at least 5 bp** (0.05 percentage points of annualized CAGR). Daily-rebalanced mixes were checked against never-rebalanced and monthly-rebalanced families on real data and dominate at every ratio (Appendix A), so the frontier is daily-only. The 5 bp margin exceeds 5× the maximum off-grid leak measured across all three families at the real rate (0.0433%). **Time-in-market floor:** below 40% the gate is not evaluated and the result is labelled INCONCLUSIVE **[ASSUMED; a labelling convention only — the frontier already excludes an always-flat rule structurally]**. The frontier-generating script, the grid specification and the DFF pull are frozen and hashed with this document (§8).
- **G4 Economic report (not gated):** CAGR, vol, Sharpe, max drawdown, round trips, time invested vs in cash, year-by-year table; V2 and V3 alongside; price-only vs total-return sensitivity; the **boundary-sensitivity list** (every evaluation month with |month-end close − SMA| within 1% of price, and the resulting position); the gap between the naive linear static-mix line and the empirical frontier *F*; Table 2's load-bearing figures reproduced verbatim; and the **blend test (H2)** — a portfolio of X% timed-SPY and (100 − X)% plain SPY at X = 25/50/100%, against 100% plain SPY.
- **G5 (E4, once):** a single named, hashed script; report only.

**Verdict categories:** **REPLICATION-FAILED** (any G2 miss, including the negative control, regardless of G3) / **value-CONFIRMED** (G2 passes and the dominance test passes) / **value-CONTRADICTED** (G2 passes and the dominance test fails) / **INCONCLUSIVE** (below the 40% time-in-market floor, or a G0/G1 issue). "value-" signals that G2's replication result is separate: a rule can replicate the historical pattern and still be value-contradicted in this window.

**Decision rule:** proceed to forward paper trading **iff the verdict is value-CONFIRMED.** REPLICATION-FAILED, value-CONTRADICTED and INCONCLUSIVE all block.

## 8. Conditions of freeze, and the freeze ceremony

**Status as of the v0.8 continuation (28 Sep 2026):**
1. Real DFF pull for E3 — **done** (CalcFi/DataHub mirror of FRED DFF, CC-BY-4.0; mean 1.074%/yr, range 0.04–4.33%/yr; plausibility assert satisfied).
2. Independent total-return cross-check — **done, passed** after one protocol-permitted fix (monthly-average basis): correlation 0.998, max annual anchor difference 0.344% against the 0.5% bound, CAGR difference 0.003%.
3. Paper-trading protocol — drafted (§8a); **needs the owner's review.**
4. Stop-rule interaction — **settled by the owner:** PR-005 is outside the PR-003/PR-004 stop rule's scope in both directions (recorded on the stop-rule Rulebook page, 28 Sep 2026).
5. PR-004 hash-sequencing precondition — hardened, self-tested, and present in PR-004's committed spec (§8 condition 2 there); the guard correctly halts in the current repo state (no `PR005_HASH.txt` yet).
6. Cross-family margin re-derivation at the real rate — **done**; the existing 5 bp margin kept as the conservative choice.
7. Faber anchor redundancy — **both required:** the owner reads back Table 2's load-bearing figures at freeze time; the verbatim rows appear separately in the G4 report.
8. **The owner-executed hash step.** Last-mile protocol: re-run every G0 assert; diff any illustrative figures in this document against final ones; assemble `docs/PR005_MANIFEST.txt` (the SHA-256 of this document, the frontier-generating script, the grid specification and the DFF pull output, one per line) and `docs/PR005_HASH.txt` (the SHA-256 of the manifest only — one 64-character hex string). **The owner personally confirms items 1–7 and computes or appends the digest, dated and initialed. The authoring session does not perform this step.**

## 8a. Paper-trading protocol

- **Purpose:** operational validation only — fill mechanics, pipeline integrity, realized slippage against the assumed 5 bp. Performance is explicitly non-evidential at the rule's expected 0–1 trades a year; no conclusion about H1 may be drawn from the paper window's returns.
- **Duration:** 12 months minimum, or 2 completed round trips, whichever is later. At 0.67 round trips a year, reaching 2 round trips has about a 14.5% chance within 12 months and does not pass 50% until roughly year 3. The owner approves a multi-year protocol knowingly; the risk of quiet abandonment over that span is named as a risk in its own right.
- **Kill criteria (operational only):** realized slippage above 3× the assumed 5 bp on 2 or more fills; any month in which the paper-traded signal diverges from the rule's own computed output; a fill-mechanism failure, meaning an order unfilled within a stated window, or filled at the wrong size or side.
- **Prohibited:** any performance-based continuation or kill decision during the window.
- **Forward use:** the paper window's data becomes a future out-of-sample dataset, evaluated once at a pre-registered point, not continuously monitored for a verdict.

## 9. Multiple testing

Three declared trials (T1, V2, V3); no deflated Sharpe — a single primary endpoint and a closed variant list do the relevant work at this trial count. The excluded-variants list in §3 may never be cited as grounds for adopting a new variant; any cross-asset extension is a new pre-registration, not a sensitivity check on this one.

## 10. Known weak points

1. **Low statistical power by construction:** monthly signals over 15–23 years give few independent observations; a working rule can fail to reach value-CONFIRMED for reasons of power alone.
2. **Single-instrument, not the cross-asset design the published evidence relies on** (§0). The drawdown-reduction claim is the part most likely to survive a single-instrument test; the return-improvement claim is the part least likely to.
3. **Faber's exact published figures cannot be replicated** (data starts in 1993); only the Table 2 pattern is checked.
4. **The 2008–2022 market narrative is not new information** to the analysts (§5).
5. **Same-author risk** on the two G1 implementations, mitigated by the owner's manual spot-check and the runtime invariants.
6. **The total-return reconstruction approximates** a true total-return index; it has been independently cross-checked (§4), and small residual tracking differences are disclosed, not treated as data errors.
7. **V2 is a different cadence** (daily vs monthly), kept as a declared trial.
8. **The drawdown claim's real power lives in G2** (containing 2008), not G3; E3's drawdown structure is set by a single episode (2020).

## 11. Relationship to PR-004

Pre-registered and to be hashed **before PR-004's G2 mechanism session runs**, enforced mechanically by PR-004's own spec (§8.1, condition 2) and its addendum (`docs/PR004_hash_precondition_addendum.md`), including a release valve (`docs/PR005_CLOSURE.txt`) so an abandoned PR-005 cannot deadlock PR-004. This item's design does not depend on PR-004's outcome. PR-004 §8.5 adds a one-way guard: PR-005's admission to paper trading may not cite PR-004's pipeline-only window, and that window discharges none of PR-005's own validation duties.

## 12. Effort

Three to four sessions: (1) data loader, total-return reconstruction, G0; (2) two implementations, G1, runtime invariants, manual spot-check; (3) G2/G3, the blend test, report; (4) E4 script and write-up, if not folded into (3).

## Sources

- Faber, M.T., "A Quantitative Approach to Tactical Asset Allocation," working paper, July 2006 (fetched and read directly: trendfollowing.com/whitepaper/CMT-Simple.pdf) — Table 2 (S&P 500 alone, 1900–2005) is the anchor; Table 5 (five-asset portfolio, 1972–2005) cited only to distinguish it.
- Moskowitz, Ooi & Pedersen, "Time Series Momentum," *Journal of Financial Economics*, 2012 — cross-asset evidence base, out of scope.
- Robert Shiller, Yale online dataset (S&P 500 monthly price and dividends) — independent total-return cross-check.
- FRED DFF via the CalcFi/DataHub mirror (CC-BY-4.0); FRED SP500 for post-2023-06 extension of Shiller.
- QQQ split history (2:1, 2000-03-20), verified by direct fetch.
- This project's `src/seal.py`, `src/market_data_store.py`, and PR-004's spec (shared conventions).

## Appendix A — Strategy-blind computations performed before freeze

None involves the T1/V2/V3 signal or any strategy P&L.

1. Sharpe invariance: Sharpe(*w*·SPY + (1 − *w*)·cash) = Sharpe(SPY) for *w* ∈ {0.15, 0.41, 0.5, 0.7, 0.99} with a time-varying risk-free rate, to machine precision (why a Sharpe-vs-buy-and-hold arm cannot distinguish skill from dilution).
2. E3 buy-and-hold (simple total-return reconstruction): CAGR 11.65%, maximum drawdown −33.70%.
3. Static-mix simulations showing a naive linear benchmark is beatable by zero-skill mixes (v0.3–v0.4), superseded by the empirical frontier.
4. Three-family frontier comparison (daily-, never-, monthly-rebalanced), first at an illustrative 1.3% cash rate and then on real DFF: no exceedance of the daily-only frontier above 1×10⁻⁶ (annualized CAGR fraction) at any ratio. The *w* = 1 identity holds exactly. (An earlier reported 0.22% exception was traced to bugs in the verification code — an off-by-one dropping the first day's return, and a timezone-stripping bug that left the monthly family never rebalancing — both fixed.)
5. Off-grid leak probes on real DFF: daily 0.00865%, never 0.00000%, monthly 0.00000%.
6. Real DFF statistics over E3 (§8 item 1) and the total-return cross-check against Shiller (§4, §8 item 2).
