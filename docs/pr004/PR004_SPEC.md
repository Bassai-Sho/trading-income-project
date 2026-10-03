# PR-004 Pre-registration — v0.7 (NOT FROZEN)

*v0.7, 2 Oct 2026: §8 rewritten for the economic-hurdle decision (consequence table, per-strategy stop bar, pipeline-only branch, ordering, interpretation locks, computation whitelist, provenance log); stop-bar step added to §7; one weak point added. Everything else is unchanged from v0.6-consolidated.*

*v0.6 consolidated 2 Oct 2026 from the full draft chain v0.1 → v0.2 → v0.3 → v0.3.1 → v0.4 → v0.5 → v0.5.1 → v0.6 (all 28 Sep 2026). Earlier committed text (commit 3ba087c) was the v0.6 change-document only: it carried §1–§5 and §9 forward "by reference" to drafts that were never in the repository. This file states every section in full so that the hash, when taken, covers the actual rules. **No rule, threshold, gate or decision rule has been changed in consolidating.** Where the chain contained a stale or contradictory sentence, the later version's explicit supersession is applied and the point is logged in `docs/pr004/CONSOLIDATION_NOTES.md` for the owner to confirm before freeze. The full recovered drafts are preserved verbatim in `docs/pr004/history/`. **Nothing has been run.***

## 0. Process

Draft → red-team review (five rounds, external reviewer) → triage, each claim checked against sources → **freeze** (this file committed, hash recorded) → build → gates run in order → one-shot out-of-sample → forward paper trading. Document review was closed at v0.5.1; v0.6 added one correction found while archiving the source (§0.1).

### 0.1 Instrument used by the replicated source (from v0.6)

Archiving the source meant reading the full page, including its comment thread:

> **Jay Eiser** (comment, 5 May 2019): *"You seem to ignore that Connors did this testing on a fixed set of ETF's. Stocks were not involved in the Connors testing. As you can see your testing on the SPY matched his results exactly."*
> **Joe Marwood** (reply, 14 May 2019): *"I have tested on ETFs various strategies by Connors. Some of them are quite good such as RSI4."*

A reader states plainly that Test One ran on SPY; the author's reply does not correct this and confirms testing "on ETFs" generally. **Weight of evidence: Marwood's Test One was run on SPY, via Norgate — not the raw S&P 500 index.** G1b is therefore a direct SPY-to-SPY comparison. This does **not** resolve whether the Norgate SPY series was dividend-adjusted, or whether sizing was fixed-notional or compounded, so annualised return and max drawdown are **descriptive, not gated** in G1b (§7). The source and its relevant text are archived at `docs/sources/marwood_rsi2_2016.md` and on the Wayback Machine (`web.archive.org/web/20260928211054/https://stocksoftresearch.com/rsi-2-trading-strategy/`).

## 1. Question and role

Owner goals: (1) income/wealth and (3) learn/build capability, with (2) diversification secondary. Paper-first: nothing trades real money until proven on paper.

This item tests the owner's "swing" idea in its **days-scale** form, using the one index rule with a published test on SPY. Holds are ~3–4 days, not weeks or months. A weeks-to-months trend-following family is a separate item, **PR-005, to be drafted and hashed before the G2 session runs** (§8, condition 2). **Whatever this item shows is evidence about one days-scale rule, not about "swing trading" as a family.**

**Stated prior:** the published post-2008 result for this rule is ~1.35% a year on capital (long-only, price-only, no costs). Even if it replicates, it is unlikely to matter for goal (1). Its expected value is mostly learning, plus a possible small diversifier. Time-box: the 21 Nov 2026 review (effort, §11).

## 2. Hypotheses

**H0 — replication (validity, not discovery).** Our implementation reproduces the published figures within the gated bands of G1b (§7), under the conventions stated there.

**H1 — mechanism, all days.** Within days above the 200-day average, a lower RSI(2) predicts a higher forward 3-day return, controlling for volatility state. *Primary endpoint: the RSI coefficient on E3 (2016-01-01 to 2022-12-31)*, as specified in §6.

**H2 — the rule as a sleeve (report only, not gated).** Net of costs, what the frozen rule delivers alone and blended with holding SPY.

## 3. Rules

**Primary, P1** — the book's first test, as replicated by Marwood **[secondary source; UNVERIFIED against the 2008 book]**:

- RSI(2): Wilder smoothing on daily closes. Trend: SMA200 of the close. Exit line: SMA5.
- **Long only.** Entry: close > SMA200 **and RSI2 < 5**. Buy at the close. Exit when the close is above SMA5 (evaluated from the day *after* entry). One position at a time; no stops; flat otherwise.
- Ties: strict inequalities; a value within 1e-9 of a threshold does not trigger. Re-entry only after flat.
- Fixed 1× notional per position (no compounding). Sharpe uses daily excess-over-cash returns, annualised by √252.

**Execution.** *Stage R (fidelity):* trade at the signal close, as published. *Stage V (viability):* enter **and exit** at the next day's open.

**Declared secondary variants** (each counts as a trial): **V2:** threshold 10 instead of 5. **V3:** a symmetric short extension (close < SMA200, RSI2 > 95, exit when close < SMA5), reported separately by side.

**Excluded to control multiplicity:** the cumulative-RSI SPY variant (the source's own replication of it failed: 127 trades against 50 published), IBS, any VIX filter, stops, RSI-based exits, parameter optimisation.

**Era-boundary convention:** a position entered on the last trading day of one era that exits into the next era is counted by its **entry date**; the exit is allowed to complete in the following era's data. Applies at every boundary (E1/E2, E2/E3, E3/E4).

## 4. Data and integrity (gate G0)

**Primary:** SPY daily open/high/low/close, *unadjusted*, plus dividends, yfinance (unofficial, single upstream). **Loader requirement:** yfinance's `end` parameter is exclusive (confirmed by two live fetches); request `end = intended_end + 1 day` and assert the last returned row's date equals the intended end date.

**Asserts:** `high ≥ max(open, close, low)`, `low ≤ min(open, close, high)`, `volume ≥ 0`; every session on the `exchange_calendars` XNYS calendar present, half-days handled (not yfinance's own trading days, which would be circular); `Close` (never `Adj Close`) used everywhere; no duplicate dates; |daily return| ≤ 15% (the same constant is reused as the G2 winsorization bound, §6); ≥ 200 prior rows before every evaluation day; 2016–2024 dividend ex-dates match the project's cached list (pre-2016 amounts not independently verified — reported, not gated).

**Cross-checks, in order of strength:**
1. **G1b against the Norgate-derived published figures, 1995–2015 — an independent vendor** (§7).
2. The project's own SIP-derived daily bars, 2016–2024: closes median ≤ 3 bps / p99 ≤ 10 bps; opens median ≤ 5 bps / p99 ≤ 20 bps **[ASSUMED tolerances]**; every exceedance individually reviewed and logged by date.
3. yfinance `^GSPC` vs SPY×10 daily-return correlation **≥ 0.999**, 1993–2024 (catches single-series errors, not upstream-wide ones).

**Optional, not required for freeze:** a second vendor (Stooq/Tiingo) for 2008–2015, availability unverified. FRED's S&P 500 series is **not usable** (it carries only 10 years of daily history).

Failure of G0 stops the item: fix the data first.

## 5. Eras (honest labelling)

| Era | Dates | Status and use |
|---|---|---|
| E1 | 1995-01-01 to 2007-12-31 | published figures exist (Norgate-derived, archived source); **not blind**; replication only |
| E2 | 2008-01-01 to 2015-12-31 | published figures exist (same source); **not blind**; replication and a *descriptive* decay report only |
| E3 | 2016-01-01 to 2022-12-31 | **no consulted source reports granular, trade-level results for this window. Two consulted sources report only aggregate, full-sample or post-2015 claims** (a self-published backtest's overall win rate; a commercial site's unquantified "slight decay" claim) — coarse, not a trade-level leak, but real and disclosed. **Primary inference.** |
| E4 | 2023-01-01 to 2024-12-31 | **one look**: a single named, hashed script |
| E5 | 2025-01-01 onward | **sealed** (`src/seal.py`); not read |

Disclosure: the analysts also know the general market history and that this rule became popular after 2008. That cannot be removed.

## 6. Costs, the G2 equation and the verdict logic

### 6.1 Costs **[ASSUMED — for review]**

Two accounting modes, both reported, neither gated (the owner's vehicle is undecided).

- **Execution:** 2 bps of notional per side (sensitivity at 1 and 4 bps). Stage V uses 3× that cost on days after a prior-day |return| > 2σ, where σ reuses `vol20`'s definition (trailing 20-day standard deviation of daily log returns through t−1).
- **Cash rate:** DFF (daily effective fed funds, FRED), floored at 0.
- **Mode F (futures-style):** long P&L = total return − cash rate; short P&L = −total return + cash rate.
- **Mode S (spread-bet-style):** longs pay DFF + 2.0%; shorts receive DFF − 2.0% (can be negative). *The 2.0% is a placeholder until the provider's published terms are sourced.*
- Flat days earn the cash rate. Buy-and-hold is total return: (close + dividend) / previous close − 1.
- Vehicle-neutral by design; the owner's vehicle is undecided.

### 6.2 G2, pinned as an equation (fixed at freeze, not adjusted after seeing E3)

```
r3(t) = a + b * RSI2(t) + c * vol20(t) + d * absret1(t) + e(t)
```
- `r3(t)`: price-only close-to-close return from day t to t+3 (ex-dividend; a total-return sensitivity is reported, not gated, to match G1b's price-only convention).
- `RSI2(t)`: as defined in §3. **Expected sign of b: negative** (lower RSI2 → higher forward return).
- `vol20(t)`: standard deviation of daily log returns over the 20 trading days *before* t (not including t).
- `absret1(t)`: |close-to-close return from day t−1 to t|.
- Winsorize `r3` and `absret1` at **±15%** (fixed constant, reused from G0 — never a sample quantile, which would use future information).
- Sample: days with close(t) > SMA200(t). Era E3 only for the primary test; E1 and E2 run the same equation separately, reported (not gated) for decay context.
- **"Pooled over E1+E2"** means one regression of this same equation on the E1 and E2 days concatenated. (Inverse-variance combining of separate era coefficients, or concatenation with era dummies, is not used; any such construction is a separate, stated sensitivity, not the primary benchmark.)
- Standard errors: Newey–West, lag 5 (exceeds the 3-day return overlap).
- **MDE is computed and printed from the actual E1+E2 data before E3 is read**, so INCONCLUSIVE is interpretable. (An order-of-magnitude estimate during review was roughly 5–10 bps per RSI point; the script computes the real number.)
- **Reported (not gated) sensitivity:** the same regression on E3 with year fixed effects.

### 6.3 Verdict logic

Let `b̂3, se3` be E3's coefficient and standard error; `b̂12, se12` the same from the pooled E1+E2 regression (expected sign of both: negative).

```
z = (b̂3 − b̂12) / sqrt(se3² + se12²)
CONTRADICTED  iff  z > 1.645     (E3's coefficient significantly WEAKER — less negative — than published-era)
CONFIRMED     iff  NOT CONTRADICTED  and  (b̂3 + 1.645·se3) < 0     (evaluated in that order — precedence matters)
INCONCLUSIVE  otherwise
```

**Verified on four cases before adoption** (b12 = −8, se12 = 2, se3 = 4, illustrative units): b̂3 = 0 (dead) → CONTRADICTED; b̂3 = −20 (much stronger) → CONFIRMED, not CONTRADICTED; b̂3 = −7 (unchanged) → CONFIRMED; b̂3 = −2 (mostly decayed) → INCONCLUSIVE. Precision dependence: with b̂12 = −8, se12 = 2, a dead effect reads CONTRADICTED only if se3 < 4.43; a halved-but-real effect (b̂12 = −10, se12 = 2, b̂3 = −4) reads CONTRADICTED at se3 = 2 (z = 2.12) but INCONCLUSIVE at se3 = 4 (z = 1.34).

**Fallback if the published-era coefficient is not negative:** if b̂12 ≥ 0, the decay contrast is undefined; the verdict is INCONCLUSIVE, the report states why, and the paper-trading decision falls to H0 plus the owner's documented review of G2's descriptive output (§8).

**Pre-written interpretation sentences** (fixed now so the report cannot slide after the fact):
- *INCONCLUSIVE branch:* "If MDE (computed from E1+E2, printed before E3 is read) exceeds the observed |b̂3|'s confidence interval width, the result is under-powered; if b̂3's point estimate sits between zero and b̂12 without either bound being decisive, the honest reading is partial decay, not absence of a mechanism."
- *CONTRADICTED branch:* the report states whether U3 = b̂3 + 1.645·se3 < 0 — i.e. whether E3's effect, though significantly weaker than published, is still individually significant — so "decayed" is never silently read as "dead."
- *Bucket interpretation:* each RSI bucket's **excess** mean forward return is the bucket mean minus the same-era, above-SMA200, all-days mean. CONFIRMED is read as "the mechanism operates in the rule's own tail" only if the RSI < 5 and RSI 5–10 buckets show positive excess **and** excess return decreases monotonically as the bucket range widens toward the middle; otherwise the report says the coefficient is driven by mid-range variation.

## 7. Gates and steps, in order

- **G0 Data integrity** (§4). Failure stops the item.
- **G1 Implementation.** Two independently coded versions from the written spec (one vectorised, one an explicit day-by-day loop) agree on 100% of position-days, trade lists and daily P&L under the tie policy; plus the owner's manual spreadsheet audit of 10 randomly chosen episodes, with the selection seed derived from this document's own commit hash once frozen.
- **G1b Replication (H0).** P1 on SPY, Stage R, price-only, costless, flat capital earns 0% (matching the source). The archived source's figures this gate reproduces — Marwood's own replication (Norgate data, Amibroker), not the book's own claim:
  - *1/1/1995 to 1/1/2008:* 49 trades, 83.7% winners, 524.4 total points, average hold 4 days, annualised return 3.91%, max drawdown −6.48%.
  - *1/1/2008 to 1/1/2016:* 30 trades, 80% winners, 184.91 total points, average hold 4 days, annualised return 1.35%, max drawdown −13.19%.
  - (Both 1 Jan 2008 and 1 Jan 2016 are NYSE holidays, so these windows are exactly calendar years 1995–2007 and 2008–2015 = E1 and E2. The book's own claim — 83.6% winners, 3-day hold — is reported for context only, since its vendor is unstated.)

  **Gated bands:** E1 trades 45–53, win rate 78–89%, average hold 3–5 days; E2 trades 26–34, win rate 72–88%, average hold 3–5 days **[ASSUMED bands around the quoted figures]**. **Annualised return and max drawdown are reported, not gated** (E2 reference range 0.5–2.5% and −9% to −17%), because the source's sizing and dividend-adjustment conventions are not stated. A descriptive statistic outside its quoted range is disclosed with an attribution attempt but does not by itself close the item. **Failure protocol:** one documented investigation (data first, then implementation, then sizing or adjustment mismatch), at most one fix-and-rerun; an unattributable miss closes the item. E3 is not read until this gate passes.
- **Event-driven replay rehearsal** (after G1b passes, before G2). Runs the frozen pipeline day-by-day on the project's own 2016–2024 1-minute store with next-open fills — a dress rehearsal for Stage V and for forward paper trading, testing the off-by-one and fill-timing failure class G0 cannot see. **It may print only:** fill-timestamp correctness, the next-open date-shift count (must be 0), the calendar-anomaly count, and a pass/fail equality check between the replay and the vectorised implementation. **It may not print** trade lists, equity curves, returns or any performance figure. Those are written to a hashed, sealed file **partitioned at the E3/E4 boundary**: the E3 segment is openable only at G2; the E4 segment only at G7, never before.
- **G2 Mechanism (H1), primary endpoint,** as pinned in §6.2–6.3, on E3. Also reported on E1 and E2 for decay context, and as a bucket table (RSI < 5, 5–10, 10–30, 30–70, > 70) with an episode-level bootstrap CI.
- **G3 Nulls**, evaluated only if E3 has ≥ 30 entries (otherwise "not evaluable — power"): sign-flip (a coding check only); vol-matched random entry; down-move-matched random entry, drawn from above-SMA200 days.
- **G4 Economic report** (not gated), on **E1+E2+E3** (E4 reported separately, never merged): net CAGR, vol, Sharpe, Jensen alpha with CI, max drawdown, time in market, both cost modes; the **blend test at 25% / 50% / 100% sleeve allocation** into a 100% SPY book; long-book loss on SPY's 20 worst days and short-book loss on its 20 worst up-days (V3 only); the mean close→next-open "gap wedge" on entry days; top-3-episode share of P&L (labelled "episode-dependent" above 60%); year table with trade counts; volatility-tercile and long/short breakdowns; every day on which the ±15% winsorization bound actually clipped a value; the full trade list as an audit appendix.
- **Stop-bar evaluation (§8.4),** computed at G4 time from the G1b-validated implementation, with numbers recorded.
- **G7 (E4, once):** a single named, hashed script; report only, in the frozen template.

**Verdict categories:** CONFIRMED / CONTRADICTED / INCONCLUSIVE (power), as defined in §6.3, with the MDE stated alongside whichever applies.

## 8. Conditions of freeze, the decision rule, and consequences

*§8 rewritten 2 Oct 2026 after a D-A-C cycle and three external review rounds on the economic-hurdle question. Every decision below was taken by the owner. The record of how it was reached is §8.9.*

### 8.1 Conditions of freeze

1. **Owner sign-off, executed as the freeze itself.** Performing the hash (condition 4) *is* the owner's sign-off. It ratifies: (a) the mechanism-gate wording below; (b) the economic-hurdle placement in §8.3, including v0.2's previously unsigned demotion of the Sharpe-vs-buy-and-hold gate; and (c) the consequence machinery in §8.2–8.7. The freeze carries the provenance log (§8.9).
   *Mechanism-gate wording:* "The mechanism gate blocks proceeding to paper trading if E3's effect is significantly weaker than the published era's. This includes a completely dead effect, but only if it's measured precisely enough — the printed MDE tells you in advance whether that will hold. It also includes a real, still-statistically-significant effect that has merely halved or more — and whether that halved-but-real case is blocked also depends on how precisely it is measured, not only on its size; the MDE is computed from 1995–2015 data as a preview of E3's expected precision, so treat it as a good-faith estimate, not a guarantee. If the published-era comparison itself turns out not to have the expected sign, the gate cannot judge decay at all, and the decision falls to replication (H0) plus the owner's own direct review of the descriptive results — not an automatic proceed."
2. **PR-005 hashed before the G2 session runs — enforced as a hard precondition, not a reminder.** Before the G2 session runs, its setup step must verify that `docs/PR005_HASH.txt` exists, is non-empty, and — case-insensitively — is exactly 64 hexadecimal characters, **or** that a logged, non-empty, dated closure record exists at `docs/PR005_CLOSURE.txt`. If neither holds, the session halts and does not run. Implementation and self-test: `docs/PR004_hash_precondition_addendum.md`. A human checklist line duplicates it as a second enforcement layer.
3. **Source archived before G1b is coded — done, both halves** (`docs/sources/marwood_rsi2_2016.md`; `web.archive.org/web/20260928211054/https://stocksoftresearch.com/rsi-2-trading-strategy/`, cross-checked line-by-line, 28 Sep 2026).
4. **The owner-executed hash step.** Once 1–3 are satisfied, the owner personally computes and records this document's hash, dated and initialed. The authoring session does not perform this step.

### 8.2 Decision rule and consequence table

Every outcome maps to exactly one state:

| State | Condition | Consequence |
|---|---|---|
| **S1** | G1b PASS; H1 verdict not CONTRADICTED; per-strategy stop bar (§8.4) PASS | Full paper-trading window per `docs/pr004/PAPER_TRADING_PROTOCOL.md` |
| **S2** | G1b PASS; H1 verdict not CONTRADICTED; stop bar computed and FAIL | **Pipeline-only window** (§8.5): 12 months, operational-only, if the reuse case holds; otherwise no window |
| **S3** | G1b does not PASS (after the G1b failure protocol is exhausted) | **Definitional stop-bar FAIL.** No window of any kind. The attribution class (implementation / data or source / indeterminate) is recorded. The failure is final for this item: post-freeze code changes are barred, and any re-attempt is a new hashed item carrying this failure history. No stop-bar statistic is ever computed from code that failed replication. |
| **S4** | G1b PASS; H1 verdict CONTRADICTED | No window. The stop bar is still computed (the code is validated) and recorded. |

In the b̂12 ≥ 0 fallback (§6.3), S1 and S2 additionally require the owner's documented review of G2's descriptive output before any window starts. Every state records both the mechanism verdict and the stop-bar verdict, with numbers (§8.6 e). Which verdict "defines" the study is not chosen after the fact. **Paper trading validates operations and detects gross failure; it cannot statistically confirm the edge** (~3.8 round trips a year).

### 8.3 Economic-hurdle placement (decision of 2 Oct 2026)

"Net Sharpe ≥ buy-and-hold SPY's Sharpe" is **reported in G4, not a gate** on entry to paper trading. Economic judgment is made per strategy by the project stop bar (§8.4), with the consequences in §8.2.

**Class rule.** This placement applies to every low-exposure strategy, defined as expected time in market ≤ 25%. At that exposure, standalone Sharpe parity with the benchmark would require in-market days with at least 2× the benchmark's Sharpe (1/√f ≥ 2). The rule binds future candidates regardless of their published economics.

**Why: the option space was searched, not avoided.**
- A standalone gate tests *replacement* of SPY, which no one has proposed. On v0.1's own 2008–2022 window, SPY's Sharpe is 0.485, so that gate needs ≈ 1.98 annualized Sharpe on in-market days. The stop bar needs ≈ 2.04. Same hurdle height within 3%; different legs; different lane.
- Add-to-the-book (marginal or blend) tests are not skill tests. For a rule in the market a fraction f of days, its correlation with SPY is √f, and a zero-skill rule sits exactly on the marginal-improvement bar. On synthetic data: correlation 0.244 vs √0.06 = 0.245; zero-skill Sharpe 0.137 vs bar 0.136.
- Skill is tested where power can be built, by G2's conditioned regression and G3's matched nulls. No economic gate for this class is both a skill test and non-redundant with them.

**Passive-benchmark reporting (conditional, per the Rulebook rule "net return must clear a passive-benchmark hurdle before being credited as edge").** If G4's blend test does not show an improvement in the net Sharpe of the 100% SPY book, the report states that PR-004 is "not proven as edge against a passive benchmark." In every state, no PR-004 performance is credited as edge before capital is committed. The live-capital decision needs its own pre-registered gate.

### 8.4 The per-strategy stop bar for PR-004 (pins)

Computed at G4, from the implementation validated by G1b, and recorded with numbers:
- **Returns:** Stage V (next-open fills), **cost Mode F**, fixed 1× notional; daily excess return over DFF (act/360); Sharpe annualized by √252.
- **In-sample (IS):** 1995-01-01 to 2015-12-31 (E1+E2). **Out-of-sample (OOS):** 2016-01-01 to 2022-12-31 (E3). E4 is not used.
- **PASS iff all of:** Sharpe_IS ≥ 0.5; Sharpe_OOS ≥ 0.5; Sharpe_OOS ≥ 0.5 × Sharpe_IS ("OOS ≥ half of IS", read as walk-forward efficiency, per the Rulebook rule "WFE = OOS Sharpe / IS Sharpe ≥ 0.50" and `src/evaluation/robustness.py`); correlation of daily returns with SPY daily total returns ≤ 0.3 over 1995–2022; maximum drawdown ≤ 15% over 1995–2022.
- This mapping uses no data beyond what G1b and G2 already open. Disclosure: the correlation leg is satisfied structurally by any rule in the market ~6% of days (correlation ≈ √f ≈ 0.245). It is logged as a Rulebook issue and does not count as evidence.

### 8.5 Pipeline-only window (state S2) and the reuse case

- **Shared components (spec level; the code does not yet exist):** daily SPY bar ingestion from the live store; XNYS trading calendar; close-of-day signal computation; next-open order generation and fill recording; ex-dividend handling; per-fill model-implied cost logging. **Build requirement:** PR-004 and PR-005 import shared modules for these, not copies.
- **Beneficiary:** PR-005 (T1 monthly, V2 daily), which also executes at the next open. At ≈ 0.67 round trips a year it generates ≈ 1.3 next-open fill events a year and cannot self-validate that machinery in any reasonable window. PR-004 generates ≈ 7.6.
- **One-way justification guard:** PR-004's window is justified by PR-005's pre-registered existence. PR-005's admission to paper trading may not cite PR-004's window, and PR-004's window discharges none of PR-005's own validation duties.
- **Contingency:** the reuse value is realized only if a reusing item is admitted. If, when S2 executes, PR-005 has a recorded closure (a final verdict other than value-CONFIRMED, or `docs/PR005_CLOSURE.txt`), the reuse case is void and S2 becomes **no window**. If the joint stop fires, a running window halts (§8.6 d).
- **Duration:** a fixed 12-month hard stop. No round-trip leg. No early termination for "success". The only early exits are the operational kill criteria in the paper protocol and the joint-stop halt.
- **Firewall:** paper P&L is recorded as data and is never narrated in any decision log, summary, tracker note or report. No continuation, kill or admission decision may cite it.

### 8.6 Ordering, and what "research stops" means

The project stop rule (Rulebook, 26 Sep 2026) fires when PR-003 and PR-004 both fail its bar.
- **Execution point:** per-strategy consequences (S1–S4) execute at joint-verdict determination, meaning once both PR-003's and PR-004's stop-bar verdicts are recorded, computed or definitional. PR-003's stop-bar mapping is in `docs/pr003/STOP_BAR_MAPPING.md` and is a dependency of this ordering.
- **Precedence:** if both FAIL, the joint stop overrides every per-strategy consequence.
- **"Research stops" means:**
  - (a) Frozen items run to completion, including each strategy's stop-bar evaluation. Completion means reaching a recorded verdict, computed or definitional.
  - (b) The joint stop fires only when both verdicts exist and both are FAIL. The 8-week time-box is not a deadline that voids unfinished evaluations.
  - (c) No new items start, including re-specifications, parameter variants and follow-on studies of stopped strategies.
  - (d) No paper windows start, and already-started non-evidential windows halt.
  - (e) Recording means numbers plus verdicts, never verdicts alone.
  - (f) Restart requires a new written charter, not a resumption.
- PR-005 sits outside the stop rule (Rulebook amendment, 28 Sep 2026). As a frozen item it is still subject to (a) and (d) if the joint stop fires.

### 8.7 Interpretation locks

- **MDE:** recorded as a limitation. It triggers no design change after freeze, whatever its size.
- **INCONCLUSIVE** means "unproven, not confirmed". It never means "confirmed by default".
- **E3 carries three pre-registered readings:** G2 (mechanism), the stop bar's OOS leg, and G4 (economics, era E1+E2+E3). All three are recorded. None is selected after the fact as "the" verdict.

### 8.8 Pre-freeze computation whitelist

- **Permitted before freeze:** benchmark and metadata statistics that do not touch PR-004's signal, plus synthetic-data checks.
- **Performed (all on 2 Oct 2026):**
  - SPY buy-and-hold daily excess Sharpe over FRED DFF, computed from yfinance SPY total returns: 1995–2007 0.521; 2008–2015 0.382; 1995–2015 0.458; 2016–2022 0.624; 2008–2022 0.485.
  - Synthetic RSI(2) seeding check.
  - Synthetic √f correlation and zero-skill marginal-Sharpe simulation.
- **Forbidden:** anything that computes PR-004's signal, positions, returns or statistics on real data.

### 8.9 Provenance log (facts only; causation not asserted)

- **27 Sep 2026:** the Rulebook adopts "net return must clear a passive-benchmark hurdle before being credited as edge."
- **28 Sep, v0.1:** G4 is a gate: net Sharpe ≥ buy-and-hold on 2008–2022, and max drawdown no worse. v0.1's sources include a self-published backtest — a different source and a different rule variant — saying the rule "loses to buy-and-hold on raw return". v0.1's own red-team packet (Q8) asks whether that hurdle is sensible for a low-exposure rule.
- **28 Sep, v0.2:** the gate is demoted to a report on the reviewer's √exposure argument. The same revision introduces the replicated source's figures (2008–2015: ~1.35%/yr, −13.19% max drawdown). The demotion was flagged as needing the owner's sign-off, which was never given before 2 Oct.
- **No version was ever hashed.**
- **2 Oct 2026:** decided after a D-A-C cycle (options A–D, then A vs B) and three external review rounds. The rounds established symmetric outcome-awareness on both options, the zero-skill property of marginal tests, the threshold convergence (1.98 vs 2.04), and a gap closure: as previously written, a per-strategy stop-bar FAIL with PR-003 passing triggered nothing and left the full window open. Every consequence in §8.2 therefore tightens the prior operative default.

## 9. Multiple testing

Single primary endpoint (G2 on E3); no correction needed within it. Family: P1 (primary) + V2 + V3 = 3 declared trials. No deflated Sharpe: it gated nothing, and the project's 1/(n−1) cross-trial-variance default makes it numerically inert; the single endpoint, the frozen variant list and the never-cite rule do this job. Sensitivity tables (thresholds, SMA length, k = 1 and 5) run **on E1+E2 only**, never on E3, and may never be cited as grounds for adopting a variant — any new variant is a new pre-registration. **Report skeleton, frozen now:** verdict; G0–G1b; replay rehearsal result; G2 by era with MDE; G4; blend test; limitations; the sentence "evidence about one days-scale rule, not swing trading as a family."

## 10. Known weak points

1. Power: ~4 entries a year; E3 alone has ~26. G2 is powered by all days, but the effect lives in the tails.
2. E1 and E2 are not blind; only E3–E4 are new evidence.
3. G1b bands and assumed costs are judgement.
4. One instrument; QQQ shares the same upstream and may not be cited as independent confirmation.
5. The 2008 book itself is unread; the rule comes from a secondary source (archived).
6. Same-author risk on the two G1 implementations, mitigated by the manual audit and by G1b being the real interpretation check.
7. G1b's sizing and dividend-adjustment conventions are unstated in the source; return and drawdown are therefore descriptive only.
8. The stop bar's correlation leg is satisfied structurally at this exposure (§8.4); it carries no evidential weight.
9. Capacity: paper-trading scale is not a capacity constraint; revisit only if any live deployment beyond paper is ever considered.

## 11. Effort

Five sessions: (1) data loader, G0, calendar/OHLC asserts; (2) two implementations, G1, manual audit; (3) G1b replication, replay rehearsal; (4) G2 equation, MDE, buckets, nulls; (5) G4 report, E4 script, write-up.

## Sources

- Joe Marwood, "Testing The RSI 2 Trading Strategy On US Stocks," stocksoftresearch.com (published 2016-01-20, modified 2020-11-10) — Test One rules, replication figures, comment thread; archived at `docs/sources/marwood_rsi2_2016.md` and on the Wayback Machine.
- LuxAlgo, StratBase, MQL5 listings — variant rules; a Substack backtest (Sep 2026, self-published) and QuantifiedStrategies (Mar 2026) — aggregate claims disclosed in §5.
- FRED series notes: SP500 carries 10 years of daily history; DFF is daily.
- This project's own code, read directly: `src/evaluation/robustness.py`, `src/seal.py`, `src/market_data_store.py`.
