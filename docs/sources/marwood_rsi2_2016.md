# Archived source for PR-004 G1b

**Archived:** 28 Sep 2026, for PR-004 pre-registration (`docs/PR004_*`).
**Original:** Joe Marwood, "Testing The RSI 2 Trading Strategy On US Stocks,"
stocksoftresearch.com/rsi-2-trading-strategy/, published 2016-01-20,
last modified 2020-11-10T11:16:07+00:00 (per the page's own meta tag).
**Retrieved:** 28 Sep 2026, via direct fetch (full text, including comments).

---

## Why this page, and the symbol correction (28 Sep 2026)

Earlier drafts of PR-004 assumed Marwood's "Test One" ran on the raw S&P 500
index (based on the article's prose: "is on the S&P 500 Index... tested for
myself using... historical data from Norgate"). A full read of the comment
thread contradicts this:

> **Jay Eiser** (May 5, 2019): "You seem to ignore that Connors did this
> testing on a fixed set of ETF's. Stocks were not involved in the Connors
> testing. As you can see your testing on the SPY matched his results
> exactly."
>
> **Joe Marwood** (May 14, 2019, reply): "I have tested on ETFs various
> strategies by Connors. Some of them are quite good such as RSI4."

Marwood does not correct Eiser's claim that his test used SPY, and confirms
testing "on ETFs" generally. Weight of evidence: **Test One was run on SPY**,
via Norgate data, not the raw index. G1b is revised accordingly (v0.6).

## Verbatim text relevant to G1b ("Test One")

> ### Test One – 2-period RSI under 5 on S&P 500
>
> The first strategy mentioned in the book is on the S&P 500 Index. It has
> the following rules:
>
> 1. The S&P 500 is above its 200-day MA
> 2. RSI 2 of the S&P 500 is below 5
> 3. Buy the S&P 500 on the close
> 4. Exit when the S&P 500 closes above its 5-day MA
>
> In other words, we want to buy the S&P 500 when it is oversold but still
> above it's 200-day moving average. This strategy is run between 1995 and
> 2008 and is shown in the book to produce the following returns:
>
> • No. of trades = 49
> • No. of Winners = 83.6%
> • Total points made = 522.92
> • Average hold time = 3 days
>
> I tested this strategy for myself using Amibroker and historical data from
> Norgate between 1/1/1995 and 1/1/2008 (without any transaction costs) and I
> achieved the following results:
>
> • No. of trades = 49
> • No. of winners = 83.7%
> • Total points made = 524.4
> • Average hold time = 4 days
> • Annualised return = 3.91%
> • Maximum drawdown = -6.48%
> • CAR/MDD = 0.60
>
> As you can see, the results are almost identical. [...]
>
> Since the test results look good so far I moved the data set forward and
> tested the strategy between 1/1/2008 and 1/1/2016. The results are shown
> below:
>
> • No. of trades = 30
> • No. of Winners = 80%
> • Total points made = 184.91
> • Average hold time = 4 days
> • Annualised return = 1.35%
> • Maximum drawdown = -13.19%
> • CAR/MDD = 0.10

## Methodology footer (whole article)

> Trading systems and charts produced with Amibroker using data from Norgate
> Premium Data. Simulations assume a cash account with no margin. Stock
> universes include historical constituents/delisted shares.

"No margin" rules out leverage on this test but does not by itself resolve
fixed-notional vs. compounding, which stays an open, unverified point.

## Structural note resolving an earlier round's worry

The Test One quote sits under the heading "Test One – 2-period RSI under 5
on S&P 500," structurally distinct from the following "Test Two – Cumulative
RSI on SPY" section. The 49-trade / 83.6–83.7% figures belong to Test One
(threshold-5, single-instrument, SMA5-exit rule), not the cumulative-RSI
variant.

## Full article text

The complete fetched text (including the full comment thread) is preserved
in this project's chat history for the session that built PR-004, dated
28 Sep 2026. A second, independent copy exists via the Internet Archive
(see the source URL above) as of the same date.
