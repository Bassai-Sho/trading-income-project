# Trading Income Project

Systematic, evidence-based day trading system built around the Opening Range Breakout (ORB) strategy, validated against academic benchmarks before any live capital is deployed.

**Hardware:** Intel NUC 14 Pro · Core Ultra 5 125H · 64GB DDR5 · Intel Arc iGPU (58GB VRAM pool) · Ubuntu 24.04  
**Academic foundation:** Zarattini, Barbon & Aziz 2024 (SSRN 4729284) — ORB + RVOL filter, Sharpe 2.81

---

## What this does

Trades SPY once per session using a 15-minute ORB signal with VWAP confirmation. Every decision is logged, analysed nightly by a local LLM D-A-C pipeline (Phase 1: director analysis → Phase 2: adversarial stress-test using a different model architecture), and validated weekly against walk-forward criteria. The system cannot deploy live capital until it clears a statistical gate (100 paper trades, WFE ≥ 0.50). The edge is verified by running the same IS sessions through seven academic strategy replications and a 10,000-path vectorised random-trader Monte Carlo benchmark.

---

## Architecture

```
Startup
  prepare_host.sh   Intel Arc drivers (stock Ubuntu only), OpenVINO GenAI, model download
  launch_models.sh  Start Phase 1 server (Port 8000) + Phase 2 server (Port 8001)
  setup.sh          Python deps, DATA/ directory, .env, health check
  verify.sh         5-step post-setup verification

Daily (automated via runner.py)
  09:00  morning_brief.py     VIX (FRED), yield curve, CBOE P/C, CFTC, OPEX, GO/NO-GO
  09:25  markov_engine.py     Regime chain refresh from FRED VIXCLS
  09:28  trading_engine.py    60s signal loop: ORB → VWAP gate → retest → entry
  15:35  session_analyser.py  D-A-C: Phase 1 (Port 8000) → Phase 2 adversarial (Port 8001)
  16:30  market_data_store.py Append today's 1-min bars to SQLite
  16:35  fred_store.py        Update FRED macro series (VIX, rates, spreads, CPI)
  16:40  sentiment_store.py   Scrape CBOE P/C today + refresh CFTC COT current year
  Sat    tournament_evaluator.py  Weekly WFE/Sharpe comparison, Design Studio

LLM inference — OpenVINO GenAI in .venv (no Docker, no Ollama registry)
  Port 8000  Phase 1 director model   (selected at prepare_host.sh time)
  Port 8001  Phase 2 adversarial model (different architecture from Phase 1)
  serve_model.py wraps openvino_genai.LLMPipeline with OpenAI /v1 API + tool calling

Data stores — all DATA/market_data.db unless noted
  market_bars         Alpaca 1-min SPY bars  2016-2024
  session_context     PDH/PDL, VIX regime, ORB range per session
  fred_observations   7 FRED series: VIXCLS DGS2 DGS10 T10YIE BAMLH0A0HYM2 FEDFUNDS CPIAUCSL
  cboe_pc_daily       CBOE equity put/call ratio  2006+
  cftc_cot_weekly     CFTC COT S&P 500 leveraged + asset manager net  2016+
  (DATA/paper_account.db)  sim_trades, dac_findings, confirmed_patterns, wfa_results,
                            random_baseline_summary, academic_replication_results
```

Full deployment guide → [`docs/SETUP.md`](docs/SETUP.md)

---

## Prerequisites

| Requirement | Where | Notes |
|---|---|---|
| Alpaca paper account | [alpaca.markets](https://alpaca.markets) | Free, no funding required |
| FRED API key | [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html) | Free, instant |
| Brave Search API key | [api.search.brave.com](https://api.search.brave.com) | Free 2,000 queries/month. Credit card required (fraud prevention only) |
| Python 3.10+, Ubuntu 24.04 | — | NUC 14 Pro recommended |
| OpenVINO GenAI | `./prepare_host.sh` | Installed into `.venv` — no Docker, no third-party repos |

---

## Quick start

```bash
# 1. Host setup and model download (once — choose a model pair)
chmod +x prepare_host.sh launch_models.sh setup.sh verify.sh git_setup.sh
./prepare_host.sh           # interactive: pair 1 / 2 / 3 | drivers + venv + download

# 2. Start LLM servers
./launch_models.sh          # starts both ports. First run: ~45s compile. After: <10s.

# 3. Fill in API keys
nano .env                   # ALPACA_API_KEY, ALPACA_SECRET_KEY, FRED_API_KEY, BRAVE_SEARCH_API_KEY

# 4. Python env + data download (once — ~2-3 hours, see docs/SETUP.md)
./setup.sh
python src/market_data_store.py --download --tickers SPY \
  --start 2016-01-01 --end 2024-12-31 --db DATA/market_data.db
python src/fred_store.py      --download --db DATA/market_data.db
python src/sentiment_store.py --download --db DATA/market_data.db

# 5. Correct, bootstrap, and validate
python src/data_corrector.py  --db DATA/market_data.db --ticker SPY \
  --start 2016-01-01 --end 2022-12-31
python src/historical_sim.py  --start 2016-01-01 --end 2022-12-31 \
  --db DATA/paper_account.db
# ↑ auto-triggers 10,000-path random baseline + 7-strategy academic comparison

# 6. Verify everything works
./verify.sh                 # 5-step check: servers, inference, thinking log, adversarial, session analyser

# 7. Run
python src/runner.py --ticker SPY --orb 15min --account 10000
streamlit run src/trading_dashboard.py   # separate terminal
```

---

## Pushing to GitHub

```bash
./git_setup.sh
```

Interactive: initialises git, verifies `.env` and `DATA/` are excluded, creates the first commit, pushes. Full guide → [`docs/GITHUB_SETUP.md`](docs/GITHUB_SETUP.md)

---

## Source files (23 Python · 18,661 lines)

### Core system
| File | Lines | Purpose |
|---|---|---|
| `runner.py` | 615 | Entry point — APScheduler daily orchestration |
| `trading_engine.py` | 1,568 | ORB signal gates, position management, fill simulation |
| `trading_dashboard.py` | 585 | 6-tab Streamlit cockpit |
| `morning_brief.py` | 392 | Pre-market CLI: VIX, macro, CBOE P/C, CFTC, OPEX, GO/NO-GO |
| `session_analyser.py` | 1,296 | D-A-C pipeline, presets for all three model pairs, tool calling |
| `broker_interface.py` | 395 | DataProvider / BrokerClient abstraction |
| `serve_model.py` | 376 | OpenVINO GenAI OpenAI /v1 bridge — tool calling, thinking token strip |

### Strategy and evolution
| File | Lines | Purpose |
|---|---|---|
| `strategy_registry.py` | 334 | 6 strategy variants, phase gate enforcement |
| `tournament_evaluator.py` | 483 | Weekly WFE/Sharpe comparison, Design Studio |
| `gene_mixer.py` | 731 | Evolutionary engine: crossover, mutate, D-A-C validate |

### Statistical and ML
| File | Lines | Purpose |
|---|---|---|
| `markov_engine.py` | 832 | Regime Markov chain, outcome chain, candle N-gram |
| `monte_carlo_extended.py` | 574 | Galton board paths, risk-of-ruin, Kelly |
| `trade_journal_extended.py` | 654 | 50-field journal, psychology flags, correlations |
| `trading_quant_toolkit_v2_4.py` | 3,682 | Core maths: EV, Kelly, WFA, DSR, Monte Carlo |

### Data pipeline
| File | Lines | Purpose |
|---|---|---|
| `market_data_store.py` | 850 | Alpaca 1-min bar store with 5 quality checks |
| `data_corrector.py` | 660 | Phantom H/L, stale bars, OHLC integrity, early-close trim |
| `historical_sim.py` | 971 | IS backtest + SimBootstrapper (seeds all learning) |
| `fred_store.py` | 527 | FRED macro: VIX, yield curve, HY spread, CPI (7 series) |
| `sentiment_store.py` | 564 | CBOE put/call (2006+) + CFTC COT S&P 500 (2016+) |

### Validation and intelligence
| File | Lines | Purpose |
|---|---|---|
| `tool_runner.py` | 709 | Brave Search primary + DDG fallback, fetch_url, 8 LLM tools |
| `findings_store.py` | 336 | D-A-C findings persistence → confirmed patterns → morning brief |
| `random_baseline_sim.py` | 751 | 10,000-path vectorised random-trader Monte Carlo benchmark |
| `academic_replications.py` | 776 | 7-strategy academic comparison (long-only + no-retest included) |

---

## Model pairs (OpenVINO GenAI — all run in 58GB iGPU pool)

Select a pair at `./prepare_host.sh` time. Both models remain resident in VRAM simultaneously (zero swap latency between D-A-C phases).

| Pair | Phase 1 — Port 8000 | Phase 2 — Port 8001 | VRAM | Best for |
|---|---|---|---|---|
| **1 — Thinking Quant** | DeepSeek-R1-Distill-32B INT4 | Mistral-Nemo-12B INT4 | ~26GB | o1-class step-by-step reasoning, quantitative audit |
| **2 — High-Speed Agentic** | Qwen3-30B-A3B MoE INT4 | Mistral-Nemo-12B INT4 | ~25GB | 30+ tok/s, fast multi-turn tool loops |
| **3 — Frontier Heavyweight** | Qwen2.5-32B-Instruct INT4 | Mistral-Small-24B INT4 | ~33GB | Peak JSON/tool precision + deep adversarial critique |

**Recommended: Pair 1.** DeepSeek-R1's `<think>` reasoning chain is well-suited to multi-step quantitative analysis (checks its own maths before committing). Thinking tokens are stripped from the response and logged to `LOGS/thinking_8000.log` for audit.

**Architectural independence:** All Phase 2 adversarial models (Mistral family) use Sliding Window Attention and a different tokeniser from Phase 1 (Qwen/DeepSeek), breaking intra-family confirmation bias.

**Why OpenVINO GenAI over IPEX-LLM Ollama:**
- Intel archived IPEX-LLM January 2026. OpenVINO is Intel's active production framework.
- No third-party display driver repositories — `prepare_host.sh` uses only stock Ubuntu 24.04 packages, eliminating the GDM3 black-screen risk.
- No Ollama registry version checks — no Error 412.
- Native INT4 symmetric quantisation compiled 1:1 into Arc DP4a vector engines: ~1.5-1.8× faster than llama.cpp Vulkan.

**Switch pairs without reinstalling drivers:**
```bash
./prepare_host.sh --pair 2 --models-only   # download pair 2 models
./launch_models.sh --pair 2                # start pair 2
```

---

## Web search

`tool_runner.py` uses **Brave Search API as primary** (independent index, reliable JSON API, 2,000 free queries/month) with DuckDuckGo as automatic fallback when `BRAVE_SEARCH_API_KEY` is absent or the request fails. Get a free key at `api.search.brave.com`.

---

## Academic foundation

| Paper | Strategy | Result | Used in |
|---|---|---|---|
| Zarattini et al. 2024 (SSRN 4729284) | ORB + RVOL filter on Stocks in Play | Sharpe 2.81, alpha 36% | Signal design |
| Zarattini et al. 2024 (SSRN 4824172) | VWAP intraday momentum on SPY | Sharpe 1.33, 19.6% p.a. | Exit design, SPY validation |
| Gao, Han, Li, Zhou 2018 (JFE) | First/last 30-min ETF momentum | Significant after fees | Academic comparison floor |
| Maroy 2025 (SSRN 5095349) | VWAP + ladder exits | Sharpe 3.0+ | Phase 3 exit enhancement |
| Barber et al. 2011 (Taiwan, 15yr) | Retail day trading outcomes | 84% lose, <1% positive net of fees | Risk framing |
| Chague et al. 2019 (Brazil) | Retail day trading outcomes | 97% lose, no learning curve | Risk framing |

**7-strategy academic comparison** (ranked by expected IS Sharpe — run automatically after bootstrap):

| Rank | Strategy | Basis | Elimination rule |
|---|---|---|---|
| 1 | Random bidirectional | Null hypothesis | — |
| 2 | Gao first/last 30-min | JFE 2018 | Floor: must beat this |
| 3 | Zarattini 5-min ORB (SPY proxy) | SSRN 4729284 | — |
| 4 | Zarattini VWAP momentum | SSRN 4824172 | If hybrid < this: simplify to VWAP |
| 5 | Long-only 15-min ORB | Independent review | If > bidirectional: prune shorts |
| 6 | 15-min ORB no-retest | Adverse selection test | If > with-retest: remove Scarface mechanic |
| 7 (target) | Hybrid 15-min ORB + VWAP | Our system | Must beat rank 4 |

---

## Data boundaries

| Window | Dates | Status |
|---|---|---|
| In-sample | 2016–2022 | Training. `SealedDataError` raised in code if crossed. |
| Validation | 2023–2024 | Walk-forward OOS — untouched until Phase 3 |
| Sealed | 2025–present | Never touched. Final live-capital evaluation only. |

---

## Phase gates

| Gate | Condition |
|---|---|
| Phase 2 → 3 | 100 live paper trades, WFE ≥ 0.50 on live data |
| Phase 3 → 4 | 3 months positive EV, Deflated Sharpe Ratio > 0 |
| Phase 4 → 5 | Live capital deployed, 6-month track record |

---

## UK residents

FCA-regulated spread betting (IG, CMC Markets, Spreadex) is classified as gambling under UK law and is exempt from Capital Gains Tax and Income Tax on profits (ITTOIA 2005 ss.6 & 10; TCGA 1992). Executing the same systematic strategy via a spread bet API delivers a 20–45% post-tax advantage over a US broker account. Consult a UK tax adviser before live deployment.
