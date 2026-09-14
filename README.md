# Trading Income Project

Systematic, evidence-based day trading system built around the Opening Range Breakout (ORB) strategy on SPY, validated against academic benchmarks before live capital deployment.

**Hardware:** Intel NUC 14 Pro · Core Ultra 5 125H · 64GB DDR5 · Intel Arc iGPU (58GB VRAM pool) · Ubuntu 24.04 LTS  
**Academic foundation:** Zarattini, Barbon & Aziz 2024 (SSRN 4729284) — ORB + RVOL filter, Sharpe 2.81  
**Runtime:** OpenVINO GenAI in-process `.venv` (Level Zero / OneAPI) — dual-port architecture with Open WebUI

---

## What this does

Trades SPY once per session using a 15-minute ORB signal with VWAP slope confirmation. Every decision is logged to an SQLite ledger, analysed post-market by a local LLM D-A-C pipeline (Phase 1: Lead Quant Reasoner → Phase 2: Adversarial Risk Auditor), and evaluated weekly in an evolutionary strategy tournament. 

The system cannot deploy live capital until it clears a strict statistical gate (100 paper trades, WFE ≥ 0.50, DSR > 80%). The edge is verified by running all in-sample sessions through seven academic strategy replications and a 10,000-path vectorised random-trader Monte Carlo benchmark.

---

## Architecture

```text
Startup
  prepare_host.sh     Intel Arc drivers (stock Ubuntu), OpenVINO GenAI nightly, model downloader
  launch_models.sh    Start Phase 1 server (Port 8000), Phase 2 server (Port 8001), WebUI (Port 8080)
  setup.sh            Self-healing directory creation (DATA/, LOGS/), .env, health check
  verify.sh           7-step pre-flight verification suite

Daily (Automated via runner.py)
  09:00  morning_brief.py     Pre-market CLI: VIX (FRED), yield curves, CBOE P/C, CFTC, OPEX, GO/NO-GO
  09:25  markov_engine.py     Regime transition matrix refresh from local FRED VIXCLS
  09:28  trading_engine.py    60s signal poll: ORB → VWAP slope gate → execution fill simulation
  15:35  session_analyser.py  Staged Dossier D-A-C: Python Ingestion (<0.05s) → Macro Scout (Port 8001)
                              → Quant Reasoner (Port 8000) → Adversarial Audit (Port 8001)
  15:40  monte_carlo_ext.py   Galton board forward equity forecast & risk-of-ruin update
  16:30  market_data_store.py Ingest today's 1-min SPY bars into SQLite
  16:35  fred_store.py        Update FRED macro observations (VIX, rates, spreads, CPI)
  16:40  sentiment_store.py   Scrape CBOE P/C today + refresh CFTC COT current year
  Sat    tournament_eval.py   Weekly WFE/Sharpe comparison, Design Studio genetic mixer

LLM Inference — OpenVINO GenAI in .venv (Dual-Port Co-Residency in 58GB VRAM)
  Port 8000  Phase 1 Lead Quant Reasoner   (Qwen3.8-27B MTP or Qwen3.6-35B MoE)
  Port 8001  Phase 2 Adversarial Auditor   (Phi-4-mini or Mistral-Nemo-12B)
  Port 8080  Open WebUI                    (ChatGPT-style browser interface)
  serve_model.py wraps openvino_genai with OpenAI-compatible /v1 endpoints & telemetry

Data Stores — All consolidated under DATA/
  DATA/market_data.db    market_bars (1-min SPY), session_context, fred_observations,
                         cboe_pc_daily, cftc_cot_weekly
  DATA/paper_account.db  positions, orders, decisions, sessions, wfa_results,
                         llm_analysis, confirmed_patterns, monte_carlo_forecasts
```

Full deployment guide → [`docs/SETUP.md`](docs/SETUP.md)

---

## Prerequisites

| Requirement | Source | Purpose |
|---|---|---|
| Alpaca Paper Account | [alpaca.markets](https://alpaca.markets) (Free) | Market data and paper order simulation |
| FRED API Key | [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html) (Free) | Authoritative CBOE VIX, yield curves, spreads |
| Brave Search API Key | [api.search.brave.com](https://api.search.brave.com) (Free 2k/mo) | Independent macro news search (DDG fallback) |
| Intel Arc Compute Stack | Stock Ubuntu 24.04 packages (`libze-intel-gpu1`) | Level Zero GPU runtime (no 3rd-party repos) |
| OpenVINO GenAI Nightly | `pip install --pre -U openvino openvino-genai` | Native Arc matrix vectorization & MTP support |

---

## Quick Start

```bash
# 1. Host setup and model pair download
chmod +x prepare_host.sh launch_models.sh setup.sh verify.sh git_setup.sh
./prepare_host.sh             # interactive pair selector (Pair 1 or 2 recommended)

# 2. Start model servers & Open WebUI
./launch_models.sh --with-webui

# 3. Environment configuration
cp .env.example .env && nano .env
# Fill in: ALPACA_API_KEY, ALPACA_SECRET_KEY, FRED_API_KEY, BRAVE_SEARCH_API_KEY

# 4. Project setup & one-time data bootstrap
./setup.sh
python3 src/market_data_store.py --download --tickers SPY --start 2016-01-01 --end 2024-12-31 --db DATA/market_data.db
python3 src/fred_store.py        --download --db DATA/market_data.db
python3 src/sentiment_store.py   --download --db DATA/market_data.db

# 5. Algorithmic bar correction & simulation bootstrap
python3 src/data_corrector.py    --db DATA/market_data.db --ticker SPY --start 2016-01-01 --end 2022-12-31
python3 src/historical_sim.py    --start 2016-01-01 --end 2022-12-31 --db DATA/paper_account.db
# ↑ Automatically seeds WFA, Markov chains, 10,000-path random baseline & 7-strategy comparison

# 6. Verify full stack
./verify.sh                   # 7-step pre-flight verification

# 7. Start live operations
python3 src/runner.py         # single command daily orchestrator
streamlit run src/trading_dashboard.py   # live trading cockpit (separate terminal)
```

---

## Model Pairs (OpenVINO GenAI — Co-Resident in 58GB VRAM)

Both models reside in the Arc iGPU memory pool simultaneously, completely eliminating model swap latency during post-session D-A-C reviews.

| Pair | Phase 1 (Port 8000) | Phase 2 (Port 8001) | Static VRAM | Characteristics |
|---|---|---|---|---|
| **1 — Most Popular** | Qwen3.8-27B INT4 (MTP built-in) | Phi-4-mini INT4 (3.8B) | ~19GB | Dense mathematical reasoning + fast auditor |
| **2 — High-Speed MoE** | Qwen3.6-35B-A3B MoE INT4 | Mistral-Nemo-12B INT4 | ~25GB | **30+ tok/s throughput**, SWA independent critic |
| **3 — Novel Adversary** | Qwen3.6-35B-A3B MoE INT4 | LFM2.5-8B-A1B INT4 | ~23GB | Liquid AI state machine (non-transformer) auditor |

**Switch pairs without reinstalling host drivers:**
```bash
./prepare_host.sh --pair 2 --models-only   # downloads pair 2
./launch_models.sh --pair 2 --with-webui   # launches pair 2
```

---

## Data Partitions (Enforced in Code)

| Window | Dates | Operational Rule |
|---|---|---|
| **In-Sample** | 2016-01-01 → 2022-12-31 | Training only. `SealedDataError` raised if crossed during exploratory testing. |
| **Validation** | 2023-01-01 → 2024-12-31 | Walk-Forward Analysis (WFA) out-of-sample window. Untouched until Phase 3 gate. |
| **Sealed** | 2025-01-01 → Present | Never touched. Reserved exclusively for final live capital evaluation. |

---

## Phase Gates

* **Phase 2 → 3:** 100 live paper trades, WFE ≥ 0.50, Monte Carlo pass rate ≥ 85%, DSR > 80%.
* **Phase 3 → 4:** 3 consecutive rolling 20-trade windows with positive EV on live paper execution.
* **Phase 4 → 5:** Live capital deployment, Sharpe ≥ 1.0 sustained over 3 consecutive months.
