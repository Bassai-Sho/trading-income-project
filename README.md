# Trading Income Project

Systematic, evidence-based day trading system built around the Opening Range Breakout (ORB) strategy on SPY, validated against academic benchmarks before live capital deployment.

**Hardware:** Intel NUC 14 Pro · Core Ultra 5 125H · 64GB DDR5 · Intel Arc iGPU (58GB VRAM pool) · Ubuntu 24.04 LTS  
**Academic Foundation:** Zarattini, Barbon & Aziz 2024 (SSRN 4729284) — ORB + RVOL filter, Sharpe 2.81  
**Runtime:** OpenVINO GenAI in-process `.venv` (Level Zero / OneAPI) — dual-port architecture with Open WebUI

---

## What This Does

Trades SPY once per session using a 15-minute ORB signal with VWAP slope confirmation. Every decision is logged to an SQLite ledger, analysed post-market by a local LLM D-A-C pipeline (Phase 1: Lead Quant Reasoner → Phase 2: Adversarial Risk Auditor), and evaluated weekly in an evolutionary strategy tournament.

The system cannot deploy live capital until it clears a strict statistical gate (100 paper trades, WFE ≥ 0.50, DSR > 80%). The edge is verified by running all in-sample sessions through seven academic strategy replications and a 10,000-path vectorised random-trader Monte Carlo benchmark.

---

## Automated Master Orchestrator (`runner.py`)

**You do not need to manually execute individual Python scripts.** 

A single command (`python3 src/runner.py` or the `trading-runner` systemd service) runs the entire system according to the US market clock (all times EST):

```text
Daily Schedule (Automated by runner.py)
  09:00  morning_brief.py     Pre-market brief: VIX (FRED), yield curves, CBOE P/C, CFTC, OPEX, GO/NO-GO
  09:25  markov_engine.py     Regime transition matrix refresh from local FRED VIXCLS
  09:28  trading_engine.py    60s signal evaluation loop: ORB → VWAP slope gate → execution fill simulation
  15:35  session_analyser.py  Staged Dossier D-A-C: Python Ingestion (<0.05s) → Macro Scout (Port 8001)
                              → Quant Reasoner (Port 8000) → Adversarial Audit (Port 8001)
  15:40  monte_carlo_ext.py   Galton board forward equity forecast & risk-of-ruin update
  16:30  market_data_store.py Ingest today's 1-min SPY bars into SQLite
  16:35  fred_store.py        Update FRED macro observations (VIX, rates, spreads, CPI)
  16:40  sentiment_store.py   Scrape CBOE P/C today + refresh CFTC COT current year
  Sat    tournament_eval.py   Weekly WFE/Sharpe comparison, Design Studio genetic mixer
```

*(Individual scripts in `src/` can still be executed manually whenever ad-hoc testing or historical simulation is desired).*

---

## Local Dual-Model Architecture (OpenVINO GenAI)

Both models reside simultaneously in the 58GB unified VRAM pool (zero swap latency between analysis and adversarial audit):

* **Port 8000 (Phase 1 — Lead Quant Reasoner):** `qwen3.8:27b` (Deep mathematical reasoning, regime synthesis, action proposals)
* **Port 8001 (Phase 2 — Adversarial Auditor):** `phi-4-mini:int4` (Independent critic enforcing the Rule 17 $n < 20$ sample-size invariant)
* **Port 8080 (Browser Chat Interface):** Open WebUI with real-time SSE token streaming

---

## Quick Start (3 Steps)

### Step 1: Run Unified Interactive Setup
```bash
chmod +x setup.sh launch_models.sh verify.sh git_setup.sh
./setup.sh
```
* Installs Level Zero GPU runtime and applies GRUB display guards.
* Sets up `.venv` with all dependencies and OpenVINO GenAI nightly.
* Interactively prompts for your API keys (`ALPACA_API_KEY`, `FRED_API_KEY`, `BRAVE_SEARCH_API_KEY`, `HF_TOKEN`, `DISCORD_WEBHOOK_URL`) and saves them to `.env`.
* Downloads your selected OpenVINO model pair into `~/models/`.
* **Automated Data Bootstrap:** Automatically downloads 10 years of SPY bars, FRED series, and CBOE sentiment, and seeds historical backtests in one shot.

### Step 2: Start Model Servers & Open WebUI
```bash
./launch_models.sh --with-webui
./verify.sh
```
* Compiles Level Zero GPU kernel blobs (<45s cold start, <5s subsequent).
* Runs 7 automated pre-flight checks with live token-per-second (`tok/s`) metrics.

### Step 3: Launch Live System
```bash
python3 src/runner.py                    # master orchestrator
streamlit run src/trading_dashboard.py   # live monitoring cockpit (separate terminal)
```

*(To auto-start on boot as a background service: `sudo systemctl enable --now trading-runner`).*

---

## Model Pairs

| Pair | Phase 1 (Port 8000) | Phase 2 (Port 8001) | Static VRAM | Characteristics |
|---|---|---|---|---|
| **1 — Most Popular** | Qwen3.8-27B INT4 (MTP built-in) | Phi-4-mini INT4 (3.8B) | ~19GB | Dense mathematical reasoning + fast auditor |
| **2 — High-Speed MoE** | Qwen3.6-35B-A3B MoE INT4 | Mistral-Nemo-12B INT4 | ~25GB | **30+ tok/s throughput**, SWA independent critic |
| **3 — Novel Adversary** | Qwen3.6-35B-A3B MoE INT4 | LFM2.5-8B-A1B INT4 | ~23GB | Liquid AI state machine (non-transformer) auditor |

**Switch pairs anytime:**
```bash
./setup.sh --pair 2 --models-only
./launch_models.sh --pair 2 --with-webui
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
