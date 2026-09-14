# Trading Income Project — Deployment Guide

**Target Hardware:** Intel NUC 14 Pro (Intel Core Ultra 5 125H · 7 Xe-cores · 64GB DDR5 · Ubuntu 24.04 LTS)  
**VRAM Configuration:** 58GB dynamically shared iGPU pool

---

## Architectural Overview

This system runs a dual-model LLM architecture hosted locally in OpenVINO GenAI:
* **Port 8000:** Phase 1 Lead Quant Strategist (Qwen3.8-27B or Qwen3.6-35B MoE)
* **Port 8001:** Phase 2 Adversarial Auditor (Phi-4-mini or Mistral-Nemo-12B)
* **Port 8080:** Open WebUI Browser Chat Interface

Data ingestion is **staged and deterministic**: SQLite tables and macroeconomic indicators are compiled in Python in `< 0.05s`, bypassing slow, timeout-prone ReAct loops.

---

## Step-by-Step Installation

### Step 1: Host Preparation & Model Downloads
```bash
chmod +x prepare_host.sh launch_models.sh setup.sh verify.sh git_setup.sh
./prepare_host.sh
```
* **Drivers:** Installs stock Ubuntu Level Zero compute drivers (`libze-intel-gpu1`, `intel-opencl-icd`). No third-party PPAs are used.
* **Kernel Guard:** Disables Panel Self Refresh (`i915.enable_psr=0`) in GRUB to prevent Meteor Lake display artifacts.
* **Model Downloads:** Downloads your chosen model pair directly from Hugging Face into `~/models/` with `HF_XET_HIGH_PERFORMANCE=1`.

*(If PSR was just disabled, reboot the machine once before proceeding).*

---

### Step 2: Start Model Servers & Open WebUI
```bash
./launch_models.sh --with-webui
```
* Compiles Level Zero GPU kernel blobs (takes ~45 seconds on cold start, < 5 seconds subsequently).
* Both models reside simultaneously in the 58GB VRAM pool.

**Verify server status:**
```bash
./launch_models.sh --status
```

---

### Step 3: Configure Environment Variables
Copy the template and add your credentials:
```bash
cp .env.example .env
nano .env
```

| Key | Purpose | Required? |
| :--- | :--- | :---: |
| `ALPACA_API_KEY` | Alpaca Paper Trading API Key | **Yes** |
| `ALPACA_SECRET_KEY` | Alpaca Paper Trading Secret Key | **Yes** |
| `FRED_API_KEY` | St. Louis Fed Macroeconomic API Key | **Yes** |
| `BRAVE_SEARCH_API_KEY` | Brave News Search API Key (DDG fallback if empty) | Optional |
| `DISCORD_WEBHOOK_URL` | Discord Channel EOD Alert Webhook | Optional |
| `LLM_BASE_URL` | `http://127.0.0.1:8000/v1` | Built-in |
| `DAC_BASE_URL` | `http://127.0.0.1:8001/v1` | Built-in |

---

### Step 4: Application Setup & Directory Verification
```bash
./setup.sh
```
* Creates self-healing directory trees: `DATA/`, `DATA/models/`, `LOGS/`, `LOGS/pids/`.
* Configures Python virtual environment (`.venv`) and verifies package dependencies.

---

### Step 5: Data Bootstrap (One-Time Run)

Execute the historical data ingestion suite:

```bash
source .venv/bin/activate

# 1. Download 1-minute historical bars for SPY
python3 src/market_data_store.py --download --tickers SPY --start 2016-01-01 --end 2024-12-31 --db DATA/market_data.db

# 2. Download FRED macro series (VIXCLS, 2Y/10Y yields, HY OAS spreads)
python3 src/fred_store.py --download --db DATA/market_data.db

# 3. Download CBOE Put/Call ratios and CFTC Commitments of Traders
python3 src/sentiment_store.py --download --db DATA/market_data.db

# 4. Correct Alpaca data anomalies (phantom spikes, stale bars, early close trims)
python3 src/data_corrector.py --db DATA/market_data.db --ticker SPY --start 2016-01-01 --end 2022-12-31

# 5. Bootstrap learning models & run academic comparison benchmarks
python3 src/historical_sim.py --start 2016-01-01 --end 2022-12-31 --db DATA/paper_account.db
```

---

### Step 6: Pre-Flight Verification
Run the automated verification suite:
```bash
./verify.sh
```
All 7 checks must report `PASSED`:
1. Port 8000 Health Check (Lead Quant Reasoner)
2. Port 8001 Health Check (Adversarial Critic)
3. Phase 1 Clean Inference (Prompt response & thinking token isolation)
4. Thinking Token Audit Logging
5. Phase 2 Adversarial Response
6. Staged Dossier Session Analyser Execution
7. Open WebUI Endpoint Accessibility

---

### Step 7: Live Operations

Launch the daily scheduled orchestrator:
```bash
python3 src/runner.py
```

Launch the Streamlit monitoring cockpit in a separate terminal:
```bash
streamlit run src/trading_dashboard.py
```

---

## Daily Schedule (All Times EST)

| Time | Script | Description |
| :--- | :--- | :--- |
| **09:00** | `morning_brief.py` | Macro regime analysis, VIX positioning, price levels, GO/NO-GO score |
| **09:25** | `markov_engine.py` | VIX transition matrix update |
| **09:28** | `trading_engine.py` | Real-time signal loop begins; awaits 09:30 open |
| **09:30–11:00** | `trading_engine.py` | 60-second AND-gate evaluation: ORB breakout + VWAP slope |
| **15:35** | `session_analyser.py` | Staged Dossier D-A-C analysis and adversarial audit |
| **15:40** | `monte_carlo_extended.py`| Monte Carlo equity forecasting & risk of ruin update |
| **16:30** | `market_data_store.py` | Daily 1-min bars stored to SQLite |
| **16:35** | `fred_store.py` | FRED macro series updated |
| **16:40** | `sentiment_store.py` | CBOE Put/Call ratios scraped |
| **Sat 08:00** | `tournament_evaluator.py` | Weekly strategy tournament & genetic parameter mixing |
