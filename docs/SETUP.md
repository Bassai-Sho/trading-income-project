# Trading Income Project — Deployment Guide

**Target Hardware:** Intel NUC 14 Pro (Intel Core Ultra 5 125H · 7 Xe-cores · 64GB DDR5 · Ubuntu 24.04 LTS)  
**VRAM Pool:** 58GB dynamically shared iGPU pool (Level Zero / OneAPI)

---

## Operational Architecture

The Trading Income Project is architected for **zero manual maintenance**:
* **One Master Runner (`runner.py`):** Automatically schedules and triggers pre-market briefs, market hours trading, and post-market D-A-C analysis.
* **Staged Dossier Pipeline:** Ingests local SQLite data in `<0.05s`, bypassing slow, timeout-prone ReAct loops.
* **Local Dual-Model Inference:** `qwen3.8:27b` (Port 8000) and `phi-4-mini:int4` (Port 8001) reside permanently in VRAM with Open WebUI (Port 8080).

---

## Step-by-Step Installation

### Step 1: Run the Unified Setup Script
```bash
chmod +x setup.sh launch_models.sh verify.sh git_setup.sh
./setup.sh
```

The interactive wizard steps through all 8 stages automatically:
1. **Directory Scaffolding:** Creates self-healing trees (`DATA/`, `DATA/models/`, `LOGS/`, `LOGS/pids/`, `~/models/`).
2. **Compute Runtime:** Installs stock Ubuntu Level Zero compute packages (`libze-intel-gpu1`, `intel-opencl-icd`) and configures GPU group permissions.
3. **Kernel Guard:** Disables Panel Self Refresh (`i915.enable_psr=0`) in GRUB to prevent Meteor Lake display artifacts.
4. **Python Environment:** Configures `.venv` with `requirements.txt` (Streamlit included) and OpenVINO GenAI nightly.
5. **Interactive Keys:** Prompts and saves `ALPACA_API_KEY`, `FRED_API_KEY`, `BRAVE_SEARCH_API_KEY`, `HF_TOKEN`, and `DISCORD_WEBHOOK_URL` into `.env`.
6. **Model Pair Download:** Downloads your selected OpenVINO model pair into `~/models/`.
7. **Automated One-Shot Data Bootstrap:** Prompts to automatically download SPY bars, FRED macro series, CBOE sentiment, and seed historical backtests in one pass.
8. **Systemd Service Option:** Optionally configures `trading-runner.service` for background daemon management.

*(If PSR was disabled during setup, reboot the machine once before launching servers).*

---

### Step 2: Start Model Servers & Open WebUI
```bash
./launch_models.sh --with-webui
```
* Compiles Level Zero GPU kernel blobs (<45s cold start, <5s subsequent).
* Both models reside simultaneously in the 58GB VRAM pool.

**Verify server health:**
```bash
./launch_models.sh --status
```

---

### Step 3: Run Pre-Flight Verification
```bash
./verify.sh
```
Verifies all 7 components:
1. Port 8000 Health (Lead Quant Reasoner)
2. Port 8001 Health (Adversarial Critic)
3. Phase 1 Clean Inference & `tok/s` Throughput
4. Thinking Token Isolation
5. Phase 2 Adversarial Response & `tok/s` Throughput
6. Staged-Dossier Engine (`session_analyser.py`)
7. Open WebUI Endpoint Accessibility

---

### Step 4: Start Daily Live Operations

You have two choices for running the automated system:

#### Option A: Run as a Linux Systemd Service (Recommended for NUC)
If you enabled the systemd option during `./setup.sh`:
```bash
sudo systemctl enable --now trading-runner
```
* Runs silently in the background.
* Starts automatically on system boot.
* Auto-restarts on unexpected crashes.
* Check status anytime: `sudo systemctl status trading-runner`
* Follow logs: `journalctl -u trading-runner -f`

#### Option B: Run in Terminal
```bash
python3 src/runner.py
```

Launch the Streamlit monitoring cockpit in a separate terminal:
```bash
streamlit run src/trading_dashboard.py
```

---

## Daily Schedule (Automated by `runner.py`)

| Time (EST) | Action | Automated Execution |
| :--- | :--- | :--- |
| **09:00** | `morning_brief.py` | Macro regime analysis, VIX positioning, price levels, GO/NO-GO score |
| **09:25** | `markov_engine.py` | VIX transition matrix updated from FRED VIXCLS |
| **09:28** | `trading_engine.py` | Real-time signal loop begins; awaits 09:30 open |
| **09:30–11:00** | `trading_engine.py` | 60-second AND-gate polling: ORB breakout + VWAP slope |
| **15:35** | `session_analyser.py` | Staged Dossier D-A-C analysis and adversarial audit |
| **15:40** | `monte_carlo_extended.py`| Monte Carlo equity forecasting & risk-of-ruin update |
| **16:30** | `market_data_store.py` | Daily 1-min bars stored to SQLite |
| **16:35** | `fred_store.py` | FRED macro series updated |
| **16:40** | `sentiment_store.py` | CBOE Put/Call ratios scraped |
| **Sat 08:00** | `tournament_evaluator.py` | Weekly strategy tournament & genetic parameter mixing |

---

## Manual Diagnostic Commands

If you ever wish to test an individual component outside the automated schedule:

```bash
# Run morning brief manually
python3 src/morning_brief.py --ticker SPY --orb 15min

# Run post-market session analysis manually
python3 src/session_analyser.py --preset nuc-pair1

# Run historical simulation report
python3 src/historical_sim.py --report --db DATA/paper_account.db

# Check local FRED macro observations
python3 src/fred_store.py --status --db DATA/market_data.db

# Inspect recent Step 5 verification log
cat /tmp/verify_session.log
```
