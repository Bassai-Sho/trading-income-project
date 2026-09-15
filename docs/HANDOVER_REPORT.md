# Trading Income Project — LLM Handover Report

**Date:** September 2026  
**Status:** Operational / Paper Trading Stage  
**Primary Architecture:** Staged Dossier D-A-C Pipeline on Intel Arc iGPU (OpenVINO GenAI)

---

## 1. System Overview

The Trading Income Project is an automated, evidence-based intraday trading system designed to trade SPY (S&P 500 ETF) on a 15-minute Opening Range Breakout (ORB) strategy with VWAP confirmation.

The system is designed around two principles:
1. **Single-Command Orchestration:** `runner.py` (or the `trading-runner` systemd daemon) manages the entire daily lifecycle automatically according to the US market clock.
2. **Deterministic Execution + Cognitive Review:** Real-time signal execution is 100% deterministic (pure Python/SQLite, zero LLM latency). Nightly post-market review uses a dual-model local LLM architecture (Lead Quant Reasoner + Independent Adversary) running on the local Intel Arc GPU.

---

## 2. Quantitative Strategy Specification

* **Instrument:** SPY (NYSE Arca)
* **Timeframe:** 5-minute bars during 09:30–11:00 EST. Maximum 1 trade per session. No overnight holds.
* **Signal AND-Gate:**
  1. *ORB Break:* 5-minute close above the 09:30–09:45 high (long) or below the low (short).
  2. *VWAP Slope Gate:* Intraday VWAP sloping in the direction of the breakout over a 3-bar lookback.
  3. *Retest Mechanic:* Price retests the breakout level within 0.15% tolerance without closing back inside the range.
* **Risk & Sizing:**
  * Base Risk: 1.0% of account equity per trade (fixed fractional; compounding).
  * Commodity Override: 0.5% max risk on commodities (GC=F, SI=F).
  * Volatility Adjustment: Sizing scales down if VIX > 25 (50%) or VIX > 35 (25%).
  * Daily Stop Rule: If cumulative session loss reaches -3.0%, the session halts immediately.
  * Consecutive Loss Rule: 3 consecutive losses force a mandatory 30-minute operational pause.
* **Reward/Risk:** Minimum 2.0:1 R:R target with 3-stage ladder partial exits (1R, 2R, 3R) and VWAP trailing stops.

---

## 3. Machine Learning & Inference Stack

### OpenVINO GenAI vs. Ollama
The project migrated from IPEX-LLM Ollama to native **OpenVINO GenAI (`.venv`)**:
* Compiled 1:1 against Intel Level Zero vector matrix engines (`DP4a`).
* Stock Ubuntu 24.04 packages only (`libze-intel-gpu1`, `intel-opencl-icd`). Eliminates third-party display driver conflicts.
* Both models reside co-resident in the 58GB unified VRAM pool on ports 8000 and 8001.

### Active Model Lineup (September 2026)

| Tier | Model ID | Port | Format | Role |
| :--- | :--- | :---: | :---: | :--- |
| **Phase 1 (Lead Quant)** | `qwen3.8:27b` | 8000 | INT4 OV | Deep quantitative reasoning, regime analysis, action proposals |
| **Phase 2 (Adversarial Critic)** | `phi-4-mini:int4` | 8001 | INT4 OV | Sceptical review, Rule 17 sample size enforcement |
| **Phase 2 Alternate** | `mistral-nemo:12b` | 8001 | INT4 OV | Sliding Window Attention (SWA) cross-architecture auditor |
| **Chat Interface** | Open WebUI | 8080 | Pip App | Side-by-side model chat, prompt verification |

---

## 4. The Staged Dossier Architecture (D-A-C Engine)

Following empirical latency testing on the NUC 14 Pro, the multi-turn ReAct agent loop in `session_analyser.py` was replaced with the **Staged Dossier Pipeline**:

```text
[ SQLite Ingestion ] ──────► Python builds complete session dossier (< 0.05s)
[ Macro Extraction ] ──────► Phi-4-mini synthesizes web news into 2 bullets (~2.0s)
[ Quant Reasoning  ] ──────► Qwen3.8-27B analyzes dossier in single pass with <think> (~30s)
[ Adversarial Audit] ──────► Phi-4-mini stress-tests report and enforces Rule 17 (~12s)
```

**Total runtime dropped from >300 seconds (with timeout errors) to <45 seconds.**

### The Empirical Sample-Size Invariant (Rule 17)
The Adversarial Validator enforces this hard rule:
> *If total historical trades in the journal for a strategy variant is fewer than 20 (n < 20), any proposed parameter mutation (stop distance, profit target, indicator threshold) is automatically classified as `REJECTED: PREMATURE DUE TO SAMPLE SIZE`.*

---

## 5. Audit History & Bug Fix Register

### Pass 1 & 2 (Historical Initial Architecture Fixes)
* Fixed dictionary slicing `TypeError` in `trading_engine.py`.
* Fixed undefined `vix_data` `NameError` in `morning_brief.py`.
* Fixed `DATA/` database splits in `runner.py` and `trading_engine.py`.
* Fixed bash operator precedence in `launch_models.sh`.
* Fixed missing `--preset` arguments in `session_analyser.py`.

### Pass 3 (Production Readiness & Infrastructure Fixes)

| # | Component | Root Cause | Resolution |
| :--- | :--- | :--- | :--- |
| **1** | `prepare_host.sh` / `setup.sh` | Heuristic `.bin > 100MB` check flagged incomplete downloads as finished. | Replaced with native `ov.Core().read_model()` graph integrity inspection. Merged into unified `setup.sh`. |
| **2** | `setup.sh` | `hf_transfer` deprecated in `huggingface_hub`. | Removed `hf_transfer` and enabled `HF_XET_HIGH_PERFORMANCE=1`. |
| **3** | `setup.sh` | Model card opset-16 mismatch on Qwen3.8 caused `SIGSEGV` exit 139. | Added OpenVINO GenAI nightly wheel index to host setup. |
| **4** | `setup.sh` | Unbound variable `$OLLAMA_INSTALLED` triggered crash under `set -u`. | Removed variable and validated OpenVINO ports 8000/8001 directly. |
| **5** | `setup.sh` | Missing `LOGS/pids` directory caused process manager failures. | Added `LOGS/pids` to self-healing directory creation list. |
| **6** | `launch_models.sh` | Unclosed quotes around `--with-webui` output broke bash parsing. | Cleaned quoting, added PID management for WebUI. |
| **7** | `launch_models.sh` | Background process crashes caused launcher to wait 180s in vain. | Added PID liveness polling to fail fast and dump logs immediately. |
| **8** | `serve_model.py` | `max_num_batched_tokens = 2**31` integer overflow in `SchedulerConfig`. | Removed invalid scheduler config; loaded as clean `LLMPipeline`. |
| **9** | `serve_model.py` | `TOOL_SYSTEM_PREFIX` contained unescaped `{ and end with }` causing `KeyError`. | Replaced `.format()` with literal `.replace("{tools_json}", ...)`. |
| **10** | `serve_model.py` | `async def` endpoints starved the asyncio event loop during GPU compute. | Converted routes to synchronous `def` protected by `threading.Lock()`. |
| **11** | `verify.sh` | Bash arithmetic post-increment `(( PASS++ ))` returned 0, triggering `set -e` abort. | Replaced all instances with `(( PASS += 1 ))` and `(( FAIL += 1 ))`. |
| **12** | `session_analyser.py` | Multi-turn ReAct loop exceeded client timeout (120s) on 27B model. | Replaced with single-pass Staged Dossier; bumped client timeout to 600s. |
| **13** | `session_analyser.py` | Missing `--discord` in `argparse` caused crash when called by `runner.py`. | Added `--discord` to parser with non-blocking webhook dispatcher. |
| **14** | `trading_engine.py` | `_trigger_analyser` hardcoded LM Studio port 1234 instead of OpenVINO. | Updated to pass `--preset nuc-pair1` dynamically. |
| **15** | `trading_engine.py` | Missing deduplication caused `_trigger_analyser` to spawn every 60s post-close. | Added `analyser_triggered` state flag. |
| **16** | `tournament_evaluator.py` | Line 217 referenced out-of-scope `report['design_studio']`. | Changed reference to `studio_result['message']`. |
| **17** | `markov_engine.py` | `fit_from_fred()` referenced undefined `log` logger. | Added `import logging; log = logging.getLogger("markov_engine")`. |
| **18** | `tool_runner.py` | Subprocess working directory set to `src/` broke `DATA/` relative paths. | Set `cwd` to project root. |
| **19** | `trading_dashboard.py` | `Styler.applymap` removed in Pandas 2.2+. | Switched to dynamic `getattr(styler, "map", getattr(styler, "applymap"))`. |
| **20** | Across 7 Modules | Bare `.db` paths defaulted to root folder. | Standardized all defaults to `"DATA/*.db"`. |

---

## 6. Operational Execution

* **Unified Bootstrap:** `./setup.sh` handles host runtime, virtual environment, API keys, model downloads, and one-shot data bootstrap.
* **Server Verification:** `./launch_models.sh --with-webui` and `./verify.sh` confirm all endpoints and display live `tok/s` metrics (7/7 passed).
* **Automated Master Loop:** `python3 src/runner.py` or `sudo systemctl start trading-runner` handles daily scheduling and execution automatically.
