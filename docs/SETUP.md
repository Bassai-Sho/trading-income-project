# Trading Income Project — Deployment Guide

**Hardware:** Intel NUC 14 Pro · Core Ultra 5 125H · 64GB DDR5 · Intel Arc iGPU (58GB VRAM pool) · Ubuntu 24.04

---

## File manifest (23 Python files · 18,661 lines)

| File | Lines | Purpose |
|---|---|---|
| `runner.py` | 615 | Entry point — APScheduler daily orchestration |
| `trading_engine.py` | 1,568 | ORB signal gates, position management, fill simulation |
| `trading_dashboard.py` | 585 | 6-tab Streamlit cockpit |
| `morning_brief.py` | 392 | Pre-market CLI: VIX, macro, CBOE P/C, CFTC, OPEX, GO/NO-GO |
| `session_analyser.py` | 1,296 | D-A-C pipeline — presets for all three model pairs |
| `broker_interface.py` | 395 | DataProvider / BrokerClient abstraction |
| `serve_model.py` | 376 | OpenVINO GenAI OpenAI /v1 bridge — tool calling, thinking token strip |
| `strategy_registry.py` | 334 | 6 strategy variants, phase gate enforcement |
| `tournament_evaluator.py` | 483 | Weekly WFE/Sharpe comparison, Design Studio |
| `gene_mixer.py` | 731 | Evolutionary engine: crossover, mutate, D-A-C validate |
| `markov_engine.py` | 832 | Regime Markov chain, outcome chain, candle N-gram |
| `monte_carlo_extended.py` | 574 | Galton board paths, risk-of-ruin, Kelly |
| `trade_journal_extended.py` | 654 | 50-field journal, psychology flags, correlations |
| `trading_quant_toolkit_v2_4.py` | 3,682 | Core maths: EV, Kelly, WFA, DSR, Monte Carlo |
| `market_data_store.py` | 850 | Alpaca 1-min bar store with 5 quality checks |
| `data_corrector.py` | 660 | Phantom H/L, stale bars, OHLC integrity, early-close trim |
| `historical_sim.py` | 971 | IS backtest + SimBootstrapper (seeds all learning) |
| `fred_store.py` | 527 | FRED macro: 7 series including VIX, yield curve, HY spread |
| `sentiment_store.py` | 564 | CBOE put/call (2006+) + CFTC COT S&P 500 (2016+) |
| `tool_runner.py` | 709 | Brave Search primary + DDG fallback, 8 LLM tools |
| `findings_store.py` | 336 | D-A-C findings → confirmed pattern accumulation → morning brief |
| `random_baseline_sim.py` | 751 | 10,000-path vectorised random-trader Monte Carlo benchmark |
| `academic_replications.py` | 776 | 7-strategy academic comparison suite |

**Shell scripts:** `prepare_host.sh`, `launch_models.sh`, `setup.sh`, `verify.sh`, `NUC_SETUP.sh` (legacy fallback), `deploy_ovms.sh` (Docker alternative), `git_setup.sh`

---

## Data stores

**`DATA/market_data.db`** — downloaded reference data, stored once, queried forever

| Table | Source | Content |
|---|---|---|
| `market_bars` | Alpaca | SPY 1-min OHLCV 2016-2024 |
| `session_context` | Derived | PDH/PDL, ORB range, VIX regime per session |
| `fred_series` / `fred_observations` | FRED API | VIXCLS, DGS2, DGS10, T10YIE, BAMLH0A0HYM2, FEDFUNDS, CPIAUCSL |
| `cboe_pc_daily` | CBOE CSV + HTML | Daily equity put/call ratio 2006+ |
| `cftc_cot_weekly` | CFTC.gov ZIPs | Weekly S&P 500 leveraged + asset manager positioning 2016+ |

**`DATA/paper_account.db`** — simulation results and live paper trading

| Table | Content |
|---|---|
| `sim_trades` | ~3,500 IS backtest virtual trades |
| `wfa_results` | Walk-forward splits with WFE scores |
| `dac_findings` + `confirmed_patterns` | Nightly D-A-C structured findings + patterns in morning brief |
| `random_baseline_paths` / `_summary` | 10,000-path random trader comparison |
| `academic_replication_results` | 7-strategy side-by-side IS results |

---

## One-time setup

### Step 1 — Host preparation, model selection and download

```bash
chmod +x prepare_host.sh launch_models.sh setup.sh verify.sh git_setup.sh
./prepare_host.sh
```

Interactive. Select a model pair — both models stay resident in the 58GB iGPU pool simultaneously:

| Pair | Phase 1 (Port 8000) | Phase 2 (Port 8001) | VRAM | Best for |
|---|---|---|---|---|
| 1 | DeepSeek-R1-Distill-32B INT4 | Mistral-Nemo-12B INT4 | ~26GB | Deep step-by-step reasoning (recommended) |
| 2 | Qwen3-30B-A3B MoE INT4 | Mistral-Nemo-12B INT4 | ~25GB | Fast multi-turn tool loops (30+ tok/s) |
| 3 | Qwen2.5-32B-Instruct INT4 | Mistral-Small-24B INT4 | ~33GB | Peak JSON precision + deep adversarial |

`prepare_host.sh` installs Intel Arc compute runtime from **stock Ubuntu 24.04 packages only** — no third-party Intel GPU repositories, which are known to break GDM3 and cause black screens on boot. Installs `openvino-genai` into `.venv` and downloads chosen model pair from HuggingFace via `huggingface-cli` (resume-capable, checksum-verified).

The script also applies the **PSR screen corruption fix** (`i915.enable_psr=0` kernel parameter), which addresses a Meteor Lake Arc regression in kernel 6.10+ that causes pixel artifacts.

**Switch pairs later (no driver reinstall):**
```bash
./prepare_host.sh --pair 2 --models-only
```

### Step 2 — Start LLM servers

```bash
./launch_models.sh
```

Starts `serve_model.py` twice — once per port — inside `.venv`. On first run, OpenVINO compiles Level Zero GPU kernel blobs (~45 seconds). Subsequent starts complete in under 10 seconds (cached blobs in `~/models/.ov_cache`).

For DeepSeek-R1 (Pair 1): `<think>...</think>` tokens are stripped from response content and logged to `LOGS/thinking_8000.log` for audit. This prevents thinking blocks (1,000–3,000 tokens) from overflowing `max_tokens`.

```bash
./launch_models.sh --status    # check both ports
./launch_models.sh --stop      # clean shutdown
./launch_models.sh --pair 2    # switch to pair 2
```

### Step 3 — API keys

```bash
nano .env
```

`session_analyser.py` and `fred_store.py` both call `load_dotenv()` automatically.

| Key | Where | Required for |
|---|---|---|
| `ALPACA_API_KEY` + `ALPACA_SECRET_KEY` | [alpaca.markets](https://alpaca.markets) — free | All market data and paper trading |
| `FRED_API_KEY` | [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html) — free | Macro data (VIX, rates, CPI) |
| `BRAVE_SEARCH_API_KEY` | [api.search.brave.com](https://api.search.brave.com) — free 2k/month | Primary web search (DDG fallback if absent) |
| `LLM_BASE_URL` | `http://127.0.0.1:8000/v1` | Phase 1 model server |
| `DAC_BASE_URL` | `http://127.0.0.1:8001/v1` | Phase 2 adversarial server |
| `LLM_MODEL_PRIMARY` | e.g. `deepseek-r1:32b` | Matches model ID in serve_model.py |
| `LLM_MODEL_ADVERSARIAL` | e.g. `mistral-nemo:12b` | Matches model ID in serve_model.py |
| `DISCORD_WEBHOOK_URL` | Discord server settings | EOD alerts (optional) |

### Step 4 — Python environment

```bash
./setup.sh
```

Installs remaining Python dependencies (`alpaca-py`, `streamlit`, `APScheduler`, etc.), creates `DATA/` and `LOGS/`, and runs a health check against both model servers.

### Step 5 — Push to GitHub

```bash
./git_setup.sh
```

Full guide → `docs/GITHUB_SETUP.md`

---

## Data bootstrap (run once, ~2-3 hours total)

### Step 6 — Download SPY 1-minute bars

```bash
python src/market_data_store.py \
  --download --tickers SPY \
  --start 2016-01-01 --end 2024-12-31 \
  --db DATA/market_data.db
```

### Step 7 — Correct Alpaca data quality issues

```bash
python src/data_corrector.py \
  --db DATA/market_data.db --ticker SPY \
  --start 2016-01-01 --end 2022-12-31
```

Four corrections: OHLC integrity, phantom H/L (FRED VIXCLS anchor), stale bar interpolation, early-close trim (`pandas_market_calendars`). Expected 2-8% of sessions corrected.

### Step 8 — Download FRED macro data

```bash
python src/fred_store.py --download --db DATA/market_data.db
```

Requires `FRED_API_KEY`. ~30 seconds. Replaces yfinance `^VIX` across the system.

### Step 9 — Download sentiment data

```bash
python src/sentiment_store.py --download --db DATA/market_data.db
```

CBOE equity put/call CSV (2006-2019) + daily HTML scrape (2019+). CFTC COT ZIPs (2016-present, no auth).

### Step 10 — Bootstrap all learning components

```bash
python src/historical_sim.py \
  --start 2016-01-01 --end 2022-12-31 \
  --db DATA/paper_account.db
```

Runs ORB strategy against IS data (~3,500 virtual trades). Automatically triggers:
- **10,000-path vectorised random baseline** (null hypothesis benchmark)
- **7-strategy academic comparison** (Gao, Zarattini 5min ORB, Zarattini VWAP, long-only, no-retest, hybrid, random)

### Step 11 — Verify everything

```bash
./verify.sh
```

Runs 5 checks: server health, Phase 1 clean inference (no thinking token leakage), thinking audit log, Phase 2 adversarial response, session analyser syntax. All must pass before paper trading begins.

```bash
python src/academic_replications.py --report --db DATA/paper_account.db
```

**Academic comparison elimination rules (applied automatically in report):**
1. Long-only Sharpe > bidirectional hybrid → prune shorts permanently
2. No-retest Sharpe > with-retest hybrid → replace Scarface mechanic with volume confirmation
3. Hybrid Sharpe < Zarattini VWAP → pivot Phase 2 to Zarattini VWAP momentum

---

## Daily operation

```bash
./launch_models.sh              # ensure servers are running
python src/runner.py --ticker SPY --orb 15min --account 10000
```

### Automatic daily schedule (all times EST)

| Time | Component | Action |
|---|---|---|
| 09:00 | `morning_brief.py` | VIX (FRED), yield curve, HY spread, CBOE P/C, CFTC net, OPEX, confirmed D-A-C patterns, GO/NO-GO |
| 09:25 | `markov_engine.py` | Regime chain refresh from FRED VIXCLS |
| 09:28 | `trading_engine.py` | ORB signal loop every 60s |
| 15:35 | `session_analyser.py` | Phase 1 (Port 8000) analysis → Phase 2 adversarial (Port 8001) → findings extraction |
| 15:40 | `monte_carlo_extended.py` | Forecast update |
| 16:30 | `market_data_store.py` | Append today's bars |
| 16:35 | `fred_store.py` | Update last 30 days of FRED macro |
| 16:40 | `sentiment_store.py` | Scrape CBOE P/C + refresh CFTC current-year ZIP |
| Saturday 08:00 | `tournament_evaluator.py` | Weekly WFE/Sharpe comparison, Design Studio trigger |

### Session analyser D-A-C presets

```bash
python src/session_analyser.py --preset nuc-pair1    # Pair 1: DeepSeek-R1 + Mistral-Nemo
python src/session_analyser.py --preset nuc-pair2    # Pair 2: Qwen3-MoE + Mistral-Nemo
python src/session_analyser.py --preset nuc-pair3    # Pair 3: Qwen2.5 + Mistral-Small
python src/session_analyser.py --probe               # test adversarial backend, exit
```

### Dashboard

```bash
streamlit run src/trading_dashboard.py   # http://localhost:8501
```

---

## Useful commands

```bash
# Server management
./launch_models.sh --status
./launch_models.sh --stop
./launch_models.sh --logs       # tail both model logs
./verify.sh --quick             # steps 1-4 only (skip session analyser)

# Data stores
python src/market_data_store.py --status   --db DATA/market_data.db
python src/fred_store.py        --status   --db DATA/market_data.db
python src/sentiment_store.py   --status   --db DATA/market_data.db
python src/fred_store.py        --date 2022-06-13 --db DATA/market_data.db

# Simulation reports
python src/historical_sim.py        --report --db DATA/paper_account.db
python src/random_baseline_sim.py   --report --db DATA/paper_account.db
python src/academic_replications.py --report --db DATA/paper_account.db

# DeepSeek-R1 thinking audit
tail -f LOGS/thinking_8000.log
```

---

## Data boundaries (enforced in code)

| Window | Dates | Status |
|---|---|---|
| In-sample | 2016-01-01 → 2022-12-31 | Training only. `SealedDataError` raised if crossed. |
| WFA validation | 2023-01-01 → 2024-12-31 | Untouched until Phase 3 gate. |
| Sealed test | 2025-01-01 → present | Never touch. Final live-capital evaluation only. |

---

## Phase gates

| Gate | Condition |
|---|---|
| Phase 2 → 3 | 100 live paper trades + WFE ≥ 0.50 on live data |
| Phase 3 → 4 | 3 months consistent positive EV, DSR > 0 |
| Phase 4 → 5 | Live capital deployed, 6-month track record |

---

## Academic sources

| Paper | Strategy | Result | Role |
|---|---|---|---|
| Zarattini et al. 2024 (SSRN 4729284) | 5-min ORB + RVOL filter (Stocks in Play) | Sharpe 2.81, alpha 36%, beta ≈ 0 | Signal mechanism |
| Zarattini et al. 2024 (SSRN 4824172) | VWAP intraday momentum on SPY | Sharpe 1.33, 19.6% p.a. | Exit design, SPY validation |
| Gao, Han, Li, Zhou 2018 (JFE) | First/last 30-min ETF momentum | Significant after fees | Academic comparison floor |
| Maroy 2025 (SSRN 5095349) | VWAP + ladder exits | Sharpe 3.0+ | Phase 3 exit enhancement |
| Barber et al. 2011 (Taiwan, 15yr) | Retail outcomes | 84% lose, <1% net positive | Risk framing |
| Chague et al. 2019 (Brazil) | Retail outcomes | 97% lose, no learning curve | Risk framing |

**Gap note:** Our strategy applies the ORB mechanism (SSRN 4729284, individual stocks) to SPY (SSRN 4824172's instrument). This combination is unvalidated in peer-reviewed literature. The 7-strategy academic comparison and WFE gate are the empirical adjudicators.

---

*Trading Income Project · Systematic ORB strategy · OpenVINO GenAI inference · validated before live capital · Phase 2 paper trading*
