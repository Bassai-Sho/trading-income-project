# Trading Income Project — LLM Handover Report

**Date:** September 2026  
**Purpose:** Self-contained review brief for an independent LLM.  
**Status:** Phase 2 — data bootstrap stage. No live trades placed yet.

---

## 1. What this project is

Adam is building a systematic day trading system with the goal of supplemental income, running alongside a completely separate passive index-fund investing programme (different capital pools, zero cross-contamination). The project is explicitly exploratory and skill-building — success is defined as determining through disciplined, data-driven testing whether active trading can reliably produce income.

**Core principle:** No single signal confers edge. Only win rate and reward-to-risk combinations that clear positive expected value, wrapped in disciplined sizing and validated against a proper data sample, constitute genuine edge.

---

## 2. Strategy design

**Instrument:** SPY (S&P 500 ETF)  
**Timeframe:** Intraday only, single trade per session, no overnight holds  
**Signal:** 15-minute Opening Range Breakout with three AND-gate filters:

```
1. ORB break confirmed  — price closes above/below the 09:30–09:45 range
2. VWAP gate           — VWAP slope in direction of break
3. Retest mechanic     — price retests the ORB level and holds
```

*Note: The retest mechanic (from practitioner Scarface Trades, not academic) is under empirical review — the `orb_15min_no_retest` variant in `academic_replications.py` will settle whether it is creating adverse selection.*

**Stop:** 1R (ORB range), **Target:** 2R, **Commission model:** 0.016R  
**Position sizing:** 1% max risk per trade  
**VIX regime:** VIX > 20 → 30-min ORB window; VIX > 35 → 25% position size

---

## 3. Academic foundation

**SSRN 4729284 (Zarattini, Barbon & Aziz 2024):** ORB on Stocks in Play (RVOL ≥ 100%, news-catalyst individual equities). Sharpe 2.81, annualised alpha 36%, beta ≈ 0. Period: 2016-2023.

**SSRN 4824172 (Zarattini, Aziz & Barbon 2024):** VWAP intraday momentum on SPY. Sharpe 1.33, 19.6% p.a. Period: 2007-2024.

**Gao et al. 2018 (JFE):** First 30-min return sign predicts last 30-min on liquid ETFs. Significant after fees.

**Maroy 2025 (SSRN 5095349):** VWAP + ladder exits. Sharpe 3.0+. Single source, no independent replication yet.

**Retail failure context (both ESTABLISHED):**  
Barber et al. 2011: 84% of 360,000 Taiwanese day traders lost over 15 years.  
Chague et al. 2019: 97% of Brazilian day traders lost; no learning curve observed.

**Critical gap:** SSRN 4729284 trades individual Stocks in Play with RVOL filters. We apply the ORB mechanism to SPY. This combination is not directly peer-reviewed. The 7-strategy academic comparison and WFE gate are the empirical adjudicators.

---

## 4. Data infrastructure

All stored in SQLite under `DATA/`. No yfinance at 09:00 — all reference data is local.

**`DATA/market_data.db`:**
- `market_bars` — Alpaca 1-min SPY bars 2016-2024
- `session_context` — PDH/PDL, ORB range, VIX regime per session
- `fred_observations` — 7 FRED series (VIXCLS, DGS2, DGS10, T10YIE, BAMLH0A0HYM2, FEDFUNDS, CPIAUCSL)
- `cboe_pc_daily` — CBOE equity put/call ratio 2006+
- `cftc_cot_weekly` — CFTC COT S&P 500 leveraged + asset manager net 2016+

**`DATA/paper_account.db`:**
- `sim_trades`, `wfa_results`, `monte_carlo_forecasts`
- `dac_findings`, `confirmed_patterns` (D-A-C self-reinforcing loop)
- `random_baseline_paths` / `_summary` (10,000-path null hypothesis)
- `academic_replication_results` (7-strategy IS comparison)

**Data timeline:**

| Window | Dates | Status |
|---|---|---|
| In-sample | 2016-2022 | Training. `SealedDataError` if crossed. |
| Validation | 2023-2024 | Untouched until Phase 3. |
| Sealed | 2025+ | Never touch. |

---

## 5. LLM inference — OpenVINO GenAI

**Why OpenVINO GenAI (not IPEX-LLM Ollama):**
- Intel archived IPEX-LLM January 2026. OpenVINO is Intel's active production framework.
- Fully in-process (`.venv`) — no Docker, no daemon, no registry version checks (no Error 412).
- `prepare_host.sh` uses only stock Ubuntu 24.04 packages — no third-party Intel GPU repo that caused GDM3 black-screen issues.
- Intel INT4 symmetric quantisation compiled 1:1 into Arc DP4a vector engines: ~1.5–1.8× faster than llama.cpp Vulkan.

**`serve_model.py`** — the OpenAI /v1 bridge:
- Wraps `openvino_genai.LLMPipeline` with FastAPI + uvicorn
- Serves `/v1/chat/completions` and `/v1/models` (standard OpenAI client compatible)
- Uses each model's own `tokenizer.apply_chat_template()` — correct for Qwen (ChatML), DeepSeek-R1, and Mistral ([INST])
- Tool calling: injects tool definitions into system prompt, parses JSON `{"name": ..., "arguments": ...}` from output, returns OpenAI `tool_calls` format
- **Thinking token handling:** strips `<think>...</think>` blocks from response content, logs to `LOGS/thinking_8000.log` — prevents 1,000–3,000 token thinking blocks from overflowing `max_tokens`
- Hardware optimisations: `PERFORMANCE_HINT=LATENCY`, `KV_CACHE_PRECISION=u8` (50% memory bandwidth reduction), `CACHE_DIR` persistent kernel blobs (< 2s warm start)

**Three frontier model pairs (both models resident in 58GB VRAM simultaneously):**

| Pair | Phase 1 Port 8000 | Phase 2 Port 8001 | VRAM | Notes |
|---|---|---|---|---|
| 1 — Thinking Quant | DeepSeek-R1-Distill-32B INT4 | Mistral-Nemo-12B INT4 | ~26GB | o1-class reasoning. Thinking logged. **Recommended.** |
| 2 — High-Speed | Qwen3-30B-A3B MoE INT4 | Mistral-Nemo-12B INT4 | ~25GB | 30+ tok/s. Fast tool loops. |
| 3 — Heavyweight | Qwen2.5-32B-Instruct INT4 | Mistral-Small-24B INT4 | ~33GB | Peak JSON precision + deep critique. |

**Architectural independence (D-A-C principle):** All Phase 2 models are Mistral-family (Sliding Window Attention, different tokeniser from Qwen/DeepSeek). This prevents intra-family sycophancy — the failure mode where a model confirms rather than challenges conclusions drawn by a model with the same training distribution.

**Session analyser presets:** `--preset nuc-pair1` / `nuc-pair2` / `nuc-pair3` configure both port endpoints and model IDs.

---

## 6. D-A-C pipeline (nightly analysis)

`session_analyser.py` runs at 15:35 EST after each session.

**Architecture:**
- Phase 1 (Port 8000): initial analysis with 8 tools
- Phase 2 (Port 8001): adversarial stress test — different model architecture, `dac_base_url` routes to separate port
- Findings extraction: second call with `format:json` at temperature 0.05
- `load_dotenv()` called at import — `.env` loaded automatically

**Tools available:** `web_search` (Brave primary, DDG fallback), `fetch_url`, `run_toolkit`, `run_script`, `get_session_data`, `get_wfa_results`, `get_rolling_stats`, `get_journal_stats`

**Reliability features:** pre-flight health check (DB/search/toolkit), tool loop guard (identical-args detection), structured findings extraction

**Self-reinforcing loop:** Findings in `dac_findings`. Same `(area, observation_hash)` appearing 3+ times in 20 sessions → `confirmed_patterns`. Surfaced in morning brief before GO/NO-GO.

**Web search:** Brave Search API primary (independent index, 2,000 free queries/month, reliable JSON). DuckDuckGo automatic fallback (no API key). Set `BRAVE_SEARCH_API_KEY` in `.env`.

---

## 7. Simulation and validation framework

**Random baseline (10,000 paths):** Truly vectorised — `R_matrix` shape `(n_sessions, max_bars, 2)` sampled via 2D NumPy advanced indexing across all paths simultaneously. No Python loops in the sampling phase. SE of 99th-percentile estimator: ~0.100 (vs 0.315 at 1,000 paths).

**7-strategy academic comparison:**

| # | Strategy | Based on | Elimination rule |
|---|---|---|---|
| 1 | Random bidirectional | Null hypothesis | Floor |
| 2 | Gao first/last 30-min | JFE 2018 | Must beat |
| 3 | Zarattini 5-min ORB (SPY proxy) | SSRN 4729284 | — |
| 4 | Zarattini VWAP momentum | SSRN 4824172 | Hybrid < this → simplify |
| 5 | Long-only 15-min ORB | Independent review | Beats bidirectional → prune shorts |
| 6 | 15-min ORB no-retest | Adverse selection test | Beats with-retest → remove mechanic |
| 7 | Hybrid 15-min ORB + VWAP | Our system | Target |

**WFE gate:** ≥ 0.50 across 6 IS splits. PSR ≥ 0.95 (Bailey & López de Prado) recommended as Phase 3 upgrade.

**DSR:** Implemented in `trading_quant_toolkit_v2_4.py`. Adjusts for multiple testing, skewness, kurtosis.

---

## 8. Bug fixes (two review passes — 18 bugs total)

### Pass 1 (12 bugs across 8 files)

| # | File | Bug | Fix |
|---|---|---|---|
| 1 | `trading_engine.py` | `ctx[:50]` slicing a dict → `TypeError` every bar | `ctx.get("reason","")[:60]` |
| 2 | `morning_brief.py` | `vix_data` undefined → `NameError` crash | Bundle `vix_data` dict at end of VIX section |
| 3 | `tournament_evaluator.py` | `_phase_summary()` references `report` out of scope | Added `studio_result` parameter |
| 4 | `trading_dashboard.py` | `re.search()` in Tab 6, `import re` missing | `import re` at top |
| 5 | `random_baseline_sim.py` | `COMMISSION_R` undefined | `COMMISSION_R: float = 0.016` at module level |
| 6 | `academic_replications.py` | `simulate_hybrid_session()` missing `vwap` arg; returned `list` not `dict` | Added `_compute_vwap`, returns `trades[0]` |
| 7 | `markov_engine.py` | `self.steady_state()` doesn't exist | `zip(self.STATES, self.stationary)` |
| 8 | `historical_sim.py` | `store_path=` kwarg but function expects `store_db=` | `store_path=` → `store_db=` |
| 9a | `trading_engine.py` | Importing `v2_2` / `v2_1` — both missing | → `v2_4` |
| 9b | `morning_brief.py` | Importing `v2_3` — missing | → `v2_4` |
| 10 | `fred_store.py` | `isinstance(cols, tuple)` never true — MultiIndex never flattened | `isinstance(cols, pd.MultiIndex)` |
| 11 | `runner.py` | Bare `"market_data.db"` splits database | → `"DATA/market_data.db"` |
| 12 | `morning_brief.py` | Duplicate `_section("2.")` header | Removed duplicate |

### Pass 2 (6 bugs in session_analyser.py)

| # | Bug | Fix |
|---|---|---|
| 1 | `AttributeError: Namespace has no 'preset'` on every EOD run | Added `--preset`, `--dac-model`, `--dac-url`, `--dac-key`, `--probe` to argparse |
| 2 | Phase 2 always used Phase 1 model (D-A-C bypass) | `dac_url/dac_key/dac_model` resolved; `model=dac_model` in stress test |
| 3 | Discord/Telegram never sent despite `.env` configured | `load_dotenv()` at import; CONFIG reads env vars |
| 4 | 1-2 tok/s on 58GB pool | `num_ctx=8192`, `num_batch=2048` in `extra_body` |
| 5 | `test_adversarial_backend()` hardcoded model | `model or os.environ.get("OLLAMA_MODEL_ADVERSARIAL", ...)` |
| 6 | DB written to `paper_account.db` in CWD | → `"DATA/paper_account.db"` |

### Additional fix (launch_models.sh)

Bash operator precedence bug: `A || B && C` evaluates as `A || (B && C)`. When `A` is true (deepseek model), `||` short-circuits and `THINK_LOG` is never set. Fixed to `if [[ ... ]] || [[ ... ]]; then THINK_LOG=...; fi`.

---

## 9. What is uncertain or potentially weak

**9.1 The RVOL filter is absent.** Zarattini 4729284 attributes most of the Sharpe 2.81 to "Stocks in Play" — abnormal RVOL from fundamental news catalysts. We apply the ORB mechanism to SPY without an equivalent filter. The `academic_replications.py` suite will reveal whether the ORB mechanism alone (without RVOL) holds on an index ETF.

**9.2 The retest mechanic may create adverse selection.** Strong trend days break the ORB and never retest (those are filtered out). Weak, choppy sessions that pull back to retest are filtered in. The `orb_15min_no_retest` variant tests this empirically.

**9.3 Long-only vs bidirectional.** SPY's structural upward drift may make shorts structurally negative EV on an index ETF. The `long_only_15min_orb` variant tests this.

**9.4 CBOE P/C data gap (2019-2024).** Historical CSV ends October 2019. Daily HTML scrape fills 2019+ but covers the gap only from when the scrape starts running.

**9.5 serve_model.py tool calling.** Tool definitions are injected into the system prompt and outputs parsed for JSON patterns. This prompt-based approach works reliably for both Qwen2.5 and Mistral NeMo but is not the same as native function-calling API support. Hallucinated tool names or malformed JSON require error handling in `tool_runner.py`.

**9.6 No live Alpaca integration tested.** `trading_engine.py` uses yfinance for live 5-minute session bars. The live Alpaca WebSocket feed has never been tested end-to-end.

**9.7 Variable commission model.** On SPY with a $0.01 spread, actual round-trip friction is ~0.033R on a $0.60 ORB range — roughly double the modelled 0.016R flat rate.

---

## 10. What is missing

1. **Volume confirmation gate** — proposed replacement for retest mechanic (>1.25× 20-day average volume on breakout bar). Not yet implemented.
2. **NYSE TICK / UVOL/DVOL** — no institutional order flow confirmation for SPY breakouts.
3. **Multi-day drawdown limit** — daily stop rule exists; no weekly or monthly loss cap.
4. **UK spread betting route** — FCA-regulated spread betting (IG, CMC, Spreadex) is exempt from UK CGT and Income Tax (ITTOIA 2005 ss.6 & 10). Structural 20-45% post-tax advantage over Alpaca. Requires `SpreadBetBrokerClient` in `broker_interface.py`. Seek UK tax advice first.
5. **Automated Notion logging** — `confirmed_patterns` not yet auto-posted to Intelligence Ledger.
6. **Ladder exit mechanic** — Maroy 2025 (Phase 3 enhancement): scale out at 1R/2R, trail remainder via VWAP.

---

## 11. Specific questions for independent review

1. Should the strategy be **long-only** on SPY? The `long_only_15min_orb` variant in `academic_replications.py` will answer this empirically. If long-only beats bidirectional on IS data, prune shorts immediately.

2. Is the **retest mechanic** adding value or creating adverse selection? The `orb_15min_no_retest` variant will answer. If no-retest beats with-retest, remove the mechanic and add volume confirmation instead.

3. Is **WFE 0.50** theoretically grounded? It is a Pardo (2008) rule of thumb. PSR ≥ 0.95 (Bailey & López de Prado) is more rigorous for Phase 3.

4. What if **hybrid < Zarattini VWAP momentum**? Abandon the hybrid immediately and pivot to the VWAP momentum strategy.

5. Is **serve_model.py tool calling** reliable enough? Prompt-based tool injection works for well-tuned instruction models but does not provide the same guarantees as a dedicated function-calling API. If the D-A-C pipeline produces malformed tool calls, the tool loop guard (in `session_analyser.py`) prevents infinite loops, but the analysis quality will degrade.

6. **UK tax structure:** FCA spread betting is 0% CGT, 0% Income Tax. Not using it is structurally suboptimal for a UK-resident trader generating supplemental income.

---

## 12. File reference

```
Root scripts:
  prepare_host.sh     One-time: Arc drivers, .venv, model pair download, PSR fix
  launch_models.sh    Start/stop/status both model servers. Reads .model_pair.
  setup.sh            Python deps, DATA/ directory, health check against model servers
  verify.sh           5-step post-setup verification (server, inference, thinking log, etc.)
  NUC_SETUP.sh        Legacy: IPEX-LLM Ollama fallback path
  deploy_ovms.sh      Alternative: Docker-based OVMS (two containers)
  git_setup.sh        Interactive GitHub init + push

src/ Python files:
  runner.py                    Entry point. APScheduler. 
  trading_engine.py            Strategy: ORB + 3 gates + fill simulation.
  trading_dashboard.py         6-tab Streamlit.
  morning_brief.py             09:00 CLI. FRED + sentiment + patterns + GO/NO-GO.
  session_analyser.py          D-A-C. load_dotenv. Presets: nuc-pair1/2/3.
  serve_model.py               OpenVINO GenAI /v1 bridge. Tool calling. Thinking strip.
  broker_interface.py          DataProvider/BrokerClient.
  strategy_registry.py         6 variants + phase gates.
  tournament_evaluator.py      Saturday tournament.
  gene_mixer.py                Evolutionary optimiser.
  markov_engine.py             Regime chain. Fitted from FRED VIX.
  monte_carlo_extended.py      Galton board + Kelly + ruin.
  trade_journal_extended.py    50-field journal.
  historical_sim.py            IS backtest + SimBootstrapper → random + academic.
  market_data_store.py         Alpaca 1-min bars. FRED VIX preferred over yfinance.
  data_corrector.py            4 corrections. pandas_market_calendars.
  trading_quant_toolkit_v2_4.py  EV, WFA, DSR, Kelly, sharpe_significance.
  tool_runner.py               8 tools. Brave Search primary. 30s timeouts.
  findings_store.py            D-A-C findings → confirmed_patterns → morning brief.
  fred_store.py                FRED macro. VIXCLS preferred.
  sentiment_store.py           CBOE P/C + CFTC COT.
  random_baseline_sim.py       10,000-path vectorised null hypothesis.
  academic_replications.py     7-strategy comparison. Long-only + no-retest included.
```

---

## 13. Environment (.env)

| Variable | Default | Purpose |
|---|---|---|
| `ALPACA_API_KEY` | — | Required: Alpaca paper account |
| `ALPACA_SECRET_KEY` | — | Required: Alpaca paper account |
| `FRED_API_KEY` | — | Required: FRED macro data |
| `BRAVE_SEARCH_API_KEY` | — | Primary web search (DDG fallback if absent) |
| `LLM_BASE_URL` | `http://127.0.0.1:8000/v1` | Phase 1 model server (serve_model.py) |
| `DAC_BASE_URL` | `http://127.0.0.1:8001/v1` | Phase 2 adversarial server |
| `LLM_MODEL_PRIMARY` | `deepseek-r1:32b` | Matches model-id in launch_models.sh |
| `LLM_MODEL_ADVERSARIAL` | `mistral-nemo:12b` | Matches model-id in launch_models.sh |
| `LLM_API_KEY` | `unused` | Required by openai client; OVMS ignores it |
| `DISCORD_WEBHOOK_URL` | empty | EOD alerts (optional) |
| `DB_PATH` | `DATA/paper_account.db` | Paper account SQLite |
| `MARKET_DATA_DB` | `DATA/market_data.db` | Reference data SQLite |

---

*Report generated September 2026. Code at `src/`. Intelligence Ledger at Notion DB `9ba49df2-d288-46b7-89fc-876cbcccb999`.*
