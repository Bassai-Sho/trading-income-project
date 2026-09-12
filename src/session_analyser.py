"""
session_analyser.py
===================
Autonomous LLM-powered post-session analysis for the Trading Income Project.

After each session the trading engine triggers this script. It:
  1. Reads the decision ledger, trade journal, and SessionLearner output
  2. Sends everything to a local LLM (LM Studio or Ollama) with tools enabled
  3. The LLM uses those tools to reason, web-search for grounding, validate
     findings against the toolkit, and check WFA health
  4. Produces a structured analysis report
  5. Sends the report to Discord and/or Telegram

The LLM can NEVER apply parameter changes. It produces recommendations for
human review. All suggestions land in the notification channel and the DB.

SETUP — LM Studio (recommended)
---------------------------------
  1. Download LM Studio from lmstudio.ai
  2. Download a model:
       Recommended: Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct, or phi-3.5-mini
  3. Go to Server tab → Start Server (port 1234 by default)
  4. Set LLM_BASE_URL = "http://localhost:1234/v1"  in CONFIG below

SETUP — Ollama (alternative, simpler)
--------------------------------------
  1. Install Ollama: https://ollama.ai
  2. Pull a model:   ollama pull qwen2.5:7b
  3. Set LLM_BASE_URL = "http://localhost:11434/v1"  in CONFIG below

USAGE
------
  # Run manually after a session:
  python session_analyser.py --date 2026-09-08 --db paper_account.db

  # Triggered automatically by trading_engine.py at session end (EOD close)
  # No manual action needed once configured.

CONFIG
------
  Edit the CONFIG dict below to set your webhook URLs and preferences.
  All fields are optional — the analyser degrades gracefully.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sqlite3
import sys
import textwrap
import time
from datetime import date, datetime, timedelta, timezone
UTC = timezone.utc
from typing import Any

# Optional modules — degrade gracefully if unavailable
try:
    from tool_runner import TOOLS as _TOOLS_EXT, dispatch_tool as _dispatch_ext
    _HAS_TOOL_RUNNER = True
except ImportError:
    _HAS_TOOL_RUNNER = False
    _TOOLS_EXT = []
    _dispatch_ext = None

try:
    from findings_store import extract_structured_findings, store_findings
    _HAS_FINDINGS_STORE = True
except ImportError:
    _HAS_FINDINGS_STORE = False

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv optional — .env vars must be set manually

log = logging.getLogger("analyser")

# ---------------------------------------------------------------------------
# CONFIG — edit these before first run
# ---------------------------------------------------------------------------
CONFIG: dict[str, Any] = {
    # LLM endpoint (LM Studio or Ollama — both expose OpenAI-compatible API)
    "llm_base_url":    os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        "llm_model":       os.environ.get("LLM_MODEL_PRIMARY", "qwen3:30b-a3b"),
    "llm_api_key":     "not-needed",       # LM Studio/Ollama don't need a real key
    "llm_max_tokens":  2048,
    "llm_temperature": 0.2,               # low = more focused, less creative
    "tool_loop_max":   12,                # max tool calls per analysis

    # Database
    "db_path":         "DATA/paper_account.db",

    # Notifications (leave blank to disable that channel)
    "discord_webhook": os.environ.get("DISCORD_WEBHOOK_URL", ""),                 # paste your Discord webhook URL
    "telegram_token":  "",                 # paste your Telegram bot token
    "telegram_chat_id": "",               # paste your Telegram chat_id

    # Behaviour
    "web_search_results": 4,              # results per search
    "send_full_report":   True,           # True = full analysis, False = summary only
    "timeout_seconds":    120,            # max time for LLM analysis
        # DAC_BASE_URL: for OVMS, point at port 8001 (Phase 2 container)
    # e.g. DAC_BASE_URL=http://127.0.0.1:8001/v3
    "dac_base_url":    os.environ.get("DAC_BASE_URL", None),  # None = reuse llm_base_url
    "dac_api_key":     None,             # None = same as llm_api_key
    # Adversarial phase: use a DIFFERENT model architecture from Phase 1.
    # Default: mistral-nemo:12b (Mistral sparse-attention, breaks Qwen intra-family bias)
    # Override via OLLAMA_MODEL_ADVERSARIAL in .env
    "dac_model":       os.environ.get("OLLAMA_MODEL_ADVERSARIAL", "mistral-nemo:12b"),
    "dac_tool_loop_max":  8,               # max tool calls for D-A-C phase
    "run_dac":            True,            # False = skip D-A-C (faster, less rigorous)
}

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are the Trading Intelligence Analyst for the Trading Income Project.
You analyse automated paper trading session data and produce actionable, evidence-based reports.

YOUR ROLE:
- Read session data (trades, decisions, skip analysis, fill bias, WFA) provided to you
- Use tools to ground your findings in real market context and research
- Identify patterns, anomalies, and improvement opportunities
- Validate or challenge the SessionLearner's automated findings
- Produce clear, honest assessments — including when results are inconclusive

MANDATORY RULES:
- You NEVER suggest applying parameter changes automatically. All suggestions are for human review.
- You ALWAYS use web_search to ground factual market claims before making them
- You ALWAYS check WFA status before commenting on strategy health
- If data is insufficient to draw a conclusion, say so explicitly
- Express uncertainty clearly. A 10-trade sample is not statistically significant.

OUTPUT FORMAT:
Produce a structured report with these sections:
  ## SESSION SUMMARY
  ## WHAT WORKED
  ## WHAT DIDN'T WORK
  ## FILL QUALITY ASSESSMENT
  ## WFA HEALTH CHECK
  ## MARKET CONTEXT (grounded by web search)
  ## ACTION ITEMS FOR HUMAN REVIEW  (numbered, specific, actionable)
  ## CONFIDENCE LEVEL (HIGH/MEDIUM/LOW with reason)

Keep each section concise. Total report: 400-600 words maximum.
"""

# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def _conn(db_path: str) -> sqlite3.Connection:
    c = sqlite3.connect(db_path, timeout=10)
    c.row_factory = sqlite3.Row
    return c

def _q(db_path: str, sql: str, params: tuple = ()) -> list[dict]:
    try:
        with _conn(db_path) as c:
            return [dict(r) for r in c.execute(sql, params).fetchall()]
    except Exception as e:
        log.warning("DB query failed: %s", e)
        return []

def _q1(db_path: str, sql: str, params: tuple = ()) -> dict | None:
    rows = _q(db_path, sql, params)
    return rows[0] if rows else None

def _save_analysis(db_path: str, session_date: str, report: str,
                    tool_calls_used: int, model_used: str) -> None:
    """Persist the LLM analysis report to the DB for dashboard display."""
    try:
        with _conn(db_path) as c:
            c.execute("""CREATE TABLE IF NOT EXISTS llm_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_date TEXT NOT NULL,
                ts TEXT NOT NULL,
                report TEXT NOT NULL,
                tool_calls_used INTEGER,
                model_used TEXT,
                sent_discord INTEGER DEFAULT 0,
                sent_telegram INTEGER DEFAULT 0
            )""")
            c.execute(
                "INSERT INTO llm_analysis (session_date,ts,report,tool_calls_used,model_used) "
                "VALUES (?,?,?,?,?)",
                (session_date, datetime.now(UTC).isoformat(),
                 report, tool_calls_used, model_used)
            )
    except Exception as e:
        log.warning("Could not save analysis to DB: %s", e)

# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def tool_web_search(query: str, n_results: int = 4) -> str:
    """Search the web and return top results as plain text."""
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=n_results):
                title = r.get("title","")
                body  = r.get("body","")[:200]
                results.append(f"[{title}] {body}")
        return "\n---\n".join(results) if results else "No results found."
    except ImportError:
        return "duckduckgo-search not installed. Run: pip install duckduckgo-search"
    except Exception as e:
        return f"Search failed: {e}"


def tool_get_session_data(db_path: str, session_date: str) -> str:
    """Fetch comprehensive session data from the DB for the given date."""
    try:
        # Session summary
        sess = _q1(db_path, "SELECT * FROM sessions WHERE session_date=?", (session_date,))
        # Trades
        trades = _q(db_path,
                    "SELECT direction,entry_price,exit_price,actual_r,exit_reason,notes "
                    "FROM positions WHERE date(opened_at)=? ORDER BY id",
                    (session_date,))
        # Decisions
        decs = _q(db_path,
                  "SELECT action,gate_orb_break,gate_vwap,gate_retest,gate_final,"
                  "vix,rvol,vwap_slope,reason FROM decisions WHERE session_date=? ORDER BY id",
                  (session_date,))
        # Session learning
        sl = _q1(db_path,
                 "SELECT * FROM session_learning WHERE session_date=? ORDER BY id DESC LIMIT 1",
                 (session_date,))

        data = {
            "session_summary": dict(sess) if sess else {},
            "trades":  trades,
            "n_decisions": len(decs),
            "decision_sample": decs[:5],   # first 5 to give context
            "skip_count": sum(1 for d in decs if d.get("action")=="SKIP"),
            "enter_count": sum(1 for d in decs if d.get("action")=="ENTER"),
            "session_learning": {
                "n_trades":          sl.get("n_trades")         if sl else None,
                "n_skips":           sl.get("n_skips")          if sl else None,
                "over_filter_flag":  bool(sl.get("over_filter_flag")) if sl else None,
                "avg_total_drag_r":  sl.get("avg_total_drag_r") if sl else None,
                "modelled_ev":       sl.get("modelled_ev")      if sl else None,
                "realistic_ev":      sl.get("realistic_ev")     if sl else None,
                "best_regime":       sl.get("best_regime")      if sl else None,
                "worst_regime":      sl.get("worst_regime")     if sl else None,
                "action_items":      json.loads(sl.get("action_items","[]")) if sl else [],
            } if sl else "No SessionLearner report found for this date.",
        }
        return json.dumps(data, indent=2, default=str)
    except Exception as e:
        return f"Error fetching session data: {e}"


def tool_get_wfa_results(db_path: str, ticker: str = "SPY", n: int = 12) -> str:
    """Fetch the most recent WFA results and compute aggregate WFE."""
    try:
        rows = _q(db_path,
                  "SELECT is_start,is_end,oos_start,oos_end,is_sharpe,oos_sharpe,wfe,wfe_verdict "
                  "FROM wfa_results WHERE ticker=? ORDER BY run_ts DESC LIMIT ?",
                  (ticker, n))
        if not rows:
            return "No WFA results found. Run: python trading_engine.py --wfa-only"
        valid_wfe = [r["wfe"] for r in rows if r.get("wfe") is not None]
        mean_wfe  = round(sum(valid_wfe)/len(valid_wfe), 3) if valid_wfe else None
        passes    = sum(1 for r in rows if r.get("wfe_verdict")=="PASS")
        summary = {
            "n_splits":       len(rows),
            "mean_wfe":       mean_wfe,
            "passes":         passes,
            "overall_verdict": ("PASS" if mean_wfe and mean_wfe >= 0.5 else
                                "FAIL" if mean_wfe is not None else "INCONCLUSIVE"),
            "most_recent_split": rows[0] if rows else None,
        }
        return json.dumps(summary, indent=2, default=str)
    except Exception as e:
        return f"Error fetching WFA results: {e}"


def tool_get_journal_stats(db_path: str, min_samples: int = 3) -> str:
    """Fetch extended journal correlations: EV by candle, time, psychology, setup grade."""
    try:
        from trade_journal_extended import get_journal_stats
        stats = get_journal_stats(db_path, min_samples=min_samples)
        import json
        # Truncate safely: return summary if full JSON too long
        full = json.dumps(stats, indent=2, default=str)
        if len(full) <= 3800:
            return full
        # Return just the most important parts under token limit
        summary = {
            "n":              stats.get("n"),
            "top_insights":   stats.get("top_insights", [])[:5],
            "psychology_alerts": stats.get("psychology_alerts", {}),
            "note":           f"Full correlations truncated ({len(full)} chars). Top 5 insights shown."
        }
        return json.dumps(summary, indent=2, default=str)
    except ImportError:
        return 'trade_journal_extended not found — extended journal not active yet'
    except Exception as e:
        return f'Journal stats error: {e}'


def tool_run_toolkit(function_name: str, kwargs_json: str) -> str:
    """
    Call a function from trading_quant_toolkit_v2_4.py directly.
    The LLM can use this to compute sharpe significance, EV, Kelly fraction, etc.
    """
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import trading_quant_toolkit_v2_4 as tk
        fn = getattr(tk, function_name, None)
        if fn is None:
            return f"Function '{function_name}' not found in toolkit v2.4.0"
        kwargs = json.loads(kwargs_json) if kwargs_json.strip() else {}
        result = fn(**kwargs)
        return json.dumps(result, indent=2, default=str)
    except Exception as e:
        return f"Toolkit call failed ({function_name}): {e}"


def tool_get_rolling_stats(db_path: str, n: int = 20) -> str:
    """Compute rolling performance stats from the last n closed trades."""
    try:
        rows = _q(db_path,
                  "SELECT actual_r FROM positions WHERE status='closed' "
                  "AND actual_r IS NOT NULL ORDER BY id DESC LIMIT ?", (n,))
        if not rows:
            return json.dumps({"n": 0, "message": "No closed trades found."})
        rs     = [r["actual_r"] for r in rows]
        wins   = [r for r in rs if r > 0]
        losses = [r for r in rs if r < 0]
        wr     = len(wins) / len(rs)
        aw     = sum(wins)   / len(wins)   if wins   else 0.0
        al     = abs(sum(losses)/len(losses)) if losses else 0.0
        ev     = (wr * aw) - ((1-wr) * al)
        import statistics as _s
        sharpe = None
        if len(rs) >= 2:
            sd = _s.stdev(rs)
            if sd > 0: sharpe = round(_s.mean(rs)/sd, 3)
        return json.dumps({
            "n": len(rs), "win_rate": round(wr,3), "avg_win_r": round(aw,3),
            "avg_loss_r": round(al,3), "ev_per_trade": round(ev,4),
            "sharpe": sharpe,
            "note": f"Based on last {n} closed trades. Statistically significant only after ~200+ trades."
        }, indent=2)
    except Exception as e:
        return f"Error computing rolling stats: {e}"


# ---------------------------------------------------------------------------
# Tool definitions (OpenAI function-calling format)
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web to ground findings in real market data, news, or research. Use this BEFORE making any claim about market conditions, economic events, or strategy research.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Specific search query. Be precise."},
                    "n_results": {"type": "integer", "description": "Number of results (1-5)", "default": 4}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_session_data",
            "description": "Fetch complete session data from the trading DB: trades, decisions, SessionLearner report.",
            "parameters": {
                "type": "object",
                "properties": {
                    "session_date": {"type": "string", "description": "Date in YYYY-MM-DD format"}
                },
                "required": ["session_date"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_wfa_results",
            "description": "Fetch Walk-Forward Analysis results. Use this to assess if the strategy edge is still valid.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Ticker symbol", "default": "SPY"},
                    "n": {"type": "integer", "description": "Number of recent splits to fetch", "default": 12}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_rolling_stats",
            "description": "Compute rolling performance statistics from the last N closed paper trades.",
            "parameters": {
                "type": "object",
                "properties": {
                    "n": {"type": "integer", "description": "Number of recent trades", "default": 20}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_journal_stats",
            "description": "Fetch extended trade journal correlations: EV by candle type, candle quality, time slot, day of week, session type, VIX regime, emotional state, setup grade, process grade, FOMO flag, revenge flag. Use this to identify behavioural patterns and market condition correlations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "min_samples": {"type": "integer", "description": "Minimum trades per bucket", "default": 3}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_toolkit",
            "description": "Call a function from trading_quant_toolkit_v2_4.py to compute statistical metrics. Useful for: kelly_fraction(), sharpe_significance_tstat_annualised(), regime_conditional_streak_probability(), deflated_sharpe_ratio_extended().",
            "parameters": {
                "type": "object",
                "properties": {
                    "function_name": {"type": "string", "description": "Exact function name from the toolkit"},
                    "kwargs_json": {"type": "string", "description": "JSON string of keyword arguments, e.g. '{\"win_rate\": 0.6, \"avg_win_r\": 2.0}'"}
                },
                "required": ["function_name", "kwargs_json"]
            }
        }
    }
]

# ---------------------------------------------------------------------------
# Tool dispatcher — delegates to tool_runner.py if available
# tool_runner.py adds: fetch_url, run_script, per-tool timeouts, richer web snippets
# ---------------------------------------------------------------------------

# Use tool_runner TOOLS list if available (adds fetch_url + run_script schemas)
if _HAS_TOOL_RUNNER:
    TOOLS = _TOOLS_EXT


def dispatch_tool(name: str, args: dict, cfg: dict) -> str:
    """Route a tool call. Prefers tool_runner.py; falls back to inline implementations."""
    if _HAS_TOOL_RUNNER:
        return _dispatch_ext(name, args, cfg)
    # Legacy fallback when tool_runner.py not found
    db_path = cfg["db_path"]
    if name == "web_search":
        return tool_web_search(args.get("query", ""), args.get("n_results", cfg["web_search_results"]))
    elif name == "get_session_data":
        return tool_get_session_data(db_path, args.get("session_date", str(date.today())))
    elif name == "get_wfa_results":
        return tool_get_wfa_results(db_path, args.get("ticker", "SPY"), args.get("n", 12))
    elif name == "get_journal_stats":
        return tool_get_journal_stats(db_path, args.get("min_samples", 3))
    elif name == "get_rolling_stats":
        return tool_get_rolling_stats(db_path, args.get("n", 20))
    elif name == "run_toolkit":
        return tool_run_toolkit(args.get("function_name", ""), args.get("kwargs_json", "{}"))
    else:
        return f"Unknown tool: {name}"

# ---------------------------------------------------------------------------
# LLM analysis loop (ReAct pattern)
# ---------------------------------------------------------------------------


def _preflight_tool_check(cfg: dict) -> str:
    """Check which tools are working. Returns status line for system prompt."""
    status = []
    try:
        import sqlite3
        with sqlite3.connect(cfg.get("db_path","paper_account.db"), timeout=3) as _c:
            _c.execute("SELECT 1")
        status.append("DB=OK")
    except Exception:
        status.append("DB=UNAVAILABLE")
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            r = list(ddgs.text("SPY ETF", max_results=1))
        status.append("web_search=OK" if r else "web_search=EMPTY")
    except Exception:
        status.append("web_search=UNAVAILABLE — use DB-only analysis")
    try:
        import trading_quant_toolkit_v2_4  # noqa
        status.append("toolkit=OK")
    except Exception:
        status.append("toolkit=UNAVAILABLE")
    return "Tool status: " + ", ".join(status)


def run_analysis(session_date: str, cfg: dict) -> tuple[str, int, str]:
    """
    Run the full LLM analysis loop.
    Returns (report_text, tool_calls_used, model_name).
    """
    try:
        from openai import OpenAI
    except ImportError:
        return ("openai package not installed. Run: pip install openai", 0, "none")

    client = OpenAI(
        base_url=cfg["llm_base_url"],
        api_key=cfg["llm_api_key"],
    )

    tool_status_line = _preflight_tool_check(cfg)
    log.info("Pre-flight: %s", tool_status_line)

    # Initial user message
    user_msg = textwrap.dedent(f"""
    Please analyse the trading session for {session_date}.

    Start by fetching the session data using get_session_data, then get the WFA results
    and rolling stats. Search the web to understand what the market was doing on {session_date}
    (SPY price action, VIX level, any major news). Then use the toolkit where needed to
    validate the SessionLearner's findings. Finally produce your structured report.

    Remember: you can SUGGEST parameter adjustments but NEVER apply them.
    """).strip()

    messages = [
        {"role": "system",  "content": SYSTEM_PROMPT + "\n\n" + tool_status_line},
        {"role": "user",    "content": user_msg},
    ]

    tool_calls_used = 0
    model_used = cfg["llm_model"]
    _seen_calls: list[str] = []  # loop guard

    for iteration in range(cfg["tool_loop_max"]):
        try:
            response = client.chat.completions.create(
                model=cfg["llm_model"],
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                max_tokens=cfg["llm_max_tokens"],
                temperature=cfg["llm_temperature"],
                timeout=cfg["timeout_seconds"],
            )
        except Exception as e:
            log.error("LLM call failed on iteration %d: %s", iteration, e)
            return (f"LLM call failed: {e}\n\nCheck that LM Studio or Ollama is running at {cfg['llm_base_url']}", 
                    tool_calls_used, model_used)

        choice  = response.choices[0]
        message = choice.message
        messages.append({"role": "assistant",
                          "content": message.content,
                          "tool_calls": [tc.model_dump() for tc in (message.tool_calls or [])]})

        # No tool calls → final answer
        if not message.tool_calls:
            report = message.content or "(empty response from LLM)"
            log.info("Analysis complete after %d tool calls", tool_calls_used)
            if _HAS_FINDINGS_STORE:
                try:
                    findings = extract_structured_findings(
                        report, tool_calls_used, cfg)
                    if findings:
                        store_findings(cfg.get("db_path","paper_account.db"),
                                      session_date, findings, tool_calls_used)
                except Exception as _fe:
                    log.warning("Findings extraction failed: %s", _fe)
            return (report, tool_calls_used, model_used)

        # Execute each tool call
        for tc in message.tool_calls:
            fn_name = tc.function.name
            try:
                fn_args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                fn_args = {}
            log.info("Tool call %d: %s(%s)", tool_calls_used+1, fn_name,
                      str(fn_args)[:80])
            call_sig = f"{fn_name}:{str(sorted(fn_args.items()))[:100]}"
            if _seen_calls.count(call_sig) >= 2:
                result = (f"[LOOP GUARD] '{fn_name}' called identically "
                          f"{_seen_calls.count(call_sig)+1}x. "
                          "Accept previous result or try different approach.")
                log.warning("Loop guard: %s", call_sig[:80])
            else:
                _seen_calls.append(call_sig)
                result = dispatch_tool(fn_name, fn_args, cfg)
            tool_calls_used += 1
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result[:4000],   # truncate very long results
            })

    # Fell through max iterations
    return ("Analysis reached tool call limit. Partial reasoning may be incomplete.",
            tool_calls_used, model_used)

# ---------------------------------------------------------------------------
# Notifications
# ---------------------------------------------------------------------------

def _truncate(text: str, max_chars: int) -> str:
    return text[:max_chars] + "…" if len(text) > max_chars else text

def send_discord(report: str, session_date: str, cfg: dict) -> bool:
    """Post the analysis report to Discord via webhook."""
    webhook = cfg.get("discord_webhook","")
    if not webhook:
        return False
    try:
        import urllib.request, json as _json
        # Discord limit: 2000 chars per message; split if needed
        header = f"📊 **Session Analysis — {session_date}**\n"
        chunks  = []
        body    = header + report
        while body:
            chunks.append(body[:1900])
            body = body[1900:]
        for chunk in chunks:
            data = _json.dumps({"content": chunk}).encode()
            req  = urllib.request.Request(webhook, data=data,
                                           headers={"Content-Type":"application/json"})
            urllib.request.urlopen(req, timeout=10)
        log.info("Discord: sent %d chunk(s)", len(chunks))
        return True
    except Exception as e:
        log.warning("Discord send failed: %s", e)
        return False

async def _telegram_send_async(token: str, chat_id: str, text: str) -> bool:
    """Async Telegram message send."""
    try:
        from telegram import Bot
        bot = Bot(token=token)
        # Telegram limit: 4096 chars; split if needed
        chunks = [text[i:i+4000] for i in range(0, len(text), 4000)]
        for chunk in chunks:
            await bot.send_message(chat_id=chat_id, text=chunk,
                                    parse_mode="Markdown")
        return True
    except Exception as e:
        log.warning("Telegram send failed: %s", e)
        return False

def send_telegram(report: str, session_date: str, cfg: dict) -> bool:
    """Post the analysis report to Telegram."""
    token   = cfg.get("telegram_token","")
    chat_id = cfg.get("telegram_chat_id","")
    if not token or not chat_id:
        return False
    text = f"📊 *Session Analysis — {session_date}*\n\n{report}"
    try:
        return asyncio.run(_telegram_send_async(token, chat_id, text))
    except Exception as e:
        log.warning("Telegram async failed: %s", e)
        return False

def send_all(report: str, session_date: str, cfg: dict) -> dict[str, bool]:
    """Send report to all configured channels."""
    results = {}
    results["discord"]  = send_discord(report, session_date, cfg)
    results["telegram"] = send_telegram(report, session_date, cfg)
    if not any(results.values()):
        log.info("No notification channels configured. Report:\n%s", report)
    return results

# ---------------------------------------------------------------------------
# Fallback report (no LLM available)
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# D-A-C STRESS TEST
# ---------------------------------------------------------------------------

DAC_SYSTEM_PROMPT = """You are the Adversarial Validator for the Trading Income Project.
You have just received a session analysis report produced by an AI analyst.
Your job is to stress-test it using the D-A-C methodology before any recommendation
is acted upon. You are deliberately sceptical.

D-A-C PROTOCOL (execute all three phases in order):

══════════════════════════════════════════════════════
PHASE D — DIVERGENT (Alternative Hypotheses)
══════════════════════════════════════════════════════
Generate EXACTLY 3 alternative hypotheses that CONTRADICT or significantly
complicate the analyst's conclusions. Each hypothesis must:
  • Be specific — name the exact claim being challenged
  • Be grounded — use web_search or run_toolkit to find supporting evidence
  • Be non-trivial — not just "maybe it's noise"

For a trading session, divergent hypotheses typically include:
  • Statistical: the result is explainable by random variance at this sample size
  • Structural: today's market conditions were anomalous (check web_search for news)
  • Measurement: the fill simulation over/underestimates the actual cost drag

══════════════════════════════════════════════════════
PHASE A — ADVERSARIAL (Challenge Each Action Item)
══════════════════════════════════════════════════════
For EACH action item in the analysis report, argue the strongest case AGAINST
implementing it. Use run_toolkit to check statistical significance where possible.

Mandatory adversarial checks:
  1. Sample size check: call run_toolkit with sharpe_significance_tstat_annualised
     or regime_conditional_streak_probability. If n < 20 trades, flag as PREMATURE.
  2. Academic conflict check: does the recommendation conflict with any logged
     academic paper (Zarattini 2024, Gao 2018, Maroy 2025)?
  3. Regime check: could today's VIX regime explain the result without needing
     a parameter change?
  4. Overfitting risk: would implementing this change reduce strategy simplicity
     (more than 5 tunable parameters violates Rule 13)?

══════════════════════════════════════════════════════
PHASE C — CONVERGENT (Grounded Recommendations)
══════════════════════════════════════════════════════
After adversarial challenge, classify EACH action item as one of:
  • GROUNDED: survives adversarial challenge + has sufficient evidence
  • PROVISIONAL: plausible but insufficient trades to act on yet
  • REJECTED: adversarial case is stronger than the original
  • MONITOR: worth watching over next 10+ sessions before acting

For each GROUNDED item, state:
  • Minimum evidence threshold before implementing (e.g., "replicated in 3+ sessions")
  • Confidence: HIGH / MEDIUM / LOW
  • One-line implementation instruction

OUTPUT FORMAT:
## D-A-C STRESS TEST
### Phase D — Divergent Hypotheses
[3 alternative hypotheses with grounding]

### Phase A — Adversarial Challenges
[Challenge each action item]

### Phase C — Grounded Recommendations
[Only surviving items, classified and with evidence thresholds]

### VERDICT
[Overall confidence in today's analysis: HIGH / MEDIUM / LOW with one-line reason]
"""

DAC_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search for market context, news, or academic validation to ground adversarial arguments.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "n_results": {"type": "integer", "default": 3}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_toolkit",
            "description": "Call toolkit functions to check statistical significance of claims. Useful: sharpe_significance_tstat_annualised(), kelly_fraction(), regime_conditional_streak_probability(), compute_metrics().",
            "parameters": {
                "type": "object",
                "properties": {
                    "function_name": {"type": "string"},
                    "kwargs_json": {"type": "string"}
                },
                "required": ["function_name", "kwargs_json"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_rolling_stats",
            "description": "Get rolling trade statistics to assess whether sample size is sufficient for a given claim.",
            "parameters": {
                "type": "object",
                "properties": {
                    "n": {"type": "integer", "default": 20}
                }
            }
        }
    }
]


def run_dac_stress_test(
    initial_report: str,
    session_date: str,
    cfg: dict,
) -> tuple[str, int]:
    """
    Run the D-A-C stress test on the initial analysis report.

    The D-A-C LLM is a sceptical adversarial validator that:
      D) Generates 3 alternative hypotheses contradicting the analysis
      A) Challenges every action item with statistical and academic arguments
      C) Promotes only surviving items to GROUNDED RECOMMENDATIONS

    Returns (dac_report_text, tool_calls_used).
    """
    try:
        from openai import OpenAI
    except ImportError:
        return ("openai not installed — D-A-C skipped.", 0)

    # Fix: adversarial phase MUST use dac_model/dac_base_url — not Phase 1 model
    dac_url   = cfg.get("dac_base_url") or cfg["llm_base_url"]
    dac_key   = cfg.get("dac_api_key")  or cfg["llm_api_key"]
    dac_model = cfg.get("dac_model", "mistral-nemo:12b")

    client = OpenAI(base_url=dac_url, api_key=dac_key)

    user_msg = f"""The following session analysis was produced for {session_date}.
Apply the D-A-C stress test protocol to validate it.

Start by calling get_rolling_stats to understand the sample size,
then use web_search to understand what happened in the market on {session_date},
then run_toolkit as needed to check statistical significance of any claims.

=== INITIAL ANALYSIS REPORT ===
{initial_report}
=== END REPORT ===

Now apply: Phase D (Divergent), Phase A (Adversarial), Phase C (Convergent).
"""

    messages = [
        {"role": "system", "content": DAC_SYSTEM_PROMPT},
        {"role": "user",   "content": user_msg},
    ]

    tool_calls_used = 0
    max_dac_tools   = cfg.get("dac_tool_loop_max", 8)

    for _ in range(max_dac_tools):
        try:
            response = client.chat.completions.create(
                model=dac_model,
                messages=messages,
                tools=DAC_TOOLS,
                tool_choice="auto",
                max_tokens=cfg.get("llm_max_tokens", 2048),
                temperature=0.1,   # more deterministic for adversarial reasoning
                timeout=cfg.get("timeout_seconds", 120),
                extra_body={"options": {
                    "num_ctx":   cfg.get("num_ctx",   8192),
                    "num_batch": cfg.get("num_batch", 2048),
                }},
            )
        except Exception as e:
            log.error("D-A-C LLM call failed: %s", e)
            return (f"D-A-C stress test failed: {e}", tool_calls_used)

        choice  = response.choices[0]
        message = choice.message
        messages.append({
            "role": "assistant",
            "content": message.content,
            "tool_calls": [tc.model_dump() for tc in (message.tool_calls or [])]
        })

        if not message.tool_calls:
            dac_text = message.content or "(empty D-A-C response)"
            log.info("D-A-C complete after %d tool calls", tool_calls_used)
            return (dac_text, tool_calls_used)

        for tc in message.tool_calls:
            fn_name = tc.function.name
            try:
                fn_args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                fn_args = {}
            log.info("D-A-C tool call %d: %s(%s)",
                     tool_calls_used+1, fn_name, str(fn_args)[:60])
            # Reuse the same dispatcher (web_search, run_toolkit, get_rolling_stats)
            result = dispatch_tool(fn_name, fn_args, cfg)
            tool_calls_used += 1
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result[:3000],
            })

    return ("D-A-C reached tool call limit.", tool_calls_used)


def parse_grounded_recommendations(dac_report: str) -> list[str]:
    """
    Extract only the GROUNDED recommendations from a D-A-C report.
    Used by the dashboard to show the highest-confidence action items.
    """
    import re
    grounded = []
    # Find lines that start with GROUNDED: or contain [GROUNDED]
    for line in dac_report.split("\n"):
        line = line.strip()
        if line.upper().startswith("GROUNDED") or "[GROUNDED]" in line.upper():
            # Strip the prefix and clean
            clean = re.sub(r"(?i)^grounded[:\s*-]*", "", line).strip()
            clean = re.sub(r"\[GROUNDED\]", "", clean, flags=re.IGNORECASE).strip()
            if len(clean) > 10:
                grounded.append(clean)
    return grounded


def generate_fallback_report(session_date: str, db_path: str) -> str:
    """
    Generate a structured text report directly from the DB without an LLM.
    Used when LM Studio / Ollama is not running.
    """
    sl = _q1(db_path,
             "SELECT * FROM session_learning WHERE session_date=? ORDER BY id DESC LIMIT 1",
             (session_date,))
    sess = _q1(db_path, "SELECT * FROM sessions WHERE session_date=?", (session_date,))
    wfa_rows = _q(db_path, "SELECT * FROM wfa_results ORDER BY run_ts DESC LIMIT 6")
    trades = _q(db_path,
                "SELECT actual_r,exit_reason,direction FROM positions "
                "WHERE date(opened_at)=? AND status='closed'", (session_date,))

    rs = [t["actual_r"] for t in trades if t.get("actual_r") is not None]
    wr = f"{len([r for r in rs if r > 0])/max(len(rs),1):.0%}" if rs else "N/A"
    ev = f"{sum(rs)/max(len(rs),1):+.3f}R" if rs else "N/A"

    wfe_vals = [r["wfe"] for r in wfa_rows if r.get("wfe")]
    mean_wfe = f"{sum(wfe_vals)/len(wfe_vals):.3f}" if wfe_vals else "N/A"

    drag = sl.get("avg_total_drag_r","?") if sl else "?"
    real_ev = sl.get("realistic_ev","?") if sl else "?"
    action_items = json.loads(sl.get("action_items","[]")) if sl else []

    lines = [
        f"## SESSION ANALYSIS — {session_date} (no LLM — structured summary)",
        "",
        "## SESSION SUMMARY",
        f"  Trades: {sess.get('n_trades',0) if sess else 0}  "
        f"Wins: {sess.get('n_wins',0) if sess else 0}  "
        f"Losses: {sess.get('n_losses',0) if sess else 0}  "
        f"P&L: {sess.get('session_pnl_r',0):+.2f}R" if sess else "  No session record",
        f"  Win rate: {wr}  EV/trade: {ev}",
        "",
        "## FILL BIAS ASSESSMENT",
        f"  Avg fill drag: {drag}R per trade",
        f"  Realistic EV: {real_ev}R  (after fill simulation)",
        "",
        "## WFA HEALTH",
        f"  Mean WFE (last {len(wfa_rows)} splits): {mean_wfe}  "
        f"{'✅ PASS' if mean_wfe != 'N/A' and float(mean_wfe) >= 0.5 else '⚠ CHECK'}",
        "",
        "## ACTION ITEMS",
    ]
    if action_items:
        for i, item in enumerate(action_items, 1):
            lines.append(f"  {i}. {item}")
    else:
        lines.append("  No action items. Continue paper trading.")

    lines += [
        "",
        "## NOTE",
        "  Configure LM Studio or Ollama to get full AI-reasoned analysis.",
        f"  Set llm_base_url in session_analyser.py CONFIG.",
    ]
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Named model presets  (usage: python session_analyser.py --preset <name>)
# ---------------------------------------------------------------------------

MODEL_PRESETS: dict[str, dict] = {
    # ── OVMS (recommended) — OpenVINO Model Server via Docker
    # ── OpenVINO GenAI pairs (no Docker, pure .venv) ─────────────────────────
    # Start with: ./launch_models.sh --pair N
    "nuc-pair1": {
        # Pair 1: Thinking Quant — DeepSeek-R1-32B (o1-class) + Mistral-Nemo (~26.3GB)
        "llm_base_url": "http://127.0.0.1:8000/v1",
        "llm_api_key":  "unused",
        "llm_model":    "deepseek-r1:32b",
        "dac_base_url": "http://127.0.0.1:8001/v1",
        "dac_api_key":  "unused",
        "dac_model":    "mistral-nemo:12b",
        "_note": "Deepest reasoning. <think> tokens stripped from response, logged to LOGS/.",
    },
    "nuc-pair2": {
        # Pair 2: High-Speed — Qwen3-30B-A3B MoE (30+ tok/s) + Mistral-Nemo (~25.2GB)
        "llm_base_url": "http://127.0.0.1:8000/v1",
        "llm_api_key":  "unused",
        "llm_model":    "qwen3:30b-a3b",
        "dac_base_url": "http://127.0.0.1:8001/v1",
        "dac_api_key":  "unused",
        "dac_model":    "mistral-nemo:12b",
        "_note": "Fastest multi-turn tool loops. Best throughput for 8+ tool-call sessions.",
    },
    "nuc-pair3": {
        # Pair 3: Frontier Heavy — Qwen2.5-32B + Mistral-Small-24B (~33.2GB)
        "llm_base_url": "http://127.0.0.1:8000/v1",
        "llm_api_key":  "unused",
        "llm_model":    "qwen2.5:32b",
        "dac_base_url": "http://127.0.0.1:8001/v1",
        "dac_api_key":  "unused",
        "dac_model":    "mistral-small:24b",
        "_note": "Peak JSON/tool precision + deepest adversarial critique (24B Mistral).",
    },
    "nuc-ovms": {  # alias for nuc-pair2 (backward compat)
        "llm_base_url": "http://127.0.0.1:8000/v1",
        "llm_api_key":  "unused",
        "llm_model":    "qwen3:30b-a3b",
        "dac_base_url": "http://127.0.0.1:8001/v1",
        "dac_api_key":  "unused",
        "dac_model":    "mistral-nemo:12b",
        "_note": "Alias for nuc-pair2. Use nuc-pair1/2/3 directly.",
    },
    # ── IPEX-LLM Ollama (fallback) — community fork
    "lmstudio-7b": {"llm_base_url":"http://localhost:1234/v1","llm_api_key":"not-needed","llm_model":"local-model","dac_model":None,"_note":"7B via LM Studio. Borderline D-A-C quality."},
    "ollama-qwen3-8b": {"llm_base_url":"http://localhost:11434/v1","llm_api_key":"ollama","llm_model":"qwen3:8b","dac_model":"qwen3:8b","_note":"Best free upgrade. Beats Qwen2.5-14B. 6GB VRAM."},
    "ollama-split": {"llm_base_url":"http://localhost:11434/v1","llm_api_key":"ollama","llm_model":"qwen3:8b","dac_base_url":"http://localhost:11434/v1","dac_api_key":"ollama","dac_model":"qwen3:14b","_note":"Best local. Phase 1: 8B. Phase 2: 14B thinking. ~16GB VRAM."},
    "openrouter-flash": {"llm_base_url":"https://openrouter.ai/api/v1","llm_api_key":os.environ.get("OPENROUTER_API_KEY",""),"llm_model":"deepseek/deepseek-v4-flash","dac_model":"deepseek/deepseek-v4-flash","_note":"~$0.001/session. Best value cloud."},
    "openrouter-split": {"llm_base_url":"https://openrouter.ai/api/v1","llm_api_key":os.environ.get("OPENROUTER_API_KEY",""),"llm_model":"deepseek/deepseek-v4-flash","dac_base_url":"https://openrouter.ai/api/v1","dac_api_key":os.environ.get("OPENROUTER_API_KEY",""),"dac_model":"anthropic/claude-haiku-4-5","_note":"~$0.002/session. Recommended cloud. Best D-A-C."},
    "openrouter-premium": {"llm_base_url":"https://openrouter.ai/api/v1","llm_api_key":os.environ.get("OPENROUTER_API_KEY",""),"llm_model":"anthropic/claude-sonnet-5","dac_model":"anthropic/claude-sonnet-5","_note":"~$0.03/session. Best quality."},
    # ── Asus NUC 14 Pro / Intel Core Ultra iGPU presets ─────────────
    # Requires IPEX-LLM Ollama (Intel GPU-accelerated build):
    # https://github.com/intel/ipex-llm/blob/main/docs/mddocs/Quickstart/ollama_quickstart.md
    # Models <= 32GB fit in the 64GB/2 = 32GB iGPU memory pool -> GPU accelerated.
    "nuc-balanced": {
        "llm_base_url": "http://localhost:11434/v1", "llm_api_key": "ollama",
        "llm_model": "qwen3:30b-a3b",   # MoE: 30B params, 3B activated -> fast
        "dac_base_url": "http://localhost:11434/v1", "dac_api_key": "ollama",
        "dac_model": "qwen3:32b",        # Dense 32B -> best D-A-C reasoning, fits iGPU
        "_note": "NUC 14 Pro (64GB): Phase 1 MoE fast, Phase 2 dense 32B. Both GPU-accelerated.",
    },
    "nuc-fast": {
        "llm_base_url": "http://localhost:11434/v1", "llm_api_key": "ollama",
        "llm_model": "qwen3:14b",        # 14B: ~9GB, very fast on iGPU
        "dac_model": "qwen3:30b-a3b",    # MoE 30B: excellent quality, still fast
        "_note": "NUC 14 Pro: Max speed. Good D-A-C. All GPU-accelerated.",
    },
    "nuc-quality": {
        "llm_base_url": "http://localhost:11434/v1", "llm_api_key": "ollama",
        "llm_model": "qwen3:32b",        # Dense 32B for both phases
        "dac_model": "qwen3:32b",
        "_note": "NUC 14 Pro: Single dense 32B for both phases. Best reasoning, fits iGPU.",
    },
    "nuc-maximum": {
        "llm_base_url": "http://localhost:11434/v1", "llm_api_key": "ollama",
        "llm_model": "qwen3:30b-a3b",   # Fast Phase 1
        "dac_base_url": "http://localhost:11434/v1", "dac_api_key": "ollama",
        "dac_model": "qwen2.5:72b",      # 72B Q4 ~45GB: CPU+iGPU mixed, max quality
        "_note": "NUC 14 Pro: Phase 2 runs 72B mixed CPU/iGPU. Slower (~3 min) but near-frontier quality.",
    },
    # ── Qwen3.8-27B presets (released Aug 2026 — newer architecture than Qwen3) ─
    # Qwen3.8-27B: 27B dense, Aug 2026. Better than Qwen3-32B for reasoning.
    # DFlash2/DSpark speculative decoding requires NVIDIA CUDA — NOT for Intel Arc NUC.
    # On NUC 14 Pro: run via IPEX-LLM Ollama (~19GB, fits in 32GB iGPU pool)
    # Speed: ~5-8 tok/s (no speculative decoding on Intel Arc). Fine for EOD analysis.
    # On NVIDIA GPU (H200/A100/DGX Spark): use DFlash2 for 3-5x speedup via SGLang.
    "nuc-qwen38": {
        "llm_base_url": "http://localhost:11434/v1", "llm_api_key": "ollama",
        "llm_model": "qwen3:30b-a3b",      # Phase 1: fast MoE (Ollama name pending for 3.8 series)
        "dac_base_url": "http://localhost:11434/v1", "dac_api_key": "ollama",
        "dac_model": "qwen3.8:27b",         # Phase 2: Aug 2026 model, newer than Qwen3-32B
        "_note": "NUC: Phase 2 = Qwen3.8-27B (newer arch, ~19GB fits iGPU). No DFlash2 on Intel Arc.",
    },
    # Qwen3.8-27B on NVIDIA via OpenRouter (DFlash2 speculation available server-side)
    "openrouter-qwen38": {
        "llm_base_url": "https://openrouter.ai/api/v1",
        "llm_api_key": os.environ.get("OPENROUTER_API_KEY", ""),
        "llm_model": "deepseek/deepseek-v4-flash",          # Phase 1: cheap + fast
        "dac_base_url": "https://openrouter.ai/api/v1",
        "dac_api_key": os.environ.get("OPENROUTER_API_KEY", ""),
        "dac_model": "qwen/qwen-3.8-27b",                   # Phase 2: Qwen3.8-27B via OpenRouter
        "_note": "Cloud: Phase 2 = Qwen3.8-27B via OpenRouter. Provider may apply DFlash2 server-side.",
    },
}


def probe_model(base_url: str, api_key: str, model: str, timeout: int = 5) -> dict:
    """Quick connectivity check. Returns {ok, latency_ms, response, error}."""
    try:
        from openai import OpenAI
        import time as _time
        client = OpenAI(base_url=base_url, api_key=api_key)
        t0 = _time.monotonic()
        resp = client.chat.completions.create(model=model,messages=[{"role":"user","content":"READY"}],max_tokens=5,temperature=0,timeout=timeout)
        ms = int((_time.monotonic()-t0)*1000)
        return {"ok":True,"latency_ms":ms,"response":(resp.choices[0].message.content or "").strip()[:50],"error":None}
    except Exception as e:
        return {"ok":False,"latency_ms":0,"response":"","error":str(e)[:120]}

def main(session_date: str, cfg: dict) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    log.info("Session analyser starting for %s", session_date)
    log.info("DB: %s", cfg["db_path"])
    log.info("LLM: %s @ %s", cfg["llm_model"], cfg["llm_base_url"])

    # Try LLM analysis; fall back to structured report if unreachable
    try:
        # Quick health check on LLM endpoint
        import urllib.request as _ur
        _ur.urlopen(cfg["llm_base_url"].replace("/v1","") + "/v1/models", timeout=3)
        log.info("LLM endpoint reachable — running full analysis")
        report, n_calls, model = run_analysis(session_date, cfg)
        log.info("LLM analysis done: %d tool calls, model=%s", n_calls, model)

        # ── D-A-C stress test on the initial analysis ──
        log.info("Running D-A-C stress test...")
        dac_report, dac_calls = run_dac_stress_test(report, session_date, cfg)
        n_calls += dac_calls
        log.info("D-A-C complete: %d additional tool calls", dac_calls)

        # Append D-A-C to main report
        separator = "\n\n" + "═"*60 + "\n"
        report = report + separator + dac_report

        # Extract and log grounded recommendations
        grounded = parse_grounded_recommendations(dac_report)
        if grounded:
            log.info("D-A-C GROUNDED RECOMMENDATIONS (%d):", len(grounded))
            for i, rec in enumerate(grounded, 1):
                log.info("  %d. %s", i, rec[:100])

    except Exception as e:
        log.warning("LLM endpoint unreachable (%s) — generating fallback report", e)
        report   = generate_fallback_report(session_date, cfg["db_path"])
        n_calls  = 0
        model    = "fallback"

    # Save to DB
    _save_analysis(cfg["db_path"], session_date, report, n_calls, model)
    log.info("Report saved to DB (llm_analysis table)")

    # Send notifications
    sent = send_all(report, session_date, cfg)
    log.info("Notifications sent: discord=%s telegram=%s",
             sent.get("discord"), sent.get("telegram"))

    # Always print to stdout (visible in trading_engine.log)
    print("\n" + "="*60)
    print(f"  SESSION ANALYSIS — {session_date}")
    print("="*60)
    print(report)
    print("="*60 + "\n")


def test_adversarial_backend(model: str | None = None,
                              base_url: str = "http://localhost:11434/v1") -> dict:
    """
    Probe the adversarial model with a structured test prompt.
    Returns dict with model, latency, and a sample challenge response.
    Usage: python -c "import session_analyser; print(session_analyser.test_adversarial_backend())"
    """
    import time
    target_model = model or os.environ.get("OLLAMA_MODEL_ADVERSARIAL", "mistral-nemo:12b")
    try:
        from openai import OpenAI
        client = OpenAI(base_url=base_url, api_key="ollama")
        t0 = time.monotonic()
        resp = client.chat.completions.create(
            model=target_model,
            messages=[{
                "role": "system",
                "content": "You are a sceptical quantitative analyst. Challenge any trading claim critically."
            }, {
                "role": "user",
                "content": (
                    "A colleague says the 15-minute ORB strategy on SPY has Sharpe 1.8 "
                    "on 2016-2022 in-sample data. In 2 sentences, what is your biggest "
                    "concern about this claim?"
                )
            }],
            max_tokens=120,
            temperature=0.1,
            timeout=60,
        )
        latency = round(time.monotonic() - t0, 2)
        reply   = resp.choices[0].message.content or ""
        return {
            "model":    target_model,
            "status":   "OK",
            "latency_s": latency,
            "response": reply.strip()[:300],
            "note": ("PASS: adversarial backend responding. "
                     "If response challenges IS overfitting or RVOL filter absence, "
                     "the architecture divergence from Qwen is working as intended."),
        }
    except Exception as e:
        return {
            "model":  target_model,
            "status": "FAIL",
            "error":  str(e),
            "note":   f"Pull model first: ollama-ipex pull {target_model}",
        }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="LLM-powered session analyser")
    p.add_argument("--date", default=str(date.today()),
                   help="Session date YYYY-MM-DD (default: today)")
    p.add_argument("--db",   default=CONFIG["db_path"],
                   help="SQLite DB path (must match trading_engine.py --db)")
    p.add_argument("--model",  default=CONFIG["llm_model"])
    p.add_argument("--url",    default=CONFIG["llm_base_url"])
    p.add_argument("--discord", default=CONFIG["discord_webhook"])
    p.add_argument("--telegram-token",   default=CONFIG["telegram_token"])
    p.add_argument("--telegram-chat-id", default=CONFIG["telegram_chat_id"])
    p.add_argument("--no-llm", action="store_true",
                   help="Skip LLM analysis, produce fallback report only")
    p.add_argument("--preset",    default=None, choices=list(MODEL_PRESETS.keys()),
                   help="Named model preset (e.g. nuc-balanced)")
    p.add_argument("--dac-model", default=None, help="Override adversarial model")
    p.add_argument("--dac-url",   default=None, help="Override adversarial base URL")
    p.add_argument("--dac-key",   default=None, help="Override adversarial API key")
    p.add_argument("--probe",     action="store_true",
                   help="Probe LLM endpoints and exit")
    args = p.parse_args()

    cfg = {**CONFIG,
           "db_path":          args.db,
           "llm_model":        args.model,
           "llm_base_url":     args.url,
           "discord_webhook":  args.discord,
           "telegram_token":   args.telegram_token,
           "telegram_chat_id": args.telegram_chat_id}

    if args.preset:
        _p = MODEL_PRESETS.get(args.preset)
        if not _p:
            log.error("Unknown preset. Options: %s", list(MODEL_PRESETS.keys()))
            sys.exit(1)
        cfg.update({k: v for k, v in _p.items() if not k.startswith("_")})
        log.info("Preset '%s': %s", args.preset, _p.get("_note",""))
    if args.dac_model: cfg["dac_model"]    = args.dac_model
    if args.dac_url:   cfg["dac_base_url"] = args.dac_url
    if args.dac_key:   cfg["dac_api_key"]  = args.dac_key

    if getattr(args, 'probe', False):
        import time
        r = probe_model(cfg['llm_base_url'], cfg['llm_api_key'], cfg['llm_model'])
        status = 'OK' if r['ok'] else 'FAIL'
        print(f"Phase 1 [{status}] {cfg['llm_model']} {r['latency_ms']}ms: {r.get('response') or r.get('error')}")
        dm = cfg.get('dac_model') or cfg['llm_model']
        du = cfg.get('dac_base_url') or cfg['llm_base_url']
        dk = cfg.get('dac_api_key')  or cfg['llm_api_key']
        if dm != cfg['llm_model'] or du != cfg['llm_base_url']:
            r2 = probe_model(du, dk, dm)
            s2 = 'OK' if r2['ok'] else 'FAIL'
            print(f"Phase 2 D-A-C [{s2}] {dm} {r2['latency_ms']}ms: {r2.get('response') or r2.get('error')}")
        sys.exit(0)

    if getattr(args, "probe", False):
        import json
        result = test_adversarial_backend()
        print(json.dumps(result, indent=2))
        raise SystemExit(0)

    if args.no_llm:
        report = generate_fallback_report(args.date, cfg["db_path"])
        _save_analysis(cfg["db_path"], args.date, report, 0, "fallback")
        send_all(report, args.date, cfg)
        print(report)
    else:
        main(args.date, cfg)
