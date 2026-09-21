"""
tool_runner.py
==============
All tool implementations for the session_analyser.py LLM D-A-C pipeline.

WHY THIS EXISTS
───────────────
The local LLM (qwen3.8:27b, qwen3:30b-a3b via Ollama) needs to call tools
to ground its analysis in real data. Without tools it hallucinates.

Three categories of tools the LLM needs:
  1. WEB  — search the web, load a full page for its content
  2. DATA — query the trading DB, run statistical toolkit functions
  3. SYS  — run whitelisted system scripts, check system status

TOOLS LIST vs DISPATCH
──────────────────────
TOOLS      — the OpenAI function-calling schema sent to the LLM
dispatch()  — the Python function that executes what the LLM asked for

IMPORT PATTERN (session_analyser.py)
─────────────────────────────────────
  from tool_runner import TOOLS, dispatch_tool

WHAT WORKS ON INTEL ARC / OFFLINE
───────────────────────────────────
All tools run on the NUC without internet if:
  - DB tools: always offline ✅
  - run_toolkit: always offline ✅
  - run_script: always offline ✅
  - web_search: needs internet (DuckDuckGo, no API key required)
  - fetch_url: needs internet
  If the NUC is offline, web/fetch tools return a clear error and the LLM
  falls back to DB + toolkit grounding only.
"""

from __future__ import annotations

import json
import logging
import os
import re
import signal
import sqlite3
import subprocess
import sys
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import date
from pathlib import Path
from typing import Any

# Optional enhanced dependencies — degrade gracefully if not installed
try:
    import yfinance as yf
    _YFINANCE = True
except ImportError:
    _YFINANCE = False

try:
    from domain_telemetry import get_domain_strategy, record_domain_result
    _TELEMETRY = True
except ImportError:
    _TELEMETRY = False
    def get_domain_strategy(url: str) -> str: return "HTTPX_TRAFILATURA"  # noqa
    def record_domain_result(*a, **kw): pass  # noqa

log = logging.getLogger("tool_runner")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TOOL_TIMEOUT     = 45    # seconds — raised from 30; ddgs needs time to try multiple backends
WEB_BODY_CHARS   = 800   # chars returned per web search result (was 200, too short)
FETCH_MAX_CHARS  = 3000  # chars returned from a full page fetch
WEB_N_RESULTS    = 10    # default number of search results — more options for multi-source fetch

# FIXED 19 Sep 2026 (revised twice — see history below): the `ddgs` package's
# DDGS().text() defaults to backend="auto", a metasearch pool (bing, brave,
# duckduckgo, google, mojeek, startpage, yandex, yahoo, wikipedia) with only
# a 5s default HTTP timeout per backend attempt.
#
# Revision history, each caught by live-testing before being trusted:
#   1. Original bug: "auto" mode hit Startpage, which proxies through Google
#      and is prone to bot-detection — timed out.
#   2. First fix attempt: pinned to backend="duckduckgo" alone. Live-tested —
#      FAILED outright. Pinning to one backend throws away fallback
#      resilience; if that one backend is itself slow/blocked on a given
#      network (confirmed for plain DuckDuckGo in the sandbox used to test
#      this — raw curl to duckduckgo.com got zero response), a single-
#      backend pin fails with no fallback, worse than the original bug.
#   3. Second fix: "duckduckgo,bing,brave,yahoo" — kept fallback resilience,
#      excluded Startpage. CONFIRMED ON THE ACTUAL PRODUCTION NUC (not just
#      the test sandbox) that this still isn't enough: it stalled on THIS
#      list's "brave" entry — ddgs's own HTML-scraping backend against
#      search.brave.com (the consumer search results page), a completely
#      different thing from _brave_search()'s call to the real API at
#      api.search.brave.com. Error was "Connection reset by peer" on a
#      follow-up fetch_url attempt — an actively torn-down connection, more
#      consistent with bot-detection/blocking than ordinary slowness.
#   4. Dropped ddgs's scraping "brave" backend too, confirmed bad on the real
#      network, not just suspected. Two scraping backends failing on the same
#      real network is a pattern, not one bad engine.
#   5. Current (20 Sep 2026, P2-083): expanded to the five backends below
#      (bing, google, duckduckgo, yahoo, ecosia), validated on the production
#      NUC by a parallel session. SearXNG public instances were considered and
#      rejected -- ddgs's multi-engine pool is the pip-native answer.
# BRAVE (corrected 20 Sep 2026): earlier revisions of this comment called setting
# BRAVE_SEARCH_API_KEY "the actual fix". Brave removed its free tier in Feb 2026
# (new accounts get $5/month in credits, ~1,000 queries, card required, billed
# beyond that), so the key is now OPTIONAL. _brave_search() is still tried first
# when the key is set, but the ddgs list below is the default path.
# ddgs v0.3+ supports 9 engines: duckduckgo, bing, google, brave, ecosia,
# qwant, yahoo, yandex, wikipedia. More engines = more resilience. If one
# is blocked/rate-limited, ddgs tries the next automatically.
# "auto" would pick for us but is unpredictable; explicit list is better.
DDG_BACKEND      = "bing,google,duckduckgo,yahoo,ecosia"


# Whitelisted scripts that run_script is allowed to execute
# Format: {alias: [python_path, *args]}
# The LLM requests by alias, not by raw command.
SCRIPT_WHITELIST: dict[str, list[str]] = {
    "historical_report":    ["-m", "historical_sim",    "--report"],
    "store_status":         ["-m", "market_data_store", "--status"],
    "store_regime":         ["-m", "market_data_store", "--regime-analysis", "--ticker", "SPY"],
    "store_validate":       ["-m", "market_data_store", "--validate",        "--ticker", "SPY"],
    "morning_brief":        ["-m", "morning_brief"],
}

# ---------------------------------------------------------------------------
# Timeout utility
# ---------------------------------------------------------------------------

def _with_timeout(fn, args: tuple = (), kwargs: dict | None = None, timeout: int = TOOL_TIMEOUT) -> Any:
    """Run fn(*args, **kwargs) with a wall-clock timeout. Returns result or raises TimeoutError."""
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(fn, *args, **(kwargs or {}))
        try:
            return fut.result(timeout=timeout)
        except FuturesTimeoutError:
            raise TimeoutError(f"Tool call timed out after {timeout}s")


# ---------------------------------------------------------------------------
# TOOL 1 — web_search
# ---------------------------------------------------------------------------

def _brave_search(query: str, n_results: int, api_key: str) -> str | None:
    """
    Brave Search API -- independent index. No free tier since Feb 2026: new accounts get
    $5/month in credits (~1,000 queries), then usage is billed.
    Returns formatted results string or None on failure.
    API key: api.search.brave.com  (credit card required; usage beyond the monthly credit is billed)
    """
    try:
        import requests as _req
        resp = _req.get(
            "https://api.search.brave.com/res/v1/web/search",
            params={"q": query, "count": min(n_results, 20), "search_lang": "en"},
            headers={
                "Accept": "application/json",
                "X-Subscription-Token": api_key,
            },
            timeout=15,
        )
        if resp.status_code == 401:
            return None   # bad key — fall through to DDG
        if resp.status_code == 429:
            return None   # rate limited — fall through to DDG
        resp.raise_for_status()
        data   = resp.json()
        items  = data.get("web", {}).get("results", [])
        if not items:
            return None
        out = []
        for r in items[:n_results]:
            title   = r.get("title", "")
            url     = r.get("url", "")
            snippet = (r.get("description") or r.get("extra_snippets", [""])[0] or "")[:WEB_BODY_CHARS]
            out.append(f"[{title}]\n{url}\n{snippet}")
        return "\n\n---\n\n".join(out)
    except Exception:
        return None


def _ddg_search(query: str, n_results: int) -> str:
    """DuckDuckGo fallback — no API key, HTML scrape, may rate-limit on heavy use."""
    try:
        try:
            from ddgs import DDGS                # new package name (replaces duckduckgo_search)
        except ImportError:
            from duckduckgo_search import DDGS   # fallback for older installs
        out = []
        # timeout on DDGS() constructor (default is 5s — too low for slow backends)
        with DDGS(timeout=20) as ddgs:
            # backend order: bing,google tried first — duckduckgo last
            for r in ddgs.text(query, max_results=n_results, backend=DDG_BACKEND):
                title = r.get("title", "")
                href  = r.get("href", "")
                body  = r.get("body", "")[:WEB_BODY_CHARS]
                out.append(f"[{title}]\n{href}\n{body}")
        return "\n\n---\n\n".join(out) if out else "No results found."
    except ImportError:
        return "DDG search not installed. Run: pip install ddgs"
    except Exception as e:
        return f"Search failed: {e}"


# Common uppercased words that look like tickers but aren't
_TICKER_BLACKLIST = {
    "NEWS", "TODAY", "STOCK", "WHAT", "BEST", "FOR", "USA", "USD", "FED",
    "THE", "AND", "SPX", "ETF", "TOP", "MARKET", "LATEST", "THIS", "WEEK",
    "NOW", "ARE", "HOW", "WHY", "FROM", "WITH", "RATE", "HIKE", "HIGH", "LOW"
}


def _yfinance_headlines(query: str, max_items: int = 3) -> str:
    """Pull verified publisher headlines direct from Yahoo Finance for any ticker in query."""
    if not _YFINANCE:
        return ""
    tickers = re.findall(r'\b[A-Z]{2,5}\b', query.upper())
    valid = [t for t in tickers if t not in _TICKER_BLACKLIST]
    if not valid:
        return ""
    lines = []
    try:
        t = yf.Ticker(valid[0])
        for n in (t.news or [])[:max_items]:
            # yfinance v0.2+ returns nested content dict
            content = n.get("content", n)  # fall back to n itself for older versions
            title = content.get("title", "")
            # canonicalUrl is more reliable than clickThroughUrl
            canon = content.get("canonicalUrl", {})
            link  = canon.get("url", "") or content.get("clickThroughUrl", {}).get("url", "")
            pub   = (content.get("provider", {}) or {}).get("displayName", "Yahoo Finance")
            if title and link:
                lines.append(f"[{pub}] {title}\nURL: {link}")
        if lines:
            log.debug("web_search: yfinance pre-flight returned %d headlines for %s",
                      len(lines), valid[0])
    except Exception as e:
        log.debug("web_search: yfinance pre-flight failed: %s", e)
    return "\n".join(lines)


def tool_web_search(query: str, n_results: int = WEB_N_RESULTS) -> str:
    """
    Search the web.

    Strategy: yfinance ticker pre-flight (if ticker detected)
              → Brave Search API (primary, independent index)
              → DuckDuckGo multi-engine fallback (bing,google,duckduckgo,yahoo,ecosia)

    Brave is preferred because it has its own index, a proper JSON API with an SLA,
    and consistent results. DuckDuckGo scrapes Bing's HTML endpoint and can be blocked
    from repeated same-IP requests.

    Set BRAVE_SEARCH_API_KEY in .env to activate Brave.
    Get a key at: https://api.search.brave.com  (no free tier since Feb 2026; $5/month credit, card required)
    """
    # yfinance pre-flight: zero-scraping official news for ticker queries
    yf_prefix = _yfinance_headlines(query)

    def _search():
        brave_key = os.environ.get("BRAVE_SEARCH_API_KEY", "").strip()

        # Primary: Brave Search API
        if brave_key:
            result = _brave_search(query, n_results, brave_key)
            if result:
                log.debug("web_search: Brave Search used")
                return result
            log.info("web_search: Brave failed/rate-limited — falling back to DuckDuckGo")

        # Fallback: ddgs multi-engine (bing,google,duckduckgo,yahoo,ecosia)
        ddg = _ddg_search(query, n_results)
        return (yf_prefix + "\n\n---\n\n" + ddg).strip() if yf_prefix else ddg

    try:
        return _with_timeout(_search, timeout=TOOL_TIMEOUT)
    except TimeoutError as e:
        return f"Search timed out: {e}"


# ---------------------------------------------------------------------------
# TOOL 2 — fetch_url
# ---------------------------------------------------------------------------

def tool_fetch_url(url: str, max_chars: int = FETCH_MAX_CHARS) -> str:
    """
    Fetch the main text content from a URL.

    Uses trafilatura for clean article extraction (strips nav, ads, boilerplate).
    Integrates domain_telemetry: skips quarantined domains, logs success/failure.
    Falls back to requests + basic HTML strip if trafilatura unavailable.

    Useful for: reading full SSRN paper abstracts, news articles found via
    web_search, FRED/VIX data pages, Zarattini blog posts, etc.
    """
    if not url.startswith(("http://", "https://")):
        return f"Invalid URL (must start with http:// or https://): {url}"

    # Telemetry gate — skip quarantined domains in <1ms
    if get_domain_strategy(url) == "SKIP":
        log.debug("fetch_url: skipping quarantined domain %s", url)
        return f"[Skipped: domain quarantined due to persistent 403/paywall — trying next source]"

    def _fetch():
        t0 = time.time()
        try:
            import requests
            resp = requests.get(
                url,
                timeout=15,
                headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                         "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"},
            )
            lat_ms = int((time.time() - t0) * 1000)

            if resp.status_code in (401, 403):
                record_domain_result(url, False, error_type="403_BOT", latency_ms=lat_ms)
                return f"HTTP fetch failed: {resp.status_code} Client Error: HTTP Forbidden for url: {url}"
            if resp.status_code != 200:
                record_domain_result(url, False,
                                     error_type=f"HTTP_{resp.status_code}", latency_ms=lat_ms)
                resp.raise_for_status()

            html = resp.text
        except Exception as e:
            return f"HTTP fetch failed: {e}"

        # Try trafilatura first — best at extracting article body
        try:
            import trafilatura
            text = trafilatura.extract(html, include_comments=False, include_tables=True)
            if text and len(text.strip()) > 100:
                record_domain_result(url, True, latency_ms=lat_ms)
                compact = " ".join(text.split())
                return compact[:max_chars].rsplit(" ", 1)[0] + "..."
        except ImportError:
            pass

        # Fallback: basic tag stripping
        try:
            import re as _re
            text = _re.sub(r"<[^>]+>", " ", html)
            text = _re.sub(r"\s+", " ", text).strip()
            if len(text) > 180:
                record_domain_result(url, True, latency_ms=lat_ms)
                return text[:max_chars]
            record_domain_result(url, False, error_type="EMPTY_CONTENT", latency_ms=lat_ms)
            return f"[No article body extracted from {url}]"
        except Exception as e:
            record_domain_result(url, False, error_type="EXTRACT_ERROR")
            return f"Content extraction failed: {e}"

    try:
        return _with_timeout(_fetch, timeout=TOOL_TIMEOUT)
    except TimeoutError as e:
        return f"URL fetch timed out: {e}"


# ---------------------------------------------------------------------------
# TOOL 3 — run_toolkit
# ---------------------------------------------------------------------------

def tool_run_toolkit(function_name: str, kwargs_json: str = "{}") -> str:
    """
    Call a function from trading_quant_toolkit_v2_4.py and return its result.

    The LLM uses this to compute:
      kelly_fraction(win_rate, avg_win_r, avg_loss_r)
      sharpe_significance_tstat_annualised(sharpe, n_trades)
      deflated_sharpe_ratio_extended(sharpe, n_trades, n_trials)
      regime_conditional_streak_probability(streak_len, win_rate, regime)
      expected_value(win_rate, avg_win_r, avg_loss_r, commission_r)
      … and any other function in the toolkit.
    """
    def _run():
        try:
            src_dir = Path(__file__).parent
            if str(src_dir) not in sys.path:
                sys.path.insert(0, str(src_dir))
            import trading_quant_toolkit_v2_4 as tk
        except ImportError:
            return "trading_quant_toolkit_v2_4 not found in src/ directory"

        fn = getattr(tk, function_name, None)
        if fn is None:
            fns = [n for n in dir(tk) if not n.startswith("_") and callable(getattr(tk, n))]
            return (f"Function '{function_name}' not found in toolkit v2.4.0.\n"
                    f"Available: {', '.join(fns[:20])}")
        try:
            kwargs = json.loads(kwargs_json) if kwargs_json.strip() else {}
            result = fn(**kwargs)
            return json.dumps(result, indent=2, default=str)
        except Exception as e:
            return f"Toolkit call failed ({function_name}): {e}"

    try:
        return _with_timeout(_run, timeout=TOOL_TIMEOUT)
    except TimeoutError as e:
        return f"Toolkit call timed out: {e}"


# ---------------------------------------------------------------------------
# TOOL 4 — run_script
# ---------------------------------------------------------------------------

def tool_run_script(script_alias: str, extra_args: dict | None = None) -> str:
    """
    Run a whitelisted system script and return its stdout output.

    The LLM can check system state, run reports, and validate data
    without being able to execute arbitrary code.

    Available aliases:
      historical_report  → python -m historical_sim --report
      store_status       → python -m market_data_store --status
      store_regime       → python -m market_data_store --regime-analysis
      store_validate     → python -m market_data_store --validate
      morning_brief      → python -m morning_brief
    """
    if script_alias not in SCRIPT_WHITELIST:
        available = ", ".join(SCRIPT_WHITELIST.keys())
        return (f"Unknown script alias '{script_alias}'.\n"
                f"Available: {available}")

    base_args = SCRIPT_WHITELIST[script_alias]

    # Inject extra --db path if needed (from cfg passed via extra_args)
    cmd_args: list[str] = [sys.executable] + base_args
    if extra_args:
        db = extra_args.get("db")
        if db:
            cmd_args += ["--db", str(db)]
        ticker = extra_args.get("ticker")
        if ticker:
            cmd_args += ["--ticker", ticker]

    def _run():
        try:
            # Set working directory to project root so DATA/ paths resolve correctly
            project_root = Path(__file__).parent.parent
            result = subprocess.run(
                cmd_args,
                capture_output=True,
                text=True,
                timeout=60,
                cwd=str(project_root),
            )
            out = (result.stdout or "").strip()
            err = (result.stderr or "").strip()
            if result.returncode != 0:
                return f"Script exited with code {result.returncode}.\nSTDERR: {err[:500]}"
            return out[:3000] or "(no output)"
        except subprocess.TimeoutExpired:
            return f"Script '{script_alias}' timed out after 60s"
        except Exception as e:
            return f"Script execution failed: {e}"

    try:
        return _with_timeout(_run, timeout=90)
    except TimeoutError as e:
        return f"Script timed out: {e}"


# ---------------------------------------------------------------------------
# TOOL 5 — get_session_data (DB)
# ---------------------------------------------------------------------------

def tool_get_session_data(db_path: str, session_date: str) -> str:
    """Fetch session data, trades, decisions, and session learning from the DB."""
    def _q(sql: str, params: tuple = ()) -> list[dict]:
        with sqlite3.connect(db_path) as c:
            c.row_factory = sqlite3.Row
            return [dict(r) for r in c.execute(sql, params).fetchall()]

    def _q1(sql: str, params: tuple = ()) -> dict | None:
        rows = _q(sql, params)
        return rows[0] if rows else None

    try:
        parts: list[str] = []

        sess = _q1("SELECT * FROM sessions WHERE session_date=?", (session_date,))
        if sess:
            parts.append("SESSION:\n" + json.dumps(sess, default=str, indent=2))

        trades = _q(
            "SELECT direction,entry_price,exit_price,actual_r,exit_reason,notes "
            "FROM positions WHERE date(opened_at)=? ORDER BY id",
            (session_date,)
        )
        if trades:
            parts.append("TRADES:\n" + json.dumps(trades, default=str, indent=2))

        decs = _q(
            "SELECT action,gate_orb_break,gate_vwap,gate_retest,gate_final,"
            "vix,rvol,vwap_slope,reason FROM decisions "
            "WHERE session_date=? ORDER BY id",
            (session_date,)
        )
        if decs:
            parts.append(f"DECISIONS ({len(decs)} evaluated):\n" +
                         json.dumps(decs, default=str, indent=2))

        sl = _q1(
            "SELECT * FROM session_learning WHERE session_date=? "
            "ORDER BY id DESC LIMIT 1",
            (session_date,)
        )
        if sl:
            parts.append("SESSION LEARNING:\n" + json.dumps(sl, default=str, indent=2))

        return "\n\n".join(parts) if parts else f"No data found for {session_date}."
    except Exception as e:
        return f"DB error: {e}"


# ---------------------------------------------------------------------------
# TOOL 6 — get_wfa_results (DB)
# ---------------------------------------------------------------------------

def tool_get_wfa_results(db_path: str, ticker: str = "SPY", n: int = 12) -> str:
    """Fetch the last N walk-forward analysis results from the DB."""
    try:
        with sqlite3.connect(db_path) as c:
            c.row_factory = sqlite3.Row
            rows = c.execute(
                "SELECT * FROM wfa_results WHERE ticker=? ORDER BY id DESC LIMIT ?",
                (ticker, n)
            ).fetchall()
        if not rows:
            return "No WFA results found. Run the historical simulation first."
        data = [dict(r) for r in rows]
        wfes = [r["wfe"] for r in data if r.get("wfe") is not None]
        summary = {
            "n_splits":      len(data),
            "mean_wfe":      round(sum(wfes) / len(wfes), 3) if wfes else None,
            "min_wfe":       round(min(wfes), 3) if wfes else None,
            "max_wfe":       round(max(wfes), 3) if wfes else None,
            "pass_count":    sum(1 for w in wfes if w >= 0.50),
            "gate_wfe_0.50": "PASS" if (wfes and sum(wfes)/len(wfes) >= 0.50) else "FAIL",
        }
        return json.dumps({"summary": summary, "splits": data}, default=str, indent=2)
    except Exception as e:
        return f"WFA DB error: {e}"


# ---------------------------------------------------------------------------
# TOOL 7 — get_rolling_stats (DB)
# ---------------------------------------------------------------------------

def tool_get_rolling_stats(db_path: str, n: int = 20) -> str:
    """Compute win rate, EV, and Sharpe from the last N closed trades."""
    try:
        with sqlite3.connect(db_path) as c:
            c.row_factory = sqlite3.Row
            rows = c.execute(
                "SELECT actual_r FROM positions WHERE status='closed' "
                "AND actual_r IS NOT NULL ORDER BY id DESC LIMIT ?",
                (n,)
            ).fetchall()
        if not rows:
            return json.dumps({"n": 0, "message": "No closed trades yet."})
        rs     = [r["actual_r"] for r in rows]
        wins   = [r for r in rs if r > 0]
        losses = [r for r in rs if r < 0]
        wr     = len(wins) / len(rs)
        aw     = sum(wins) / len(wins)       if wins   else 0.0
        al     = abs(sum(losses)/len(losses)) if losses else 0.0
        ev     = (wr * aw) - ((1-wr) * al)
        import statistics as _s
        sharpe = round(_s.mean(rs) / _s.stdev(rs), 3) if len(rs) >= 2 and _s.stdev(rs) > 0 else None
        return json.dumps({
            "n": len(rs), "win_rate": round(wr, 3),
            "avg_win_r": round(aw, 3), "avg_loss_r": round(al, 3),
            "ev_per_trade": round(ev, 4), "sharpe": sharpe,
            "note": f"Last {n} trades. Meaningful at 50+ trades; significant at 200+."
        }, indent=2)
    except Exception as e:
        return f"Error computing rolling stats: {e}"


# ---------------------------------------------------------------------------
# TOOL 8 — get_journal_stats (DB)
# ---------------------------------------------------------------------------

def tool_get_journal_stats(db_path: str, min_samples: int = 3) -> str:
    """
    Return correlation stats from the extended trade journal:
    win rate by candle type, time slot, entry confidence, emotional state, etc.
    """
    try:
        with sqlite3.connect(db_path) as c:
            c.row_factory = sqlite3.Row
            rows = c.execute(
                "SELECT * FROM trade_journal_extended WHERE actual_r IS NOT NULL"
            ).fetchall()
        if not rows:
            return json.dumps({"n": 0, "message": "No journal entries yet."})

        def _groupby(field: str) -> dict:
            groups: dict[str, list[float]] = {}
            for r in rows:
                key = r[field] if r[field] is not None else "unknown"
                groups.setdefault(str(key), []).append(r["actual_r"])
            out = {}
            for k, rs in groups.items():
                if len(rs) >= min_samples:
                    wr = sum(1 for r in rs if r > 0) / len(rs)
                    ev = sum(rs) / len(rs)
                    out[k] = {"n": len(rs), "wr": round(wr, 3), "ev": round(ev, 4)}
            return dict(sorted(out.items(), key=lambda x: x[1]["ev"], reverse=True))

        correlations: dict[str, Any] = {}
        for field in [
            "entry_candle_type", "entry_candle_colour", "vwap_slope_at_entry",
            "time_slot", "day_of_week", "emotional_state",
            "confidence_pre", "setup_grade", "process_grade",
        ]:
            try:
                result = _groupby(field)
                if result:
                    correlations[field] = result
            except Exception:
                pass

        return json.dumps({
            "n_total": len(rows),
            "min_samples_filter": min_samples,
            "correlations": correlations,
        }, indent=2)
    except Exception as e:
        return f"Journal stats error: {e}"


# ---------------------------------------------------------------------------
# Tool schema (OpenAI function-calling format — works with Ollama)
# ---------------------------------------------------------------------------

TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web. Primary: Brave Search (set BRAVE_SEARCH_API_KEY, "
                "2,000 free queries/month). Fallback: DuckDuckGo (no key, may rate-limit). "
                "Use for: VIX levels, SPY price action, macro news, SSRN abstracts, "
                "Fed announcements, earnings releases."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query":     {"type": "string", "description": "Search query"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": (
                "Fetch and extract the main text content from a URL. "
                "Use after web_search when you need to read the full article, "
                "paper, or data page rather than just the search snippet. "
                "Works on SSRN papers, news articles, FRED data pages, Zarattini blog posts."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url":       {"type": "string",  "description": "Full URL to fetch (must start with https://)"},
                    "max_chars": {"type": "integer", "description": "Max characters to return (default 3000)", "default": 3000},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_toolkit",
            "description": (
                "Call a function from trading_quant_toolkit_v2_4.py to compute "
                "a statistical metric. Key functions: "
                "kelly_fraction(win_rate, avg_win_r, avg_loss_r), "
                "sharpe_significance_tstat_annualised(sharpe, n_trades), "
                "deflated_sharpe_ratio_extended(sharpe, n_trades, n_trials), "
                "expected_value(win_rate, avg_win_r, avg_loss_r, commission_r)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "function_name": {"type": "string", "description": "Exact function name"},
                    "kwargs_json":   {"type": "string", "description": "JSON object of keyword arguments"},
                },
                "required": ["function_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_script",
            "description": (
                "Run a whitelisted system script and return its output. "
                "Use to check system status, run reports, validate data. "
                "Available aliases: historical_report, store_status, store_regime, "
                "store_validate, morning_brief."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "script_alias": {
                        "type": "string",
                        "description": "Script alias",
                        "enum": list(SCRIPT_WHITELIST.keys()),
                    },
                    "extra_args": {
                        "type": "object",
                        "description": "Optional: {db: '/path/to/db', ticker: 'SPY'}",
                    },
                },
                "required": ["script_alias"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_session_data",
            "description": "Fetch session data, trades, decisions, and session learning from the DB for a given date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "session_date": {"type": "string", "description": "Date in YYYY-MM-DD format"},
                },
                "required": ["session_date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_wfa_results",
            "description": "Fetch walk-forward analysis results from the DB. Shows WFE scores per split, mean WFE, and whether the strategy passes the WFE ≥ 0.50 gate.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Ticker symbol (default SPY)"},
                    "n":      {"type": "integer", "description": "Number of most recent splits to return (default 12)"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_rolling_stats",
            "description": "Compute win rate, EV, and Sharpe from the last N closed trades.",
            "parameters": {
                "type": "object",
                "properties": {
                    "n": {"type": "integer", "description": "Number of recent trades to use (default 20)"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_journal_stats",
            "description": "Return correlation stats from the trade journal: win rate and EV broken down by candle type, time slot, emotional state, setup grade, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "min_samples": {"type": "integer", "description": "Minimum samples per group (default 3)"},
                },
            },
        },
    },
]

# ---------------------------------------------------------------------------
# Dispatcher — maps LLM tool call name → implementation
# ---------------------------------------------------------------------------

def dispatch_tool(name: str, args: dict, cfg: dict) -> str:
    """
    Route a tool call from the LLM to its implementation.
    cfg must contain at minimum: {"db_path": "...", "web_search_results": 5}

    Returns a string result (never raises — errors returned as readable text).
    """
    db_path  = cfg.get("db_path", "DATA/paper_account.db")
    n_web    = cfg.get("web_search_results", WEB_N_RESULTS)

    try:
        if name == "web_search":
            return tool_web_search(args.get("query", ""), n_web)  # always use WEB_N_RESULTS, ignore model's n_results

        if name == "fetch_url":
            return tool_fetch_url(args.get("url", ""), args.get("max_chars", FETCH_MAX_CHARS))

        if name == "run_toolkit":
            return tool_run_toolkit(
                args.get("function_name", ""),
                args.get("kwargs_json", "{}"),
            )

        if name == "run_script":
            return tool_run_script(
                args.get("script_alias", ""),
                args.get("extra_args") or {"db": db_path},
            )

        if name == "get_session_data":
            return tool_get_session_data(db_path, args.get("session_date", str(date.today())))

        if name == "get_wfa_results":
            return tool_get_wfa_results(db_path, args.get("ticker", "SPY"), args.get("n", 12))

        if name == "get_rolling_stats":
            return tool_get_rolling_stats(db_path, args.get("n", 20))

        if name == "get_journal_stats":
            return tool_get_journal_stats(db_path, args.get("min_samples", 3))

        return f"Unknown tool: '{name}'. Available: {[t['function']['name'] for t in TOOLS]}"

    except Exception as e:
        log.error("Tool dispatch error (%s): %s", name, e, exc_info=True)
        return f"Tool '{name}' raised an unexpected error: {e}"
