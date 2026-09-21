#!/usr/bin/env python3
"""
chainlit_app.py — Trading Income Project
=========================================
Chainlit UI with:
- ChatProfiles: D-A-C (thinking), Research (web search), Quick Chat
- ToolProgress: ONE collapsed, self-labelling section for all tool calls (P2-090)
- LaTeX delimiter fix (Mindfire pattern)
- Collect-then-render for thinking (solves ordering issue)
- Per-session state isolation
- Conversation starters
- cl.Action quick-action buttons
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from pathlib import Path

import datetime
import sqlite3

import httpx
import chainlit as cl
from chainlit.input_widget import Slider, Switch
from openai import AsyncOpenAI

# ── Backend config ─────────────────────────────────────────────────────────────
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
MODEL_ID    = os.getenv("MODEL_ID", "qwen3.8:27b")
DEBUG_MODE  = os.getenv("CHAINLIT_DEBUG", "0") == "1"
# Scout (the port-8001 pre-filter) is OFF by default -- P2-098, option B. In every live run examined
# (20-21 Sep) it cost 1.6-16 s per query and never contributed to an answer: its output was empty,
# discarded, or ungrounded (and the model on port 8001 differs per pair; on Phi-4-mini it is also sent
# a malformed prompt, P2-099). Set SCOUT_ENABLED=1 to turn it back on for an A/B comparison.
SCOUT_ENABLED = os.getenv("SCOUT_ENABLED", "0") == "1"

# Logger — verbose when CHAINLIT_DEBUG=1 (set by launch_models.sh --debug)
import logging as _logging
_logging.basicConfig(level=_logging.DEBUG if DEBUG_MODE else _logging.INFO)
log = _logging.getLogger("chainlit_app")

# Timeouts configurable from .env (OrionBelt pattern)
_BACKEND_TIMEOUT = float(os.getenv("BACKEND_TIMEOUT", "600"))
_TOOL_TIMEOUT    = float(os.getenv("TOOL_TIMEOUT", "30"))

_timeout = httpx.Timeout(_BACKEND_TIMEOUT, connect=10.0,
                          read=_BACKEND_TIMEOUT, write=30.0)
_http    = httpx.AsyncClient(timeout=_timeout)
_client  = AsyncOpenAI(
    base_url=f"{BACKEND_URL}/v1",
    api_key="local",
    http_client=_http,
)

# ── Path setup ────────────────────────────────────────────────────────────────
# chainlit_app.py lives in src/ — project root is one level up
_project_root = Path(__file__).parent.parent
_src = str(Path(__file__).parent)   # src/ itself
if _src not in sys.path:
    sys.path.insert(0, _src)

from tool_progress import ToolProgress  # src/tool_progress.py  (P2-090)

try:
    from dotenv import load_dotenv
    load_dotenv(_project_root / ".env", override=False)
except ImportError:
    pass

# ── SQLite chat history ────────────────────────────────────────────────────────
_DB_PATH = str(_project_root / "DATA" / "chainlit_history.db")

def _db_save_message(thread_id: str, role: str, content: str,
                     step_type: str = "message", profile: str = "Research") -> None:
    """Persist a message to the local SQLite chat history (fire-and-forget)."""
    try:
        Path(_DB_PATH).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(_DB_PATH, timeout=5) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS threads (
                    id TEXT PRIMARY KEY, created_at TEXT, name TEXT,
                    profile TEXT, user_id TEXT DEFAULT 'local')
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS steps (
                    id TEXT PRIMARY KEY, thread_id TEXT, role TEXT,
                    content TEXT, created_at TEXT,
                    step_type TEXT DEFAULT 'message', metadata TEXT DEFAULT '{}')
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_steps_thread ON steps(thread_id, created_at)
            """)
            ts = datetime.datetime.utcnow().isoformat()
            conn.execute(
                "INSERT OR IGNORE INTO threads VALUES (?,?,?,?,?)",
                (thread_id, ts, None, profile, "local")
            )
            conn.execute(
                "INSERT INTO steps VALUES (?,?,?,?,?,?,?)",
                (f"{thread_id}_{ts}_{role}", thread_id, role, content,
                 ts, step_type, "{}")
            )
            conn.commit()
    except Exception:
        pass  # persistence is best-effort, never crash the chat

def _db_load_thread(thread_id: str) -> list[dict]:
    """Load message history for a thread from SQLite."""
    try:
        with sqlite3.connect(_DB_PATH, timeout=5) as conn:
            rows = conn.execute(
                "SELECT role, content FROM steps WHERE thread_id=? "
                "AND step_type='message' ORDER BY created_at",
                (thread_id,)
            ).fetchall()
            return [{"role": r[0], "content": r[1]} for r in rows]
    except Exception:
        return []

# ── LaTeX fix (Mindfire Technology pattern) ───────────────────────────────────
def fix_latex(text: str) -> str:
    """Convert LaTeX delimiters to Chainlit-compatible format.
    Qwen3 outputs \\[...\\] and \\(...\\) — Chainlit only renders $$...$$ and $...$
    Critical for D-A-C sessions with Kelly criterion / probability calculations.
    """
    if not text:
        return text
    text = text.replace(r'\[', '$$').replace(r'\]', '$$')
    text = text.replace(r'\(', '$').replace(r'\)', '$')
    return text


# ── Plotly chart detection ────────────────────────────────────────────────────
def _try_render_plotly(result: str) -> dict | None:
    """
    Detect if a tool result contains Plotly-compatible data.
    Returns a Plotly figure dict if found, else None.
    Inspired by OrionBelt Chat's multi-strategy extraction approach.
    """
    import json as _json
    stripped = result.strip()
    # Try parsing as JSON directly
    if stripped.startswith("{") or stripped.startswith("["):
        try:
            obj = _json.loads(stripped)
            # Plotly figure dict has 'data' and optionally 'layout'
            if isinstance(obj, dict) and "data" in obj:
                return obj
            # List of trace objects
            if isinstance(obj, list) and obj and isinstance(obj[0], dict) and "type" in obj[0]:
                return {"data": obj, "layout": {"template": "plotly_dark"}}
        except Exception:
            pass
    return None


async def _maybe_render_chart(result: str, msg: cl.Message) -> None:
    """If tool result contains chart data, attach a Plotly element to the message."""
    fig = _try_render_plotly(result)
    if fig:
        # Apply trading dark theme to the chart
        if "layout" not in fig:
            fig["layout"] = {}
        fig["layout"].update({
            "template": "plotly_dark",
            "paper_bgcolor": "rgba(0,0,0,0)",
            "plot_bgcolor": "rgba(0,0,0,0)",
        })
        await cl.Plotly(name="chart", figure=fig, display="inline").send(for_id=msg.id)


# ── Thinking split ────────────────────────────────────────────────────────────
async def _collect_and_split(stream) -> tuple[str, str]:
    """Collect all streaming tokens, then split at </think>.
    Returns (thinking_text, answer_text).
    Used for non-thinking mode and as fallback.
    """
    buffer = ""
    async for chunk in stream:
        buffer += chunk.choices[0].delta.content or ""

    if "<think>" in buffer and "</think>" in buffer:
        s = buffer.find("<think>") + len("<think>")
        e = buffer.find("</think>")
        return buffer[s:e].strip(), buffer[e + len("</think>"):].strip()
    elif "<think>" in buffer:
        return buffer.replace("<think>", "").strip(), ""
    else:
        return "", buffer.strip()


async def _stream_thinking_then_answer(
    stream,
    thinking: bool,
    thinking_step_name: str = "💭 Thinking",
) -> tuple[str, str]:
    """
    Live-stream thinking tokens into a cl.Step accordion, then stream the
    answer tokens into a cl.Message bubble — in that guaranteed order.

    This is the confirmed production pattern from the Chainlit docs:
        async with cl.Step(...) as step:
            await step.stream_token(token)  # streams live inside accordion
        # step closes here → collapses → THEN answer bubble opens

    Returns (thinking_text, answer_text).
    """
    think_buf = ""
    answer_buf = ""
    in_think = False
    preamble = ""   # any text before <think> (e.g. "Let me think...")

    # ── Phase A: collect tokens, stream thinking live into Step ──────────────
    if thinking:
        step_ctx = cl.Step(name=thinking_step_name, type="run", show_input=False)
        await step_ctx.__aenter__()
        think_started = False

        async for chunk in stream:
            token = chunk.choices[0].delta.content or ""
            if not token:
                continue

            if not in_think and "<think>" in token:
                # Split on <think> — anything before it is preamble
                before, _, after = token.partition("<think>")
                if before.strip():
                    preamble += before
                in_think = True
                think_started = True
                # Stream any content after the opening tag
                if after:
                    after_clean = after.lstrip("\n")
                    think_buf += after_clean
                    if after_clean:
                        await step_ctx.stream_token(after_clean)
                continue

            if in_think and "</think>" in token:
                # End of thinking block
                before_end, _, after_end = token.partition("</think>")
                if before_end:
                    think_buf += before_end
                    await step_ctx.stream_token(before_end)
                in_think = False
                # Anything after </think> is the start of the answer
                answer_start = after_end.lstrip("\n")
                if answer_start:
                    answer_buf += answer_start
                continue

            if in_think:
                think_buf += token
                await step_ctx.stream_token(token)
            elif think_started:
                # Post-</think> answer content
                answer_buf += token
            else:
                # Pre-<think> or no think block at all
                preamble += token

        # Close the thinking step — it collapses in the UI
        await step_ctx.__aexit__(None, None, None)

        # Remaining tokens in answer_buf go to the answer bubble
        # (already collected above after </think>)
        return think_buf.strip(), answer_buf.strip()

    else:
        # No thinking mode — just collect everything for the answer
        async for chunk in stream:
            answer_buf += chunk.choices[0].delta.content or ""
        return "", answer_buf.strip()


# ── Tool helpers ──────────────────────────────────────────────────────────────
async def _get_tools() -> list[dict]:
    try:
        async with httpx.AsyncClient(timeout=5.0) as h:
            r = await h.get(f"{BACKEND_URL}/v1/tools")
            return r.json().get("tools", []) if r.status_code == 200 else []
    except Exception:
        return []


# run_tool() returns failure/stub STRINGS instead of raising. fetch_url's quarantine skip and
# empty-extraction stubs start with "[" -- they were once counted as successful pages (a live run
# showed a 0.0s "✓" for a quarantined domain), wasting one of the three source slots.
_TOOL_FAIL_PREFIXES = ("Tool error", "Search failed", "Search timed out",
                       "HTTP fetch failed", "Error", "Invalid URL", "403", "401",
                       "[Skipped", "[No article body")


def _tool_failed(result: str) -> bool:
    """run_tool() returns error STRINGS rather than raising — detect them."""
    return (not result) or result.startswith(_TOOL_FAIL_PREFIXES)


# -- Content-quality gates (P2-096) ------------------------------------------
# fetch_url can "succeed" with a cookie wall or a minified-JS shell (a live run counted a YouTube
# consent page and a TradingView script blob as two of its three sources), and the small Scout
# model can invent a "dossier" that replaces the real context. Both gates fail SAFE: when in
# doubt, skip the page / pass the raw fetched text to synthesis.
_JUNK_MARKERS = ("before you continue to", "we use cookies", "accept all cookies", "enable javascript",
                 "please enable cookies", "verify you are human", "checking your browser", "access denied",
                 # sign-up / marketing walls (a StocksToTrade quote page counted as a good source)
                 "enter a valid email address", "i agree to receive", "sign up for access",
                 "subscribe to continue", "sign in to continue", "create a free account")
_JS_HINTS = ("window.", "document.", "function(", "=>{", "var ", "const ", "initdata", "settimeout(")


# Domains that fetch_url has told THIS process are quarantined. The telemetry DB is the source of
# truth, but the tool runs in the model-server process and this process reads the DB separately; a
# live run (21 Sep) showed the DB pre-filter not taking effect, so the stub reply itself is also
# treated as a signal. Either way a quarantined domain is attempted at most once per process.
_QUARANTINE_SEEN: set = set()
_QUARANTINE_DIAG_DONE = False


def _domain_of(url: str) -> str:
    from urllib.parse import urlparse
    return urlparse(url).netloc.lower().removeprefix("www.")


def _is_quarantined(url: str) -> bool:
    """True if this domain is known to be quarantined (learned from a stub, or per domain_telemetry)."""
    global _QUARANTINE_DIAG_DONE
    if _domain_of(url) in _QUARANTINE_SEEN:
        return True
    try:
        import domain_telemetry as _dt
        state = _dt.get_domain_strategy(url)
        if not _QUARANTINE_DIAG_DONE:          # logged once per process (visible with --debug)
            _QUARANTINE_DIAG_DONE = True
            _db = getattr(_dt, "DB_PATH", None)
            log.info("quarantine lookup: cwd=%s db=%s exists=%s  %s -> %s", os.getcwd(),
                     _db.resolve() if _db else "?", _db.exists() if _db else "?", _domain_of(url), state)
        return state == "SKIP"
    except Exception as _e:
        if not _QUARANTINE_DIAG_DONE:
            _QUARANTINE_DIAG_DONE = True
            log.warning("quarantine lookup FAILED: %s: %s", type(_e).__name__, _e)
        return False


def _looks_like_junk_page(text: str) -> bool:
    """True for consent walls / script shells that carry no article text."""
    t = (text or "").lower()
    if not t.strip():
        return True
    if any(m in t for m in _JUNK_MARKERS):
        return True
    if sum(h in t for h in _JS_HINTS) >= 3:
        return True
    code_chars = sum(t.count(ch) for ch in "{}();=<>")
    return len(t) >= 200 and code_chars / len(t) > 0.05


_FIGURE_RE = re.compile(r"\$\d[\d,]*(?:\.\d+)?|\d[\d,]*\.\d+%?|\d+(?:\.\d+)?%")
_URL_RE = re.compile(r"https?://[^\s\]\)<>\"']+")


def _grounding_anchors(text: str) -> set[str]:
    """Checkable specifics in a text: price/percent figures and source domains."""
    from urllib.parse import urlparse
    anchors = set(_FIGURE_RE.findall(text))
    for u in _URL_RE.findall(text):
        d = urlparse(u).netloc.lower().removeprefix("www.")
        if d:
            anchors.add(d)
    return anchors


def _scout_dossier_usable(scout_raw: str, raw_context: str, min_anchors: int = 2) -> bool:
    """A Scout dossier may replace the raw context only if it is XML AND repeats at least
    min_anchors figures/domains that really occur in the fetched text. An ungrounded dossier
    (e.g. a 1.5B model 'defining' what SPY trading is) would otherwise become the ONLY thing the
    synthesis model sees."""
    if "<dossier>" not in scout_raw:
        return False
    raw_anchors = _grounding_anchors(raw_context)
    if len(raw_anchors) < min_anchors:
        return False                       # nothing to verify against -> do not trust it
    low = scout_raw.lower()
    return sum(1 for a in raw_anchors if a.lower() in low) >= min_anchors


def _call_detail(name: str, args: dict) -> str:
    """Short label for the section headline: the query, or the page's domain."""
    from urllib.parse import urlparse
    if not isinstance(args, dict):
        return ""
    if args.get("url"):
        return urlparse(str(args["url"])).netloc.removeprefix("www.") or str(args["url"])
    if args.get("query"):
        return str(args["query"])
    return next((str(v) for v in args.values() if isinstance(v, str)), "")


async def run_tool(name: str, args: dict) -> str:
    """Execute a tool call with up to 3 retries.

    No longer a @cl.step: the caller wraps it in ToolProgress.call(), which owns
    the (collapsed) UI row — a decorator here would nest a duplicate step inside it.
    """
    last_error = ""
    for attempt in range(3):
        try:
            async with httpx.AsyncClient(timeout=30.0) as h:
                r = await h.post(f"{BACKEND_URL}/v1/tools/call",
                                 json={"name": name, "arguments": args})
                if r.status_code == 200:
                    result = r.json().get("result", "(empty result)")
                    if result and not result.startswith("Tool error"):
                        return result
                    last_error = result
                else:
                    last_error = f"HTTP {r.status_code}"
        except Exception as e:
            last_error = str(e)
        if attempt < 2:
            await asyncio.sleep(1.0 * (attempt + 1))  # 1s, 2s backoff

    # Direct fallback after retries exhausted
    try:
        from tool_runner import dispatch_tool
        cfg = {"db_path": "DATA/paper_account.db", "web_search_results": 5}
        return await asyncio.to_thread(dispatch_tool, name, args, cfg) or "(no result)"
    except Exception as e:
        return f"Tool error after 3 attempts: {last_error} | fallback: {e}"


def _extract_top_urls(search_result: str, n: int = 3) -> list[str]:
    """Return up to n fetchable URLs — one per domain for diversity."""
    import re as _re
    from urllib.parse import urlparse as _up
    urls = _re.findall(r"https?://[^ \t\n\r<>]+", search_result)
    urls = [u.rstrip(".,)]") for u in urls]

    skip = {"wikipedia.org", "google.com", "google.co.uk", "yahoo.com", "bing.com",
            "bloomberg.com", "ft.com", "wsj.com", "nytimes.com",
            "economist.com", "thetimes.co.uk", "telegraph.co.uk",
            "reuters.com", "marketwatch.com", "barrons.com", "seekingalpha.com",
            "tipranks.com", "fool.com", "thestreet.com", "investopedia.com"}

    article_pats = [r"/\d{4}/\d{2}/\d{2}/", r"/article/", r"/story/",
                    r"/news/[^/]+/[^/]+", r"-\d{8}"]

    news_fallback = ["bbc.", "apnews.", "nbcnews.", "cbsnews.", "theguardian.",
                     "euronews.", "cnbc.", "cnn.", "independent.", "sky.com",
                     "thehill.", "axios.", "politico.", "npr.org"]

    def _is_skip(url):
        d = _up(url).netloc.removeprefix("www.")
        return any(s in d for s in skip)

    def _is_homepage(url):
        p = _up(url)
        path = p.path.rstrip("/")
        # Pure domain root with no path or query
        if path == "" and not p.query:
            return True
        # Generic section roots with no sub-path
        if path in ("/news", "/world", "/news/world", "/latest", "/markets"):
            return True
        # Yahoo Finance quote root (but /quote/SPY/news is NOT a homepage)
        if "/quote/" in path and path.endswith("/quote/" + path.split("/quote/")[-1].split("/")[0]):
            return True
        return False

    results: list = []

    domain_count: dict = {}

    def _try_add(url, max_per_domain=1):
        d = _up(url).netloc.removeprefix("www.")
        if _is_skip(url):
            return False
        if domain_count.get(d, 0) >= max_per_domain:
            return False
        domain_count[d] = domain_count.get(d, 0) + 1
        results.append(url)
        return len(results) >= n

    # Pass 1: article-pattern URLs, 1 per domain
    for url in urls:
        if not _is_homepage(url) and any(_re.search(p, url) for p in article_pats):
            if _try_add(url, max_per_domain=1): return results
    # Pass 2: any non-homepage URL, 1 per domain
    for url in urls:
        if not _is_homepage(url) and url not in results:
            if _try_add(url, max_per_domain=1): return results
    # Pass 3: known news domains even if homepage-like, 1 per domain
    for url in urls:
        d = _up(url).netloc.removeprefix("www.")
        if url not in results and any(nd in d for nd in news_fallback):
            if _try_add(url, max_per_domain=1): return results
    # Pass 4: relax to 2 per domain if still short
    for url in urls:
        if not _is_homepage(url) and url not in results:
            if _try_add(url, max_per_domain=2): return results
    # Pass 5: include homepage-like pages as last resort
    for url in urls:
        if url not in results and not _is_skip(url):
            if _try_add(url, max_per_domain=2): return results
    return results


# ── Chat Profiles ─────────────────────────────────────────────────────────────
@cl.set_chat_profiles
async def set_chat_profiles():
    return [
        cl.ChatProfile(
            name="Research",
            markdown_description="**Web Search enabled** — searches the web and fetches articles for grounded answers. Best for news, prices, and current market data.",
            icon="🔍",
            starters=[
                cl.Starter(label="📰 Top news today",
                           message="What are the top 3 market-moving news items today?"),
                cl.Starter(label="📈 SPY right now",
                           message="What is SPY trading at today and what are analysts saying?"),
                cl.Starter(label="📅 Earnings this week",
                           message="Which notable companies report earnings this week?"),
                cl.Starter(label="⚡ VIX check",
                           message="What is the current VIX level and what does it signal?"),
            ],
        ),
        cl.ChatProfile(
            name="D-A-C",
            markdown_description="**Deep Analysis** — thinking mode on, web search enabled. Slower but much deeper reasoning. Use for Divergence-Adversarial-Convergence sessions, strategy review, and complex analysis.",
            icon="🧠",
            starters=[
                cl.Starter(label="🧠 Start D-A-C Gate 0",
                           message="Run a D-A-C review. Start with Gate 0: explain the full methodology."),
                cl.Starter(label="⚔️ Stress-test my setup",
                           message="Adversarially stress-test my SPY 0DTE approach. Find the 3 most likely failure modes."),
                cl.Starter(label="📐 Kelly sizing review",
                           message="Review my position sizing using Kelly Criterion. What adjustments given recent drawdown?"),
                cl.Starter(label="📊 V-D-U-R analysis",
                           message="Has my edge degraded? Run a V-D-U-R analysis on my last 20 trades."),
            ],
        ),
        cl.ChatProfile(
            name="Quick Chat",
            markdown_description="**No tools, fast responses** — direct conversation without web search or extended reasoning. Best for quick questions and explanations.",
            icon="⚡",
            starters=[
                cl.Starter(label="📖 Explain VWAP",
                           message="Explain VWAP and how I should use it for intraday entries."),
                cl.Starter(label="🎯 What is EV?",
                           message="Explain Expected Value in the context of trading and why it matters."),
                cl.Starter(label="📏 ATR usage",
                           message="How do I use ATR for position sizing and stop placement?"),
                cl.Starter(label="🔢 Options Greeks",
                           message="Explain the key Greeks I need to understand for 0DTE trading."),
            ],
        ),
    ]


# Note: @cl.set_commands requires Chainlit v2.0.5+ — using @cl.on_message
# command detection instead for version compatibility.
# Quick-action buttons are sent as cl.Action on the welcome message.


@cl.on_chat_start
async def on_chat_start():
    profile = cl.context.session.chat_profile or "Research"

    # Profile-driven defaults
    use_tools   = profile != "Quick Chat"
    thinking    = profile == "D-A-C"
    temperature = 0.6 if profile == "D-A-C" else 0.2
    max_tokens  = 1024

    # Fetch which model is ACTUALLY running on each port before building the
    # Settings panel — this must happen BEFORE ChatSettings is sent, not after.
    # FIXED 19 Sep 2026 (P2-076): the panel used to hard-code
    # "8000 = Qwen3.8-27B (deep analysis) · 8001 = Phi-4-mini (fast)" as a
    # static string, regardless of which pair launch_models.sh actually
    # started. That's actively misleading once more than one pair exists
    # (Pair 4 is Qwen2.5-Coder-7B on port 8000, not Qwen3.8-27B) — confirmed
    # via a live screenshot showing the stale label while Pair 4 was running.
    # serve_model.py's /health endpoint already returns {"model": ...}; this
    # reuses that instead of a second hardcoded guess.
    p1_ok, p2_ok = False, False
    p1_model_id, p2_model_id = "unknown (port 8000)", "unknown (port 8001)"
    p1_url = BACKEND_URL
    p2_url = BACKEND_URL.replace(":8000", ":8001")
    try:
        async with httpx.AsyncClient(timeout=3.0) as h:
            r = await h.get(f"{p1_url}/health")
            p1_ok = True
            p1_model_id = r.json().get("model", p1_model_id)
    except Exception:
        pass
    try:
        async with httpx.AsyncClient(timeout=3.0) as h:
            r = await h.get(f"{p2_url}/health")
            p2_ok = True
            p2_model_id = r.json().get("model", p2_model_id)
    except Exception:
        pass

    cl.user_session.set("p1_model_id", p1_model_id)
    cl.user_session.set("p2_model_id", p2_model_id)

    from chainlit.input_widget import Select
    await cl.ChatSettings([
        Select(
            id="model_port",
            label="Model",
            values=["8000", "8001"],
            initial_index=0,
            description=f"8000 = {p1_model_id} · 8001 = {p2_model_id}"
        ),
        Switch(id="use_tools",     label="🔍 Web Search & Tools", initial=use_tools,
               description="Search the web and fetch article content"),
        Switch(id="thinking_mode", label="🧠 Thinking Mode (D-A-C)", initial=thinking,
               description="Chain-of-thought reasoning — slower, deeper. Required for D-A-C."),
        Slider(id="temperature",   label="Temperature", initial=temperature,
               min=0.0, max=1.0, step=0.05, description="0.0 = focused, 1.0 = creative"),
        Slider(id="max_tokens",    label="Max Tokens", initial=max_tokens,
               min=64, max=1024, step=64, description="Response length cap"),
    ]).send()

    cl.user_session.set("history", [])
    cl.user_session.set("use_tools",     use_tools)
    cl.user_session.set("thinking_mode", thinking)
    cl.user_session.set("temperature",   temperature)
    cl.user_session.set("max_tokens",    max_tokens)
    cl.user_session.set("profile",       profile)

    # Only send a status message when something needs attention.
    # When all is well, stay silent so starters appear above the input.
    if not p1_ok:
        await cl.Message(
            content="⏳ Primary model still loading — allow 3-4 minutes after launch.",
            author="System"
        ).send()
    elif not p2_ok:
        await cl.Message(
            content="⚠️ Secondary model (port 8001) offline — primary only.",
            author="System"
        ).send()
    # else: silent — starters render above input automatically


@cl.on_settings_update
async def on_settings_update(settings: dict):
    for k, v in settings.items():
        cl.user_session.set(k, v)


# Tracks the current active httpx client so on_stop can cancel it
_active_client: httpx.AsyncClient | None = None

@cl.on_stop
async def on_stop():
    """Cancel the active request when user clicks Stop or presses Escape."""
    global _active_client
    if _active_client and not _active_client.is_closed:
        await _active_client.aclose()
        _active_client = None


# ── Main message handler ──────────────────────────────────────────────────────
@cl.on_message
async def on_message(message: cl.Message):
    # ── Handle slash commands ────────────────────────────────────────────────
    cmd = message.content.strip().lstrip("/").lower()

    if cmd == "think":
        thinking = not cl.user_session.get("thinking_mode", False)
        cl.user_session.set("thinking_mode", thinking)
        state = "🟢 ON" if thinking else "🔴 OFF"
        await cl.Message(
            content=f"🧠 Thinking mode {state}",
            author="System"
        ).send()
        return

    if cmd == "clear":
        cl.user_session.set("history", [])
        await cl.Message(content="🗑️ Chat history cleared.", author="System").send()
        return

    if cmd == "dac":
        message = cl.Message(content=(
            "Run a Divergence-Adversarial-Convergence review of my current trading setup. "
            "Start with Gate 0: explain the full D-A-C methodology before we begin."
        ))

    elif cmd == "toolkit":
        message = cl.Message(content=(
            "Run the trading toolkit analysis: calculate current EV, Kelly position size, "
            "and Sharpe ratio from my recent session data."
        ))

    elif cmd == "search":
        cl.user_session.set("use_tools", True)
        await cl.Message(content="🔍 Web search enabled for this session.", author="System").send()

    elif cmd == "model":
        port = cl.user_session.get("model_port", "8000")
        new_port = "8001" if port == "8000" else "8000"
        cl.user_session.set("model_port", new_port)
        model = cl.user_session.get(
            "p2_model_id" if new_port == "8001" else "p1_model_id",
            f"unknown (port {new_port})"
        )
        await cl.Message(content=f"🔄 Switched to `{model}` on port `{new_port}`", author="System").send()
        return

    elif cmd == "help":
        _p1 = cl.user_session.get("p1_model_id", "port 8000")
        _p2 = cl.user_session.get("p2_model_id", "port 8001")
        await cl.Message(
            content=(
                "**Available commands:**\n"
                "- `/think` — toggle 🧠 thinking mode on/off\n"
                "- `/dac` — start a D-A-C review session\n"
                "- `/toolkit` — run trading toolkit analysis\n"
                "- `/search` — enable web search\n"
                f"- `/model` — switch between `{_p1}` and `{_p2}`\n"
                "- `/clear` — clear chat history\n"
                "- `/help` — show this message"
            ),
            author="System"
        ).send()
        return

    # ── Normal message processing ─────────────────────────────────────────────
    history:     list[dict] = cl.user_session.get("history", [])
    use_tools:   bool       = cl.user_session.get("use_tools", True)
    thinking:    bool       = cl.user_session.get("thinking_mode", False)
    temperature: float      = float(cl.user_session.get("temperature", 0.2))
    max_tokens:  int        = int(cl.user_session.get("max_tokens", 1024))

    history.append({"role": "user", "content": message.content})

    # Load system prompt from file if available, otherwise use default
    _prompt_file = _project_root / "system_prompt.md"
    if _prompt_file.exists():
        _prompt_text = _prompt_file.read_text().strip()
    else:
        _prompt_text = (
            "You are a trading research assistant with web search capability. "
            "RULES: "
            "1. For news/current events/prices: call web_search with a SHORT query (3-6 words) "
            "   that returns ARTICLE pages. Good: 'Reuters top stories today'. "
            "   Bad: the user's full question verbatim. "
            "2. Always include the 'query' argument when calling web_search. "
            "3. Never say you lack real-time access — search first. "
            "4. Cite sources in your answer."
        )
    # Apply model port from settings (allows switching between Qwen and Phi mid-session)
    model_port = cl.user_session.get("model_port", "8000")
    effective_url = BACKEND_URL.rsplit(":", 1)[0] + f":{model_port}"
    active_client = AsyncOpenAI(
        base_url=f"{effective_url}/v1",
        api_key="local",
        http_client=httpx.AsyncClient(timeout=_timeout),
    )

    SYSTEM = {"role": "system", "content": _prompt_text}
    messages_with_system = [SYSTEM] + history

    tools = await _get_tools() if use_tools else []
    full_text = ""

    # ── Phase 1: Tool intent detection ────────────────────────────────────────
    if use_tools and tools:
        try:
            # Auto-reconnect on connection failure (OrionBelt pattern)
            for _attempt in range(2):
                try:
                    resp = await _client.chat.completions.create(
                        model=MODEL_ID,
                        messages=messages_with_system,
                        tools=tools,
                        temperature=0.0,
                        max_tokens=512,  # raised: 128 truncated tool XML before <parameter=query>
                        stream=False,
                        extra_body={"thinking": False},
                    )
                    break
                except httpx.ConnectError:
                    if _attempt == 0:
                        # Brief pause then retry — handles serve_model restart mid-session
                        await asyncio.sleep(3.0)
                        continue
                    raise
        except Exception as e:
            # Status-code-specific hints (OrionBelt pattern)
            err_str = str(e)
            if "503" in err_str or "overload" in err_str.lower():
                hint = "The model server is busy — the GPU is likely processing another request. Wait ~30s and retry."
            elif "502" in err_str or "connect" in err_str.lower():
                hint = "Cannot reach the model server — it may still be loading (allow 3-4 minutes after start)."
            elif "timeout" in err_str.lower():
                hint = "Request timed out — the model may be generating a very long response. Try a shorter prompt."
            elif "lock" in err_str.lower():
                hint = "Model is locked by another request — wait a moment and retry."
            else:
                hint = "Check that serve_model.py is running on port 8000."
            err_msg = cl.Message(
                content=f"❌ Backend error: {e}\n\n{hint}",
                author="System",
                actions=[cl.Action(name="retry", payload={"value": "retry"}, label="🔄 Retry")]
            )
            await err_msg.send()
            history.pop()
            cl.user_session.set("history", history)
            return

        choice = resp.choices[0]

        if choice.finish_reason == "tool_calls" and choice.message.tool_calls:
            history.append(choice.message.model_dump(exclude_none=True))

            # Show pre-tool reasoning if present
            if thinking and choice.message.content:
                think_text = (choice.message.content
                              .replace("<think>", "").replace("</think>", "").strip())
                if think_text:
                    async with cl.Step(name="💭 Reasoning chain", type="run",
                                       show_input=False) as s:
                        s.output = fix_latex(think_text)

            # Execute tools inside ONE collapsed, self-labelling section (P2-090):
            # the title tracks the latest call while running, then becomes a
            # summary.  The section is CLOSED before synthesis so the thinking
            # step and answer bubble are not nested inside it.
            async with ToolProgress() as progress:
                for tc in choice.message.tool_calls:
                    fn_name = tc.function.name
                    try:
                        fn_args = json.loads(tc.function.arguments or "{}")
                        if isinstance(fn_args, dict) and "name" in fn_args and "arguments" in fn_args:
                            fn_args = fn_args["arguments"]
                    except Exception:
                        fn_args = {"query": str(tc.function.arguments)}

                    async with progress.call(fn_name, _call_detail(fn_name, fn_args),
                                             args=fn_args) as c:
                        raw_result = await run_tool(fn_name, fn_args)
                        c.result(raw_result)
                        if _tool_failed(raw_result):
                            c.fail()

                    history.append({
                        "role": "tool",
                        "content": raw_result[:1600],
                        "tool_call_id": tc.id,
                    })

                    # Auto fetch_url after web_search — fetch up to 3 sources in parallel
                    # for news queries to ensure mix of results even if some block/401.
                    if fn_name == "web_search" and not raw_result.startswith(("Search failed", "Search timed out")):
                        # Fetch up to 6 candidate URLs, retrying on failure to get 3 successes
                        # Ask for extra candidates, then drop quarantined domains up front so the slots
                        # go to sources that can actually be fetched (no more 0.0s "skipped" rows).
                        candidate_urls = [u for u in _extract_top_urls(raw_result, n=12)
                                          if not _is_quarantined(u)][:6]
                        combined_articles = []
                        for fetch_url_candidate in candidate_urls:
                            if len(combined_articles) >= 3:
                                break
                            if _is_quarantined(fetch_url_candidate):
                                continue        # learned earlier in this run: no row, no wasted call
                            try:
                                fetch_args = {"url": fetch_url_candidate, "max_chars": 1500}
                                async with progress.call("fetch_url",
                                                         _call_detail("fetch_url", fetch_args),
                                                         args=fetch_args) as c:
                                    page = await run_tool("fetch_url", fetch_args)
                                    c.result(page)
                                    if page.startswith("[Skipped: domain quarantined"):
                                        _QUARANTINE_SEEN.add(_domain_of(fetch_url_candidate))
                                    page_ok = not _tool_failed(page) and not _looks_like_junk_page(page)
                                    if not page_ok:
                                        c.fail()
                                        if not _tool_failed(page):
                                            c.note("(cookie wall / script shell -- not an article; skipped)")
                                if page_ok:
                                    combined_articles.append(f"[{fetch_url_candidate}]\n{page[:1500]}")
                            except Exception:
                                continue  # try next URL
                        if combined_articles:
                            history.append({
                                "role": "tool",
                                "content": "Fetched articles:\n\n" + "\n\n---\n\n".join(combined_articles),
                                "tool_call_id": f"auto_fetch_{tc.id}",
                            })

                # ── Scout: Phi-4-mini port 8001 (Research profile only) ──────────
                profile = cl.user_session.get("chat_profile", "Research")
                raw_context = "\n\n".join(
                    m["content"] for m in history
                    if m["role"] == "tool" and m.get("content")
                )
                dossier = raw_context  # default: pass raw context if Scout skipped
                scout_used = False

                if SCOUT_ENABLED and profile == "Research" and raw_context.strip():
                    # Port 8001 is always Phi-4-mini regardless of active_client port
                    from urllib.parse import urlparse as _up
                    _parsed = _up(BACKEND_URL)
                    scout_url = f"{_parsed.scheme}://{_parsed.hostname}:8001"
                    log.info("Scout: firing to %s (raw_context=%d chars)", scout_url, len(raw_context))
                    scout_payload = {
                        "model": "phi-4-mini:int4",
                        "messages": [
                            {
                                "role": "user",
                                "content": (
                                    f"Extract key facts from this search data and output ONLY valid XML.\n"
                                    f"Tickers such as SPY, QQQ or NVDA are stock/ETF symbols, not ordinary words. "
                                    f"State ONLY facts that appear in the Data, each with its [source domain]; "
                                    f"never define terms from memory.\n\n"
                                    f"Question: {cmd}\n\n"
                                    f"Data:\n{raw_context[:3000]}\n\n"
                                    "Output format (XML only, no other text):\n"
                                    "<dossier>\n"
                                    "  <key_facts>\n"
                                    "  - fact one [source domain]\n"
                                    "  - fact two [source domain]\n"
                                    "  </key_facts>\n"
                                    "  <gaps>any gaps or contradictions, or NONE</gaps>\n"
                                    "</dossier>"
                                )
                            }
                        ],
                        "max_tokens": 350,
                        "temperature": 0.1,
                        "stream": False,
                        "thinking": False,
                    }
                    try:
                        n_sources = len([m for m in history if m["role"] == "tool"])
                        async with progress.call(
                            "scout", cl.user_session.get("p2_model_id") or "Scout model",
                            args={"chars": len(raw_context), "sources": n_sources},
                        ) as sc:
                            log.info("Scout: sending POST to %s/v1/chat/completions", scout_url)
                            # Hard 30s timeout — Scout must not block synthesis
                            async with httpx.AsyncClient(
                                timeout=httpx.Timeout(30.0, connect=5.0)
                            ) as _sc:
                                scout_resp = await _sc.post(
                                    f"{scout_url}/v1/chat/completions",
                                    json=scout_payload
                                )
                            log.info("Scout: response status %s", scout_resp.status_code)
                            scout_data = scout_resp.json()
                            scout_raw = scout_data["choices"][0]["message"]["content"].strip()
                            # Strip Phi-4-mini EOS tokens from display
                            import re as _re
                            scout_raw = _re.sub(r"<\|?im_end\|?>", "", scout_raw).strip()
                            # Use Scout's dossier only if it is valid XML AND grounded in the fetched text
                            _has_xml = "<dossier>" in scout_raw
                            _usable = _scout_dossier_usable(scout_raw, raw_context)
                            if _usable:
                                dossier = scout_raw
                                scout_used = True
                                log.info("Scout: grounded XML dossier (%d chars)", len(dossier))
                            else:
                                dossier = raw_context
                                log.warning("Scout output %s -- using raw context",
                                            "is not grounded in the fetched pages" if _has_xml
                                            else "was prose, not XML")
                            # Always leave visible text in the row (an empty output renders blank)
                            sc.result(scout_raw or "(Scout returned an empty response)")
                            if not scout_raw:
                                sc.fail()
                            elif not _has_xml:
                                sc.note("-> not a valid XML dossier; raw context passed to synthesis")
                            elif not _usable:
                                sc.note("-> dossier not grounded in the fetched pages (no matching figures or "
                                        "sources -- possible hallucination); raw context passed to synthesis")
                    except Exception as _scout_err:
                        dossier = raw_context  # fall back to raw context on Scout failure
                        log.warning("Scout (port 8001) FAILED: %s — %s", type(_scout_err).__name__, _scout_err)

            # ── Phase 2: Synthesis (Qwen3.8-27B port 8000) ───────────────────
            _ctx_label = "pre-filtered by Scout" if scout_used else "retrieved from the web"
            synthesis_messages = [{
                "role": "system",
                "content": (
                    "You have retrieved real-time information from the web. "
                    "Use ONLY this retrieved information — do NOT fall back to training data. "
                    "Summarise clearly and cite source URLs."
                ),
            }] + [m for m in history if m["role"] not in ("tool",)] + [{
                "role": "user",
                "content": (
                    f"Research intelligence ({_ctx_label}):\n{dossier}\n\n"
                    f"Answer the question: {cmd}"
                )
            }]

            stream = await active_client.chat.completions.create(
                model=MODEL_ID,
                messages=synthesis_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                extra_body={"thinking": thinking},
            )
            # Live stream thinking into collapsible Step, then stream answer
            think_text, full_text = await _stream_thinking_then_answer(
                stream, thinking, thinking_step_name="💭 Reasoning"
            )
            full_text = fix_latex(full_text)

            # Answer bubble — opens AFTER thinking Step has closed
            msg = cl.Message(content=full_text)
            await msg.send()

        elif choice.message.content:
            # Model chose not to use tools
            full_text = fix_latex(choice.message.content)
            msg = cl.Message(content=full_text)
            await msg.send()
        else:
            # Fall through to direct streaming
            use_tools = False

    # ── Phase 3: Direct streaming (no tools) ──────────────────────────────────
    if not use_tools or not tools:
        stream = await active_client.chat.completions.create(
            model=MODEL_ID,
            messages=messages_with_system,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            extra_body={"thinking": thinking},
        )
        # Live stream thinking into collapsible Step, then stream answer
        think_text, full_text = await _stream_thinking_then_answer(
            stream, thinking, thinking_step_name="💭 Thinking"
        )
        full_text = fix_latex(full_text)

        msg = cl.Message(content=full_text)
        await msg.send()

    if full_text:
        history.append({"role": "assistant", "content": full_text})
        cl.user_session.set("history", history)

        # Persist to SQLite incrementally (last user + assistant pair)
        thread_id = cl.context.session.id
        profile = cl.user_session.get("profile", "Research")
        if len(history) >= 2:
            _db_save_message(thread_id, history[-2]["role"],
                             history[-2]["content"], profile=profile)
        _db_save_message(thread_id, "assistant", full_text, profile=profile)

        # Actions replaced by /commands (more discoverable, ChatGPT-style)
        # /commands handled in on_message command dispatcher above


# ── Action callbacks ──────────────────────────────────────────────────────────
@cl.action_callback("think")
async def on_think_action(action):
    thinking = not cl.user_session.get("thinking_mode", False)
    cl.user_session.set("thinking_mode", thinking)
    state = "🟢 ON" if thinking else "🔴 OFF"
    await cl.Message(content=f"🧠 Thinking mode {state}", author="System").send()


@cl.action_callback("help")
async def on_help(action):
    _p1 = cl.user_session.get("p1_model_id", "port 8000")
    _p2 = cl.user_session.get("p2_model_id", "port 8001")
    await cl.Message(
        content=(
            "**Available commands:**\n"
            "- `/think` — toggle 🧠 thinking mode on/off\n"
            "- `/dac` — start a D-A-C review session\n"
            "- `/toolkit` — run trading toolkit analysis\n"
            "- `/search` — enable web search\n"
            f"- `/model` — switch between `{_p1}` and `{_p2}`\n"
            "- `/clear` — clear chat history\n"
            "- `/help` — show this message"
        ),
        author="System"
    ).send()


@cl.action_callback("retry")
async def on_retry(action):
    history = cl.user_session.get("history", [])
    if len(history) >= 2:
        last_user = next((m["content"] for m in reversed(history)
                          if m["role"] == "user"), None)
        if last_user:
            cl.user_session.set("history", history[:-2])
            await on_message(cl.Message(content=last_user))


@cl.action_callback("clear")
async def on_clear(action):
    cl.user_session.set("history", [])
    await cl.Message(content="🗑️ Chat history cleared.", author="System").send()


@cl.action_callback("dac")
async def on_dac(action):
    await on_message(cl.Message(
        content="Run a Divergence-Adversarial-Convergence review of my current setup. Start with Gate 0."
    ))


@cl.action_callback("toolkit")
async def on_toolkit(action):
    await on_message(cl.Message(
        content="Run the trading toolkit analysis: calculate current EV, Kelly position size, and Sharpe ratio from my recent session data."
    ))
