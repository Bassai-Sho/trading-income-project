#!/usr/bin/env python3
"""
chainlit_app.py — Trading Income Project
=========================================
Chainlit UI with:
- ChatProfiles: D-A-C (thinking), Research (web search), Quick Chat
- @cl.step decorator for reliable tool Step ordering
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


@cl.step(type="tool")
async def run_tool(name: str, args: dict) -> str:
    """Execute a tool call with up to 3 retries — @cl.step guarantees render order."""
    # Update step name to show which tool is being called
    if cl.context.current_step:
        cl.context.current_step.name = name
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
        d = _up(url).netloc.lstrip("www.")
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

    seen: set = set()
    results: list = []

    domain_count: dict = {}

    def _try_add(url, max_per_domain=1):
        d = _up(url).netloc.lstrip("www.")
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
        d = _up(url).netloc.lstrip("www.")
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


def _extract_best_url(search_result: str) -> str | None:
    """Pick the best article URL from search results, skipping paywalls and homepages."""
    urls = re.findall(r'https?://[^\s\])\'"…]+', search_result)
    if not urls:
        return None

    skip = {'wikipedia.org', 'google.com', 'google.co.uk', 'yahoo.com', 'bing.com',
            'bloomberg.com', 'ft.com', 'wsj.com', 'nytimes.com',
            'economist.com', 'thetimes.co.uk', 'telegraph.co.uk',
            'reuters.com',   # 401 Forbidden on direct fetch
            'marketwatch.com', 'barrons.com', 'seekingalpha.com'}  # paywall/bot-block

    article_pats = [r'/\d{4}/\d{2}/\d{2}/', r'/article/', r'/story/',
                    r'/news/[^/]+/[^/]+', r'/world/[^/]+', r'/business/[^/]+',
                    r'/markets/[^/]+', r'-\d{8}']

    def _skip(url):
        from urllib.parse import urlparse
        d = urlparse(url).netloc.lstrip('www.')
        return any(s in d for s in skip)

    def _homepage(url):
        from urllib.parse import urlparse
        p = urlparse(url)
        path = p.path.rstrip('/')
        return (path == '' and not p.query) or \
               (path in ('/news', '/world', '/news/world', '/latest') and not p.query)

    news_domains = ['bbc.co.uk', 'bbc.com', 'reuters.com', 'apnews.com',
                    'theguardian.com', 'cnbc.com', 'cnn.com', 'euronews.com',
                    'marketwatch.com', 'seekingalpha.com', 'nbcnews.com', 'cbsnews.com']
    news_fallback = ['bbc.', 'reuters.', 'apnews.', 'nbcnews.', 'cbsnews.',
                     'theguardian.', 'euronews.', 'sky.com', 'independent.']

    # P1: article-like URL
    for url in urls:
        if _skip(url) or _homepage(url): continue
        if any(re.search(p, url) for p in article_pats): return url
    # P2: news domain non-homepage
    for url in urls:
        from urllib.parse import urlparse
        d = urlparse(url).netloc.lstrip('www.')
        if _skip(url) or _homepage(url): continue
        if any(nd in d for nd in news_domains): return url
    # P3: any non-skip non-homepage
    for url in urls:
        if _skip(url) or _homepage(url): continue
        return url
    # P4: fallback — any news domain even if homepage
    for url in urls:
        from urllib.parse import urlparse
        d = urlparse(url).netloc.lstrip('www.')
        if _skip(url): continue
        if any(nd in d for nd in news_fallback): return url
    return None


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

    from chainlit.input_widget import Select, TextInput
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

            # Execute tools using @cl.step decorator for reliable ordering
            for tc in choice.message.tool_calls:
                fn_name = tc.function.name
                try:
                    fn_args = json.loads(tc.function.arguments or "{}")
                    if isinstance(fn_args, dict) and "name" in fn_args and "arguments" in fn_args:
                        fn_args = fn_args["arguments"]
                except Exception:
                    fn_args = {"query": str(tc.function.arguments)}

                raw_result = await run_tool(fn_name, fn_args)

                history.append({
                    "role": "tool",
                    "content": raw_result[:1600],
                    "tool_call_id": tc.id,
                })

                # Auto fetch_url after web_search — fetch up to 3 sources in parallel
                # for news queries to ensure mix of results even if some block/401.
                if fn_name == "web_search" and not raw_result.startswith(("Search failed", "Search timed out")):
                    # Fetch up to 6 candidate URLs, retrying on failure to get 3 successes
                    candidate_urls = _extract_top_urls(raw_result, n=6)
                    combined_articles = []
                    for fetch_url_candidate in candidate_urls:
                        if len(combined_articles) >= 3:
                            break
                        try:
                            page = await run_tool("fetch_url", {"url": fetch_url_candidate, "max_chars": 1500})
                            if page and not page.startswith(("HTTP fetch failed", "Error", "Invalid URL", "403", "401")):
                                combined_articles.append(f"[{fetch_url_candidate}]\n{page[:1500]}")
                        except Exception:
                            continue  # try next URL
                    if combined_articles:
                        history.append({
                            "role": "tool",
                            "content": "Fetched articles:\n\n" + "\n\n---\n\n".join(combined_articles),
                            "tool_call_id": f"auto_fetch_{tc.id}",
                        })

            # ── Phase 2: Synthesis ────────────────────────────────────────────
            synthesis_messages = [{
                "role": "system",
                "content": (
                    "You have retrieved real-time information from the web. "
                    "Use ONLY this retrieved information — do NOT fall back to training data. "
                    "Summarise clearly and cite source URLs."
                ),
            }] + history

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
