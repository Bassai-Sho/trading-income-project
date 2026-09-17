#!/usr/bin/env python3
"""
chainlit_app.py
===============
Chainlit UI for the Trading Income Project OpenVINO GenAI backend.

Features:
- Async SSE streaming via openai AsyncClient with 600s timeout
- Tool calling: Brave/DDG web search, URL fetch, toolkit functions
- Visual cl.Step accordions for tool execution
- Per-session state isolation via cl.user_session
- cl.ChatSettings for runtime parameter control
- Conversation starters for common trading queries
- try/finally streaming guards (no orphaned cursors)
- asyncio.to_thread() for any sync tool_runner calls

Architecture:
    Browser ←── WebSocket ──→ chainlit_app.py (:8080)
                                    ↓ async httpx (600s timeout)
                            serve_model.py (:8000)
                            OpenVINO VLMPipeline (Qwen3.8-27B-int4)
                                    ↓
                            tool_runner.py (web search, fetch_url)
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import httpx
import chainlit as cl
from chainlit.input_widget import Slider, Switch
from openai import AsyncOpenAI

# ── Backend config ─────────────────────────────────────────────────────────────
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
MODEL_ID    = os.getenv("MODEL_ID", "qwen3.8:27b")

# 600s total timeout — at 2.7 tok/s a 1024-token response takes ~380 seconds
_timeout = httpx.Timeout(600.0, connect=10.0, read=600.0, write=30.0)
_http    = httpx.AsyncClient(timeout=_timeout)
_client  = AsyncOpenAI(
    base_url=f"{BACKEND_URL}/v1",
    api_key="local",            # serve_model.py doesn't check keys
    http_client=_http,
)

# ── Tool runner path setup ─────────────────────────────────────────────────────
# Add src/ to path so tool_runner.py is importable when running from project root
_src = str(Path(__file__).parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

# Load .env so tool_runner gets BRAVE_SEARCH_API_KEY etc.
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env", override=False)
except ImportError:
    pass


# ── Helpers ────────────────────────────────────────────────────────────────────

async def _get_tools() -> list[dict]:
    """Fetch available tool schemas from serve_model.py /v1/tools endpoint."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as h:
            r = await h.get(f"{BACKEND_URL}/v1/tools")
            if r.status_code == 200:
                return r.json().get("tools", [])
    except Exception:
        pass
    return []


async def _call_tool(name: str, args: dict) -> str:
    """
    Execute a tool via serve_model.py /v1/tools/call.
    Falls back to direct tool_runner.dispatch_tool if the endpoint is unavailable.
    Uses asyncio.to_thread for any sync dispatch to avoid blocking the event loop.
    """
    try:
        async with httpx.AsyncClient(timeout=30.0) as h:
            r = await h.post(
                f"{BACKEND_URL}/v1/tools/call",
                json={"name": name, "arguments": args},
            )
            if r.status_code == 200:
                return r.json().get("result", "(empty result)")
    except Exception:
        pass

    # Direct fallback: call tool_runner.dispatch_tool in a thread
    try:
        from tool_runner import dispatch_tool
        cfg = {"db_path": "DATA/paper_account.db", "web_search_results": 5}
        result = await asyncio.to_thread(dispatch_tool, name, args, cfg)
        return result or "(no result)"
    except ImportError:
        return f"Tool '{name}' unavailable — tool_runner.py not found"
    except Exception as e:
        return f"Tool error: {e}"


# ── Chainlit hooks ─────────────────────────────────────────────────────────────

@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="Market overview",
            message="What are the key market headlines and major index movements today?",
        ),
        cl.Starter(
            label="SPY analysis",
            message="What is SPY trading at today and what are analysts saying about near-term direction?",
        ),
        cl.Starter(
            label="Earnings this week",
            message="Which notable companies report earnings this week and what are expectations?",
        ),
        cl.Starter(
            label="D-A-C session",
            message="Run a Divergence-Adversarial-Convergence review of my current trading setup. Start with Gate 0 instructions.",
        ),
    ]


@cl.on_chat_start
async def on_chat_start():
    """Initialise per-session state and UI controls."""
    await cl.ChatSettings([
        Switch(
            id="use_tools",
            label="Web Search & Tools",
            initial=True,
            description="Enable Brave/DDG web search and URL fetching",
        ),
        Slider(
            id="temperature",
            label="Temperature",
            initial=0.2,
            min=0.0,
            max=1.0,
            step=0.05,
            description="0.0 = focused, 1.0 = creative",
        ),
        Slider(
            id="max_tokens",
            label="Max Tokens",
            initial=512,
            min=64,
            max=1024,
            step=64,
            description="1024 token cap — ~380s at 2.7 tok/s on Arc iGPU",
        ),
    ]).send()

    cl.user_session.set("history", [])
    cl.user_session.set("use_tools", True)
    cl.user_session.set("temperature", 0.2)
    cl.user_session.set("max_tokens", 512)

    # Check backend health
    try:
        async with httpx.AsyncClient(timeout=3.0) as h:
            r = await h.get(f"{BACKEND_URL}/health")
            if r.status_code == 200:
                d = r.json()
                await cl.Message(
                    content=f"✅ Connected to `{d.get('model', 'unknown')}` on `{BACKEND_URL}`",
                    author="System",
                ).send()
            else:
                await cl.Message(
                    content=f"⚠️ Backend returned HTTP {r.status_code}",
                    author="System",
                ).send()
    except Exception as e:
        await cl.Message(
            content=f"❌ Cannot reach backend at `{BACKEND_URL}`: {e}",
            author="System",
        ).send()


@cl.on_settings_update
async def on_settings_update(settings: dict):
    for key, val in settings.items():
        cl.user_session.set(key, val)


@cl.on_stop
async def on_stop():
    """
    Called when the user clicks the Stop button during generation.
    The httpx client will raise a CancelledError which our try/finally catches.
    This hook exists so we can add future cleanup (e.g. signal _infer_lock release).
    """
    pass


@cl.on_message
async def on_message(message: cl.Message):
    history:     list[dict] = cl.user_session.get("history", [])
    use_tools:   bool       = cl.user_session.get("use_tools", True)
    temperature: float      = float(cl.user_session.get("temperature", 0.2))
    max_tokens:  int        = int(cl.user_session.get("max_tokens", 512))

    history.append({"role": "user", "content": message.content})
    tools = await _get_tools() if use_tools else []

    # ── Phase 1: Tool intent detection (non-streaming) ─────────────────────────
    if use_tools and tools:
        try:
            resp = await _client.chat.completions.create(
                model=MODEL_ID,
                messages=history,
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
            )
        except Exception as e:
            await cl.Message(content=f"❌ Backend error: {e}", author="System").send()
            history.pop()   # remove the user message on error
            cl.user_session.set("history", history)
            return

        choice = resp.choices[0]

        if choice.finish_reason == "tool_calls" and choice.message.tool_calls:
            # Save assistant's tool-calling intent
            history.append(choice.message.model_dump(exclude_none=True))

            for tc in choice.message.tool_calls:
                fn_name = tc.function.name
                try:
                    fn_args = json.loads(tc.function.arguments or "{}")
                except Exception:
                    fn_args = {"query": str(tc.function.arguments)}

                # Visual step card — initialized immediately to keep WebSocket alive
                async with cl.Step(name=fn_name, type="tool", show_input=True) as step:
                    step.input = json.dumps(fn_args, indent=2)
                    raw_result = await _call_tool(fn_name, fn_args)
                    # Truncate display only — full result goes to model
                    step.output = (
                        raw_result[:1200] +
                        ("\n…(truncated for display)" if len(raw_result) > 1200 else "")
                    )

                # Bound context fed to model to avoid saturating iGPU KV cache
                bounded = raw_result[:1600] if len(raw_result) > 1600 else raw_result
                history.append({
                    "role":         "tool",
                    "content":      bounded,
                    "tool_call_id": tc.id,
                })

            # ── Phase 2: Streaming synthesis after tool results ───────────────
            msg = cl.Message(content="")
            await msg.send()
            full_text = ""

            try:
                stream = await _client.chat.completions.create(
                    model=MODEL_ID,
                    messages=history,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=True,
                )
                async for chunk in stream:
                    token = chunk.choices[0].delta.content or ""
                    if token:
                        full_text += token
                        await msg.stream_token(token)
            finally:
                await msg.update()

            history.append({"role": "assistant", "content": full_text})
            cl.user_session.set("history", history)
            return

        # Model chose not to use tools — use its text response directly
        if choice.message.content:
            text = choice.message.content
            history.append({"role": "assistant", "content": text})
            await cl.Message(content=text).send()
            cl.user_session.set("history", history)
            return

    # ── Phase 3: Direct streaming (no tools, or tools disabled) ───────────────
    msg = cl.Message(content="")
    await msg.send()
    full_text = ""

    try:
        stream = await _client.chat.completions.create(
            model=MODEL_ID,
            messages=history,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        async for chunk in stream:
            token = chunk.choices[0].delta.content or ""
            if token:
                full_text += token
                await msg.stream_token(token)
    except asyncio.CancelledError:
        pass   # user clicked Stop — finalize whatever we have
    except Exception as e:
        full_text += f"\n\n❌ Error: {e}"
    finally:
        await msg.update()

    if full_text:
        history.append({"role": "assistant", "content": full_text})
    cl.user_session.set("history", history)
