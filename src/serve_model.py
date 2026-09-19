#!/usr/bin/env python3
"""
serve_model.py
==============
OpenAI /v1/chat/completions server wrapping OpenVINO GenAI.
Supports both LLMPipeline (text) and VLMPipeline (VLM text-only) on Intel Arc GPU.

FIXES (P2-059) — 2026-09-15
----------------------------
1. SYCL_DEVICE_FILTER=gpu  — set before any OpenVINO import; blocks silent CPU
   fallback at driver level. If GPU OOM the process crashes rather than silently
   degrading, making the failure visible immediately.

2. Streaming fix  — streamer_cb previously built chunk_payload but never yielded
   it. A SimpleQueue fix was attempted but deadlocked: FastAPI's StreamingResponse
   runs a sync generator in a threadpool executor, and queue.get() blocked the
   uvicorn event loop. Final fix: async generator + asyncio.Queue +
   loop.call_soon_threadsafe(). The background inference thread pushes tokens
   into the asyncio queue thread-safely; the async generator awaits them without
   blocking the event loop. Route handler is async. OpenWebUI now receives tokens
   in real time.

3. Keepalive thread  — Intel Arc iGPU reclaims shared memory pages when the GPU
   goes idle. A 1-token generation every 55s keeps the model pinned in iGPU
   memory between requests, preventing the eviction that caused cold-reload
   latency on subsequent requests.
"""

from __future__ import annotations

# ── P2-059 Fix 1: SYCL guard — must be set before openvino_genai import ──────
import os
os.environ.setdefault("SYCL_DEVICE_FILTER", "gpu")      # block CPU fallback
os.environ.setdefault("SYCL_CACHE_PERSISTENT", "1")     # persist compiled cache
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import asyncio
import json
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(override=False)   # load .env so tool_runner gets BRAVE_SEARCH_API_KEY etc.
except ImportError:
    pass

try:
    import openvino_genai as ov_genai
except ImportError as e:
    raise SystemExit("openvino-genai not installed.") from e

app         = FastAPI(title="OpenVINO GenAI — OpenAI Bridge")

# Allow browser fetch from any local origin (file://, localhost:*, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
_pipe:      ov_genai.LLMPipeline | ov_genai.VLMPipeline | None = None
_tok:       Any   = None
_model_id:  str   = "ov-model"
_is_vlm:    bool  = False
_cache_dir: str   = os.path.expanduser("~/models/.ov_cache")
# ADDED (P2-068, 2026-09-17): needed by the new /admin/reload_pipeline endpoint
# below, so pipeline reconstruction can happen outside main() too.
_model_dir: str | None = None
_device:    str  = "GPU"
_force_llm: bool = False
_infer_lock = threading.Lock()   # ensures one inference at a time; concurrent
# FIX (P2-068, 2026-09-17): timeout for lock acquisition, so a stuck worker
# thread can no longer wedge the server permanently. If a generate() call is
# genuinely still running, a new request should wait for it — this is set
# generously high (280s, just under the streaming path's own 300s queue
# timeout) so a normal, slow generation is never falsely rejected. But if the
# lock is held by a thread that will NEVER release it (the actual failure
# mode this bug is chasing), every request now fails clearly and fast-ish
# instead of hanging silently forever. Tune _INFER_LOCK_TIMEOUT_S if normal
# generations routinely take longer than this.
_INFER_LOCK_TIMEOUT_S = 280
                                  # requests block here until the lock is free


# ===========================================================================
# Request / Response models (unchanged)
# ===========================================================================

class Message(BaseModel):
    role:    str
    content: Any = ""

class ToolFunction(BaseModel):
    name:        str
    description: str | None = None
    parameters:  dict | None = None

class Tool(BaseModel):
    type:     str = "function"
    function: ToolFunction

class ResponseFormat(BaseModel):
    type: str = "text"
    json_schema: dict | None = None

class ChatRequest(BaseModel):
    model:           str | None = None
    messages:        list[Message]
    tools:           list[Tool] | None = None
    tool_choice:     str | dict | None = None
    max_tokens:      int   = Field(default=1024, ge=1, le=16384)
    temperature:     float = Field(default=0.2,  ge=0.0, le=2.0)
    stream:          bool  = False
    response_format: ResponseFormat | None = None
    thinking:        bool  = False   # True = enable Qwen3 chain-of-thought
    thinking:        bool  = False   # True = enable Qwen3 chain-of-thought reasoning

TOOL_SYSTEM_PREFIX = """You are a trading research assistant with access to the following tools:
{tools_json}

CRITICAL RULES FOR TOOL USE:
- For ANY question about current events, news, prices, market conditions, or recent data: call web_search FIRST, then answer using the results. Never answer these from memory.
- For questions requiring a specific URL or article: call fetch_url after web_search.
- For trading statistics or calculations: call run_toolkit.
- After receiving tool results, synthesize them into a clear, grounded answer.
- If no tool is needed (e.g. explaining a concept you know well), answer directly.
- Do NOT say "I don't have access to real-time information" — use web_search instead.
"""


# ===========================================================================
# Helpers (unchanged)
# ===========================================================================

def _extract_text_content(content: Any) -> str:
    if not content:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [str(p.get("text", "")) for p in content if isinstance(p, dict) and "text" in p]
        return "".join(parts)
    return str(content)

def _build_prompt(req: ChatRequest) -> str:
    """Build the prompt string for generate().

    When req.thinking=True (D-A-C mode): Qwen3 chain-of-thought is enabled.
    The model reasons through the problem before responding. Slower but
    produces deeper adversarial analysis and convergence scoring.

    When req.thinking=False (default): /no_think suppresses the reasoning
    chain for faster, more direct responses (tool calls, web search, Q&A).

    For VLMPipeline: we build the prompt via _tok.apply_chat_template() and
    set apply_chat_template=False in GenerationConfig so VLMPipeline uses
    our pre-built string rather than re-applying the template internally.
    """
    messages = [{"role": m.role, "content": _extract_text_content(m.content)} for m in req.messages]

    # Non-thinking mode: inject /no_think into system message.
    # Skipped when thinking=True so the model reasons freely.
    if not req.thinking:
        if messages and messages[0]["role"] == "system":
            if "/no_think" not in messages[0]["content"]:
                messages[0]["content"] = messages[0]["content"].rstrip() + "\n/no_think"
        else:
            messages.insert(0, {"role": "system", "content": "/no_think"})

    if req.tools:
        tools_json = json.dumps(
            [{"name": t.function.name, "description": t.function.description or "",
              "parameters": t.function.parameters or {}}
             for t in req.tools], indent=2
        )
        tool_block = TOOL_SYSTEM_PREFIX.replace("{tools_json}", tools_json)
        if messages and messages[0]["role"] == "system":
            messages[0]["content"] = tool_block + "\n\n" + messages[0]["content"]
        else:
            messages.insert(0, {"role": "system", "content": tool_block})

    if _tok is not None:
        try:
            prompt = str(_tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=req.thinking,   # honour the thinking flag
            ))
            # Safety net: if non-thinking but template left open think block, close it
            if not req.thinking and prompt.endswith("<think>\n"):
                prompt += "\n</think>\n\n"
            return prompt
        except TypeError:
            try:
                prompt = str(_tok.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                ))
                return prompt
            except Exception:
                pass
        except Exception:
            pass

    # ChatML fallback
    lines = [f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>" for m in messages]
    if req.thinking:
        lines.append("<|im_start|>assistant\n<think>\n")   # open — model reasons
    else:
        lines.append("<|im_start|>assistant\n<think>\n\n</think>\n\n")  # closed
    return "\n".join(lines)

def _count_tokens(text: str) -> int:
    try:
        if _tok is not None:
            return len(_tok.encode(text).input_ids)
    except Exception:
        pass
    return max(1, len(text.split()))

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)

def _strip_thinking(text: str) -> tuple[str, str]:
    thoughts = "\n\n".join(_THINK_RE.findall(text))
    answer   = _THINK_RE.sub("", text).strip()
    return thoughts, answer

def _parse_tool_call(text: str, model_id: str | None = None) -> dict | None:
    """
    Parse a tool call from raw model output into canonical form.

    Uses a model capability registry (forge-guardrails / Arcanum pattern):
    each model family declares which formats it produces, and we try them
    in order, stopping at the first successful parse.

    Formats handled:
      xml_fn   — <tool_call><function=web_search><parameter=query>text
      xml_tag  — <tool_call> <web_search> text <parameter=n_results> 5
      xml_json — <tool_call>{"query":"text","n_results":5}</tool_call>
      json_bare— {"name":"web_search","arguments":{"query":"text"}}
      fenced   — ```json {"name":...} ```

    Auto-close (vLLM qwen3_xml pattern): parsers tolerate truncated output.
    """
    stripped = text.strip()
    if not stripped:
        return None

    # Strip preamble before the tool call marker
    for marker in ("<tool_call>", "<function="):
        if marker in stripped:
            idx = stripped.find(marker)
            if idx > 0:
                stripped = stripped[idx:]
            break

    # Check for direct tool-tag preamble
    for tool in _KNOWN_TOOLS:
        tag = f"<{tool}>"
        if tag in stripped and not stripped.startswith("<tool_call>"):
            idx = stripped.find(tag)
            if idx > 0:
                stripped = stripped[idx:]
            break

    # Per-model format order from capability registry
    fmts = _model_formats(model_id)
    for fmt in fmts:
        result = _try_parse_format(stripped, fmt)
        if result:
            return result
    return None


# ── Model capability registry ─────────────────────────────────────────────────
# Declares tool-call format order per model family.
# Approach from Arcanum / forge-guardrails: normalise at the serving layer.
_MODEL_TOOL_FORMATS: dict[str, list[str]] = {
    "qwen3":   ["xml_tag", "xml_fn", "xml_json", "json_bare"],
    "phi":     ["json_bare", "xml_fn"],
    "phi-4":   ["json_bare", "xml_fn"],
    "mistral": ["json_bare", "fenced"],
    "default": ["xml_fn", "xml_tag", "xml_json", "json_bare", "fenced"],
}

_KNOWN_TOOLS = [
    "web_search", "fetch_url", "run_toolkit", "run_script",
    "get_session_data", "get_wfa_results", "get_rolling_stats", "get_journal_stats",
]

# Tool-name → distinctive required argument (for inferring name from bare-arg JSON)
_TOOL_SIGNATURE: dict[str, str] = {
    "web_search":        "query",
    "fetch_url":         "url",
    "run_toolkit":       "function_name",
    "run_script":        "script_name",
    "get_session_data":  "session_date",
    "get_wfa_results":   "pair",
    "get_rolling_stats": "n_trades",
    "get_journal_stats": "stat_type",
}


def _model_formats(model_id: str | None) -> list[str]:
    mid = (model_id or "").lower()
    for key, fmts in _MODEL_TOOL_FORMATS.items():
        if key in mid:
            return fmts
    return _MODEL_TOOL_FORMATS["default"]


def _coerce(val: str):
    if val.lower() in ("true", "false"):
        return val.lower() == "true"
    try:
        return int(val)
    except ValueError:
        pass
    try:
        return float(val)
    except ValueError:
        pass
    return val


def _infer_tool_name(args: dict) -> str:
    for name, sig_key in _TOOL_SIGNATURE.items():
        if sig_key in args:
            return name
    return "web_search"


def _try_parse_format(text: str, fmt: str) -> dict | None:
    """Try one specific format parser. Returns canonical dict or None."""

    if fmt == "xml_fn":
        fn = re.search(r"<function=(\w+)", text)
        if not fn:
            return None
        name = fn.group(1)
        params: dict = {}
        for pm in re.finditer(
            r"<parameter=(\w+)>\s*(.*?)\s*(?:</parameter>|<parameter|</function|$)",
            text, re.DOTALL
        ):
            key = pm.group(1)
            val = re.sub(r"<.*", "", pm.group(2)).strip()
            if val:
                params[key] = _coerce(val)
        # Return even with empty params — model named the tool at least
        return {"name": name, "arguments": params}

    if fmt == "xml_tag":
        for tool in _KNOWN_TOOLS:
            tag = f"<{tool}>"
            if tag not in text:
                continue
            params: dict = {}
            # Text immediately after tag = query
            m = re.search(rf"<{tool}>\s*(.*?)(?:<parameter|</|$)", text, re.DOTALL)
            if m:
                q = m.group(1).strip()
                if q and not q.startswith("<"):
                    params["query"] = q
            # Named params — tolerate missing closing tags (auto-close)
            for pm in re.finditer(
                r"<parameter=(\w+)>\s*(.*?)(?:\s*</parameter>|\s*<|\s*$)",
                text, re.DOTALL
            ):
                key = pm.group(1)
                val = re.sub(r"<.*", "", pm.group(2)).strip()
                if val:
                    params[key] = _coerce(val)
            if params:
                return {"name": tool, "arguments": params}
        return None

    if fmt == "xml_json":
        m = re.search(r"<tool_call>\s*(\{.*?\})\s*(?:</tool_call>|</parameter>|$)",
                      text, re.DOTALL)
        if not m:
            return None
        try:
            args = json.loads(m.group(1))
            if isinstance(args, dict):
                return {"name": _infer_tool_name(args), "arguments": args}
        except Exception:
            pass
        return None

    if fmt == "json_bare":
        # Strip code fences first
        if "```" in text:
            m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
            if m:
                text = m.group(1).strip()
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            obj = json.loads(text[start:end + 1])
            if not isinstance(obj, dict):
                return None
            if "name" in obj and "arguments" in obj:
                args = obj["arguments"]
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        pass
                # Unwrap double-wrapped args (double-encoding bug)
                if isinstance(args, dict) and "name" in args and "arguments" in args:
                    args = args["arguments"]
                return {"name": obj["name"], "arguments": args}
            # Bare args dict — infer tool name
            if any(k in obj for k in _TOOL_SIGNATURE.values()):
                return {"name": _infer_tool_name(obj), "arguments": obj}
        except Exception:
            pass
        return None

    if fmt == "fenced":
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if not m:
            return None
        return _try_parse_format(m.group(1).strip(), "json_bare")

    return None


# ===========================================================================
# P2-059 Fix 3: Keepalive thread
# Sends a 1-token generation every 55 seconds while the server is idle.
# Prevents the Intel Arc iGPU driver from reclaiming shared memory pages
# between requests, eliminating the cold-reload latency on subsequent prompts.
# ===========================================================================

def _keepalive_worker() -> None:
    """Keep the model pinned in iGPU memory by generating 1 token every 55s.

    Skips the cycle if a real request is in flight (_infer_lock held) so
    the keepalive never queues behind or delays active inference.
    """
    while True:
        time.sleep(55)
        if _pipe is None:
            continue
        if _infer_lock.locked():
            continue   # real request in flight — skip, don't queue behind it
        try:
            cfg = ov_genai.GenerationConfig()
            cfg.max_new_tokens = 1
            cfg.do_sample      = False
            # FIX (P2-068): `_infer_lock.locked()` above and this acquire are not
            # atomic — a real request could grab the lock in between, and if that
            # request then hangs, the OLD blocking `with _infer_lock:` here would
            # wait forever too, silently killing the keepalive loop for good (this
            # thread never gets to its next time.sleep(55)). Keepalive should only
            # ever skip a cycle, never block one — short timeout, not the long
            # request-facing one.
            if not _infer_lock.acquire(timeout=2):
                continue   # lost the race to a real request — skip this cycle
            try:
                _pipe.generate("The market opens at", generation_config=cfg)
            finally:
                _infer_lock.release()
        except Exception:
            pass   # never crash the keepalive thread


# ===========================================================================
# Routes
# ===========================================================================

@app.get("/health")
def health():
    return {"status": "ok", "model": _model_id}

# ADDED (P2-068, 2026-09-17): manual pipeline-reload endpoint.
#
# WHY THIS EXISTS: upstream issue openvinotoolkit/openvino#37736 documents a
# real, actively-investigated (not yet resolved as of this writing) OpenVINO
# GenAI GPU-plugin bug class: a long-lived LLMPipeline/VLMPipeline instance
# can hang indefinitely on some later generate() call — one CPU core spins,
# GPU-side memory usage freezes, no exception is raised — and the only
# confirmed workaround found upstream is deleting and reconstructing the
# pipeline instance. That issue's specific trigger (MoE OFFLOAD_RATIO) does
# NOT apply here (this project uses neither MoE offloading nor an A3B-style
# sparse model), but the STRUCTURAL precondition matches exactly: `_pipe`
# here is a single global instance, constructed once at startup, reused for
# every request for the server's entire lifetime — exactly the shape needed
# for that upstream bug class to potentially manifest, whether or not the
# exact trigger is the same.
#
# This endpoint does NOT claim to fix or explain the hang. It exists so the
# upstream-validated workaround can actually be tested here: if a hang is
# ever observed, calling this endpoint (from a second terminal — this itself
# only needs `_infer_lock`, so it will wait behind a genuinely slow request
# but should NOT wait behind a truly stuck one thanks to the P2-068 lock
# timeout fix above) and then retrying the original request is now a real,
# concrete diagnostic step: if the retry succeeds where it didn't before,
# that's strong evidence this project has hit the same bug class as #37736.
@app.post("/admin/reload_pipeline")
def reload_pipeline():
    global _pipe, _tok
    if _model_dir is None:
        raise HTTPException(status_code=503, detail="Pipeline was never initialized — nothing to reload.")
    if not _infer_lock.acquire(timeout=_INFER_LOCK_TIMEOUT_S):
        raise HTTPException(
            status_code=503,
            detail=f"Could not acquire the inference lock within {_INFER_LOCK_TIMEOUT_S}s "
                   f"to reload — a generation may be stuck. If this itself times out, the "
                   f"process likely needs a hard restart rather than a reload.",
        )
    try:
        t0 = time.monotonic()
        old_pipe = _pipe
        _pipe = None   # drop the reference before constructing the new one
        del old_pipe
        if _is_vlm:
            _pipe = ov_genai.VLMPipeline(_model_dir, _device, CACHE_DIR=_cache_dir)
        else:
            _pipe = ov_genai.LLMPipeline(_model_dir, _device, CACHE_DIR=_cache_dir, PERFORMANCE_HINT="LATENCY")
        _tok = _pipe.get_tokenizer()
        elapsed = round(time.monotonic() - t0, 1)
        print(f"[RELOAD] Pipeline reconstructed in {elapsed}s", flush=True)
        return {"status": "reloaded", "elapsed_seconds": elapsed}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reload failed: {e}") from e
    finally:
        _infer_lock.release()

@app.get("/v1/tools")
def list_tools():
    """Return the available tools schema (from tool_runner.TOOLS)."""
    try:
        import sys, pathlib
        src_dir = str(pathlib.Path(__file__).parent)
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)
        from tool_runner import TOOLS
        return {"tools": TOOLS}
    except ImportError:
        return {"tools": [], "warning": "tool_runner.py not found"}

class ToolCallRequest(BaseModel):
    name:      str
    arguments: dict = {}
    cfg:       dict = {}

@app.post("/v1/tools/call")
def call_tool(req: ToolCallRequest):
    """
    Execute a tool call and return the result.
    The chat UI calls this when the model returns a tool_calls response.
    Proxies to tool_runner.dispatch_tool() which has Brave/DDG web search,
    URL fetching, toolkit functions, and DB queries.
    """
    try:
        import sys, pathlib, dotenv, os
        src_dir = str(pathlib.Path(__file__).parent)
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)
        # Load .env so BRAVE_SEARCH_API_KEY is available
        env_path = pathlib.Path(__file__).parent.parent / ".env"
        if env_path.exists():
            dotenv.load_dotenv(env_path, override=False)
        from tool_runner import dispatch_tool
        cfg = {
            "db_path": req.cfg.get("db_path", "DATA/paper_account.db"),
            "web_search_results": req.cfg.get("web_search_results", 5),
        }
        result = dispatch_tool(req.name, req.arguments, cfg)
        return {"result": result, "tool": req.name}
    except ImportError as e:
        return {"result": f"tool_runner not available: {e}", "tool": req.name}
    except Exception as e:
        return {"result": f"Tool error: {e}", "tool": req.name}

@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{
            "id":       _model_id,
            "object":   "model",
            "created":  int(time.time()),
            "owned_by": "openvino-arc-gpu",
        }],
    }


# ── /v1/completions — Fill-in-the-Middle tab autocomplete (VS Code Continue) ──
class CompletionRequest(BaseModel):
    model:       str | None = None
    prompt:      str  = ""
    suffix:      str  = ""           # FIM suffix for tab autocomplete
    max_tokens:  int   = Field(default=128, ge=1, le=512)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    stream:      bool  = False
    stop:        list[str] | None = None


@app.post("/v1/completions")
async def completions(req: CompletionRequest, request: Request):
    """
    FIM (Fill-in-the-Middle) endpoint for VS Code tab autocomplete.
    Wraps prompt/suffix into a chat message and returns a completion.
    """
    # Build FIM prompt — Qwen2.5-Coder uses <|fim_prefix|> tokens
    if req.suffix:
        content = f"<|fim_prefix|>{req.prompt}<|fim_suffix|>{req.suffix}<|fim_middle|>"
    else:
        content = req.prompt

    # Reuse chat completion logic via an internal ChatRequest
    chat_req = ChatRequest(
        model=req.model,
        messages=[Message(role="user", content=content)],
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        stream=req.stream,
    )

    if req.stream:
        return StreamingResponse(
            _stream_chat(chat_req, request),
            media_type="text/event-stream",
        )

    result = await _complete_chat(chat_req)
    # Reformat as completions response
    return {
        "id":      result["id"].replace("chatcmpl", "cmpl"),
        "object":  "text_completion",
        "created": result["created"],
        "model":   result["model"],
        "choices": [{
            "text":          result["choices"][0]["message"]["content"],
            "index":         0,
            "finish_reason": result["choices"][0]["finish_reason"],
        }],
        "usage": result.get("usage", {}),
    }

@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    if _pipe is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    prompt       = _build_prompt(req)
    prompt_tokens = _count_tokens(prompt)

    config = ov_genai.GenerationConfig()
    config.max_new_tokens    = req.max_tokens
    config.apply_chat_template = False   # _build_prompt() already returns a fully-templated
                                          # string in BOTH branches (thinking or not) — this
                                          # must always be False so VLMPipeline uses our string
                                          # as-is rather than re-templating it internally.
                                          # FIXED 2026-09-17 (P2-068): was `not req.thinking`,
                                          # which inverted the mapping and double-templated the
                                          # default (thinking=False) path instead of the
                                          # thinking=True path the original comment claimed.

    if req.temperature > 0.05:
        config.do_sample  = True
        config.temperature = req.temperature
        config.top_p       = 0.95
    else:
        config.do_sample = False

    # ── Streaming handler ─────────────────────────────────────────────────────
    # Architecture: sync generator running directly in FastAPI's threadpool.
    # The streamer callback fires synchronously inside _pipe.generate() on the
    # SAME thread as the generator — no queue, no background thread, no
    # cancellation race. Tokens are collected into a list by the callback and
    # yielded by the generator via a shared buffer checked after each callback.
    #
    # This is the simplest pattern that works: generate() blocks the threadpool
    # thread until complete, the streamer callback appends tokens to a deque,
    # and the generator yields them. FastAPI flushes each yield to the client.
    # ──────────────────────────────────────────────────────────────────────────
    if req.stream:
        import collections

        def stream_generator():
            call_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
            created = int(time.time())
            buf: collections.deque[str] = collections.deque()
            done = [False]
            error = [None]

            def streamer_cb(subword: str):
                buf.append(subword)
                return ov_genai.StreamingStatus.RUNNING

            import queue as _q
            _stream_q: _q.SimpleQueue[str | None] = _q.SimpleQueue()

            def _gen():
                # FIX (P2-068): timed lock acquisition instead of `with _infer_lock:`.
                # If a prior generate() call is permanently stuck holding the lock,
                # this now fails clearly after _INFER_LOCK_TIMEOUT_S instead of the
                # new request's background thread blocking forever unseen.
                acquired = _infer_lock.acquire(timeout=_INFER_LOCK_TIMEOUT_S)
                if not acquired:
                    print(f"[LOCK TIMEOUT] Could not acquire _infer_lock within "
                          f"{_INFER_LOCK_TIMEOUT_S}s — a prior request may be stuck.",
                          flush=True)
                    _stream_q.put("\n\n[Server busy: a previous generation appears "
                                  "stuck and did not release in time. Try again; if "
                                  "this persists, the server needs a restart.]")
                    _stream_q.put(None)
                    return
                try:
                    def cb(subword: str):
                        _stream_q.put(subword)
                        return ov_genai.StreamingStatus.RUNNING
                    _pipe.generate(prompt, generation_config=config, streamer=cb)
                except Exception as exc:
                    import traceback
                    print(f"[STREAM ERROR] {type(exc).__name__}: {exc}", flush=True)
                    print(traceback.format_exc(), flush=True)
                    _stream_q.put(f"\n\n[Generation error: {exc}]")
                finally:
                    _infer_lock.release()
                    _stream_q.put(None)

            _t = threading.Thread(target=_gen, daemon=True)
            _t.start()

            def _emit(content: str):
                return f"data: {json.dumps({'id': call_id, 'object': 'chat.completion.chunk', 'created': created, 'model': req.model or _model_id, 'choices': [{'index': 0, 'delta': {'content': content}, 'finish_reason': None}]})}\n\n"

            if req.thinking:
                yield _emit("<think>\n")

            try:
                while True:
                    try:
                        subword = _stream_q.get(timeout=300)
                    except _q.Empty:
                        print("[STREAM TIMEOUT] Queue empty after 300s", flush=True)
                        break
                    if subword is None:
                        break
                    yield _emit(subword)
            except GeneratorExit:
                print("[STREAM] GeneratorExit — client disconnected", flush=True)
                raise
            except Exception as exc:
                print(f"[STREAM YIELD ERROR] {type(exc).__name__}: {exc}", flush=True)
                raise
            finally:
                _t.join(timeout=10)

            if req.thinking:
                yield _emit("\n</think>\n\n")

            final = {
                "id": call_id, "object": "chat.completion.chunk",
                "created": created, "model": req.model or _model_id,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(final)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
            headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
        )

    # ── Non-streaming handler ─────────────────────────────────────────────────
    t0 = time.monotonic()
    # FIX (P2-068): this path previously had `with _infer_lock:` and NO timeout
    # at all — if the lock was ever stuck, this blocked the calling FastAPI
    # worker thread forever (worse than the streaming path, which at least had
    # the queue's 300s timeout as a backstop). Now uses the same timed-acquire
    # pattern as the streaming path.
    if not _infer_lock.acquire(timeout=_INFER_LOCK_TIMEOUT_S):
        raise HTTPException(
            status_code=503,
            detail=f"Server busy: could not acquire the inference lock within "
                   f"{_INFER_LOCK_TIMEOUT_S}s. A previous generation may be stuck. "
                   f"Try again; if this persists, the server needs a restart.",
        )
    try:
        if _is_vlm:
            result = _pipe.generate(
                prompt,
                generation_config=config,   # apply_chat_template=False is key
            )
        else:
            result = _pipe.generate(prompt, generation_config=config)
        raw = str(result.texts[0]).strip() if hasattr(result, "texts") else str(result).strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation error: {e}") from e
    finally:
        _infer_lock.release()

    elapsed = max(round(time.monotonic() - t0, 3), 0.001)

    if req.thinking:
        # Thinking mode: keep the <think> block visible so the user can see
        # the reasoning chain (valuable for D-A-C review)
        thoughts, answer = _strip_thinking(raw)
        if thoughts:
            # Prepend reasoning as a collapsible block using markdown details
            raw = f"<details>\n<summary>💭 Reasoning chain</summary>\n\n{thoughts}\n\n</details>\n\n{answer}"
        else:
            raw = answer
    else:
        _, raw = _strip_thinking(raw)

    for eos in ("<|im_end|>", "</s>", "[/INST]", "<|endoftext|>"):
        raw = raw.replace(eos, "").strip()

    # Strip preamble text before XML tool calls (e.g. "I'll search for you.\n\n<tool_call>...")
    # so the tool call parser sees a clean input
    raw_for_tool = raw
    if "<tool_call>" in raw:
        tc_idx = raw.find("<tool_call>")
        raw_for_tool = raw[tc_idx:]

    tool_call   = _parse_tool_call(raw_for_tool, model_id=_model_id)

    # Fallback: if web_search has no query, extract from last user message
    if tool_call and tool_call.get("name") == "web_search" \
            and not tool_call["arguments"].get("query"):
        for msg in reversed(req.messages):
            user_text = _extract_text_content(msg.content).strip()
            if msg.role == "user" and user_text:
                tool_call["arguments"]["query"] = user_text
                break
    comp_tokens = _count_tokens(raw)
    tok_per_sec = round(comp_tokens / elapsed, 1)

    usage_stats = {
        "prompt_tokens":          prompt_tokens,
        "completion_tokens":      comp_tokens,
        "total_tokens":           prompt_tokens + comp_tokens,
        "inference_duration_sec": elapsed,
        "tokens_per_second":      tok_per_sec,
    }

    if tool_call and req.tools:
        call_id = f"call_{uuid.uuid4().hex[:8]}"
        # Include reasoning chain in content when thinking=True
        # Chainlit reads this to render the reasoning Step before tool execution
        thinking_content = None
        if req.thinking:
            thoughts, _ = _strip_thinking(raw)
            if thoughts:
                thinking_content = f"<think>\n{thoughts}\n</think>"

        return {
            "id":      f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object":  "chat.completion",
            "created": int(time.time()),
            "model":   req.model or _model_id,
            "choices": [{
                "index": 0,
                "message": {
                    "role":       "assistant",
                    "content":    thinking_content,
                    "tool_calls": [{
                        "id":       call_id,
                        "type":     "function",
                        "function": {
                            "name":      tool_call["name"],
                            "arguments": json.dumps(tool_call["arguments"]),
                        },
                    }],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": usage_stats,
        }

    return {
        "id":      f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object":  "chat.completion",
        "created": int(time.time()),
        "model":   req.model or _model_id,
        "choices": [{
            "index":         0,
            "message":       {"role": "assistant", "content": raw},
            "finish_reason": "stop",
        }],
        "usage": usage_stats,
    }


# ===========================================================================
# Startup
# ===========================================================================

def main() -> None:
    global _pipe, _tok, _model_id, _cache_dir, _is_vlm, _model_dir, _device, _force_llm

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path",       required=True)
    parser.add_argument("--model-id",         required=True)
    parser.add_argument("--port",             type=int, default=8000)
    parser.add_argument("--device",           default="GPU")
    parser.add_argument("--cache-dir",        default=os.path.expanduser("~/models/.ov_cache"))
    parser.add_argument("--think-log",        default=None)
    parser.add_argument("--draft-model-path", default=None)
    parser.add_argument("--draft-tokens",     type=int, default=2)
    parser.add_argument("--force-llm",        action="store_true",
                        help="Force LLMPipeline even when vision embeddings are present. "
                             "Use for text-only inference on VLM model directories.")
    args, _ = parser.parse_known_args()

    _model_id  = args.model_id
    _cache_dir = args.cache_dir
    os.makedirs(_cache_dir, exist_ok=True)

    # Confirm SYCL guard is active
    sycl_filter = os.environ.get("SYCL_DEVICE_FILTER", "not set")
    print(f"\n[serve_model] SYCL_DEVICE_FILTER = {sycl_filter}")
    print(f"[serve_model] Loading '{args.model_id}' onto {args.device}...")
    t0 = time.monotonic()

    model_dir = Path(args.model_path)
    vision_xml = model_dir / "openvino_vision_embeddings_model.xml"
    _is_vlm   = vision_xml.exists() and not args.force_llm
    _model_dir = str(model_dir)
    _device    = args.device
    _force_llm = args.force_llm

    if _is_vlm:
        # Qwen3.8-27B is a native multimodal model — must use VLMPipeline.
        print(f"  Architecture: VLMPipeline (Qwen3.8 native multimodal)")
        _pipe = ov_genai.VLMPipeline(
            str(model_dir),
            args.device,
            CACHE_DIR=_cache_dir,
        )
        # Thinking is OFF by default (via apply_chat_template=False + manual
        # prompt build with enable_thinking=False in _build_prompt()) but is a
        # real, supported per-request option (ChatRequest.thinking) — see
        # _build_prompt() for both branches. apply_chat_template=False avoids
        # Minja's "Unknown type for 'is' operator: undefined" error when the
        # template uses 'is undefined'. FIXED (P2-068, 2026-09-17): this print
        # used to say "...text-only, non-thinking" unconditionally, which was
        # stale/misleading now that thinking mode is a deliberately supported
        # feature, not just an off-by-default implementation detail.
        print("  ✓ Thinking mode: off by default, available per-request via `thinking: true`")
    else:
        if args.force_llm and vision_xml.exists():
            print(f"  Architecture: LLMPipeline (--force-llm: vision files present but ignored)")
        else:
            print(f"  Architecture: LLMPipeline")
        _pipe = ov_genai.LLMPipeline(
            str(model_dir),
            args.device,
            CACHE_DIR=_cache_dir,
            PERFORMANCE_HINT="LATENCY",
        )

    try:
        _tok = _pipe.get_tokenizer()
    except Exception:
        try:
            _tok = ov_genai.Tokenizer(str(model_dir))
        except Exception:
            _tok = None

    elapsed_load = time.monotonic() - t0
    print(f"  ✓ Model ready in {elapsed_load:.1f}s on {args.device}")
    print(f"  ✓ Streaming: queue-based SSE (P2-059 fix)")
    print(f"  ✓ Keepalive: 1-token ping every 55s (iGPU memory retention)")
    print(f"  →  http://127.0.0.1:{args.port}/v1\n")

    # Start keepalive thread after model is loaded
    threading.Thread(target=_keepalive_worker, daemon=True).start()

    uvicorn.run(app, host="0.0.0.0", port=args.port,
                log_level="warning", access_log=False)


if __name__ == "__main__":
    main()
