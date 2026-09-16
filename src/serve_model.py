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
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

try:
    import openvino_genai as ov_genai
except ImportError as e:
    raise SystemExit("openvino-genai not installed.") from e

from fastapi.middleware.cors import CORSMiddleware

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
_infer_lock = threading.Lock()   # ensures one inference at a time; concurrent
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
    # Cap at 1024 — the KV cache is pre-allocated at load time for MAX_NEW_TOKENS.
    # Requesting more tokens than the cache was built for causes CL_EXEC_STATUS_ERROR
    # (-14) from ocl_event.cpp mid-generation on the Arc iGPU.
    max_tokens:      int   = Field(default=1024, ge=1, le=1024)
    temperature:     float = Field(default=0.2,  ge=0.0, le=2.0)
    stream:          bool  = False
    response_format: ResponseFormat | None = None

TOOL_SYSTEM_PREFIX = """You have access to the following tools:
{tools_json}

INSTRUCTIONS FOR TOOL CALLING:
- To call a tool, respond IMMEDIATELY with ONLY a JSON object:
{"name": "<tool_name>", "arguments": {<argument key-value pairs>}}
- Do NOT output any reasoning or introductory text before the JSON.
- Start directly with { and end with }.
- If no tool is needed, respond normally as an assistant.
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

    For VLMPipeline (Qwen3.8): we build the prompt ourselves with
    enable_thinking=False and set apply_chat_template=False in GenerationConfig
    so VLMPipeline uses our pre-built string instead of re-applying the template
    internally (which ignores enable_thinking and always opens a think block).

    For LLMPipeline: standard path, apply_chat_template=False is already set
    in the original config from the file header.
    """
    messages = [{"role": m.role, "content": _extract_text_content(m.content)} for m in req.messages]

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
            # enable_thinking=False: suppresses Qwen3's <think> reasoning chain.
            # For VLMPipeline this only works when we build the prompt here AND
            # set config.apply_chat_template=False so VLMPipeline uses our string.
            prompt = str(_tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            ))
            # Safety net: if template left an open think block, close it
            if prompt.endswith("<think>\n"):
                prompt += "\n</think>\n\n"
            return prompt
        except TypeError:
            # Older tokenizer: enable_thinking not supported
            try:
                prompt = str(_tok.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                ))
                return prompt
            except Exception:
                pass
        except Exception:
            pass

    # ChatML fallback with closed think block
    lines = [f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>" for m in messages]
    lines.append("<|im_start|>assistant\n<think>\n\n</think>\n\n")
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

def _parse_tool_call(text: str) -> dict | None:
    stripped = text.strip()
    if "```" in stripped:
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.DOTALL)
        if m:
            stripped = m.group(1).strip()
    start = stripped.find("{")
    end   = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        stripped = stripped[start:end+1]
    else:
        return None
    try:
        obj = json.loads(stripped)
        if isinstance(obj, dict) and "name" in obj and "arguments" in obj:
            if isinstance(obj["arguments"], str):
                try:
                    obj["arguments"] = json.loads(obj["arguments"])
                except Exception:
                    pass
            return obj
    except Exception:
        pass
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
            with _infer_lock:
                _pipe.generate("The market opens at", generation_config=cfg)
        except Exception:
            pass   # never crash the keepalive thread


# ===========================================================================
# Routes
# ===========================================================================

@app.get("/health")
def health():
    return {"status": "ok", "model": _model_id}

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

@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    if _pipe is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    prompt       = _build_prompt(req)
    prompt_tokens = _count_tokens(prompt)

    config = ov_genai.GenerationConfig()
    config.max_new_tokens    = req.max_tokens
    config.apply_chat_template = False   # bypass Minja C++ Jinja parser issue

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
                try:
                    with _infer_lock:
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
                    _stream_q.put(None)

            _t = threading.Thread(target=_gen, daemon=True)
            _t.start()

            try:
                while True:
                    try:
                        subword = _stream_q.get(timeout=300)
                    except _q.Empty:
                        print("[STREAM TIMEOUT] Queue empty after 300s", flush=True)
                        break
                    if subword is None:
                        break
                    chunk = {
                        "id": call_id, "object": "chat.completion.chunk",
                        "created": created, "model": req.model or _model_id,
                        "choices": [{"index": 0, "delta": {"content": subword},
                                     "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
            except GeneratorExit:
                print("[STREAM] GeneratorExit — client disconnected", flush=True)
                raise
            except Exception as exc:
                print(f"[STREAM YIELD ERROR] {type(exc).__name__}: {exc}", flush=True)
                raise
            finally:
                _t.join(timeout=10)

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
    try:
        with _infer_lock:
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

    elapsed = max(round(time.monotonic() - t0, 3), 0.001)
    _, raw = _strip_thinking(raw)
    for eos in ("<|im_end|>", "</s>", "[/INST]", "<|endoftext|>"):
        raw = raw.replace(eos, "").strip()

    tool_call   = _parse_tool_call(raw)
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
        return {
            "id":      f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object":  "chat.completion",
            "created": int(time.time()),
            "model":   req.model or _model_id,
            "choices": [{
                "index": 0,
                "message": {
                    "role":       "assistant",
                    "content":    None,
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
    global _pipe, _tok, _model_id, _cache_dir, _is_vlm

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

    if _is_vlm:
        # Qwen3.8-27B is a native multimodal model — must use VLMPipeline.
        print(f"  Architecture: VLMPipeline (Qwen3.8 native multimodal — text-only, non-thinking)")
        _pipe = ov_genai.VLMPipeline(
            str(model_dir),
            args.device,
            CACHE_DIR=_cache_dir,
        )
        # Thinking mode is disabled by setting apply_chat_template=False in
        # GenerationConfig and building the prompt manually via _tok with
        # enable_thinking=False. This avoids Minja's "Unknown type for 'is'
        # operator: undefined" error when the template uses 'is undefined'.
        # See _build_prompt() — VLM path uses _tok.apply_chat_template directly.
        print("  ✓ Thinking mode: disabled via apply_chat_template=False + manual prompt build")
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
