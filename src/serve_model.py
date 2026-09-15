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
   it. OpenWebUI received silence until [DONE], then made a second non-streaming
   request, causing the 100% CPU spike and iGPU memory eviction observed in
   intel_gpu_top. Fixed with a SimpleQueue bridge: callback pushes tokens onto
   the queue; stream_generator() pulls and yields them as SSE chunks in real time.

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
import json
import queue as _queue_mod
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

app         = FastAPI(title="OpenVINO GenAI — OpenAI Bridge")
_pipe:      ov_genai.LLMPipeline | ov_genai.VLMPipeline | None = None
_tok:       Any   = None
_model_id:  str   = "ov-model"
_is_vlm:    bool  = False
_cache_dir: str   = os.path.expanduser("~/models/.ov_cache")
_infer_lock = threading.Lock()


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
    max_tokens:      int   = Field(default=2048, ge=1, le=8192)
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
            return str(_tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        except Exception:
            pass

    # ChatML fallback
    lines = [f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>" for m in messages]
    lines.append("<|im_start|>assistant\n<think>\n</think>")
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
    """Keep the model pinned in iGPU memory by generating 1 token every 55s."""
    while True:
        time.sleep(55)
        if _pipe is None:
            continue
        try:
            cfg = ov_genai.GenerationConfig()
            cfg.max_new_tokens = 1
            cfg.do_sample = False
            with _infer_lock:
                _pipe.generate(".", generation_config=cfg)
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

    # ── P2-059 Fix 2: Streaming handler ──────────────────────────────────────
    # Previously: streamer_cb built chunk_payload but never yielded it.
    # Now: a SimpleQueue bridges the openvino callback thread to the SSE
    # generator, so tokens flow to OpenWebUI in real time as they are produced.
    # ──────────────────────────────────────────────────────────────────────────
    if req.stream:
        def stream_generator():
            call_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
            created = int(time.time())
            q: _queue_mod.SimpleQueue[str | None] = _queue_mod.SimpleQueue()

            def _run_generation() -> None:
                """Run inference in a background thread, pushing tokens onto q."""
                try:
                    with _infer_lock:
                        def streamer_cb(subword: str) -> bool:
                            q.put(subword)
                            return False   # returning True would abort generation

                        _pipe.generate(prompt, generation_config=config,
                                       streamer=streamer_cb)
                except Exception as exc:
                    # Push a sentinel with error info so the generator can stop
                    q.put(f"\n\n[Generation error: {exc}]")
                finally:
                    q.put(None)   # sentinel — generation complete

            threading.Thread(target=_run_generation, daemon=True).start()

            # Pull tokens from queue and yield as SSE chunks
            while True:
                subword = q.get()          # blocks until next token or sentinel
                if subword is None:
                    break
                chunk = {
                    "id":      call_id,
                    "object":  "chat.completion.chunk",
                    "created": created,
                    "model":   req.model or _model_id,
                    "choices": [{
                        "index":         0,
                        "delta":         {"content": subword},
                        "finish_reason": None,
                    }],
                }
                yield f"data: {json.dumps(chunk)}\n\n"

            # Final chunk signalling completion
            final = {
                "id":      call_id,
                "object":  "chat.completion.chunk",
                "created": created,
                "model":   req.model or _model_id,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(final)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    # ── Non-streaming handler (unchanged) ─────────────────────────────────────
    t0 = time.monotonic()
    try:
        with _infer_lock:
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
    _is_vlm   = (model_dir / "openvino_vision_embeddings_model.xml").exists()

    if _is_vlm:
        print(f"  Architecture: VLMPipeline")
        _pipe = ov_genai.VLMPipeline(str(model_dir), args.device)
    else:
        print(f"  Architecture: LLMPipeline")
        _pipe = ov_genai.LLMPipeline(
            str(model_dir),
            args.device,
            CACHE_DIR=_cache_dir,
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

    uvicorn.run(app, host="127.0.0.1", port=args.port,
                log_level="warning", access_log=False)


if __name__ == "__main__":
    main()
