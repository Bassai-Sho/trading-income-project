#!/usr/bin/env python3
"""
serve_model.py
==============
Lightweight OpenAI /v1/chat/completions server wrapping OpenVINO GenAI.
Designed for the Trading Income Project D-A-C pipeline on the NUC 14 Pro
Intel Arc iGPU (58GB unified VRAM pool).

Optimizations baked in at pipeline init:
  CACHE_DIR            — persistent compiled Level Zero blobs (< 2s cold start)
  PERFORMANCE_HINT     — LATENCY mode: lowest token-to-token dispatch delay
  KV_CACHE_PRECISION   — u8 INT8 KV cache: 50% memory bandwidth reduction

Tool calling:
  The D-A-C pipeline uses OpenAI tool calling. Since openvino_genai does not
  natively emit function_call JSON, this server injects tool definitions into
  the system prompt and parses JSON tool invocations from the model's output.
  Both Qwen2.5 and Mistral NeMo respond reliably to JSON tool prompting.

Chat templates:
  Uses the model's own tokenizer.apply_chat_template() so Qwen2.5 (ChatML)
  and Mistral NeMo ([INST]) are formatted correctly without hardcoding.

Usage:
    .venv/bin/python3 src/serve_model.py \\
        --model-path ~/models/qwen2.5-32b \\
        --model-id  qwen2.5:32b \\
        --port 8000
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import uuid
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

try:
    import openvino_genai as ov_genai
except ImportError as e:
    raise SystemExit(
        "openvino-genai not installed. Run: pip install openvino-genai"
    ) from e

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

app        = FastAPI(title="OpenVINO GenAI — OpenAI Bridge")
_pipe:     ov_genai.LLMPipeline | None = None
_tok:      Any   = None        # tokenizer (for chat template + token count)
_model_id: str   = "ov-model"
_cache_dir: str  = os.path.expanduser("~/models/.ov_cache")

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class Message(BaseModel):
    role:    str
    content: str | None = None

class ToolFunction(BaseModel):
    name:        str
    description: str | None = None
    parameters:  dict | None = None

class Tool(BaseModel):
    type:     str = "function"
    function: ToolFunction

class ResponseFormat(BaseModel):
    type: str = "text"             # "text" | "json_object" | "json_schema"
    json_schema: dict | None = None


class ChatRequest(BaseModel):
    model:           str | None = None
    messages:        list[Message]
    tools:           list[Tool] | None = None
    tool_choice:     str | dict | None = None
    max_tokens:      int   = Field(default=2048, ge=1, le=8192)
    temperature:     float = Field(default=0.2,  ge=0.0, le=2.0)
    stream:          bool  = False
    response_format: ResponseFormat | None = None   # json_object forces JSON output

# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

TOOL_SYSTEM_PREFIX = """You have access to the following tools. Call a tool by responding ONLY with a JSON object of this exact form (no other text before or after):

{"name": "<tool_name>", "arguments": {<argument key-value pairs>}}

Available tools:
{tools_json}

If the user request does not require a tool, respond normally as an assistant.
"""

def _build_prompt(req: ChatRequest) -> str:
    """
    Build a formatted prompt using the model's own chat template.
    Tool definitions are injected into the system message so both Qwen2.5 and
    Mistral NeMo can handle tool calling via JSON prompting.
    """
    messages = [m.model_dump() for m in req.messages]

    # Inject tool definitions into a system message prepend
    if req.tools:
        tools_json = json.dumps(
            [{"name": t.function.name,
              "description": t.function.description or "",
              "parameters": t.function.parameters or {}}
             for t in req.tools],
            indent=2
        )
        tool_block = TOOL_SYSTEM_PREFIX.format(tools_json=tools_json)

        # Prepend to existing system message or insert new one
        if messages and messages[0]["role"] == "system":
            messages[0]["content"] = tool_block + "\n\n" + (messages[0]["content"] or "")
        else:
            messages.insert(0, {"role": "system", "content": tool_block})

    # JSON output mode — inject instruction when response_format is json_object
    if req.response_format and req.response_format.type in ("json_object", "json_schema"):
        json_instruction = "Respond ONLY with valid JSON. No preamble, no explanation, no markdown."
        if req.response_format.type == "json_schema" and req.response_format.json_schema:
            schema_str = json.dumps(req.response_format.json_schema.get("schema", {}), indent=2)
            json_instruction += f"\nRequired JSON schema:\n{schema_str}"
        if messages and messages[-1]["role"] == "system":
            messages[-1]["content"] = (messages[-1]["content"] or "") + "\n\n" + json_instruction
        else:
            messages.insert(0, {"role": "system", "content": json_instruction})

    # Use the tokenizer's own chat template (handles ChatML, [INST], etc.)
    try:
        formatted = _tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return str(formatted)
    except Exception:
        # Fallback: basic ChatML (works for Qwen2.5 and most instruct models)
        lines = []
        for m in messages:
            role    = m.get("role", "user")
            content = m.get("content") or ""
            lines.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        lines.append("<|im_start|>assistant\n")
        return "\n".join(lines)


def _count_tokens(text: str) -> int:
    """Tokenize text and return token count. Falls back to word-count estimate."""
    try:
        return len(_tok.encode(text).input_ids)
    except Exception:
        return max(1, len(text.split()))


# ---------------------------------------------------------------------------
# Tool call parsing
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Thinking token handling (DeepSeek-R1 / QwQ emit <think>...</think> blocks)
# ---------------------------------------------------------------------------

_THINK_RE   = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_THINK_LOG: str | None = None  # optionally set at startup via --think-log


def _strip_thinking(text: str) -> tuple[str, str]:
    """
    Separate <think>...</think> reasoning from the final answer.
    DeepSeek-R1 and QwQ emit large thinking blocks before the answer.
    These are stripped from content (they overflow max_tokens) and
    optionally written to LOGS/thinking_<port>.log for inspection.
    Returns (thinking_content, clean_answer).
    """
    thoughts = "\n\n".join(_THINK_RE.findall(text))
    answer   = _THINK_RE.sub("", text).strip()
    return thoughts, answer


_TOOL_CALL_RE = re.compile(
    r'\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{[^}]*\})\s*\}',
    re.DOTALL,
)


def _parse_tool_call(text: str) -> dict | None:
    """
    Detect and parse a tool call JSON from the model's output.
    Returns a dict with 'name' and 'arguments' or None if not a tool call.
    """
    stripped = text.strip()
    # Quick gate: must look like JSON
    if not stripped.startswith("{"):
        return None
    m = _TOOL_CALL_RE.search(stripped)
    if not m:
        try:
            obj = json.loads(stripped)
            if "name" in obj and "arguments" in obj:
                return obj
        except json.JSONDecodeError:
            pass
        return None
    try:
        args = json.loads(m.group(2))
    except json.JSONDecodeError:
        args = {}
    return {"name": m.group(1), "arguments": args}


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{
            "id":       _model_id,
            "object":   "model",
            "created":  int(time.time()),
            "owned_by": "openvino-arc-gpu",
        }],
    }


@app.get("/health")
async def health():
    return {"status": "ok", "model": _model_id}


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    if _pipe is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialised")

    prompt = _build_prompt(req)

    config = ov_genai.GenerationConfig()
    config.max_new_tokens = req.max_tokens
    config.temperature    = max(req.temperature, 0.01)

    t0 = time.monotonic()
    try:
        raw = str(_pipe.generate(prompt, config)).strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation error: {e}") from e
    elapsed = round(time.monotonic() - t0, 3)

    # Extract and log thinking tokens (DeepSeek-R1 / QwQ)
    thinking_content, raw = _strip_thinking(raw)
    if thinking_content and _THINK_LOG:
        try:
            with open(_THINK_LOG, "a") as _tf:
                _tf.write(f"\n--- {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
                _tf.write(thinking_content + "\n")
        except Exception:
            pass

    # Strip residual end-of-turn tokens
    for eos in ("<|im_end|>", "</s>", "[/INST]", "<|endoftext|>"):
        raw = raw.replace(eos, "").strip()

    # Check if the model issued a tool call
    tool_call = _parse_tool_call(raw)

    prompt_tokens = _count_tokens(prompt)
    comp_tokens   = _count_tokens(raw)

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
            "usage": {
                "prompt_tokens":     prompt_tokens,
                "completion_tokens": comp_tokens,
                "total_tokens":      prompt_tokens + comp_tokens,
                "inference_duration_sec": elapsed,
            },
        }

    return {
        "id":      f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object":  "chat.completion",
        "created": int(time.time()),
        "model":   req.model or _model_id,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": raw},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens":     prompt_tokens,
            "completion_tokens": comp_tokens,
            "total_tokens":      prompt_tokens + comp_tokens,
            "inference_duration_sec": elapsed,
        },
    }


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

def main() -> None:
    global _pipe, _tok, _model_id, _cache_dir

    parser = argparse.ArgumentParser(
        description="OpenVINO GenAI — OpenAI /v1 Bridge Server"
    )
    parser.add_argument("--model-path", required=True,
                        help="Local directory containing OpenVINO model files")
    parser.add_argument("--model-id",   required=True,
                        help="Model identifier returned by /v1/models")
    parser.add_argument("--port",       type=int, default=8000)
    parser.add_argument("--think-log",  default=None,
                        help="Log file for <think> tokens (DeepSeek-R1/QwQ). Optional.")
    parser.add_argument("--draft-model-path", default=None,
                        help="Path to small draft model for speculative decoding. "
                             "Must share tokenizer family with main model. "
                             "Recommended: Qwen3-0.6B for Pair 2, Qwen2.5-1.5B for Pairs 1+3.")
    parser.add_argument("--device",     default="GPU",
                        help="OpenVINO device: GPU, CPU, or AUTO")
    parser.add_argument("--cache-dir",
                        default=os.path.expanduser("~/models/.ov_cache"),
                        help="Directory for compiled Level Zero kernel blobs")
    args = parser.parse_args()

    _model_id  = args.model_id
    _cache_dir = args.cache_dir
    _THINK_LOG = args.think_log
    os.makedirs(_cache_dir, exist_ok=True)

    print(f"\n[serve_model] Loading '{args.model_id}' onto {args.device}...")
    print(f"  Model path:  {args.model_path}")
    print(f"  Cache dir:   {_cache_dir}")
    print(f"  Opts:        LATENCY | KV u8 | CACHE_DIR")

    t0 = time.monotonic()
    pipeline_kwargs = {
        "CACHE_DIR":             _cache_dir,
        "PERFORMANCE_HINT":      "LATENCY",    # lowest token-to-token dispatch delay
        "KV_CACHE_PRECISION":    "u8",         # 50% KV memory bandwidth reduction
        "enable_save_ov_model":  True,         # serialise IR on first run → < 2s subsequent loads
    }

    if args.draft_model_path:
        # Speculative decoding: small GPU draft proposes tokens, large model verifies
        # 2-3× decode throughput. Draft must share tokenizer family with main model.
        print(f"  Draft model: {args.draft_model_path} (speculative decoding on {args.device})")
        draft = ov_genai.draft_model(args.draft_model_path, args.device)
        _pipe = ov_genai.LLMPipeline(args.model_path, args.device, draft_model=draft, **pipeline_kwargs)
    else:
        _pipe = ov_genai.LLMPipeline(args.model_path, args.device, **pipeline_kwargs)
    
    _tok = _pipe.get_tokenizer()
    print(f"  ✓ Ready in {time.monotonic()-t0:.1f}s  →  http://127.0.0.1:{args.port}/v1")

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=args.port,
        log_level="warning",
        access_log=False,
    )


if __name__ == "__main__":
    main()
