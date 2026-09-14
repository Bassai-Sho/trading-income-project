#!/usr/bin/env python3
"""
serve_model.py
==============
Lightweight OpenAI /v1/chat/completions server wrapping OpenVINO GenAI.
Designed for the Trading Income Project on the NUC 14 Pro Intel Arc iGPU.

Optimizations:
  LATENCY mode | INT8 u8 KV cache | KV prefix caching | Live throughput stats
"""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

try:
    import openvino_genai as ov_genai
except ImportError as e:
    raise SystemExit("openvino-genai not installed. Run: pip install openvino-genai") from e

# ---------------------------------------------------------------------------
# Global state & Mutex
# ---------------------------------------------------------------------------

app         = FastAPI(title="OpenVINO GenAI — OpenAI Bridge")
_pipe:      ov_genai.LLMPipeline | ov_genai.VLMPipeline | None = None
_tok:       Any   = None
_model_id:  str   = "ov-model"
_is_vlm:    bool  = False
_has_draft: bool  = False
_cache_dir: str   = os.path.expanduser("~/models/.ov_cache")
_infer_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

TOOL_SYSTEM_PREFIX = """You have access to the following tools:
{tools_json}

INSTRUCTIONS FOR TOOL CALLING:
- To call a tool, respond IMMEDIATELY with ONLY a JSON object:
{"name": "<tool_name>", "arguments": {<argument key-value pairs>}}
- Do NOT output any reasoning, chain of thought, or introductory text before the JSON.
- Start directly with { and end with }.
- If no tool is needed, respond normally as an assistant.
"""

def _extract_text_content(content: Any) -> str:
    if not content:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict) and "text" in p:
                parts.append(str(p["text"]))
            elif isinstance(p, str):
                parts.append(p)
        return "".join(parts)
    return str(content)

def _build_prompt(req: ChatRequest) -> str:
    messages = [{"role": m.role, "content": _extract_text_content(m.content)} for m in req.messages]

    if req.tools:
        tools_json = json.dumps(
            [{"name": t.function.name, "description": t.function.description or "", "parameters": t.function.parameters or {}}
             for t in req.tools], indent=2
        )
        # Safe string replacement avoids Python format string brace collisions
        tool_block = TOOL_SYSTEM_PREFIX.replace("{tools_json}", tools_json)
        if messages and messages[0]["role"] == "system":
            messages[0]["content"] = tool_block + "\n\n" + messages[0]["content"]
        else:
            messages.insert(0, {"role": "system", "content": tool_block})

    if req.response_format and req.response_format.type in ("json_object", "json_schema"):
        json_instruction = "Respond ONLY with valid JSON. No preamble, no explanation, no markdown."
        if req.response_format.type == "json_schema" and req.response_format.json_schema:
            schema_str = json.dumps(req.response_format.json_schema.get("schema", {}), indent=2)
            json_instruction += f"\nRequired JSON schema:\n{schema_str}"
        if messages and messages[-1]["role"] == "system":
            messages[-1]["content"] += "\n\n" + json_instruction
        else:
            messages.insert(0, {"role": "system", "content": json_instruction})

    if _tok is not None:
        try:
            return str(_tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        except Exception:
            pass

    # ChatML fallback
    lines = []
    for m in messages:
        lines.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>")
    lines.append("<|im_start|>assistant\n")
    return "\n".join(lines)

def _count_tokens(text: str) -> int:
    try:
        if _tok is not None:
            return len(_tok.encode(text).input_ids)
    except Exception:
        pass
    return max(1, len(text.split()))

# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

_THINK_RE   = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_THINK_LOG:  str | None = None
_DRAFT_TOKENS: int      = 2

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
    end = stripped.rfind("}")
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

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

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
        raise HTTPException(status_code=503, detail="Pipeline not initialised")

    prompt = _build_prompt(req)
    prompt_tokens = _count_tokens(prompt)
    print(f"  [{_model_id}] Received request: {prompt_tokens} prompt tokens, max_tokens={req.max_tokens}...")

    config = ov_genai.GenerationConfig()
    config.max_new_tokens = req.max_tokens

    if req.temperature > 0.05:
        config.do_sample = True
        config.temperature = req.temperature
        config.top_p = 0.95
    else:
        config.do_sample = False

    if _has_draft:
        config.num_assistant_tokens = _DRAFT_TOKENS
        config.assistant_confidence_threshold = 0.0

    t0 = time.monotonic()
    try:
        with _infer_lock:
            if _is_vlm:
                result = _pipe.generate(prompt, images=[], generation_config=config)
                raw = str(result.texts[0]).strip() if hasattr(result, "texts") else str(result).strip()
            else:
                raw = str(_pipe.generate(prompt, config)).strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation error: {e}") from e

    elapsed = max(round(time.monotonic() - t0, 3), 0.001)

    thinking_content, raw = _strip_thinking(raw)
    if thinking_content and _THINK_LOG:
        try:
            with open(_THINK_LOG, "a") as _tf:
                _tf.write(f"\n--- {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n{thinking_content}\n")
        except Exception:
            pass

    for eos in ("<|im_end|>", "</s>", "[/INST]", "<|endoftext|>"):
        raw = raw.replace(eos, "").strip()

    tool_call = _parse_tool_call(raw)
    comp_tokens = _count_tokens(raw)
    tok_per_sec = round(comp_tokens / elapsed, 1)

    print(f"  [{_model_id}] Completed in {elapsed:.1f}s | {comp_tokens} tokens ({tok_per_sec} tok/s)")

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
            "index": 0,
            "message": {"role": "assistant", "content": raw},
            "finish_reason": "stop",
        }],
        "usage": usage_stats,
    }

# ---------------------------------------------------------------------------
# Server Startup
# ---------------------------------------------------------------------------

def main() -> None:
    global _pipe, _tok, _model_id, _cache_dir, _is_vlm, _has_draft, _THINK_LOG, _DRAFT_TOKENS

    parser = argparse.ArgumentParser(description="OpenVINO GenAI — OpenAI Bridge Server")
    parser.add_argument("--model-path", required=True, help="Local directory with OpenVINO model")
    parser.add_argument("--model-id",   required=True, help="Model ID for /v1/models")
    parser.add_argument("--port",       type=int, default=8000)
    parser.add_argument("--think-log",  default=None)
    parser.add_argument("--draft-model-path", default=None,
                        help="Optional separate small draft model directory (contains openvino_model.xml)")
    parser.add_argument("--draft-tokens", type=int, default=2)
    parser.add_argument("--device",     default="GPU")
    parser.add_argument("--cache-dir",  default=os.path.expanduser("~/models/.ov_cache"))
    args = parser.parse_args()

    _model_id     = args.model_id
    _cache_dir    = args.cache_dir
    _THINK_LOG    = args.think_log
    _DRAFT_TOKENS = args.draft_tokens
    os.makedirs(_cache_dir, exist_ok=True)

    print(f"\n[serve_model] Loading '{args.model_id}' onto {args.device}...")
    print(f"  Model path:  {args.model_path}")
    print(f"  Cache dir:   {_cache_dir}")
    print(f"  Opts:        LATENCY | KV u8 | CACHE_DIR | PREFIX_CACHING")

    t0 = time.monotonic()
    model_dir = Path(args.model_path)
    _is_vlm = (model_dir / "openvino_vision_embeddings_model.xml").exists()

    draft = None
    if args.draft_model_path:
        draft_path = Path(args.draft_model_path)
        if (draft_path / "openvino_model.xml").exists():
            print(f"  Speculative: Draft model enabled ({draft_path})")
            draft = ov_genai.draft_model(str(draft_path), args.device)
            _has_draft = True
        else:
            print(f"  ⚠ Draft model path missing openvino_model.xml — running standard decode")

    if _is_vlm:
        print(f"  Architecture: VLMPipeline (vision-language model)")
        _pipe = ov_genai.VLMPipeline(
            str(model_dir),
            args.device,
            CACHE_DIR=_cache_dir
        )
    else:
        print(f"  Architecture: LLMPipeline (text-only)")
        llm_kwargs = {
            "CACHE_DIR":          _cache_dir,
            "PERFORMANCE_HINT":   "LATENCY",
            "KV_CACHE_PRECISION": "u8",
        }
        sched_cfg = ov_genai.SchedulerConfig()
        sched_cfg.enable_prefix_caching = True

        if draft is not None:
            _pipe = ov_genai.LLMPipeline(
                str(model_dir),
                args.device,
                draft_model=draft,
                scheduler_config=sched_cfg,
                **llm_kwargs
            )
        else:
            _pipe = ov_genai.LLMPipeline(
                str(model_dir),
                args.device,
                scheduler_config=sched_cfg,
                **llm_kwargs
            )

    try:
        _tok = _pipe.get_tokenizer()
    except Exception:
        try:
            _tok = ov_genai.Tokenizer(str(model_dir))
        except Exception:
            _tok = None

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
