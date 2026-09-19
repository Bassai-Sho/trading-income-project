# CPU Crash Diagnostic — NUC14RVH-B Trading AI

## Symptom
When a tool-chain request fires (web_search → fetch_url → synthesis), the Arc iGPU
compute drops to ~2% and CPU spikes to 100% mid-generation. No response is returned
to Chainlit. `curl http://localhost:8000/health` still returns OK but the model is
stuck in a CPU spin loop.

## Environment
- **Node:** adam-NUC14RVH-B, Intel Meteor Lake, Arc 140V iGPU
- **OS:** Ubuntu 24.04, Python 3.12
- **OpenVINO:** 2026.5.0.0 dev nightly
- **Model:** Qwen3.8-27B-int4, VLMPipeline
- **serve_model.py:** `~/github/trading-income-project/src/serve_model.py`
- **Log:** `~/github/trading-income-project/LOGS/model_8000.log`

## What is known

### The crash pattern
1. User sends a news/research query with tools enabled
2. Phase 1: non-streaming tool-detection call — completes OK (30-60s)
3. `web_search` executes — completes OK
4. `fetch_url` executes — completes OK  
5. Phase 2: streaming synthesis call with tool results in context
6. GPU goes to 0% compute, CPU goes to 100% — **crash happens here**
7. `health` endpoint still responds (FastAPI main thread alive)
8. `_infer_lock` is held — times out after 280s
9. No response returned to client

### Context at crash point
The synthesis call (Phase 2) has a large context:
- System message (~200 tokens)
- User message (~20 tokens)
- Tool intent (assistant message with tool_calls, ~50 tokens)
- web_search result (~400-800 tokens)
- fetch_url article content (~500-1000 tokens)
- Synthesis system message (~80 tokens)
**Total context: ~1250-2150 tokens before generation starts**

### Known OpenVINO issues
- OpenCL error -14 (`CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST`) occurs when
  the KV cache overflows on the Arc iGPU
- The KV cache is pre-allocated at model load time for `max_new_tokens`
- VLMPipeline has no `max_new_tokens` constructor arg — it's set in GenerationConfig
- When total context (prompt tokens + generated tokens) exceeds the cache,
  the OpenCL kernel fails silently and compute falls back to CPU

### What was changed recently (may have introduced regression)
- `max_tokens` ceiling raised from `le=1024` to `le=16384` — this allows much
  larger generation requests than the iGPU can handle
- `_parse_tool_call` now passes model_id — minor change, unlikely cause
- Synthesis call sends tool results (fetch_url content up to 2000 chars)
  which significantly increases the prompt token count

---

## Diagnostic Steps

### Step 1: Confirm crash is OpenCL -14 (not a Python exception)
```bash
# Run serve_model with verbose logging and trigger the crash
cd ~/github/trading-income-project
./launch_models.sh --stop
sleep 3

# Start with stderr captured
.venv/bin/python3 src/serve_model.py \
  --model-path ~/models/qwen3.8-27b-int4 \
  --model-id qwen3.8:27b \
  --device GPU --port 8000 2>&1 | tee /tmp/serve_verbose.log &

sleep 180  # wait for model to load

# Trigger the crash with a large context manually
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.8:27b",
    "messages": [
      {"role": "system", "content": "You are a trading assistant. Use the tool results to answer."},
      {"role": "user", "content": "What are the top 3 market-moving news items today?"},
      {"role": "assistant", "content": null, "tool_calls": [{"id": "call_test", "type": "function", "function": {"name": "web_search", "arguments": "{\"query\": \"top market news today\"}"}}]},
      {"role": "tool", "content": "BBC NEWS: Fed raises rates. Reuters: SPY falls 2%. Bloomberg: VIX spikes to 25. AP: Jobs report beats expectations. CNBC: Tech selloff continues. Guardian: UK markets hit. CNN: Oil prices surge. FT: Dollar strengthens.", "tool_call_id": "call_test"},
      {"role": "tool", "content": "ARTICLE FROM BBC: The Federal Reserve raised interest rates by 25 basis points today in a unanimous decision. Markets reacted sharply with SPY falling 2.1%. The VIX spiked to 25 indicating elevated fear. Technology stocks led the decline with the Nasdaq falling 3.2%. Bond yields rose sharply. Oil prices surged 4% on supply concerns. The dollar strengthened against major currencies. Analysts expect further rate hikes.", "tool_call_id": "auto_fetch_call_test"},
      {"role": "system", "content": "Synthesise the above tool results into a clear answer."}
    ],
    "max_tokens": 512,
    "stream": true,
    "thinking": false
  }' &

# Watch what happens to the GPU
watch -n 1 'ps aux | grep serve_model | grep -v grep | awk "{print \$3, \$4}" | head -3'
```

After triggering, check:
```bash
# Did it crash with OpenCL error?
grep -i "opencl\|CL_\|error\|-14\|crash\|exception\|traceback" /tmp/serve_verbose.log | tail -20

# What is the GPU doing?
cat /tmp/serve_verbose.log | tail -50
```

### Step 2: Determine the token count at crash
```bash
# Count tokens in the synthesis context
.venv/bin/python3 << 'PYEOF'
import sys
sys.path.insert(0, 'src')

# Load the tokenizer
from pathlib import Path
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(
    str(Path.home() / 'models/qwen3.8-27b-int4'),
    trust_remote_code=True
)

# Simulate the synthesis context
messages = [
    {"role": "system", "content": "You are a trading assistant. Use the tool results to answer."},
    {"role": "user", "content": "What are the top 3 market-moving news items today?"},
    {"role": "assistant", "content": "Let me search for that.", "tool_calls": []},
    {"role": "tool", "content": "BBC NEWS: Fed raises rates. Reuters: SPY falls 2%. " * 10},
    {"role": "tool", "content": "ARTICLE: The Federal Reserve raised rates today. " * 20},
    {"role": "system", "content": "Synthesise the above tool results into a clear answer."},
]

text = " ".join(m["content"] or "" for m in messages)
tokens = tok(text, return_tensors=None)["input_ids"]
print(f"Estimated context tokens: {len(tokens)}")
print(f"With 512 max_tokens: total = {len(tokens) + 512}")
print(f"With 1024 max_tokens: total = {len(tokens) + 1024}")
PYEOF
```

### Step 3: Check the GenerationConfig max token settings
```bash
.venv/bin/python3 << 'PYEOF'
import sys, json
sys.path.insert(0, 'src')
from pathlib import Path

cfg_path = Path.home() / 'models/qwen3.8-27b-int4/generation_config.json'
with open(cfg_path) as f:
    cfg = json.load(f)
print(json.dumps(cfg, indent=2))
# Look for: max_length, max_new_tokens, max_position_embeddings
PYEOF
```

### Step 4: Test with progressively larger contexts to find the limit
```bash
# Test 1: Small context (should work)
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.8:27b","messages":[{"role":"user","content":"Say hi."}],"max_tokens":50,"stream":false}' \
  | python3 -m json.tool | grep "finish_reason\|tokens_per"

# Test 2: Medium context (~500 prompt tokens)
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"qwen3.8:27b\",\"messages\":[{\"role\":\"user\",\"content\":\"$(python3 -c \"print('word ' * 400)\")\"}],\"max_tokens\":100,\"stream\":false}" \
  | python3 -m json.tool | grep "finish_reason\|tokens_per"

# Test 3: Large context (~1500 prompt tokens) — this is what tool chain produces
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"qwen3.8:27b\",\"messages\":[{\"role\":\"user\",\"content\":\"$(python3 -c \"print('word ' * 1200)\")\"}],\"max_tokens\":100,\"stream\":false}" \
  | python3 -m json.tool | grep "finish_reason\|tokens_per"
```

**If Test 3 crashes but Test 2 works: the KV cache limit is between 500-1500 prompt tokens.**

### Step 5: Check OpenVINO KV cache configuration
```bash
.venv/bin/python3 << 'PYEOF'
import openvino_genai as ov

# What cache parameters does VLMPipeline support?
import inspect
sig = inspect.signature(ov.VLMPipeline.__init__)
print("VLMPipeline init params:", list(sig.parameters.keys()))

cfg = ov.GenerationConfig()
print("\nGenerationConfig attributes:")
for attr in dir(cfg):
    if not attr.startswith('_'):
        try:
            val = getattr(cfg, attr)
            if not callable(val):
                print(f"  {attr} = {val}")
        except:
            pass
PYEOF
```

### Step 6: Check if the crash is reproducible with thinking=True vs thinking=False
```bash
# thinking=False synthesis (current) — does it crash?
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.8:27b","messages":[{"role":"system","content":"Summarise this: BBC Fed raises rates. Reuters SPY falls. Bloomberg VIX spikes. AP Jobs beat. CNBC Tech selloff. Guardian UK markets. CNN Oil surge. FT Dollar up. Article: Fed raised rates 25bp. Markets fell 2%. VIX hit 25. Nasdaq down 3.2%. Bond yields rose. Oil up 4%. Dollar strengthened."},{"role":"user","content":"What are the top 3 news items?"}],"max_tokens":512,"stream":true,"thinking":false}' \
  > /dev/null &
sleep 5
# Check GPU
cat /proc/$(pgrep -f "serve_model.*8000" | head -1)/status | grep State
```

---

## Likely Fixes to Test

### Fix A: Cap prompt tokens in chainlit_app.py
In `src/chainlit_app.py`, reduce `fetch_url` content before adding to history:
```python
# Current: 2000 chars
"content": f"Article from {best_url}:\n\n{page_content[:2000]}",
# Try: 800 chars max
"content": f"Article from {best_url}:\n\n{page_content[:800]}",
```

### Fix B: Reset max_tokens in ChatRequest back to le=1024
In `src/serve_model.py`:
```python
# Change back to:
max_tokens: int = Field(default=1024, ge=1, le=1024)
```
The 16384 ceiling was added for VS Code but causes the synthesis call to attempt
too many tokens with a large prompt.

### Fix C: Set explicit KV cache limit in GenerationConfig
In `src/serve_model.py`, in the `_complete_chat` / `_stream_chat` functions:
```python
config = ov.GenerationConfig()
config.max_new_tokens = min(req.max_tokens, 512)  # hard cap
# Also try setting:
# config.max_length = 2048  # if supported
```

### Fix D: Truncate the synthesis context
In `src/chainlit_app.py`, before the Phase 2 synthesis call, truncate history
to the last N messages to keep context within safe bounds:
```python
# Keep only last 6 messages for synthesis (system + user + tool intent + 2 tool results + synthesis system)
synthesis_messages = synthesis_messages[-6:] if len(synthesis_messages) > 6 else synthesis_messages
```

---

## Reporting Back

After running the diagnostics, report:
1. Output of Step 1 — did `/tmp/serve_verbose.log` show OpenCL -14 or another error?
2. Output of Step 2 — how many prompt tokens at crash point?
3. Output of Step 3 — what does `generation_config.json` say for max_length?
4. Which of Tests 1/2/3 in Step 4 crashed first?
5. Which Fix (A/B/C/D) resolved the crash?

With that information we can apply the correct permanent fix.
