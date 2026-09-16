#!/usr/bin/env bash
# =============================================================================
# debug_openwebui.sh — OpenWebUI streaming diagnostic (browser path)
# =============================================================================
set -euo pipefail

ROOT="$HOME/github/trading-income-project"
VENV="$ROOT/.venv"
PYTHON="$VENV/bin/python3"
MODEL_PATH="$HOME/models/qwen3.8-27b-int4"
MODEL_ID="qwen3.8:27b"
PORT_MODEL=8000
PORT_WEBUI=8080
EMAIL="adamsharpe1707@gmail.com"

TS=$(date +%Y%m%d_%H%M%S)
LOGDIR="/tmp/debug_$TS"
mkdir -p "$LOGDIR"

echo "========================================================"
echo "  OpenWebUI Streaming Debug (Browser Path)"
echo "  Logs: $LOGDIR"
echo "========================================================"

cleanup() {
    echo ""
    echo "========= SUMMARY ========="
    echo ""
    echo "--- Direct stream test ---"
    cat "$LOGDIR/direct_stream.txt" 2>/dev/null | grep -E "DONE|error|Hello|content" | head -5
    echo ""
    echo "--- API chat test (simulates browser) ---"
    cat "$LOGDIR/api_chat.txt" 2>/dev/null | grep -E "DONE|error|Hello|content|500|400" | head -10
    echo ""
    echo "--- WebUI errors during test ---"
    grep -E "ERROR|500|upstream|disconnect|generate_chat|title|tasks|Server" \
        "$LOGDIR/webui.log" 2>/dev/null | grep -v "8001\|11434\|ollama" | head -30
    echo ""
    echo "--- serve_model output ---"
    grep -E "STREAM|ERROR|error|Warning|Architecture|ready|patch" \
        "$LOGDIR/serve_model.log" 2>/dev/null | head -20
    echo ""
    echo "Full logs: $LOGDIR"
    kill "$SERVE_PID" 2>/dev/null || true
    kill "$WEBUI_PID" 2>/dev/null || true
}
trap cleanup EXIT

# ── Kill everything ──────────────────────────────────────────────────────────
echo "[1/7] Clearing ports..."
fuser -k "${PORT_MODEL}/tcp" 2>/dev/null || true
fuser -k "${PORT_WEBUI}/tcp" 2>/dev/null || true
pkill -f "serve_model.py" 2>/dev/null || true
pkill -f "open-webui" 2>/dev/null || true
sleep 3

# ── Start serve_model ────────────────────────────────────────────────────────
echo "[2/7] Starting serve_model.py..."
PYTHONUNBUFFERED=1 "$PYTHON" "$ROOT/src/serve_model.py" \
    --model-path "$MODEL_PATH" --model-id "$MODEL_ID" \
    --device GPU --port "$PORT_MODEL" \
    > "$LOGDIR/serve_model.log" 2>&1 &
SERVE_PID=$!

echo "      Waiting for model to load (max 5 min)..."
ELAPSED=0
while (( ELAPSED < 300 )); do
    if ! kill -0 "$SERVE_PID" 2>/dev/null; then
        echo "      CRASHED. Log:"; tail -10 "$LOGDIR/serve_model.log"; exit 1
    fi
    curl -sf "http://127.0.0.1:$PORT_MODEL/health" >/dev/null 2>&1 && break
    sleep 5; (( ELAPSED += 5 ))
    printf "\r      %ds..." "$ELAPSED"
done
echo "      ✓ serve_model ready (${ELAPSED}s)"

# ── Direct stream test ───────────────────────────────────────────────────────
echo "[3/7] Direct streaming test (bypasses OpenWebUI)..."
curl -s -N "http://127.0.0.1:$PORT_MODEL/v1/chat/completions" \
    -H "Content-Type: application/json" --max-time 30 \
    -d "{\"model\":\"$MODEL_ID\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hi.\"}],\"max_tokens\":10,\"stream\":true}" \
    > "$LOGDIR/direct_stream.txt" 2>&1
grep -q "DONE" "$LOGDIR/direct_stream.txt" \
    && echo "      ✓ Direct streaming WORKS" \
    || { echo "      ✗ Direct streaming FAILED"; cat "$LOGDIR/direct_stream.txt"; }

# ── Start OpenWebUI with all background tasks disabled ───────────────────────
echo "[4/7] Starting OpenWebUI..."
ENABLE_OLLAMA_API=false \
ENABLE_TITLE_GENERATION=false \
ENABLE_TAGS_GENERATION=false \
ENABLE_FOLLOW_UP_GENERATION=false \
ENABLE_ORJSON=false \
ENABLE_WEBSOCKET_SUPPORT=false \
WEBUI_SECRET_KEY=trading-income-debug-key \
AIOHTTP_CLIENT_TIMEOUT=600 \
AIOHTTP_CLIENT_STREAM_IDLE_TIMEOUT=600 \
OPENAI_API_BASE_URLS="http://127.0.0.1:$PORT_MODEL/v1" \
WEBUI_PORT="$PORT_WEBUI" \
"$VENV/bin/open-webui" serve > "$LOGDIR/webui.log" 2>&1 &
WEBUI_PID=$!

ELAPSED=0
while (( ELAPSED < 60 )); do
    curl -sf "http://127.0.0.1:$PORT_WEBUI/health" >/dev/null 2>&1 && break
    sleep 2; (( ELAPSED += 2 ))
done
echo "      ✓ OpenWebUI ready (${ELAPSED}s)"
sleep 5

# ── Auth ─────────────────────────────────────────────────────────────────────
echo "[5/7] Getting auth token..."
read -s -p "      Password for $EMAIL: " WEBUI_PASS; echo ""

TOKEN=$(curl -s -X POST "http://127.0.0.1:$PORT_WEBUI/api/v1/auths/signin" \
    -H "Content-Type: application/json" \
    --data-raw "{\"email\":\"$EMAIL\",\"password\":\"$WEBUI_PASS\"}" \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('token','FAILED'))" 2>/dev/null)
[[ "$TOKEN" == "FAILED" || -z "$TOKEN" ]] \
    && echo "      ✗ Auth failed" \
    || echo "      ✓ Token: ${TOKEN:0:20}..."

# Check models
echo "      Models visible to OpenWebUI:"
curl -s "http://127.0.0.1:$PORT_WEBUI/api/models" \
    -H "Authorization: Bearer $TOKEN" \
    | python3 -m json.tool 2>/dev/null | grep '"id"' | head -5 || echo "      (none)"

# ── Test via /api/chat/completions (browser path, non-streaming) ─────────────
echo "[6/7] Testing /api/chat/completions non-streaming (what browser uses first)..."
curl -s "http://127.0.0.1:$PORT_WEBUI/api/chat/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $TOKEN" \
    --max-time 60 \
    -d "{\"model\":\"$MODEL_ID\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hi.\"}],\"max_tokens\":10,\"stream\":false}" \
    > "$LOGDIR/api_nonstream.txt" 2>&1
echo "      Result: $(cat "$LOGDIR/api_nonstream.txt" | python3 -m json.tool 2>/dev/null | grep -E 'content|error' | head -3)"

# ── Test via /api/chat (the BROWSER endpoint, not /api/chat/completions) ─────
echo "[7/7] Testing /api/chat BROWSER endpoint with streaming..."
# This is what the browser actually calls - different from /api/chat/completions
CHAT_PAYLOAD=$(python3 -c "
import json
payload = {
    'model': '$MODEL_ID',
    'messages': [{'role': 'user', 'content': 'Say hi in 3 words.'}],
    'stream': True,
    'session_id': 'debug-test',
    'chat_id': 'debug-chat-001',
    'id': 'debug-msg-001'
}
print(json.dumps(payload))
")

curl -v -s -N "http://127.0.0.1:$PORT_WEBUI/api/chat" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $TOKEN" \
    -H "Accept: text/event-stream" \
    --max-time 60 \
    -d "$CHAT_PAYLOAD" \
    > "$LOGDIR/api_chat.txt" 2>&1

echo ""
echo "=== /api/chat response (first 60 lines) ==="
head -60 "$LOGDIR/api_chat.txt"

echo ""
echo "=== WebUI log (all non-GET entries during test) ==="
grep -vE "^2.*\" (GET|OPTIONS)" "$LOGDIR/webui.log" | tail -40

echo ""
echo "Press Ctrl+C for summary."
# Keep running so WebUI stays up for manual browser test
echo ""
echo ">>> OpenWebUI is at http://localhost:$PORT_WEBUI — send a prompt in browser now <<<"
echo ">>> Watching WebUI log for browser request errors... (Ctrl+C to stop) <<<"
tail -f "$LOGDIR/webui.log" | grep -vE "GET|304|version|sentence|BertModel|LOAD|Loading"
