#!/usr/bin/env bash
# =============================================================================
# Trading Income Project — Dual-Port OpenVINO GenAI Launcher
# =============================================================================
# Starts both model servers concurrently in the project .venv.
#
# Pair 1: Qwen3.8-27B-int4    + Phi-4-mini-int4   (~19GB)
# Pair 2: Qwen3.6-35B-A3B-int4 + Mistral-7B-v0.1-int4 (~21GB)
# Pair 3: Qwen3.6-35B-A3B-int4 + LFM2.5-8B-A1B    (~23GB)
#
# Usage:
#   ./launch_models.sh                    # use pair from .model_pair
#   ./launch_models.sh --pair 1           # override pair selection
#   ./launch_models.sh --with-chainlit    # launch Chainlit chat UI on port 8080 (primary)
#   ./launch_models.sh --with-chat        # launch standalone trading_chat.html on port 3000
#   ./launch_models.sh --with-webui       # launch Open WebUI on port 8080 (legacy)
#   ./launch_models.sh --stop             # stop all servers
#   ./launch_models.sh --status           # check health
#   ./launch_models.sh --logs             # tail log files
#   ./launch_models.sh --restart          # stop then start
# =============================================================================
set -euo pipefail

BOLD="\033[1m"; DIM="\033[2m"
GREEN="\033[32m"; YELLOW="\033[33m"; CYAN="\033[36m"; RED="\033[31m"
RESET="\033[0m"

ok()   { echo -e "  ${GREEN}✓${RESET}  $*"; }
warn() { echo -e "  ${YELLOW}⚠${RESET}  $*"; }
fail() { echo -e "  ${RED}✗${RESET}  $*"; }
info() { echo -e "  ${DIM}    $*${RESET}"; }

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

MODELS_DIR="${MODELS_DIR:-$HOME/models}"
VENV="$ROOT_DIR/.venv"
SERVE_SCRIPT="$ROOT_DIR/src/serve_model.py"
LOG_DIR="$ROOT_DIR/LOGS"
PID_DIR="$ROOT_DIR/LOGS/pids"
mkdir -p "$LOG_DIR" "$PID_DIR"

P1_PORT=8000
P2_PORT=8001
WEBUI_PORT=8080

PAIR_OVERRIDE=""
WITH_WEBUI=false
WITH_CHAT=false
WITH_CHAINLIT=false
CHAT_PORT=3000
CHAINLIT_PORT=8080
ACTION="start"

for arg in "$@"; do
    case "$arg" in
        --pair)             shift; PAIR_OVERRIDE="${1:-}"; shift ;;
        --pair=*)           PAIR_OVERRIDE="${arg#--pair=}" ;;
        --with-webui)       WITH_WEBUI=true ;;
        --with-chat)        WITH_CHAT=true ;;
        --with-chainlit)    WITH_CHAINLIT=true ;;
        --stop|stop)        ACTION="stop" ;;
        --status|status)    ACTION="status" ;;
        --logs|logs)        ACTION="logs" ;;
        --restart|restart)  ACTION="restart" ;;
    esac
done

declare -A DEFAULT_P1_DIR=([1]='qwen3.8-27b-int4'  [2]='qwen3.6-35b-a3b'  [3]='qwen3.6-35b-a3b'  [4]='qwen2.5-coder-7b-int4')
declare -A DEFAULT_P1_ID=( [1]='qwen3.8:27b'       [2]='qwen3.6:35b-a3b'  [3]='qwen3.6:35b-a3b'  [4]='qwen2.5-coder:7b')
declare -A DEFAULT_P2_DIR=([1]='phi-4-mini-int4'   [2]='mistral-7b-v01-int4' [3]='lfm2.5-8b-a1b' [4]='qwen2.5-coder-1.5b-int4')
declare -A DEFAULT_P2_ID=( [1]='phi-4-mini:int4'   [2]='mistral-7b:int4'     [3]='lfm2.5:8b'     [4]='qwen2.5-coder:1.5b')
declare -A PAIR_NOTE=(
    [1]='Qwen3.8-27B-int4 (MTP built-in) + Phi-4-mini (~19GB)'
    [2]='Qwen3.6-35B-A3B-int4 MoE + Mistral-7B-v0.1-int4 (~21GB)'
    [3]='Qwen3.6-35B-A3B-int4 MoE + LFM2.5-8B-A1B (~23GB)'
    [4]='Qwen2.5-Coder-7B-int4 + Qwen2.5-Coder-1.5B-int4 (~6GB) [15-20 t/s]'
)

PAIR="$PAIR_OVERRIDE"
if [[ -z "$PAIR" ]] && [[ -f "$ROOT_DIR/.model_pair" ]]; then
    PAIR=$(grep '^ACTIVE_PAIR=' "$ROOT_DIR/.model_pair" | cut -d= -f2 || echo "1")
fi
PAIR="${PAIR:-1}"

P1_DIR_VAL=""
P1_ID_VAL=""
P2_DIR_VAL=""
P2_ID_VAL=""

if [[ -f "$ROOT_DIR/.model_pair" ]] && [[ -z "$PAIR_OVERRIDE" ]]; then
    P1_DIR_VAL=$(grep '^P1_MODEL_DIR=' "$ROOT_DIR/.model_pair" | cut -d= -f2 || echo "")
    P1_ID_VAL=$(grep '^P1_MODEL_ID=' "$ROOT_DIR/.model_pair" | cut -d= -f2 || echo "")
    P2_DIR_VAL=$(grep '^P2_MODEL_DIR=' "$ROOT_DIR/.model_pair" | cut -d= -f2 || echo "")
    P2_ID_VAL=$(grep '^P2_MODEL_ID=' "$ROOT_DIR/.model_pair" | cut -d= -f2 || echo "")
fi

MODEL_P1="$MODELS_DIR/${P1_DIR_VAL:-${DEFAULT_P1_DIR[$PAIR]}}"
ID_P1="${P1_ID_VAL:-${DEFAULT_P1_ID[$PAIR]}}"
MODEL_P2="$MODELS_DIR/${P2_DIR_VAL:-${DEFAULT_P2_DIR[$PAIR]}}"
ID_P2="${P2_ID_VAL:-${DEFAULT_P2_ID[$PAIR]}}"

P1_LOG="$LOG_DIR/model_8000.log"
P2_LOG="$LOG_DIR/model_8001.log"
WEBUI_LOG="$LOG_DIR/webui_8080.log"
P1_PID="$PID_DIR/model_8000.pid"
P2_PID="$PID_DIR/model_8001.pid"
WEBUI_PID="$PID_DIR/webui.pid"

THINK_LOG=""
if [[ "${ID_P1}" == *deepseek* ]] || [[ "${ID_P1}" == *qwq* ]]; then
    THINK_LOG="$LOG_DIR/thinking_8000.log"
fi

DRAFT_MODEL_PATH=""
if [[ -f "$ROOT_DIR/.model_pair" ]]; then
    DRAFT_DIR=$(grep "^DRAFT_MODEL_DIR=" "$ROOT_DIR/.model_pair" | cut -d= -f2 || echo "")
    if [[ -n "$DRAFT_DIR" ]] && [[ -d "$MODELS_DIR/$DRAFT_DIR" ]]; then
        DRAFT_MODEL_PATH="$MODELS_DIR/$DRAFT_DIR"
    fi
fi

# ── Helpers ───────────────────────────────────────────────────────────────────
server_pid() { [[ -f "$1" ]] && cat "$1" 2>/dev/null || echo ""; }
server_alive() { local p; p=$(server_pid "$1"); [[ -n "$p" ]] && kill -0 "$p" 2>/dev/null; }

stop_server() {
    local pid_file="$1" label="$2"
    if server_alive "$pid_file"; then
        local pid; pid=$(server_pid "$pid_file")
        kill "$pid" 2>/dev/null || true; sleep 0.5
        kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
        rm -f "$pid_file"
        ok "Stopped $label (PID $pid)"
    elif [[ -f "$pid_file" ]]; then
        rm -f "$pid_file"
    fi
}

free_port() {
    local port="$1"
    # Kill any lingering process holding the target port
    fuser -k "${port}/tcp" >/dev/null 2>&1 || true
}

# ── Actions ───────────────────────────────────────────────────────────────────
case "$ACTION" in
    stop)
        echo -e "\n${BOLD}Stopping model servers...${RESET}"
        stop_server "$P1_PID" "Port $P1_PORT"
        stop_server "$P2_PID" "Port $P2_PORT"
        stop_server "$WEBUI_PID" "Open WebUI"
        stop_server "$PID_DIR/chat_ui.pid" "Chat UI"
        stop_server "$PID_DIR/chainlit.pid" "Chainlit"
        pkill -f "serve_model.py" 2>/dev/null || true
        pkill -f "open-webui"     2>/dev/null || true
        pkill -f "chainlit"       2>/dev/null || true
        fuser -k "${CHAT_PORT}/tcp" 2>/dev/null || true
        ok "All servers stopped"; exit 0 ;;
    status)
        echo -e "\n${BOLD}Model Server Status (Pair $PAIR: ${PAIR_NOTE[$PAIR]})${RESET}"
        for entry in "$P1_PORT:$P1_PID:$ID_P1" "$P2_PORT:$P2_PID:$ID_P2"; do
            IFS=: read -r port pid_file model_id <<< "$entry"
            if server_alive "$pid_file"; then
                pid=$(server_pid "$pid_file")
                curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1 \
                    && ok "Port $port — $model_id (PID $pid) — READY" \
                    || warn "Port $port — $model_id (PID $pid) — INITIALIZING"
            else
                warn "Port $port — $model_id — NOT RUNNING"
            fi
        done
        if server_alive "$WEBUI_PID"; then
            ok "Port $WEBUI_PORT — Open WebUI (PID $(server_pid "$WEBUI_PID")) — READY"
        fi
        if server_alive "$PID_DIR/chainlit.pid"; then
            ok "Port $CHAINLIT_PORT — Chainlit (PID $(server_pid "$PID_DIR/chainlit.pid")) — READY"
        fi
        exit 0 ;;
    logs)
        tail -f "$P1_LOG" "$P2_LOG"; exit 0 ;;
    restart)
        "$0" --stop; sleep 1; exec "$0" ;;
esac

# ── Pre-flight ────────────────────────────────────────────────────────────────
echo -e "\n${BOLD}OpenVINO GenAI — Launching Pair $PAIR${RESET}"
info "${PAIR_NOTE[$PAIR]}\n"

[[ -d "$VENV" ]]         || { fail ".venv not found. Run ./prepare_host.sh"; exit 1; }
[[ -f "$SERVE_SCRIPT" ]] || { fail "src/serve_model.py not found"; exit 1; }
[[ -d "$MODEL_P1" ]]     || { fail "Phase 1 model not found: $MODEL_P1\nRun: ./setup.sh --pair $PAIR --models-only"; exit 1; }
if [[ ! -d "$MODEL_P2" ]]; then
    warn "Phase 2 model not found: $MODEL_P2"
    warn "Run: ./setup.sh --pair $PAIR --models-only  to download it"
    warn "Continuing with Phase 1 only..."
    SKIP_P2=true
else
    SKIP_P2=false
fi

stop_server "$P1_PID" "existing port $P1_PORT"
stop_server "$P2_PID" "existing port $P2_PORT"
free_port "$P1_PORT"
free_port "$P2_PORT"

# ── Launch Phase 1 ────────────────────────────────────────────────────────────
THINK_ARGS=()
[[ -n "$THINK_LOG" ]] && THINK_ARGS=(--think-log "$THINK_LOG")
DRAFT_ARGS=()
[[ -n "$DRAFT_MODEL_PATH" ]] && DRAFT_ARGS=(--draft-model-path "$DRAFT_MODEL_PATH")

info "Starting Port $P1_PORT ($ID_P1)..."

"$VENV/bin/python3" "$SERVE_SCRIPT" \
    --model-path "$MODEL_P1" --model-id "$ID_P1" \
    --port "$P1_PORT" --device GPU "${THINK_ARGS[@]}" "${DRAFT_ARGS[@]}" \
    > "$P1_LOG" 2>&1 &
P1_PID_VAL=$!
echo "$P1_PID_VAL" > "$P1_PID"

# ── Launch Phase 2 ────────────────────────────────────────────────────────────
if ! $SKIP_P2; then
    info "Starting Port $P2_PORT ($ID_P2)..."
    "$VENV/bin/python3" "$SERVE_SCRIPT" \
        --model-path "$MODEL_P2" --model-id "$ID_P2" \
        --port "$P2_PORT" --device GPU \
        > "$P2_LOG" 2>&1 &
    P2_PID_VAL=$!
    echo "$P2_PID_VAL" > "$P2_PID"
fi

# ── Optional: Chainlit UI (primary chat interface, port 8080) ─────────────────
if $WITH_CHAINLIT; then
    if [[ -f "$ROOT_DIR/src/chainlit_app.py" ]] && [[ -x "$VENV/bin/chainlit" ]]; then
        stop_server "$PID_DIR/chainlit.pid" "existing Chainlit"
        free_port "$CHAINLIT_PORT"
        info "Starting Chainlit UI on port $CHAINLIT_PORT..."
        cd "$ROOT_DIR"
        BACKEND_URL="http://127.0.0.1:$P1_PORT" \
        MODEL_ID="$ID_P1" \
        "$VENV/bin/chainlit" run src/chainlit_app.py \
            --host 0.0.0.0 --port "$CHAINLIT_PORT" --headless \
            > "$LOG_DIR/chainlit_8080.log" 2>&1 &
        echo "$!" > "$PID_DIR/chainlit.pid"
        ok "Chainlit UI launched — http://localhost:${CHAINLIT_PORT} (PID $!)"
        cd - >/dev/null
    else
        warn "src/chainlit_app.py or chainlit binary not found — skipping"
        warn "Install with: pip install chainlit"
    fi
fi

# ── Optional: Standalone Chat UI (fallback, port 3000) ────────────────────────
if $WITH_CHAT; then
    if [[ -f "$ROOT_DIR/trading_chat.html" ]]; then
        fuser -k "${CHAT_PORT}/tcp" 2>/dev/null || true
        sleep 1
        cd "$ROOT_DIR"
        python3 -m http.server "$CHAT_PORT" > "$LOG_DIR/chat_ui.log" 2>&1 &
        echo "$!" > "$PID_DIR/chat_ui.pid"
        ok "Chat UI launched — http://localhost:${CHAT_PORT}/trading_chat.html (PID $!)"
        cd - >/dev/null
    else
        warn "trading_chat.html not found — skipping standalone chat UI"
    fi
fi

# ── Optional: Open WebUI (legacy, port 8080) ──────────────────────────────────
if $WITH_WEBUI; then
    if [[ -x "$VENV/bin/open-webui" ]]; then
        stop_server "$WEBUI_PID" "existing Open WebUI"
        free_port "$WEBUI_PORT"
        info "Starting Open WebUI on port $WEBUI_PORT..."
        export OPENAI_API_BASE_URLS="http://127.0.0.1:$P1_PORT/v1;http://127.0.0.1:$P2_PORT/v1"
        export WEBUI_PORT="$WEBUI_PORT"
        export WEBUI_SECRET_KEY="$(cat "$ROOT_DIR/.webui_secret_key" 2>/dev/null || echo 'trading-income-local-secret-key')"
        export ENABLE_OLLAMA_API="false"
        export ENABLE_TITLE_GENERATION="false"
        export ENABLE_TAGS_GENERATION="false"
        export ENABLE_FOLLOW_UP_GENERATION="false"
        export ENABLE_ORJSON="false"
        export ENABLE_WEBSOCKET_SUPPORT="false"
        export AIOHTTP_CLIENT_TIMEOUT="600"
        export AIOHTTP_CLIENT_STREAM_IDLE_TIMEOUT="600"
        export AIOHTTP_CLIENT_TIMEOUT_MODEL_LIST="10"
        "$VENV/bin/open-webui" serve > "$WEBUI_LOG" 2>&1 &
        echo "$!" > "$WEBUI_PID"
        ok "Open WebUI launched (PID $!)"
    else
        warn "Open WebUI binary not found in .venv — skipping WebUI"
    fi
fi

# ── Health Poll with Fast-Fail ────────────────────────────────────────────────
echo ""
info "Loading weights into Arc iGPU memory..."
info "Cold start compiles Level Zero blobs (~45s). Subsequent starts: <5s."
echo ""

MAX_WAIT=180
ELAPSED=0

while (( ELAPSED < MAX_WAIT )); do
    # Fast-fail if Python crashed rather than waiting 3 minutes
    if ! kill -0 "$P1_PID_VAL" 2>/dev/null; then
        echo ""
        fail "Phase 1 server on port $P1_PORT crashed on startup. Error log ($P1_LOG):"
        tail -n 12 "$P1_LOG" | while IFS= read -r l; do echo -e "    ${DIM}$l${RESET}"; done
        exit 1
    fi
    if ! $SKIP_P2 && ! kill -0 "$P2_PID_VAL" 2>/dev/null; then
        echo ""
        fail "Phase 2 server on port $P2_PORT crashed on startup. Error log ($P2_LOG):"
        tail -n 12 "$P2_LOG" | while IFS= read -r l; do echo -e "    ${DIM}$l${RESET}"; done
        exit 1
    fi

    P1_UP=false
    P2_UP=false
    curl -sf "http://127.0.0.1:$P1_PORT/health" >/dev/null 2>&1 && P1_UP=true
    $SKIP_P2 && P2_UP=true || { curl -sf "http://127.0.0.1:$P2_PORT/health" >/dev/null 2>&1 && P2_UP=true; }

    if $P1_UP && $P2_UP; then
        break
    fi

    sleep 2
    (( ELAPSED += 2 ))
    printf "\r  ${DIM}%ds — Port 8000: %s  |  Port 8001: %s${RESET}   " \
        "$ELAPSED" \
        "$($P1_UP && echo READY || echo loading...)" \
        "$($P2_UP && echo READY || echo loading...)"
done

printf "\r\033[K"

# ── Summary ───────────────────────────────────────────────────────────────────
echo -e "${BOLD}${CYAN}══════════════════════════════════════════════════════════════${RESET}"
echo -e "${BOLD}${GREEN} OpenVINO GenAI — Pair $PAIR Active & Ready${RESET}"
echo -e "${BOLD}${CYAN}══════════════════════════════════════════════════════════════${RESET}"
echo ""
echo -e "  ${YELLOW}Phase 1 (Port $P1_PORT):${RESET}  $ID_P1"
echo -e "  ${YELLOW}Phase 2 (Port $P2_PORT):${RESET}  $ID_P2"
echo ""
echo -e "  ${DIM}Both models resident in Arc iGPU VRAM simultaneously${RESET}"
echo ""
if $WITH_CHAINLIT; then
    echo -e "  ${YELLOW}Chainlit Chat UI:${RESET}      http://localhost:${CHAINLIT_PORT}"
fi
if $WITH_CHAT; then
    echo -e "  ${YELLOW}Standalone Chat UI:${RESET}    http://localhost:${CHAT_PORT}/trading_chat.html"
fi
if $WITH_WEBUI && [[ -x "$VENV/bin/open-webui" ]]; then
    echo -e "  ${YELLOW}Open WebUI (legacy):${RESET}   http://localhost:$WEBUI_PORT"
fi
echo ""
echo -e "  Status:  ${CYAN}./launch_models.sh --status${RESET}"
echo -e "  Stop:    ${CYAN}./launch_models.sh --stop${RESET}"
echo -e "  Logs:    ${CYAN}./launch_models.sh --logs${RESET}"
echo ""
