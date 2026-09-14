#!/usr/bin/env bash
# =============================================================================
# Trading Income Project — Dual-Port OpenVINO GenAI Launcher
# =============================================================================
# Starts both model servers concurrently in the project .venv.
#
# Pair 1: Qwen3.8-27B-int4    + Phi-4-mini-int4   (~19GB)
# Pair 2: Qwen3.6-35B-A3B-int4 + Mistral-Nemo-12B (~25GB)
# Pair 3: Qwen3.6-35B-A3B-int4 + LFM2.5-8B-A1B    (~23GB)
#
# Usage:
#   ./launch_models.sh                  # use pair from .model_pair
#   ./launch_models.sh --pair 1         # override pair selection
#   ./launch_models.sh --with-webui     # launch Open WebUI on port 8080
#   ./launch_models.sh --stop           # stop all servers
#   ./launch_models.sh --status         # check health
#   ./launch_models.sh --logs           # tail log files
#   ./launch_models.sh --restart        # stop then start
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
ACTION="start"

for arg in "$@"; do
    case "$arg" in
        --pair)           shift; PAIR_OVERRIDE="${1:-}"; shift ;;
        --pair=*)         PAIR_OVERRIDE="${arg#--pair=}" ;;
        --with-webui)     WITH_WEBUI=true ;;
        --stop|stop)      ACTION="stop" ;;
        --status|status)  ACTION="status" ;;
        --logs|logs)      ACTION="logs" ;;
        --restart|restart) ACTION="restart" ;;
    esac
done

declare -A DEFAULT_P1_DIR=([1]='qwen3.8-27b-int4'  [2]='qwen3.6-35b-a3b'  [3]='qwen3.6-35b-a3b')
declare -A DEFAULT_P1_ID=( [1]='qwen3.8:27b'       [2]='qwen3.6:35b-a3b'  [3]='qwen3.6:35b-a3b')
declare -A DEFAULT_P2_DIR=([1]='phi-4-mini-int4'   [2]='mistral-nemo-12b' [3]='lfm2.5-8b-a1b')
declare -A DEFAULT_P2_ID=( [1]='phi-4-mini:int4'   [2]='mistral-nemo:12b' [3]='lfm2.5:8b')
declare -A PAIR_NOTE=(
    [1]='Qwen3.8-27B-int4 (MTP built-in) + Phi-4-mini (~19GB)'
    [2]='Qwen3.6-35B-A3B-int4 MoE + Mistral-Nemo-12B (~25GB)'
    [3]='Qwen3.6-35B-A3B-int4 MoE + LFM2.5-8B-A1B (~23GB)'
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
        pkill -f "serve_model.py" 2>/dev/null || true
        pkill -f "open-webui" 2>/dev/null || true
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
[[ -d "$MODEL_P1" ]]     || { fail "Phase 1 model not found: $MODEL_P1\nRun: ./prepare_host.sh --pair $PAIR --models-only"; exit 1; }
[[ -d "$MODEL_P2" ]]     || { fail "Phase 2 model not found: $MODEL_P2\nRun: ./prepare_host.sh --pair $PAIR --models-only"; exit 1; }

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
info "Starting Port $P2_PORT ($ID_P2)..."
"$VENV/bin/python3" "$SERVE_SCRIPT" \
    --model-path "$MODEL_P2" --model-id "$ID_P2" \
    --port "$P2_PORT" --device GPU \
    > "$P2_LOG" 2>&1 &
P2_PID_VAL=$!
echo "$P2_PID_VAL" > "$P2_PID"

# ── Launch WebUI (Optional) ───────────────────────────────────────────────────
if $WITH_WEBUI; then
    if [[ -x "$VENV/bin/open-webui" ]]; then
        stop_server "$WEBUI_PID" "existing Open WebUI"
        free_port "$WEBUI_PORT"
        info "Starting Open WebUI on port $WEBUI_PORT..."
        export OPENAI_API_BASE_URLS="http://127.0.0.1:$P1_PORT/v1;http://127.0.0.1:$P2_PORT/v1"
        export WEBUI_PORT="$WEBUI_PORT"
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
    if ! kill -0 "$P2_PID_VAL" 2>/dev/null; then
        echo ""
        fail "Phase 2 server on port $P2_PORT crashed on startup. Error log ($P2_LOG):"
        tail -n 12 "$P2_LOG" | while IFS= read -r l; do echo -e "    ${DIM}$l${RESET}"; done
        exit 1
    fi

    P1_UP=false
    P2_UP=false
    curl -sf "http://127.0.0.1:$P1_PORT/health" >/dev/null 2>&1 && P1_UP=true
    curl -sf "http://127.0.0.1:$P2_PORT/health" >/dev/null 2>&1 && P2_UP=true

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
if $WITH_WEBUI && [[ -x "$VENV/bin/open-webui" ]]; then
    echo -e "  ${YELLOW}Chat WebUI:${RESET}            http://localhost:$WEBUI_PORT"
fi
echo ""
echo -e "  Status:  ${CYAN}./launch_models.sh --status${RESET}"
echo -e "  Stop:    ${CYAN}./launch_models.sh --stop${RESET}"
echo -e "  Logs:    ${CYAN}./launch_models.sh --logs${RESET}"
echo ""
