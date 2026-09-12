#!/usr/bin/env bash
# =============================================================================
# Trading Income Project — Dual-Port OpenVINO GenAI Launcher
# =============================================================================
# Starts both model servers concurrently in the project .venv.
# Reads active pair from .model_pair (written by prepare_host.sh) or accepts
# --pair N override.
#
# Pair 1: DeepSeek-R1-32B   + Mistral-Nemo-12B   (~26.3GB, ~32GB headroom)
# Pair 2: Qwen3-30B-A3B     + Mistral-Nemo-12B   (~25.2GB, ~33GB headroom)
# Pair 3: Qwen2.5-32B       + Mistral-Small-24B  (~33.2GB, ~25GB headroom)
#
# Usage:
#   ./launch_models.sh              # use pair from .model_pair
#   ./launch_models.sh --pair 1     # override pair selection
#   ./launch_models.sh --stop       # stop both servers
#   ./launch_models.sh --status     # check health
#   ./launch_models.sh --logs       # tail both log files
#   ./launch_models.sh --restart    # stop then start
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

PAIR_OVERRIDE=""
for arg in "$@"; do
    case "$arg" in
        --pair)  shift; PAIR_OVERRIDE="${1:-}"; shift ;;
        --pair=*) PAIR_OVERRIDE="${arg#--pair=}" ;;
    esac
done

# ── Load pair config ──────────────────────────────────────────────────────────
declare -A P1_DIR=([1]='deepseek-r1-32b' [2]='qwen3-30b-a3b'  [3]='qwen2.5-32b')
declare -A P1_ID=( [1]='deepseek-r1:32b' [2]='qwen3:30b-a3b'  [3]='qwen2.5:32b')
declare -A P2_DIR=([1]='mistral-nemo-12b' [2]='mistral-nemo-12b' [3]='mistral-small-24b')
declare -A P2_ID=( [1]='mistral-nemo:12b' [2]='mistral-nemo:12b' [3]='mistral-small:24b')
declare -A PAIR_NOTE=(
    [1]='DeepSeek-R1-32B (thinking) + Mistral-Nemo-12B (~26.3GB)'
    [2]='Qwen3-30B-A3B MoE + Mistral-Nemo-12B (~25.2GB, 30+ tok/s)'
    [3]='Qwen2.5-32B + Mistral-Small-24B (~33.2GB, peak tool precision)'
)

PAIR="$PAIR_OVERRIDE"
if [[ -z "$PAIR" ]] && [[ -f "$ROOT_DIR/.model_pair" ]]; then
    PAIR=$(grep '^ACTIVE_PAIR=' "$ROOT_DIR/.model_pair" | cut -d= -f2)
fi
PAIR="${PAIR:-1}"

MODEL_P1="$MODELS_DIR/${P1_DIR[$PAIR]}"
ID_P1="${P1_ID[$PAIR]}"
MODEL_P2="$MODELS_DIR/${P2_DIR[$PAIR]}"
ID_P2="${P2_ID[$PAIR]}"

P1_LOG="$LOG_DIR/model_8000.log"
P2_LOG="$LOG_DIR/model_8001.log"
P1_PID="$PID_DIR/model_8000.pid"
P2_PID="$PID_DIR/model_8001.pid"

# Thinking log for DeepSeek-R1 / QwQ models
# NOTE: use if/then to avoid || vs && precedence trap in bash
THINK_LOG=""
if [[ "${ID_P1}" == *deepseek* ]] || [[ "${ID_P1}" == *qwq* ]]; then
    THINK_LOG="$LOG_DIR/thinking_8000.log"
fi

# Speculative decoding draft model (optional — downloaded by prepare_host.sh)
DRAFT_MODEL_DIR=""
if [[ -f "$ROOT_DIR/.model_pair" ]]; then
    DRAFT_MODEL_DIR=$(grep "^DRAFT_MODEL_DIR=" "$ROOT_DIR/.model_pair" | cut -d= -f2 || echo "")
fi
DRAFT_MODEL_PATH=""
if [[ -n "$DRAFT_MODEL_DIR" ]] && [[ -d "$MODELS_DIR/$DRAFT_MODEL_DIR" ]]; then
    DRAFT_MODEL_PATH="$MODELS_DIR/$DRAFT_MODEL_DIR"
fi

# ── Helpers ───────────────────────────────────────────────────────────────────
server_pid() { [[ -f "$1" ]] && cat "$1" 2>/dev/null || echo ""; }
server_alive() { local p; p=$(server_pid "$1"); [[ -n "$p" ]] && kill -0 "$p" 2>/dev/null; }
stop_server() {
    local pid_file="$1" label="$2"
    if server_alive "$pid_file"; then
        local pid; pid=$(server_pid "$pid_file")
        kill "$pid" 2>/dev/null || true; sleep 1
        kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
        rm -f "$pid_file"
        ok "Stopped $label (PID $pid)"
    fi
}

# ── CLI dispatch ──────────────────────────────────────────────────────────────
case "${1:-start}" in
    --stop|stop)
        echo -e "\n${BOLD}Stopping model servers...${RESET}"
        stop_server "$P1_PID" "Port $P1_PORT"
        stop_server "$P2_PID" "Port $P2_PORT"
        pkill -f "serve_model.py" 2>/dev/null || true
        ok "All servers stopped"; exit 0 ;;
    --status|status)
        echo -e "\n${BOLD}Model Server Status  (Pair $PAIR: ${PAIR_NOTE[$PAIR]})${RESET}"
        for entry in "$P1_PORT:$P1_PID:$ID_P1" "$P2_PORT:$P2_PID:$ID_P2"; do
            IFS=: read -r port pid_file model_id <<< "$entry"
            if server_alive "$pid_file"; then
                pid=$(server_pid "$pid_file")
                curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1 \
                    && ok "Port $port — $model_id (PID $pid) — READY" \
                    || warn "Port $port — $model_id (PID $pid) — LOADING"
            else
                warn "Port $port — $model_id — NOT RUNNING"
            fi
        done
        exit 0 ;;
    --logs|logs)
        tail -f "$P1_LOG" "$P2_LOG"; exit 0 ;;
    --restart|restart)
        "$0" --stop; sleep 2; exec "$0" ;;
esac

# ── Pre-flight ────────────────────────────────────────────────────────────────
echo -e "\n${BOLD}OpenVINO GenAI — Pair $PAIR: ${PAIR_NOTE[$PAIR]}${RESET}\n"

[[ -d "$VENV" ]]         || { warn ".venv not found. Run ./prepare_host.sh"; exit 1; }
[[ -f "$SERVE_SCRIPT" ]] || { warn "src/serve_model.py not found"; exit 1; }
[[ -d "$MODEL_P1" ]]     || { warn "Phase 1 model not found: $MODEL_P1"; warn "Run: ./prepare_host.sh --pair $PAIR --models-only"; exit 1; }
[[ -d "$MODEL_P2" ]]     || { warn "Phase 2 model not found: $MODEL_P2"; warn "Run: ./prepare_host.sh --pair $PAIR --models-only"; exit 1; }

stop_server "$P1_PID" "existing port $P1_PORT"
stop_server "$P2_PID" "existing port $P2_PORT"
pkill -f "serve_model.py" 2>/dev/null || true; sleep 1

# ── Launch Phase 1 ────────────────────────────────────────────────────────────
THINK_ARGS=()
[[ -n "$THINK_LOG" ]] && THINK_ARGS=(--think-log "$THINK_LOG")
info "Starting Port $P1_PORT: $ID_P1"
[[ -n "$THINK_LOG" ]] && info "Thinking tokens logged to $THINK_LOG"
DRAFT_ARGS=()
[[ -n "$DRAFT_MODEL_PATH" ]] && DRAFT_ARGS=(--draft-model-path "$DRAFT_MODEL_PATH")
[[ -n "$DRAFT_MODEL_PATH" ]] && info "Speculative decoding: $DRAFT_MODEL_PATH"
"$VENV/bin/python3" "$SERVE_SCRIPT" \
    --model-path "$MODEL_P1" --model-id "$ID_P1" \
    --port "$P1_PORT" --device GPU "${THINK_ARGS[@]}" "${DRAFT_ARGS[@]}" \
    > "$P1_LOG" 2>&1 &
echo "$!" > "$P1_PID"
ok "Port $P1_PORT launched (PID $!)"

# ── Launch Phase 2 ────────────────────────────────────────────────────────────
info "Starting Port $P2_PORT: $ID_P2"
"$VENV/bin/python3" "$SERVE_SCRIPT" \
    --model-path "$MODEL_P2" --model-id "$ID_P2" \
    --port "$P2_PORT" --device GPU \
    > "$P2_LOG" 2>&1 &
echo "$!" > "$P2_PID"
ok "Port $P2_PORT launched (PID $!)"

# ── Health poll ───────────────────────────────────────────────────────────────
echo ""
info "Waiting for models to load into VRAM..."
info "First run: ~45-90s (Level Zero kernel compilation). Subsequent: < 10s."
echo ""

MAX_WAIT=180; ELAPSED=0
while (( ELAPSED < MAX_WAIT )); do
    P1_UP=false; P2_UP=false
    curl -sf "http://127.0.0.1:$P1_PORT/health" >/dev/null 2>&1 && P1_UP=true
    curl -sf "http://127.0.0.1:$P2_PORT/health" >/dev/null 2>&1 && P2_UP=true
    $P1_UP && $P2_UP && break
    sleep 3; (( ELAPSED += 3 ))
    printf "\r  ${DIM}%ds — 8000: %s  8001: %s${RESET}   " \
        "$ELAPSED" \
        "$($P1_UP && echo READY || echo loading...)" \
        "$($P2_UP && echo READY || echo loading...)"
done; echo ""

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${CYAN}══════════════════════════════════════════════════════════${RESET}"
echo -e "${BOLD}${GREEN} OpenVINO GenAI — Pair $PAIR active${RESET}"
echo -e "${BOLD}${CYAN}══════════════════════════════════════════════════════════${RESET}"
echo ""
echo -e "  ${YELLOW}Phase 1 (Port $P1_PORT):${RESET}  $ID_P1"
echo -e "  ${YELLOW}Phase 2 (Port $P2_PORT):${RESET}  $ID_P2"
echo ""
echo -e "  ${DIM}Both models resident in Arc iGPU VRAM simultaneously${RESET}"
[[ -n "$THINK_LOG" ]] && echo -e "  ${DIM}Thinking tokens logged to $THINK_LOG${RESET}"
echo ""
echo -e "  Status:  ${CYAN}./launch_models.sh --status${RESET}"
echo -e "if $WITH_WEBUI; then
  echo -e "  ${YELLOW}Chat UI:${RESET}  http://localhost:8080"
fi
  Stop:    ${CYAN}./launch_models.sh --stop${RESET}"
echo -e "  Switch:  ${CYAN}./launch_models.sh --pair 2${RESET}"
echo ""
