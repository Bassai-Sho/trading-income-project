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
#   ./launch_models.sh --debug            # verbose logging everywhere: serve_model.py
#                                          #   tool-call-parsing details + CHAINLIT_DEBUG=1
#                                          #   for chainlit_app.py's own logger. Off by
#                                          #   default, noisy, for active debugging only.
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
DEBUG_LOGGING=false
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
        --debug)            DEBUG_LOGGING=true ;;
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
        # FIXED: was `exec "$0"` with no args, silently dropping --pair/
        # --with-chainlit/--debug/etc. on every restart. Reconstructed
        # explicitly from the already-parsed state below rather than
        # replaying raw "$@" — that would both loop forever (still contains
        # --restart) and lose --pair's value (already consumed by `shift`
        # above during arg parsing).
        RESTART_ARGS=()
        [[ -n "$PAIR_OVERRIDE" ]] && RESTART_ARGS+=(--pair "$PAIR_OVERRIDE")
        $WITH_CHAINLIT && RESTART_ARGS+=(--with-chainlit)
        $WITH_CHAT     && RESTART_ARGS+=(--with-chat)
        $WITH_WEBUI    && RESTART_ARGS+=(--with-webui)
        $DEBUG_LOGGING && RESTART_ARGS+=(--debug)
        "$0" --stop; sleep 1; exec "$0" "${RESTART_ARGS[@]}" ;;
esac

# ── Pre-flight ────────────────────────────────────────────────────────────────
echo -e "\n${BOLD}OpenVINO GenAI — Launching Pair $PAIR${RESET}"
info "${PAIR_NOTE[$PAIR]}\n"

[[ -d "$VENV" ]]         || { fail ".venv not found. Run ./setup.sh"; exit 1; }
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

# ── GPU pre-flight ────────────────────────────────────────────────────────────
# Fail fast (~2s) with the real cause instead of starting servers that crash on
# startup with "[GPU] Can't get PERFORMANCE_HINT ... no supported devices found".
# Bypass with SKIP_GPU_CHECK=1.
if [[ "${SKIP_GPU_CHECK:-0}" != "1" ]]; then
    if "$VENV/bin/python3" -c "import sys, openvino as ov; sys.exit(0 if 'GPU' in ov.Core().available_devices else 1)" >/dev/null 2>&1; then
        ok "OpenVINO sees the Arc iGPU"
    else
        fail "OpenVINO cannot see the GPU -- the model servers would crash immediately."
        if [[ -f "$ROOT_DIR/gpu.sh" ]]; then
            bash "$ROOT_DIR/gpu.sh" status 2>&1 | sed 's/^/    /' || true
        fi
        info "Fix:    bash gpu.sh install   (installs the Intel OpenCL runtime properly and holds it)"
        info "Other:  bash gpu.sh diag      (render group / kernel driver / wedged GPU)"
        exit 1
    fi
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
DEBUG_ARGS=()
$DEBUG_LOGGING && DEBUG_ARGS=(--debug)

info "Starting Port $P1_PORT ($ID_P1)..."
$DEBUG_LOGGING && info "Debug logging enabled — verbose per-request output in $P1_LOG"

# `python3 -u` (unbuffered stdout/stderr): without it, Python fully
# block-buffers output when it's redirected to a file rather than a
# terminal — print() calls can sit in memory indefinitely instead of
# reaching the log file, making live log-tailing and post-hoc debugging
# unreliable for anything that isn't explicitly flush=True.
"$VENV/bin/python3" -u "$SERVE_SCRIPT" \
    --model-path "$MODEL_P1" --model-id "$ID_P1" \
    --port "$P1_PORT" --device GPU "${THINK_ARGS[@]}" "${DRAFT_ARGS[@]}" "${DEBUG_ARGS[@]}" \
    > "$P1_LOG" 2>&1 &
P1_PID_VAL=$!
echo "$P1_PID_VAL" > "$P1_PID"

# ── Launch Phase 2 ────────────────────────────────────────────────────────────
if ! $SKIP_P2; then
    info "Starting Port $P2_PORT ($ID_P2)..."
    "$VENV/bin/python3" -u "$SERVE_SCRIPT" \
        --model-path "$MODEL_P2" --model-id "$ID_P2" \
        --port "$P2_PORT" --device GPU "${DEBUG_ARGS[@]}" \
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
        CHAINLIT_DEBUG="$($DEBUG_LOGGING && echo 1 || echo 0)" \
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

# ── Update VS Code Continue config ──────────────────────────────────────────
# Points Continue at whichever models are actually live right now. Runs every
# launch (not just setup) since the live model changes with --pair.
#
# IMPORTANT (found 19 Sep 2026): modern Continue versions use
# ~/.continue/config.yaml as the ACTIVE config, not config.json — confirmed
# directly, since a user's real Continue installation was showing an
# Ollama-based config (Llama 3.1 8B / Qwen2.5-Coder 1.5B / Nomic Embed) that
# an earlier version of this script's config.json automation had no effect
# on whatsoever, despite running successfully and updating config.json every
# time. The YAML schema is also structurally different: models are a flat
# list, each with a `roles:` array (chat/edit/apply/autocomplete/embed),
# rather than JSON's separate top-level tabAutocompleteModel block.
#
# This writes BOTH files now: config.yaml (the one that actually matters on
# current Continue versions) and config.json (kept as cheap backward-compat
# insurance for older installs — costs nothing to also keep correct).
#
# NOTE: Continue can silently fail to read config.yaml at all if VS Code's
# YAML extension (redhat.vscode-yaml) isn't installed — if the config looks
# right but Continue still isn't picking it up, check that extension is
# present. This script can't detect or fix that from the outside.
if kill -0 "$P1_PID_VAL" 2>/dev/null && curl -sf "http://127.0.0.1:$P1_PORT/health" >/dev/null 2>&1; then
    P2_LIVE=false
    if ! $SKIP_P2 && kill -0 "${P2_PID_VAL:-0}" 2>/dev/null && curl -sf "http://127.0.0.1:$P2_PORT/health" >/dev/null 2>&1; then
        P2_LIVE=true
    fi

    CONTINUE_UPDATE=$(python3 - "$ID_P1" "$P1_PORT" "$ID_P2" "$P2_PORT" "$P2_LIVE" << 'PYEOF' 2>&1
import json, sys
from pathlib import Path

p1_model, p1_port, p2_model, p2_port, p2_live = sys.argv[1:6]
# FIXED 19 Sep 2026: was `p2_live == "True"` (Python's capitalized boolean
# string) but bash's $P2_LIVE is lowercase ("true"/"false" — matching bash's
# own true/false-as-commands convention used elsewhere in this script for
# `$X && ...` checks). That comparison never matched, so the autocomplete
# entry was silently never added regardless of whether Phase 2 was actually
# live — confirmed live: the success message correctly said "autocomplete
# -> ..." (bash's `true && echo ...` really did evaluate true) while the
# written config.yaml had no autocomplete entry at all (Python's string
# comparison against the wrong case never matched). Case-insensitive now so
# this specific mismatch can't recur even if the calling convention changes.
p2_live = p2_live.strip().lower() == "true"
warnings = []

CHAT_NAME  = "Local Qwen (Arc iGPU) — Chat"
AUTO_NAME  = "Local Qwen (Arc iGPU) — Autocomplete"

chat_entry_yaml = {
    "name": CHAT_NAME, "provider": "openai", "model": p1_model,
    "apiBase": f"http://127.0.0.1:{p1_port}/v1", "apiKey": "none",
    "roles": ["chat", "edit", "apply"],
}
auto_entry_yaml = {
    "name": AUTO_NAME, "provider": "openai", "model": p2_model,
    "apiBase": f"http://127.0.0.1:{p2_port}/v1", "apiKey": "none",
    "roles": ["autocomplete"],
}

# ── config.yaml (the one that actually matters) ─────────────────────────────
try:
    import yaml
    yaml_path = Path.home() / ".continue" / "config.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)

    if yaml_path.exists():
        try:
            config = yaml.safe_load(yaml_path.read_text()) or {}
        except yaml.YAMLError:
            backup = yaml_path.with_suffix(".yaml.bak")
            yaml_path.rename(backup)
            warnings.append(f"existing config.yaml was invalid YAML — backed up to {backup.name}")
            config = {}
    else:
        config = {"name": "Main Config", "version": "1.0.0", "schema": "v1"}

    models = [m for m in config.get("models", [])
              if m.get("name") not in (CHAT_NAME, AUTO_NAME)]
    models.append(chat_entry_yaml)
    if p2_live:
        models.append(auto_entry_yaml)
    config["models"] = models

    yaml_path.write_text(yaml.safe_dump(config, sort_keys=False, default_flow_style=False, allow_unicode=True))
    print(f"YAML_OK model={p1_model}" + (f"+{p2_model}" if p2_live else ""))
except ImportError:
    warnings.append("PyYAML not installed — config.yaml NOT updated. Run: pip install pyyaml")
except Exception as e:
    warnings.append(f"config.yaml update failed: {e}")

# ── config.json (legacy, cheap backward-compat) ─────────────────────────────
try:
    json_path = Path.home() / ".continue" / "config.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    if json_path.exists():
        try:
            jconfig = json.loads(json_path.read_text())
        except json.JSONDecodeError:
            backup = json_path.with_suffix(".json.bak")
            json_path.rename(backup)
            warnings.append(f"existing config.json was invalid JSON — backed up to {backup.name}")
            jconfig = {}
    else:
        jconfig = {}

    TITLE = "Local Qwen Coder (Arc iGPU)"
    jmodels = [m for m in jconfig.get("models", []) if m.get("title") != TITLE]
    jmodels.append({
        "title": TITLE, "provider": "openai", "model": p1_model,
        "apiBase": f"http://127.0.0.1:{p1_port}/v1", "apiKey": "none",
    })
    jconfig["models"] = jmodels
    if p2_live:
        jconfig["tabAutocompleteModel"] = {
            "title": "Local Qwen Autocomplete", "provider": "openai", "model": p2_model,
            "apiBase": f"http://127.0.0.1:{p2_port}/v1", "apiKey": "none",
        }
    json_path.write_text(json.dumps(jconfig, indent=2) + "\n")
except Exception as e:
    warnings.append(f"config.json update failed: {e}")

for w in warnings:
    print(f"WARN: {w}")
print("OK")
PYEOF
)
    if [[ "$CONTINUE_UPDATE" == *"OK" ]]; then
        if [[ "$CONTINUE_UPDATE" == *"YAML_OK"* ]]; then
            ok "VS Code Continue config.yaml updated — chat/edit/apply -> $ID_P1$($P2_LIVE && echo ", autocomplete -> $ID_P2")"
        fi
        echo "$CONTINUE_UPDATE" | grep '^WARN:' | while IFS= read -r line; do warn "${line#WARN: }"; done
    else
        warn "Could not update Continue config automatically:"
        info "$CONTINUE_UPDATE"
    fi
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo -e "${BOLD}${CYAN}══════════════════════════════════════════════════════════════${RESET}"
echo -e "${BOLD}${GREEN} OpenVINO GenAI — Pair $PAIR Active & Ready${RESET}"
echo -e "${BOLD}${CYAN}══════════════════════════════════════════════════════════════${RESET}"
echo ""
echo -e "  ${YELLOW}Phase 1 (Port $P1_PORT):${RESET}  $ID_P1"
echo -e "  ${YELLOW}Phase 2 (Port $P2_PORT):${RESET}  $ID_P2"
echo ""
echo -e "  ${DIM}Both models resident in Arc iGPU VRAM simultaneously${RESET}"
if $DEBUG_LOGGING; then
    echo -e "  ${YELLOW}Debug logging: ON${RESET}  — verbose output in LOGS/model_800{0,1}.log + Chainlit console"
fi
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
