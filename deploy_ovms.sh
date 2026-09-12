#!/usr/bin/env bash
# =============================================================================
# Trading Income Project — OpenVINO Model Server Deployment
# =============================================================================
# Launches two OVMS GPU containers on the Intel NUC 14 Pro Arc iGPU:
#
#   Port 8000 — Phase 1  (qwen3.6-35b-a3b  MoE INT4  ~18-19GB)
#   Port 8001 — Phase 2  (mistral-nemo-12b  dense INT4 ~7GB)
#
# Why two containers instead of one multi-model config?
#   - Simpler: each uses --source_model which auto-downloads from HuggingFace
#   - Cleaner: session_analyser.py just points phase1/phase2 at different ports
#   - Safer:   each model is independent; one crash doesn't kill the other
#
# Why OVMS instead of IPEX-LLM Ollama?
#   - Intel's active production framework (IPEX-LLM archived Jan 2026)
#   - Fully containerised: no host display driver conflicts, no black screens
#   - No Ollama registry version checks: no Error 412
#   - Native OpenAI /v3/chat/completions API in pure C++
#
# ENDPOINT NOTE:  OVMS generation API is /v3/chat/completions  (NOT /v1)
#                 Set LLM_BASE_URL=http://127.0.0.1:8000/v3 in .env
#
# FIRST RUN:   Both containers will download their models from HuggingFace.
#              Phase 1  ~18-19GB   Phase 2 ~7GB  — ensure internet access.
#              Subsequent runs skip the download (cache in $MODELS_DIR).
#
# Usage:
#   chmod +x deploy_ovms.sh
#   ./deploy_ovms.sh             # start both containers
#   ./deploy_ovms.sh --stop      # stop and remove both containers
#   ./deploy_ovms.sh --status    # check container health
# =============================================================================
set -euo pipefail

BOLD="\033[1m"; DIM="\033[2m"
GREEN="\033[32m"; YELLOW="\033[33m"; RED="\033[31m"; CYAN="\033[36m"
RESET="\033[0m"

ok()   { echo -e "  ${GREEN}✓${RESET}  $*"; }
warn() { echo -e "  ${YELLOW}⚠${RESET}  $*"; }
fail() { echo -e "  ${RED}✗${RESET}  $*"; }
info() { echo -e "  ${DIM}    $*${RESET}"; }
hdr()  { echo -e "\n${BOLD}${CYAN}── $* ${RESET}"; }

# Model configuration
MODELS_DIR="${OVMS_MODELS_DIR:-$HOME/models}"

# Phase 1: MoE fast analysis
P1_NAME="ovms-phase1"
P1_PORT=8000
P1_MODEL="OpenVINO/Qwen3.6-35B-A3B-int4-ov"
P1_DESC="Qwen3.6 35B A3B MoE INT4 — Phase 1 session analysis"

# Phase 2: Dense adversarial (different architecture from Phase 1)
P2_NAME="ovms-phase2"
P2_PORT=8001
P2_MODEL="OpenVINO/Mistral-Nemo-Instruct-2407-int4-ov"
P2_DESC="Mistral NeMo 12B INT4 — Phase 2 adversarial D-A-C (Mistral SWA ≠ Qwen3)"

OVMS_IMAGE="openvino/model_server:latest-gpu"

# ── CLI handling ──────────────────────────────────────────────────────────────
case "${1:-start}" in
    --stop|stop)
        echo -e "\n${BOLD}Stopping OVMS containers...${RESET}"
        docker stop "$P1_NAME" "$P2_NAME" 2>/dev/null || true
        docker rm   "$P1_NAME" "$P2_NAME" 2>/dev/null || true
        ok "Containers stopped and removed"
        exit 0
        ;;
    --status|status)
        echo -e "\n${BOLD}OVMS Container Status${RESET}"
        for name in "$P1_NAME" "$P2_NAME"; do
            if docker ps --format '{{.Names}}' 2>/dev/null | grep -q "^$name$"; then
                status=$(docker inspect "$name" --format '{{.State.Status}}' 2>/dev/null)
                ok "$name — $status"
            else
                warn "$name — not running"
            fi
        done
        echo ""
        info "Phase 1 health:  curl -sf http://localhost:$P1_PORT/v1/config | python3 -m json.tool"
        info "Phase 2 health:  curl -sf http://localhost:$P2_PORT/v1/config | python3 -m json.tool"
        info "Phase 1 test:    curl -s http://localhost:$P1_PORT/v3/chat/completions -H 'Content-Type: application/json' -d '{\"model\": \"$P1_MODEL\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello\"}], \"max_tokens\": 10}'"
        exit 0
        ;;
esac

echo -e "\n${BOLD}OpenVINO Model Server — D-A-C Model Fleet${RESET}"
echo -e "${DIM}Phase 1: $P1_DESC${RESET}"
echo -e "${DIM}Phase 2: $P2_DESC${RESET}"

# ── Prerequisites ─────────────────────────────────────────────────────────────
hdr "Prerequisites"

# Docker
if ! command -v docker &>/dev/null; then
    warn "Docker not found — installing..."
    sudo apt-get update -qq
    sudo apt-get install -y -qq docker.io
    sudo usermod -aG docker "$USER"
    ok "Docker installed — NOTE: log out/in for group membership, or use 'newgrp docker'"
fi
ok "Docker: $(docker --version | grep -oP '[\d.]+')"

# Render device for Intel Arc iGPU
if [[ ! -c /dev/dri/renderD128 ]]; then
    fail "/dev/dri/renderD128 not found — Intel Arc drivers may not be installed"
    info "Run ./NUC_SETUP.sh first to install Intel GPU compute runtime"
    exit 1
fi
RENDER_GID=$(stat -c "%g" /dev/dri/renderD128)
ok "Intel Arc render device: /dev/dri/renderD128 (gid=$RENDER_GID)"

# Models directory
mkdir -p "$MODELS_DIR"
ok "Models directory: $MODELS_DIR"

# ── Pull OVMS Docker image ────────────────────────────────────────────────────
hdr "OVMS Docker image"
if docker image inspect "$OVMS_IMAGE" &>/dev/null 2>&1; then
    ok "Image cached: $OVMS_IMAGE"
else
    info "Pulling $OVMS_IMAGE (~2-3GB)..."
    docker pull "$OVMS_IMAGE"
    ok "Image ready: $OVMS_IMAGE"
fi

# ── Launch containers ─────────────────────────────────────────────────────────
launch_container() {
    local name="$1" port="$2" model="$3" desc="$4"

    hdr "Launching $name ($desc)"

    # Stop existing if running
    docker stop "$name" 2>/dev/null || true
    docker rm   "$name" 2>/dev/null || true

    info "Container: $name"
    info "Port:      $port → 8000 (internal)"
    info "Model:     $model"
    info "Note:      First run downloads model from HuggingFace (~downloads on first start)"
    echo ""

    docker run -d \
        --name "$name" \
        --restart unless-stopped \
        --device /dev/dri/renderD128 \
        --group-add "$RENDER_GID" \
        -p "${port}:8000" \
        -v "${MODELS_DIR}:/models" \
        -e "HF_HOME=/models/.cache" \
        "$OVMS_IMAGE" \
        --source_model "$model" \
        --model_repository_path /models \
        --task text_generation \
        --target_device GPU \
        --rest_port 8000 \
        --plugin_config '{"PERFORMANCE_HINT": "LATENCY", "CACHE_DIR": "/models/.ov_cache"}'

    ok "$name started — waiting for model to load..."
}

launch_container "$P1_NAME" "$P1_PORT" "$P1_MODEL" "$P1_DESC"
launch_container "$P2_NAME" "$P2_PORT" "$P2_MODEL" "$P2_DESC"

# ── Health check ──────────────────────────────────────────────────────────────
hdr "Health check (models may take 2-5 minutes to load on first run)"

check_ready() {
    local port="$1" name="$2" max_wait=300 elapsed=0
    echo -e "  ${DIM}Waiting for $name on port $port...${RESET}"
    while (( elapsed < max_wait )); do
        if curl -sf "http://localhost:$port/v1/config" &>/dev/null; then
            state=$(curl -sf "http://localhost:$port/v1/config" 2>/dev/null \
                    | python3 -c "import sys,json; d=json.load(sys.stdin); \
                      states=[v['model_version_status'][0]['state'] \
                              for v in d.values() if v.get('model_version_status')]; \
                      print(states[0] if states else 'LOADING')" 2>/dev/null || echo "LOADING")
            if [[ "$state" == "AVAILABLE" ]]; then
                ok "$name ready on port $port"
                return 0
            fi
        fi
        sleep 5; (( elapsed += 5 ))
        printf "\r  ${DIM}%ds — waiting for %s...${RESET}   " "$elapsed" "$name"
    done
    echo ""
    warn "$name: timeout after ${max_wait}s — check logs: docker logs $name"
    return 1
}

check_ready "$P1_PORT" "$P1_NAME" &
WAIT_P1=$!
check_ready "$P2_PORT" "$P2_NAME" &
WAIT_P2=$!
wait $WAIT_P1 || true
wait $WAIT_P2 || true

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${CYAN}══════════════════════════════════════════════════════════════${RESET}"
echo -e "${BOLD}${GREEN} OpenVINO Model Server active${RESET}"
echo -e "${BOLD}${CYAN}══════════════════════════════════════════════════════════════${RESET}"
echo ""
echo -e "  ${YELLOW}Phase 1 endpoint:${RESET}  http://127.0.0.1:${P1_PORT}/v3/chat/completions"
echo -e "  ${YELLOW}Phase 2 endpoint:${RESET}  http://127.0.0.1:${P2_PORT}/v3/chat/completions"
echo ""
echo -e "  ${DIM}⚠ IMPORTANT: OVMS uses /v3/chat/completions (not /v1)${RESET}"
echo -e "  ${DIM}Set in .env:${RESET}"
echo -e "  ${DIM}  LLM_BASE_URL=http://127.0.0.1:${P1_PORT}/v3${RESET}"
echo -e "  ${DIM}  DAC_BASE_URL=http://127.0.0.1:${P2_PORT}/v3${RESET}"
echo ""
echo -e "  ${YELLOW}Logs:${RESET}  docker logs -f $P1_NAME"
echo -e "         docker logs -f $P2_NAME"
echo -e "  ${YELLOW}Stop:${RESET}  ./deploy_ovms.sh --stop"
echo ""
