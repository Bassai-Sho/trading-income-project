#!/usr/bin/env bash
# =============================================================================
# Trading Income Project — Universal Interactive Setup
# =============================================================================
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"

BOLD="\033[1m"; DIM="\033[2m"
GREEN="\033[32m"; YELLOW="\033[33m"; RED="\033[31m"; CYAN="\033[36m"
RESET="\033[0m"

ok()   { echo -e "  ${GREEN}✓${RESET}  $*"; }
warn() { echo -e "  ${YELLOW}⚠${RESET}  $*"; }
fail() { echo -e "  ${RED}✗${RESET}  $*"; }
skip() { echo -e "  ${DIM}○  $* — already done${RESET}"; }
info() { echo -e "  ${DIM}    $*${RESET}"; }
hdr()  { echo -e "\n${BOLD}${CYAN}── $* ${RESET}"; }

ask() {
    local prompt="$1" default="${2:-Y}"
    local opts; [[ "$default" == "Y" ]] && opts="[Y/n]" || opts="[y/N]"
    read -r -p "$(echo -e "     ${prompt} ${DIM}${opts}${RESET} ")" reply
    reply="${reply:-$default}"
    [[ "$reply" =~ ^[Yy]$ ]]
}

enter() {
    read -r -p "$(echo -e "     $1: ")" REPLY
}

set_env_val() {
    local key="$1" val="$2" file=".env"
    python3 -c '
import sys
key, val, filepath = sys.argv[1], sys.argv[2], sys.argv[3]
lines = []
try:
    with open(filepath, "r") as f:
        lines = f.readlines()
except FileNotFoundError:
    pass

found = False
new_lines = []
for line in lines:
    if line.startswith(f"{key}="):
        new_lines.append(f"{key}={val}\n")
        found = True
    else:
        new_lines.append(line)

if not found:
    new_lines.append(f"{key}={val}\n")

with open(filepath, "w") as f:
    f.writelines(new_lines)
' "$key" "$val" "$file"
}

detect_hardware() {
    OS="$(uname -s)"
    HW_TYPE="generic_cpu"
    if [[ "$OS" == "Darwin" ]]; then
        HW_TYPE="apple_silicon"
    elif [[ "$OS" == "Linux" ]]; then
        if [ -d "/sys/bus/pci/drivers/xe" ] || lspci 2>/dev/null | grep -iE "VGA.*Intel.*(Arc|Graphics)" >/dev/null; then
            HW_TYPE="intel_arc"
        elif command -v nvidia-smi &>/dev/null || lspci 2>/dev/null | grep -i "VGA.*NVIDIA" >/dev/null; then
            HW_TYPE="nvidia_gpu"
        fi
    fi
    echo "$HW_TYPE"
}

get_system_ram_gb() {
    local os; os="$(uname -s)"
    if [[ "$os" == "Darwin" ]]; then
        echo "$(( $(sysctl -n hw.memsize) / 1024 / 1024 / 1024 ))"
    elif [[ -f "/proc/meminfo" ]]; then
        awk '/MemTotal/ {printf "%.0f", $2/1024/1024}' /proc/meminfo
    else
        echo "16"
    fi
}

[[ -f "requirements.txt" && -d "src" ]] || {
    echo -e "${RED}Run this from the project root (where setup.sh lives).${RESET}"
    exit 1
}

echo -e "\n${BOLD}Trading Income Project — Application Setup${RESET}"

# 1 / 6 Hardware
hdr "1 / 6  Hardware & Platform Detection"
HW=$(detect_hardware)
RAM_GB=$(get_system_ram_gb)
case "$HW" in
    "intel_arc")     ok "Hardware: Intel Arc iGPU (Meteor Lake / NUC detected)" ;;
    "nvidia_gpu")    ok "Hardware: NVIDIA Discrete GPU detected" ;;
    "apple_silicon") ok "Hardware: Apple Silicon (macOS Metal detected)" ;;
    *)               ok "Hardware: Generic CPU / Cloud Host" ;;
esac
ok "Total Host Memory: ${RAM_GB}GB"

# 2 / 6 Venv
hdr "2 / 6  Python & Virtual Environment (.venv)"
VENV_DIR=".venv"
[[ -d "$VENV_DIR" ]] || python3 -m venv "$VENV_DIR"
VENV_PY="$VENV_DIR/bin/python3"
VENV_PIP="$VENV_DIR/bin/pip"
ok "Virtual environment active ($VENV_DIR/)"

"$VENV_PIP" install --upgrade pip --quiet 2>/dev/null || true
if ask "Install / verify dependencies from requirements.txt?"; then
    "$VENV_PIP" install -r requirements.txt --quiet && ok "Dependencies installed" || warn "pip had warnings"
fi

# 3 / 6 Directories (Self-healing)
hdr "3 / 6  Project Directories"
for dir in DATA DATA/models LOGS LOGS/pids; do
    if [[ ! -d "$dir" ]]; then
        mkdir -p "$dir"
        ok "Created $dir/"
    else
        skip "$dir/"
    fi
done

# 4 / 6 Environment
hdr "4 / 6  Environment Configuration (.env)"
[[ -f ".env" ]] || touch .env
set_env_val "LLM_BASE_URL" "http://127.0.0.1:8000/v1"
set_env_val "DAC_BASE_URL" "http://127.0.0.1:8001/v1"
set_env_val "LLM_PRESET"   "nuc-pair1"
ok "OpenVINO GenAI local bridge configured in .env"

# 5 / 6 Backend Check
hdr "5 / 6  LLM Backend (OpenVINO GenAI)"
PRIMARY_URL="http://127.0.0.1:8000/v1"
DAC_URL="http://127.0.0.1:8001/v1"

if curl -sf "${PRIMARY_URL}/health" >/dev/null 2>&1 && curl -sf "${DAC_URL}/health" >/dev/null 2>&1; then
    ok "Phase 1 (Port 8000) & Phase 2 (Port 8001) servers READY"
else
    warn "Model servers not responding — start with: ./launch_models.sh --with-webui"
fi

# 6 / 6 Source check
hdr "6 / 6  Source Verification"
for f in src/runner.py src/trading_engine.py src/market_data_store.py \
         src/historical_sim.py src/data_corrector.py src/session_analyser.py; do
    "$VENV_PY" -c "import ast; ast.parse(open('$f').read())" 2>/dev/null && ok "$f" || fail "$f — syntax error"
done

echo -e "\n${BOLD}${GREEN}Setup Complete.${RESET}\n"
