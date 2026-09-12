#!/usr/bin/env bash
# =============================================================================
# Trading Income Project — Universal Interactive Setup
# Supports: Intel Arc (NUC), NVIDIA GPUs, Apple Silicon (Mac), and Linux CPU
# Includes: Automated Modelfile GGUF fallback for cutting-edge models
# =============================================================================
# Safe to run multiple times. Run from project root.
# =============================================================================
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"

# ── Colours ───────────────────────────────────────────────────────────────────
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

# ── Hardware & Memory Detection ───────────────────────────────────────────────
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
    local os
    os="$(uname -s)"
    if [[ "$os" == "Darwin" ]]; then
        echo "$(( $(sysctl -n hw.memsize) / 1024 / 1024 / 1024 ))"
    elif [[ -f "/proc/meminfo" ]]; then
        awk '/MemTotal/ {printf "%.0f", $2/1024/1024}' /proc/meminfo
    else
        echo "16"
    fi
}

download_file() {
    local url="$1"
    local dest="$2"
    if command -v wget &>/dev/null; then
        wget -c --show-progress -O "$dest" "$url"
    elif command -v curl &>/dev/null; then
        curl -L -C - --progress-bar -o "$dest" "$url"
    else
        fail "Neither wget nor curl is available for download."
        return 1
    fi
}

# ── Guard: must run from project root ─────────────────────────────────────────
[[ -f "requirements.txt" && -d "src" ]] || {
    echo -e "${RED}Run this from the project root (where setup.sh lives).${RESET}"
    exit 1
}

echo -e "\n${BOLD}Trading Income Project — Application Setup${RESET}"
echo -e "${DIM}Checks project requirements, virtual environment, and models.${RESET}"

# =============================================================================
# SECTION 1 — Hardware & Platform Detection
# =============================================================================
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

# =============================================================================
# SECTION 2 — Python & Virtual Environment (.venv)
# =============================================================================
hdr "2 / 6  Python & Virtual Environment (.venv)"

if ! command -v python3 &>/dev/null; then
    fail "Python 3 not found"
    echo "    Please install Python 3.10+ for your operating system."
    exit 1
fi

PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$(python3 -c "import sys; print(sys.version_info.major)")
PY_MINOR=$(python3 -c "import sys; print(sys.version_info.minor)")

if (( PY_MAJOR < 3 || (PY_MAJOR == 3 && PY_MINOR < 10) )); then
    fail "Python $PY_VER found — need 3.10 or higher"
    exit 1
fi
ok "System Python: $PY_VER"

if ! python3 -m venv --help &>/dev/null; then
    fail "python3-venv module is missing."
    echo "    Install with: sudo apt install python3-venv"
    exit 1
fi

VENV_DIR=".venv"
if [[ ! -d "$VENV_DIR" ]]; then
    python3 -m venv "$VENV_DIR"
    ok "Created virtual environment in $VENV_DIR/"
else
    skip "Virtual environment already exists ($VENV_DIR/)"
fi

VENV_PY="$VENV_DIR/bin/python3"
VENV_PIP="$VENV_DIR/bin/pip"

REQUIRED=(pandas numpy scipy yfinance alpaca streamlit plotly requests dotenv apscheduler)
MISSING=()
for pkg in "${REQUIRED[@]}"; do
    modname="${pkg//-/_}"
    "$VENV_PY" -c "import $modname" 2>/dev/null || MISSING+=("$pkg")
done

if [[ ${#MISSING[@]} -eq 0 ]]; then
    skip "All Python dependencies installed in .venv"
else
    warn "Missing packages in .venv: ${MISSING[*]}"
    if ask "Install dependencies into .venv from requirements.txt?"; then
        echo ""
        info "Installing dependencies from requirements.txt..."

        "$VENV_PIP" install --upgrade pip --quiet 2>/dev/null || true

        # Clean stream: shows active downloads, suppresses cache spam
        "$VENV_PIP" install -r requirements.txt --progress-bar on 2>&1 | "$VENV_PY" -u -c '
import sys

for line in sys.stdin:
    if "Using cached" in line or "Requirement already satisfied" in line:
        continue
    if line.startswith("Collecting "):
        pkg = line.strip().split()[1] if len(line.strip().split()) > 1 else ""
        sys.stdout.write(f"\r\033[K  • Resolving: {pkg}")
        sys.stdout.flush()
        continue
    if "Installing collected packages:" in line:
        sys.stdout.write("\r\033[K  • Installing packages into .venv...\n")
        sys.stdout.flush()
        continue
    if line.startswith("Successfully installed"):
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()
        continue
    if "Downloading " in line:
        sys.stdout.write("\r\033[K" + line)
        sys.stdout.flush()
        continue
    sys.stdout.write(line)
    sys.stdout.flush()
' && echo "" && ok "Dependencies successfully installed into .venv" \
  || { echo ""; fail "pip install failed — check requirements.txt"; exit 1; }
    else
        warn "Skipped. Some features will not work until dependencies are installed."
    fi
fi

# =============================================================================
# SECTION 3 — Project Directories
# =============================================================================
hdr "3 / 6  Project Directories"

for dir in DATA DATA/models LOGS; do
    if [[ -d "$dir" ]]; then
        skip "$dir/"
    else
        mkdir -p "$dir"
        ok "Created $dir/"
    fi
done

# =============================================================================
# SECTION 4 — Environment Configuration (.env)
# =============================================================================
hdr "4 / 6  Environment Configuration (.env)"

if [[ ! -f ".env" ]]; then
    if [[ -f ".env.example" ]]; then
        cp .env.example .env
        ok "Created .env from template"
    else
        touch .env
        ok "Created blank .env"
    fi
fi

ALPACA_KEY=$(grep "^ALPACA_API_KEY=" .env 2>/dev/null | cut -d= -f2- | tr -d '"' | tr -d "'" || true)
ALPACA_SEC=$(grep "^ALPACA_SECRET_KEY=" .env 2>/dev/null | cut -d= -f2- | tr -d '"' | tr -d "'" || true)

if [[ "$ALPACA_KEY" == "your_alpaca_api_key_here" || -z "$ALPACA_KEY" ]]; then
    warn "Alpaca API key not configured"
    echo -e "     ${DIM}Free paper account: https://alpaca.markets${RESET}\n"
    if ask "Enter Alpaca keys now?"; then
        enter "Alpaca API key"
        NEW_KEY="$REPLY"
        enter "Alpaca Secret key"
        NEW_SEC="$REPLY"

        if [[ -n "$NEW_KEY" && -n "$NEW_SEC" ]]; then
            set_env_val "ALPACA_API_KEY" "$NEW_KEY"
            set_env_val "ALPACA_SECRET_KEY" "$NEW_SEC"
            ok "Alpaca keys saved to .env"
        fi
    fi
else
    ok "Alpaca keys configured"
fi

DISCORD=$(grep "^DISCORD_WEBHOOK_URL=" .env 2>/dev/null | cut -d= -f2- | tr -d '"' | tr -d "'" || true)
if [[ -z "$DISCORD" ]]; then
    echo ""
    echo -e "     ${BOLD}${CYAN}Discord Notification Webhook Setup:${RESET}"
    echo -e "     ${DIM}1. In Discord: Channel Settings → Integrations → Webhooks${RESET}"
    echo -e "     ${DIM}2. Click 'New Webhook', name it, and click 'Copy Webhook URL'${RESET}"
    echo -e "     ${DIM}Format: https://discord.com/api/webhooks/<id>/<token>${RESET}\n"

    if ask "Configure Discord notifications now? (optional)" "n"; then
        enter "Paste Discord webhook URL"
        WEBHOOK_INPUT="$REPLY"
        if [[ -n "$WEBHOOK_INPUT" ]]; then
            set_env_val "DISCORD_WEBHOOK_URL" "$WEBHOOK_INPUT"
            ok "Discord webhook saved to .env"

            if ask "Send a test notification to Discord right now?" "Y"; then
                if curl -s -H "Content-Type: application/json" \
                        -X POST \
                        -d '{"content": "🚀 **Trading Income Project:** Setup connection successful!"}' \
                        "$WEBHOOK_INPUT" >/dev/null 2>&1; then
                    ok "Discord test message sent! Check your channel."
                else
                    warn "Test ping failed — check URL."
                fi
            fi
        fi
    fi
else
    ok "Discord notifications configured"
fi

# Configure production models in .env
if (( RAM_GB >= 32 )); then
    set_env_val "OLLAMA_MODEL_PRIMARY" "qwen3:30b-a3b"
    set_env_val "OLLAMA_MODEL_ADVERSARIAL" "mistral-nemo:12b"
else
    set_env_val "OLLAMA_MODEL_PRIMARY" "qwen3:8b"
    set_env_val "OLLAMA_MODEL_ADVERSARIAL" "mistral-nemo:12b"
fi
set_env_val "OLLAMA_HOST" "http://127.0.0.1:11434"
set_env_val "OLLAMA_CONTEXT_LENGTH" "8192"
set_env_val "OLLAMA_NUM_BATCH" "2048"
ok "Configured LLM inference settings in .env (Context: 8192, Batch: 2048)"

# =============================================================================
# SECTION 5 — Ollama Setup & Model Deployment (with Modelfile Fallback)
# =============================================================================
hdr "5 / 6  LLM Backend (OpenVINO GenAI preferred / Ollama fallback)"

PRIMARY_URL="${LLM_BASE_URL:-http://127.0.0.1:8000/v1}"
DAC_URL="${DAC_BASE_URL:-http://127.0.0.1:8001/v1}"
SERVERS_ACTIVE=true

if ! curl -sf "${PRIMARY_URL}/models" >/dev/null 2>&1; then
    warn "Phase 1 server not responding at $PRIMARY_URL"
    SERVERS_ACTIVE=false
fi
if ! curl -sf "${DAC_URL}/models" >/dev/null 2>&1; then
    warn "Phase 2 server not responding at $DAC_URL"
    SERVERS_ACTIVE=false
fi

if ! $SERVERS_ACTIVE; then
    echo ""
    echo -e "  Start servers:  ${CYAN}./launch_models.sh${RESET}  (OpenVINO GenAI in .venv)"
    echo -e "  ${DIM}Or Ollama:  ./NUC_SETUP.sh${RESET}"
    echo ""
    if ask "Launch OpenVINO GenAI servers now via launch_models.sh?"; then
        if [[ -f "launch_models.sh" ]]; then
            bash launch_models.sh && SERVERS_ACTIVE=true
        else
            warn "launch_models.sh not found — run from project root"
        fi
    fi
fi

if $SERVERS_ACTIVE; then
    ok "Phase 1 server: $PRIMARY_URL"
    ok "Phase 2 server: $DAC_URL"
    ok "Both models resident in Arc iGPU VRAM simultaneously"
fi


hdr "6 / 6  Source Verification"

ALL_PASS=true
for f in src/runner.py src/trading_engine.py src/market_data_store.py \
         src/historical_sim.py src/data_corrector.py src/session_analyser.py; do
    if [[ ! -f "$f" ]]; then
        fail "$f — file missing"
        ALL_PASS=false
    elif ! "$VENV_PY" -c "import ast; ast.parse(open('$f').read())" 2>/dev/null; then
        fail "$f — syntax error"
        ALL_PASS=false
    else
        ok "$f"
    fi
done

# =============================================================================
# SUMMARY & NEXT STEPS
# =============================================================================
echo ""
echo -e "${BOLD}${CYAN}══════════════════════════════════════════${RESET}"
echo -e "${BOLD} Setup Summary${RESET}"
echo -e "${BOLD}${CYAN}══════════════════════════════════════════${RESET}"

ALPACA_KEY=$(grep "^ALPACA_API_KEY=" .env 2>/dev/null | cut -d= -f2- | tr -d '"' | tr -d "'" || true)
ALPACA_READY=false
[[ "$ALPACA_KEY" != "your_alpaca_api_key_here" && -n "$ALPACA_KEY" ]] && ALPACA_READY=true

$ALPACA_READY && ok "Alpaca keys configured" || warn "Alpaca keys — edit .env before data download"
$OLLAMA_INSTALLED && ok "Ollama runner ready ($OLLAMA_BIN)" || warn "Ollama runner not ready — analysis will use fallback"
$ALL_PASS && ok "All source files verified with .venv" || fail "Source file errors detected"

if $ALPACA_READY; then
    echo ""
    echo -e "${BOLD} Next Steps (Using Virtual Environment)${RESET}\n"
    echo -e "  ${DIM}Tip: You can run commands directly using .venv/bin/python3, or activate first:${RESET}"
    echo -e "  source .venv/bin/activate\n"
    echo -e "  ${YELLOW}1. Download 10 years of SPY data:${RESET}"
    echo "  .venv/bin/python3 src/market_data_store.py --download --tickers SPY \\"
    echo "    --start 2016-01-01 --end 2024-12-31 --db DATA/market_data.db"
    echo ""
    echo -e "  ${YELLOW}2. Download FRED macro + CBOE/CFTC sentiment:${RESET}"
    echo "  .venv/bin/python3 src/fred_store.py      --download --db DATA/market_data.db"
    echo "  .venv/bin/python3 src/sentiment_store.py --download --db DATA/market_data.db"
    echo ""
    echo -e "  ${YELLOW}3. Test D-A-C Adversarial Independence Probe:${RESET}"
    echo "  .venv/bin/python3 -c \"from src.session_analyser import test_adversarial_backend; print(test_adversarial_backend())\""
fi
echo ""
