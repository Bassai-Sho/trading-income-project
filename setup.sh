#!/usr/bin/env bash
# =============================================================================
# Trading Income Project — Universal Setup & Data Bootstrapper
# =============================================================================
# Sets up host runtime, .venv, keys, models, data bootstrap, and systemd service.
# Safe to run multiple times. Run from project root.
#
# Usage:
#   ./setup.sh                  # full interactive setup
#   ./setup.sh --pair 1         # select Pair 1 non-interactively
#   ./setup.sh --models-only    # skip drivers/keys/service, download models only
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

trap 'tput cnorm 2>/dev/null || true' EXIT INT TERM

run_spinner() {
    local msg="$1"; shift
    local spin_chars='|/-\'
    local i=0
    local t_start; t_start=$(date +%s)
    local log_file; log_file=$(mktemp /tmp/prep_XXXXXX.log 2>/dev/null || echo "/tmp/prep_$$.log")

    "$@" > "$log_file" 2>&1 &
    local pid=$!

    tput civis 2>/dev/null || true
    while kill -0 "$pid" 2>/dev/null; do
        local elapsed=$(( $(date +%s) - t_start ))
        local char="${spin_chars:i++%4:1}"
        printf "\r  ${CYAN}%s${RESET}  %s ${DIM}(%ds)${RESET}" "$char" "$msg" "$elapsed"
        sleep 0.2
    done
    tput cnorm 2>/dev/null || true
    printf "\r\033[K"

    local total_elapsed=$(( $(date +%s) - t_start ))
    local ret=0
    if ! wait "$pid"; then ret=$?; fi

    if (( ret == 0 )); then
        ok "$msg ${DIM}(${total_elapsed}s)${RESET}"
        rm -f "$log_file"
        return 0
    else
        fail "$msg failed (${total_elapsed}s, exit code $ret)"
        if [[ -f "$log_file" ]]; then
            info "Recent output from $log_file:"
            tail -n 10 "$log_file" | while IFS= read -r line; do echo -e "    ${DIM}${line}${RESET}"; done
            rm -f "$log_file"
        fi
        return $ret
    fi
}

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
import sys, os, re
key, val, filepath = sys.argv[1], sys.argv[2], sys.argv[3]
content = open(filepath, "r").read() if os.path.exists(filepath) else ""
if re.search(rf"^{key}=", content, flags=re.M):
    content = re.sub(rf"^{key}=.*", f"{key}={val}", content, flags=re.M)
else:
    content += f"\n{key}={val}\n"
open(filepath, "w").write(content.strip() + "\n")
' "$key" "$val" "$file"
}

get_env_val() {
    local key="$1"
    if [[ -f ".env" ]]; then
        grep "^${key}=" .env 2>/dev/null | cut -d= -f2- | tr -d '"' | tr -d "'" || echo ""
    else
        echo ""
    fi
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
MODELS_DIR="${MODELS_DIR:-$HOME/models}"
MODELS_ONLY=false
CHOSEN_PAIR=""

for arg in "$@"; do
    case "$arg" in
        --pair)        shift; CHOSEN_PAIR="${1:-}"; shift ;;
        --pair=*)      CHOSEN_PAIR="${arg#--pair=}" ;;
        --models-only) MODELS_ONLY=true ;;
    esac
done

declare -A PAIR_LABEL=(
    [1]='Most Popular    — Qwen3.8-27B-int4 (MTP built-in) + Phi-4-mini (~19GB)'
    [2]='MoE Speed       — Qwen3.6-35B-A3B-int4 + Mistral-Nemo-int4   (~25GB)'
    [3]='Novel Adversary — Qwen3.6-35B-A3B-int4 + LFM2.5-8B-A1B-int4 (~23GB)'
)
declare -A P1_HF_REPO=([1]='OpenVINO/Qwen3.8-27B-int4-ov' [2]='OpenVINO/Qwen3.6-35B-A3B-int4-ov' [3]='OpenVINO/Qwen3.6-35B-A3B-int4-ov')
declare -A P1_DIR=([1]='qwen3.8-27b-int4' [2]='qwen3.6-35b-a3b' [3]='qwen3.6-35b-a3b')
declare -A P1_ID=([1]='qwen3.8:27b' [2]='qwen3.6:35b-a3b' [3]='qwen3.6:35b-a3b')
declare -A P2_HF_REPO=([1]='OpenVINO/Phi-4-mini-instruct-int4-ov' [2]='OpenVINO/Mistral-Nemo-Instruct-2407-int4-ov' [3]='OpenVINO/LFM2.5-8B-A1B-int4-ov')
declare -A P2_DIR=([1]='phi-4-mini-int4' [2]='mistral-nemo-12b' [3]='lfm2.5-8b-a1b')
declare -A P2_ID=([1]='phi-4-mini:int4' [2]='mistral-nemo:12b' [3]='lfm2.5:8b')
declare -A DRAFT_REPO=([1]='' [2]='OpenVINO/Qwen3-0.6B-int4-ov' [3]='OpenVINO/Qwen3-0.6B-int4-ov')
declare -A DRAFT_LOCAL=([1]='' [2]='draft-qwen3-0.6b' [3]='draft-qwen3-0.6b')

echo -e "\n${BOLD}Trading Income Project — Unified System Setup${RESET}"

# 1. Directories
hdr "1 / 8  Directory Scaffolding"
for d in DATA DATA/models LOGS LOGS/pids "$MODELS_DIR"; do
    [[ ! -d "$d" ]] && { mkdir -p "$d"; ok "Created $d/"; } || skip "$d/"
done

# 2. Host Drivers
if ! $MODELS_ONLY; then
    hdr "2 / 8  Compute Drivers & GPU Access"
    if [ -f "/etc/default/grub" ] && ! grep -q "i915.enable_psr=0" /proc/cmdline 2>/dev/null && ! grep -q "i915.enable_psr=0" /etc/default/grub 2>/dev/null; then
        if ask "Disable PSR in GRUB to prevent Meteor Lake display artifacts?"; then
            sudo sed -i 's/GRUB_CMDLINE_LINUX_DEFAULT="\(.*\)"/GRUB_CMDLINE_LINUX_DEFAULT="\1 i915.enable_psr=0"/' /etc/default/grub
            sudo update-grub 2>/dev/null || true
            ok "PSR disabled in GRUB (reboot required later)"
        fi
    fi

    if ! ldconfig -p 2>/dev/null | grep -E "libze_intel_gpu" >/dev/null; then
        echo -e "  ${DIM}Requesting sudo privileges for Level Zero compute drivers...${RESET}"
        sudo -v
        for stale in /etc/apt/sources.list.d/intel-gpu*.list /etc/apt/sources.list.d/oneAPI.list; do [[ -f "$stale" ]] && sudo rm -f "$stale"; done
        run_spinner "Updating package lists" sudo apt-get update -qq
        run_spinner "Installing Level Zero & OpenCL runtime" sudo apt-get install -y -q intel-opencl-icd libze1 libze-intel-gpu1 libze-dev mesa-vulkan-drivers clinfo
        sudo ldconfig
    fi
    ok "Level Zero GPU runtime active"

    if ! id -nG "$USER" | grep -qw "render"; then
        sudo usermod -aG render,video "$USER"
        ok "Added $USER to render/video groups"
    else
        ok "GPU group permissions verified"
    fi
fi

# 3. Python Virtualenv & Deps
hdr "3 / 8  Python Environment (.venv) & Packages"
VENV_DIR="$ROOT_DIR/.venv"
[[ ! -d "$VENV_DIR" ]] && { python3 -m venv "$VENV_DIR"; ok "Created .venv/"; } || skip ".venv/"
source "$VENV_DIR/bin/activate"

run_spinner "Upgrading pip and installing base packages" pip install --upgrade pip setuptools wheel --quiet

if [[ -f "requirements.txt" ]]; then
    run_spinner "Installing application dependencies from requirements.txt" pip install -r requirements.txt --quiet
fi

# Ensure OpenVINO GenAI Nightly (Required for Qwen 3.8 / 3.6 architectures)
run_spinner "Installing OpenVINO GenAI nightly runtime" \
    pip install --pre -U openvino openvino-genai openvino-tokenizers \
    --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly --quiet
ok "All dependencies & Chainlit active"

# 4. API Keys
if ! $MODELS_ONLY; then
    hdr "4 / 8  API Key Configuration (.env)"
    [[ -f ".env" ]] || { cp .env.example .env 2>/dev/null || touch .env; ok "Created .env"; }

    CURR_ALPACA=$(get_env_val "ALPACA_API_KEY")
    if [[ -z "$CURR_ALPACA" || "$CURR_ALPACA" == "your_alpaca_api_key_here" ]]; then
        info "Alpaca API Keys (free at https://alpaca.markets):"
        enter "Enter Alpaca API Key"; [[ -n "$REPLY" ]] && set_env_val "ALPACA_API_KEY" "$REPLY"
        enter "Enter Alpaca Secret Key"; [[ -n "$REPLY" ]] && set_env_val "ALPACA_SECRET_KEY" "$REPLY"
        set_env_val "ALPACA_BASE_URL" "https://paper-api.alpaca.markets"
    fi

    CURR_FRED=$(get_env_val "FRED_API_KEY")
    if [[ -z "$CURR_FRED" ]]; then
        info "FRED Macro Key (free at https://fred.stlouisfed.org/docs/api/api_key.html):"
        enter "Enter FRED API Key (press Enter to skip)"; [[ -n "$REPLY" ]] && set_env_val "FRED_API_KEY" "$REPLY"
    fi

    CURR_BRAVE=$(get_env_val "BRAVE_SEARCH_API_KEY")
    if [[ -z "$CURR_BRAVE" ]]; then
        info "Brave News Search Key (free at https://api.search.brave.com):"
        enter "Enter Brave Search Key (press Enter for DuckDuckGo fallback)"; [[ -n "$REPLY" ]] && set_env_val "BRAVE_SEARCH_API_KEY" "$REPLY"
    fi

    CURR_HF=$(get_env_val "HF_TOKEN")
    if [[ -z "$CURR_HF" ]]; then
        info "Hugging Face Read Token (free at https://huggingface.co/settings/tokens):"
        enter "Enter Hugging Face Token (press Enter to skip)"; [[ -n "$REPLY" ]] && { set_env_val "HF_TOKEN" "$REPLY"; export HF_TOKEN="$REPLY"; }
    else
        export HF_TOKEN="$CURR_HF"
    fi

    set_env_val "MARKET_DATA_DB" "DATA/market_data.db"
    set_env_val "PAPER_ACCOUNT_DB" "DATA/paper_account.db"
    set_env_val "LLM_BASE_URL" "http://127.0.0.1:8000/v1"
    set_env_val "DAC_BASE_URL" "http://127.0.0.1:8001/v1"
    ok "Environment variables configured in .env"
fi

# 5. Model Download
hdr "5 / 8  Model Selection & Download"
if [[ -z "$CHOSEN_PAIR" ]]; then
    echo ""
    echo -e "${BOLD}Select OpenVINO Model Pair for 58GB VRAM Pool:${RESET}"
    printf "  ${CYAN}1${RESET}  %s\n" "${PAIR_LABEL[1]}"
    printf "  ${CYAN}2${RESET}  %s\n" "${PAIR_LABEL[2]}"
    printf "  ${CYAN}3${RESET}  %s\n\n" "${PAIR_LABEL[3]}"
    read -r -p "     Choose [1/2/3, default=1]: " reply
    CHOSEN_PAIR="${reply:-1}"
fi

cat > "$ROOT_DIR/.model_pair" << EOF
ACTIVE_PAIR=$CHOSEN_PAIR
P1_MODEL_DIR=${P1_DIR[$CHOSEN_PAIR]}
P1_MODEL_ID=${P1_ID[$CHOSEN_PAIR]}
P2_MODEL_DIR=${P2_DIR[$CHOSEN_PAIR]}
P2_MODEL_ID=${P2_ID[$CHOSEN_PAIR]}
EOF
set_env_val "LLM_PRESET" "nuc-pair${CHOSEN_PAIR}"
set_env_val "LLM_MODEL_PRIMARY" "${P1_ID[$CHOSEN_PAIR]}"
set_env_val "LLM_MODEL_ADVERSARIAL" "${P2_ID[$CHOSEN_PAIR]}"

download_model_verified() {
    local hf_repo="$1" local_dir="$2" label="$3"
    local full_path="$MODELS_DIR/$local_dir"
    local is_intact=false
    if [[ -d "$full_path" ]]; then
        if python3 - "$full_path" << 'PYEOF' 2>/dev/null
import sys
from pathlib import Path
import openvino as ov
model_dir = Path(sys.argv[1])
xml = next((model_dir / x for x in ("openvino_model.xml", "openvino_language_model.xml") if (model_dir / x).exists()), None)
if not xml: sys.exit(1)
ov.Core().read_model(str(xml))
sys.exit(0)
PYEOF
        then
            is_intact=true
        fi
    fi

    if $is_intact; then
        ok "$label: verified on disk"
        return 0
    fi

    info "Downloading $label from $hf_repo..."
    python3 -u - "$hf_repo" "$full_path" "$label" << 'PYEOF'
import os, sys
from huggingface_hub import snapshot_download
repo_id, dest_path, label = sys.argv[1], sys.argv[2], sys.argv[3]
os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
token = os.environ.get("HF_TOKEN") or None
try:
    snapshot_download(repo_id=repo_id, local_dir=dest_path, ignore_patterns=["*.gguf", "*.safetensors", "*.pt"], token=token)
    print(f"  ✓  {label} download finished")
except Exception as e:
    print(f"  ✗  Download failed: {e}", file=sys.stderr); sys.exit(1)
PYEOF
}

download_model_verified "${P1_HF_REPO[$CHOSEN_PAIR]}" "${P1_DIR[$CHOSEN_PAIR]}" "Phase 1: ${P1_ID[$CHOSEN_PAIR]}"
download_model_verified "${P2_HF_REPO[$CHOSEN_PAIR]}" "${P2_DIR[$CHOSEN_PAIR]}" "Phase 2: ${P2_ID[$CHOSEN_PAIR]}"
if [[ -n "${DRAFT_REPO[$CHOSEN_PAIR]}" ]]; then
    download_model_verified "${DRAFT_REPO[$CHOSEN_PAIR]}" "${DRAFT_LOCAL[$CHOSEN_PAIR]}" "Draft Model"
    echo "DRAFT_MODEL_DIR=${DRAFT_LOCAL[$CHOSEN_PAIR]}" >> "$ROOT_DIR/.model_pair"
fi

# 6. Automated One-Shot Data Bootstrap
hdr "6 / 8  Automated Data Bootstrap (Historical Bars & Macro Feeds)"
ALPACA_KEY=$(get_env_val "ALPACA_API_KEY")
if [[ -n "$ALPACA_KEY" && "$ALPACA_KEY" != "your_alpaca_api_key_here" ]]; then
    if ask "Run the automated data bootstrap now (download SPY bars, FRED macro, CBOE data, and seed backtests)?"; then
        run_spinner "1/5 Ingesting SPY 1-minute historical bars (Alpaca)" \
            "$VENV_DIR/bin/python3" src/market_data_store.py --download --tickers SPY --start 2016-01-01 --end 2024-12-31 --db DATA/market_data.db
        run_spinner "2/5 Ingesting FRED macro series (VIX, yield curves)" \
            "$VENV_DIR/bin/python3" src/fred_store.py --download --db DATA/market_data.db
        run_spinner "3/5 Ingesting CBOE Put/Call & CFTC positioning" \
            "$VENV_DIR/bin/python3" src/sentiment_store.py --download --db DATA/market_data.db
        run_spinner "4/5 Running algorithmic bar data correction" \
            "$VENV_DIR/bin/python3" src/data_corrector.py --db DATA/market_data.db --ticker SPY --start 2016-01-01 --end 2022-12-31
        run_spinner "5/5 Bootstrapping Markov engines & academic comparison benchmarks" \
            "$VENV_DIR/bin/python3" src/historical_sim.py --start 2016-01-01 --end 2022-12-31 --db DATA/paper_account.db
        ok "Data bootstrap and historical seeding complete"
    else
        info "Data bootstrap skipped"
    fi
else
    info "Alpaca API keys not yet populated — skipping data bootstrap"
fi

# 7. Systemd Service Auto-Start Configuration (Linux)
if ! $MODELS_ONLY && command -v systemctl &>/dev/null; then
    hdr "7 / 8  Systemd Service Auto-Start (Linux Daemon)"
    SERVICE_FILE="/etc/systemd/system/trading-runner.service"
    if [[ -f "$SERVICE_FILE" ]]; then
        ok "Systemd service 'trading-runner.service' is already installed"
    else
        info "Auto-start runner.py on system boot as a background service."
        if ask "Install systemd service (trading-runner.service)?" "n"; then
            echo -e "  ${DIM}Requesting sudo privileges to install systemd service...${RESET}"
            sudo -v
            sudo tee "$SERVICE_FILE" >/dev/null << EOF
[Unit]
Description=Trading Income Project Master Runner
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$ROOT_DIR
ExecStart=$ROOT_DIR/.venv/bin/python3 $ROOT_DIR/src/runner.py
Restart=always
RestartSec=10
EnvironmentFile=$ROOT_DIR/.env

[Install]
WantedBy=multi-user.target
EOF
            sudo systemctl daemon-reload
            ok "Installed /etc/systemd/system/trading-runner.service"

            if ask "Enable and start trading-runner service immediately?" "n"; then
                sudo systemctl enable --now trading-runner >/dev/null 2>&1 || true
                ok "trading-runner service active (check: sudo systemctl status trading-runner)"
            else
                info "Start anytime: sudo systemctl enable --now trading-runner"
            fi
        else
            info "Systemd service installation skipped"
        fi
    fi
fi

# 8. Verification
hdr "8 / 8  Source Code Verification"
for f in src/runner.py src/trading_engine.py src/market_data_store.py \
         src/historical_sim.py src/data_corrector.py src/session_analyser.py \
         src/serve_model.py src/market_calendar.py chainlit_app.py; do
    [[ -f "$f" ]] || { warn "$f — not found (skipping)"; continue; }
    "$VENV_DIR/bin/python3" -c "import ast; ast.parse(open('$f').read())" 2>/dev/null && ok "$f" || fail "$f — syntax error"
done

echo ""
echo -e "${BOLD}${CYAN}══════════════════════════════════════════════════════════════${RESET}"
echo -e "${BOLD}${GREEN} Setup Complete — Ready for Launch${RESET}"
echo -e "${BOLD}${CYAN}══════════════════════════════════════════════════════════════${RESET}"
echo ""
echo -e "  1. Start Model Servers:  ${CYAN}./launch_models.sh --with-chainlit${RESET}"
echo -e "     Chat UI (Chainlit):   ${CYAN}http://localhost:8080${RESET}"
echo -e "  2. Pre-Flight Check:     ${CYAN}./verify.sh${RESET}"
echo -e "  3. Start Live System:    ${CYAN}python3 src/runner.py${RESET}"
if [[ -f "/etc/systemd/system/trading-runner.service" ]]; then
    echo -e "  4. Service Management:   ${CYAN}sudo systemctl status trading-runner${RESET}"
fi
echo ""
