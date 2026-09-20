#!/usr/bin/env bash
# =============================================================================
# Universal Setup & Data Bootstrapper — with VS Code OpenVINO Infrastructure
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
CACHE_DIR="$HOME/.cache/openvino"
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
    [1]='Most Popular    — Qwen3.8-27B-int4 + Phi-4-mini (~19GB) [~2.7 t/s]'
    [2]='MoE Speed       — Qwen3.6-35B-A3B-int4 + Mistral-7B-v0.1-int4 (~21GB)'
    [3]='Novel Adversary — Qwen3.6-35B-A3B-int4 + LFM2.5-8B-A1B-int4 (~23GB)'
    [4]='Fast VS Code    — Qwen2.5-Coder-7B-int4 + Qwen2.5-Coder-1.5B-int4 (~6GB) [15-20 t/s]'
)

declare -A P1_HF_REPO=(
    [1]='OpenVINO/Qwen3.8-27B-int4-ov'
    [2]='OpenVINO/Qwen3.6-35B-A3B-int4-ov'
    [3]='OpenVINO/Qwen3.6-35B-A3B-int4-ov'
    [4]='OpenVINO/Qwen2.5-Coder-7B-Instruct-int4-ov'
)
declare -A P1_DIR=([1]='qwen3.8-27b-int4' [2]='qwen3.6-35b-a3b' [3]='qwen3.6-35b-a3b' [4]='qwen2.5-coder-7b-int4')
declare -A P1_ID=([1]='qwen3.8:27b' [2]='qwen3.6:35b-a3b' [3]='qwen3.6:35b-a3b' [4]='qwen2.5-coder:7b')

declare -A P2_HF_REPO=(
    [1]='OpenVINO/Phi-4-mini-instruct-int4-ov'
    [2]='OpenVINO/mistral-7b-instruct-v0.1-int4-ov'
    [3]='OpenVINO/LFM2.5-8B-A1B-int4-ov'
    [4]='OpenVINO/Qwen2.5-Coder-1.5B-Instruct-int4-ov'
)
declare -A P2_DIR=([1]='phi-4-mini-int4' [2]='mistral-7b-v01-int4' [3]='lfm2.5-8b-a1b' [4]='qwen2.5-coder-1.5b-int4')
declare -A P2_ID=([1]='phi-4-mini:int4' [2]='mistral-7b:int4' [3]='lfm2.5:8b' [4]='qwen2.5-coder:1.5b')

echo -e "\n${BOLD}Trading Income Project — Unified System & VS Code Setup${RESET}"

# 1. Directories
hdr "1 / 7  Directory Scaffolding"
for d in DATA DATA/models LOGS LOGS/pids "$MODELS_DIR" "$CACHE_DIR"; do
    [[ ! -d "$d" ]] && { mkdir -p "$d"; ok "Created $d/"; } || skip "$d/"
done

# 2. Python Virtualenv
# NOTE: this now runs BEFORE the driver/GPU step. It used to run after, which
# meant the GPU-visibility check in that step tested a bare `python3` that
# could never have had `openvino` installed yet on a first-time run — every
# fresh install saw a false "OpenVINO cannot see GPU" warning regardless of
# actual driver/runtime state. Creating the venv first fixes that at the root.
hdr "2 / 7  Python Environment (.venv) & Packages"
VENV_DIR="$ROOT_DIR/.venv"
[[ ! -d "$VENV_DIR" ]] && { python3 -m venv "$VENV_DIR"; ok "Created .venv/"; } || skip ".venv/"
source "$VENV_DIR/bin/activate"

run_spinner "Upgrading pip and installing base packages" pip install --upgrade pip setuptools wheel --quiet

if [[ -f "requirements.txt" ]]; then
    run_spinner "Installing application dependencies from requirements.txt" pip install -r requirements.txt --quiet
fi

run_spinner "Installing OpenVINO GenAI nightly runtime & FastAPI" \
    pip install --pre -U openvino openvino-genai openvino-tokenizers fastapi uvicorn huggingface_hub \
    --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly --quiet
ok "Runtime dependencies installed"

# 3. Host Drivers & Hardware
if ! $MODELS_ONLY; then
    hdr "3 / 7  Compute Drivers & GPU Access"

    # ── 2a. xe kernel driver migration (i915 → xe) ───────────────────────────
    # Meteor Lake / Xe-LPG architecture requires the xe driver.
    # i915 has a 10s fence watchdog that kills GPU compute during long prefills
    # (27B model attention on ~1800 token context takes 25-40s → kernel kills it).
    # xe supports native preemption and has no watchdog limit.
    # Reference: OpenVINO Issue #36260, #36404
    GPU_ID=$(lspci -nn -s 00:02.0 2>/dev/null | grep -oP '\[8086:\K[0-9a-f]+(?=\])' || echo "")
    CURRENT_DRIVER=$(lspci -k -s 00:02.0 2>/dev/null | grep "Kernel driver in use:" | awk '{print $NF}' || echo "unknown")

    if [[ -n "$GPU_ID" && "$CURRENT_DRIVER" != "xe" ]]; then
        warn "Arc iGPU using legacy i915 driver (GPU ID: $GPU_ID) — upgrade recommended"
        info "The xe driver prevents GPU fence watchdog crashes during large model inference."
        if ask "Migrate Arc iGPU from i915 to xe driver? (requires reboot)"; then
            if ! grep -q "xe.force_probe" /etc/default/grub 2>/dev/null; then
                sudo sed -i "s/GRUB_CMDLINE_LINUX_DEFAULT=\"\(.*\)\"/GRUB_CMDLINE_LINUX_DEFAULT=\"\1 i915.enable_psr=0 i915.force_probe=!${GPU_ID} xe.force_probe=${GPU_ID}\"/" /etc/default/grub
                sudo update-grub 2>/dev/null || true
                ok "GRUB updated for xe driver (i915.force_probe=!${GPU_ID} xe.force_probe=${GPU_ID})"
                warn "Reboot required to activate xe driver — run: sudo reboot"
            else
                skip "xe driver already configured in GRUB"
            fi
        fi
    elif [[ "$CURRENT_DRIVER" == "xe" ]]; then
        ok "Arc iGPU using xe driver (no fence watchdog)"
    fi

    # ── 2b. Intel Compute Runtime (NEO) — xe-compatible version ─────────────
    # xe driver requires NEO ≥ 24.52.32224.5 with IGC ≥ v2.5.6
    # Check if OpenCL can see the GPU.
    # Uses $VENV_DIR/bin/python3 explicitly (not bare python3) since Step 2
    # already installed openvino there. Distinguishes "module not found" from
    # a genuine "GPU not visible" so the warning below is actually honest
    # about what failed, instead of a bare `2>/dev/null` hiding the reason.
    GPU_VISIBLE=false
    GPU_CHECK_OUT=$("$VENV_DIR/bin/python3" -c "import openvino as ov; print('GPU' in ov.Core().available_devices)" 2>&1)
    GPU_CHECK_RC=$?
    if [[ $GPU_CHECK_RC -eq 0 && "$GPU_CHECK_OUT" == "True" ]]; then
        GPU_VISIBLE=true
        ok "OpenVINO GPU device visible (compute runtime OK)"
    elif echo "$GPU_CHECK_OUT" | grep -q "ModuleNotFoundError"; then
        fail "openvino not importable in .venv — Step 2 (Python Environment) may have failed. Re-run setup.sh, or check pip install output above."
    else
        warn "OpenVINO cannot see GPU — installing updated Intel Compute Runtime (NEO)"
        echo -e "  ${DIM}This provides OpenCL/Level Zero userspace for the xe driver${RESET}"
        sudo -v

        NEO_TMP=$(mktemp -d /tmp/neo_XXXXXX)
        cd "$NEO_TMP"

        # IGC v2.5.6 — Intel Graphics Compiler
        run_spinner "Downloading Intel Graphics Compiler v2.5.6" bash -c "
            wget -q https://github.com/intel/intel-graphics-compiler/releases/download/v2.5.6/intel-igc-core-2_2.5.6+18417_amd64.deb
            wget -q https://github.com/intel/intel-graphics-compiler/releases/download/v2.5.6/intel-igc-opencl-2_2.5.6+18417_amd64.deb
        "

        # NEO 24.52.32224.5 — Compute Runtime
        run_spinner "Downloading Intel Compute Runtime 24.52.32224.5" bash -c "
            wget -q https://github.com/intel/compute-runtime/releases/download/24.52.32224.5/intel-level-zero-gpu_1.6.32224.5_amd64.deb
            wget -q https://github.com/intel/compute-runtime/releases/download/24.52.32224.5/intel-opencl-icd_24.52.32224.5_amd64.deb
            wget -q https://github.com/intel/compute-runtime/releases/download/24.52.32224.5/libigdgmm12_22.5.2_amd64.deb
        "

        run_spinner "Installing Intel Compute Runtime (NEO)" sudo dpkg -i *.deb
        sudo ldconfig
        cd "$ROOT_DIR"
        rm -rf "$NEO_TMP"

        # Verify (same honest check as above)
        GPU_CHECK_OUT=$("$VENV_DIR/bin/python3" -c "import openvino as ov; print('GPU' in ov.Core().available_devices)" 2>&1)
        if [[ "$GPU_CHECK_OUT" == "True" ]]; then
            ok "GPU now visible to OpenVINO after NEO update"
        else
            warn "GPU still not visible — a reboot may be required if xe driver was just activated"
            warn "After reboot, re-run: ./setup.sh --models-only"
        fi
    fi

    # ── 2c. GPU group permissions ─────────────────────────────────────────────
    if ! id -nG "$USER" | grep -qw "render"; then
        sudo usermod -aG render,video "$USER"
        ok "Added $USER to render/video groups (re-login required)"
    else
        ok "GPU group permissions verified"
    fi

    # ── 2d. PSR disable (legacy i915 screen artifact prevention) ─────────────
    if [[ "$CURRENT_DRIVER" != "xe" ]] && \
       [[ -f "/etc/default/grub" ]] && \
       ! grep -q "i915.enable_psr=0" /proc/cmdline 2>/dev/null && \
       ! grep -q "i915.enable_psr=0" /etc/default/grub 2>/dev/null; then
        if ask "Disable PSR in GRUB to prevent Meteor Lake display artifacts?"; then
            sudo sed -i 's/GRUB_CMDLINE_LINUX_DEFAULT="\(.*\)"/GRUB_CMDLINE_LINUX_DEFAULT="\1 i915.enable_psr=0"/' /etc/default/grub
            sudo update-grub 2>/dev/null || true
            ok "PSR disabled in GRUB (takes effect on next boot)"
        fi
    fi
fi

# 4. API Keys & Tokens Configuration
hdr "4 / 7  API Keys & Tokens Configuration (.env)"
[[ -f ".env" ]] || { cp .env.example .env 2>/dev/null || touch .env; ok "Created .env"; }

if ! $MODELS_ONLY; then
    # Alpaca Keys
    CURR_ALPACA=$(get_env_val "ALPACA_API_KEY")
    if [[ -z "$CURR_ALPACA" || "$CURR_ALPACA" == "your_alpaca_api_key_here" ]]; then
        info "Alpaca API Keys (free at https://alpaca.markets):"
        enter "Enter Alpaca API Key"; [[ -n "$REPLY" ]] && set_env_val "ALPACA_API_KEY" "$REPLY"
        enter "Enter Alpaca Secret Key"; [[ -n "$REPLY" ]] && set_env_val "ALPACA_SECRET_KEY" "$REPLY"
        set_env_val "ALPACA_BASE_URL" "https://paper-api.alpaca.markets"
    else
        ok "Alpaca API Key already set"
    fi

    # FRED Key
    CURR_FRED=$(get_env_val "FRED_API_KEY")
    if [[ -z "$CURR_FRED" || "$CURR_FRED" == "your_fred_api_key_here" ]]; then
        info "FRED Macro Key (free at https://fred.stlouisfed.org/docs/api/api_key.html):"
        enter "Enter FRED API Key (press Enter to skip)"; [[ -n "$REPLY" ]] && set_env_val "FRED_API_KEY" "$REPLY"
    else
        ok "FRED API Key already set"
    fi

    # Brave Search Key
    CURR_BRAVE=$(get_env_val "BRAVE_SEARCH_API_KEY")
    if [[ -z "$CURR_BRAVE" || "$CURR_BRAVE" == "your_brave_api_key_here" ]]; then
        info "Brave News Search Key (free at https://api.search.brave.com):"
        enter "Enter Brave Search Key (press Enter for DuckDuckGo fallback)"; [[ -n "$REPLY" ]] && set_env_val "BRAVE_SEARCH_API_KEY" "$REPLY"
    else
        ok "Brave Search Key already set"
    fi
fi

# Hugging Face Token (Always checked, needed for Section 5 download speed & rate limits)
CURR_HF=$(get_env_val "HF_TOKEN")
if [[ -z "$CURR_HF" || "$CURR_HF" == "your_hf_token_here" ]]; then
    info "Hugging Face Read Token (free at https://huggingface.co/settings/tokens):"
    enter "Enter Hugging Face Token (press Enter to skip)"; 
    if [[ -n "$REPLY" ]]; then
        set_env_val "HF_TOKEN" "$REPLY"
        export HF_TOKEN="$REPLY"
        ok "Hugging Face token saved"
    else
        warn "Proceeding without Hugging Face token (unauthenticated download limits apply)"
    fi
else
    export HF_TOKEN="$CURR_HF"
    ok "Hugging Face token verified from .env"
fi

set_env_val "MARKET_DATA_DB" "DATA/market_data.db"
set_env_val "PAPER_ACCOUNT_DB" "DATA/paper_account.db"
set_env_val "LLM_BASE_URL" "http://127.0.0.1:8000/v1"
set_env_val "DAC_BASE_URL" "http://127.0.0.1:8001/v1"
ok "Environment variables active"

# 5. Model Selection & Verification
hdr "5 / 7  Model Selection & Download"
if [[ -z "$CHOSEN_PAIR" ]]; then
    echo ""
    echo -e "${BOLD}Select OpenVINO Model Configuration:${RESET}"
    printf "  ${CYAN}1${RESET}  %s\n" "${PAIR_LABEL[1]}"
    printf "  ${CYAN}2${RESET}  %s\n" "${PAIR_LABEL[2]}"
    printf "  ${CYAN}3${RESET}  %s\n" "${PAIR_LABEL[3]}"
    printf "  ${CYAN}4${RESET}  %s\n\n" "${PAIR_LABEL[4]}"
    read -r -p "     Choose [1-4, default=4 (Recommended for fast coding)]: " reply
    CHOSEN_PAIR="${reply:-4}"
fi

cat > "$ROOT_DIR/.model_pair" << EOF
ACTIVE_PAIR=$CHOSEN_PAIR
P1_MODEL_DIR=${P1_DIR[$CHOSEN_PAIR]}
P1_MODEL_ID=${P1_ID[$CHOSEN_PAIR]}
P2_MODEL_DIR=${P2_DIR[$CHOSEN_PAIR]}
P2_MODEL_ID=${P2_ID[$CHOSEN_PAIR]}
EOF

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
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
token = os.environ.get("HF_TOKEN")
if token and not token.strip():
    token = None

try:
    snapshot_download(
        repo_id=repo_id,
        local_dir=dest_path,
        ignore_patterns=["*.gguf", "*.safetensors", "*.pt"],
        token=token
    )
    print(f"  ✓  {label} download complete")
except Exception as e:
    print(f"  ✗  Download failed: {e}", file=sys.stderr)
    sys.exit(1)
PYEOF
}

download_model_verified "${P1_HF_REPO[$CHOSEN_PAIR]}" "${P1_DIR[$CHOSEN_PAIR]}" "Primary: ${P1_ID[$CHOSEN_PAIR]}"
download_model_verified "${P2_HF_REPO[$CHOSEN_PAIR]}" "${P2_DIR[$CHOSEN_PAIR]}" "Secondary/FIM: ${P2_ID[$CHOSEN_PAIR]}"

# ── Fixed Qwen3 chat template (eemin/Qwen-Fixed-Chat-Templates v22.1) ────────
# Fixes the infinite <tool_call><web_search> loop (empty-think poisoning).
# Reference: https://huggingface.co/eemin/Qwen-Fixed-Chat-Templates
P1_FULL_PATH="$MODELS_DIR/${P1_DIR[$CHOSEN_PAIR]}"
TEMPLATE_TARGET="$P1_FULL_PATH/chat_template.jinja"

# Only apply to Qwen3 chat models — not Phi-4-mini or Qwen2.5-Coder
if [[ "${P1_ID[$CHOSEN_PAIR]}" == *"qwen3"* ]]; then
    if [[ -f "$TEMPLATE_TARGET" ]]; then
        skip "Fixed chat template already present in $(basename $P1_FULL_PATH)"
    else
        info "Downloading fixed Qwen3 chat template (fixes tool call infinite loop)..."
        python3 -c "
import sys, shutil, os
from pathlib import Path
from huggingface_hub import hf_hub_download
dest = Path('$P1_FULL_PATH')
token = os.environ.get('HF_TOKEN') or None
try:
    p = hf_hub_download('eemin/Qwen-Fixed-Chat-Templates', 'chat_template.jinja', token=token)
    shutil.copy(p, dest / 'chat_template.jinja')
    print('ok')
except Exception as e:
    print(f'failed: {e}', file=sys.stderr)
    sys.exit(1)
" && ok "Fixed chat template installed → $(basename $P1_FULL_PATH)/chat_template.jinja" \
      || warn "Template download failed — run: bash scripts/download_fixed_template.sh $P1_FULL_PATH"
    fi
fi


# 6. Data Bootstrap (Optional)
hdr "6 / 7  Data Bootstrap"
info "Skipped by default during setup. Run historical_sim.py when ready."

# 7. Verification
hdr "7 / 7  Installation Verification"

# Remove any systemd service from a previous version of this script.
# This project now runs exactly one model pair at a time via launch_models.sh;
# a persistent, auto-restarting background service on a fixed port (which is
# what the old Step 7 installed) collides with that by design — it's a
# different, independent use case (VS Code Continue autocomplete) that
# happened to hardcode the same port launch_models.sh also uses.
if command -v systemctl &>/dev/null; then
    if systemctl --user list-unit-files 2>/dev/null | grep -q "openvino-coder.service"; then
        warn "Found an existing openvino-coder systemd service from a previous setup — removing it"
        systemctl --user disable --now openvino-coder.service 2>/dev/null || true
        rm -f "$HOME/.config/systemd/user/openvino-coder.service"
        systemctl --user daemon-reload 2>/dev/null || true
        ok "Removed openvino-coder.service — launch_models.sh now has sole ownership of the model server ports"
    fi
fi

ok "Setup complete."

echo ""
echo -e "${BOLD}${CYAN}══════════════════════════════════════════════════════════════${RESET}"
echo -e "${BOLD}${GREEN} Ready — start models with launch_models.sh${RESET}"
echo -e "${BOLD}${CYAN}══════════════════════════════════════════════════════════════${RESET}"
echo ""
echo -e "  This project runs ${BOLD}one model pair at a time${RESET}, started manually:"
echo ""
echo -e "    ${CYAN}./launch_models.sh --pair $CHOSEN_PAIR --with-chainlit${RESET}"
echo ""
echo -e "  Switching pairs is safe to do directly — launching a new pair"
echo -e "  automatically stops whatever's currently running first. To stop"
echo -e "  everything without starting a new pair: ${CYAN}./launch_models.sh --stop${RESET}"
echo ""
echo -e "  To use the local model in VS Code's Continue extension, just launch a"
echo -e "  pair — ${CYAN}~/.continue/config.json${RESET} is updated automatically every time"
echo -e "  ${CYAN}launch_models.sh${RESET} starts, pointed at whatever model is actually live."
echo -e "  (Your other Continue models/settings are preserved, not overwritten.)"
echo ""
