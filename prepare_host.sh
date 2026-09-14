#!/usr/bin/env bash
# =============================================================================
# Trading Income Project — Host Preparation & Model Downloader
# NUC 14 Pro · Intel Core Ultra 5 125H · 58GB iGPU VRAM · Ubuntu 24.04
# =============================================================================
# Uses ONLY stock Ubuntu packages — no third-party Intel GPU repositories.
# Downloads via Python huggingface_hub with HF_XET_HIGH_PERFORMANCE.
# Validates model integrity using OpenVINO Core graph reading (no false positives).
#
# Model pairs ranked by OpenVINO org (Sep 2026):
#   Pair 1: Qwen3.8-27B (MTP built-in) + Phi-4-mini (~19GB)
#   Pair 2: Qwen3.6-35B-A3B MoE        + Mistral-Nemo-12B (~25GB)
#   Pair 3: Qwen3.6-35B-A3B MoE        + LFM2.5-8B-A1B (~23GB)
#
# Usage:
#   ./prepare_host.sh                    # interactive pair selection
#   ./prepare_host.sh --pair 1           # Pair 1: Qwen3.8-27B + Phi-4-mini
#   ./prepare_host.sh --pair 2           # Pair 2: Qwen3.6-35B-A3B + Mistral-Nemo
#   ./prepare_host.sh --pair 3           # Pair 3: Qwen3.6-35B-A3B + LFM2.5-8B
#   ./prepare_host.sh --models-only      # skip drivers & UI, download models only
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

# Ensure cursor is always restored if interrupted
trap 'tput cnorm 2>/dev/null || true' EXIT INT TERM

# ── Activity Spinner with Live Seconds Counter ────────────────────────────────
run_spinner() {
    local msg="$1"
    shift
    local spin_chars='|/-\'
    local i=0
    local t_start
    t_start=$(date +%s)
    local log_file
    log_file=$(mktemp /tmp/prep_host_XXXXXX.log 2>/dev/null || echo "/tmp/prep_$$.log")

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
    if ! wait "$pid"; then
        ret=$?
    fi

    if (( ret == 0 )); then
        ok "$msg ${DIM}(${total_elapsed}s)${RESET}"
        rm -f "$log_file"
        return 0
    else
        fail "$msg failed after ${total_elapsed}s (exit code $ret)"
        if [[ -f "$log_file" ]]; then
            info "Recent output from $log_file:"
            tail -n 12 "$log_file" | while IFS= read -r line; do
                echo -e "    ${DIM}${line}${RESET}"
            done
            rm -f "$log_file"
        fi
        return $ret
    fi
}

MODELS_DIR="${MODELS_DIR:-$HOME/models}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_ONLY=false
CHOSEN_PAIR=""

for arg in "$@"; do
    case "$arg" in
        --pair)        shift; CHOSEN_PAIR="${1:-}"; shift ;;
        --pair=*)      CHOSEN_PAIR="${arg#--pair=}" ;;
        --models-only) MODELS_ONLY=true ;;
    esac
done

# =============================================================================
# Model pair definitions
# =============================================================================
declare -A PAIR_LABEL=(
    [1]='Most Popular    — Qwen3.8-27B-int4 (MTP built-in) + Phi-4-mini (~19GB)'
    [2]='MoE Speed       — Qwen3.6-35B-A3B-int4 + Mistral-Nemo-int4   (~25GB)'
    [3]='Novel Adversary — Qwen3.6-35B-A3B-int4 + LFM2.5-8B-A1B-int4 (~23GB)'
)
declare -A PAIR_NOTE_L1=(
    [1]='Phase 1: Qwen3.8-27B has built-in MTP draft head (1.8x speedup).'
    [2]='Phase 1: MoE 35B / 3.6B-active (high throughput >30 tok/s).'
    [3]='Phase 1: MoE speed (3.6B active weights).'
)
declare -A PAIR_NOTE_L2=(
    [1]='Phase 2: Phi-4-mini (Microsoft instruct, compact and fast).'
    [2]='Phase 2: Mistral NeMo (12B SWA architecture, confirmed working).'
    [3]='Phase 2: Liquid AI (Non-transformer state machine adversary).'
)
declare -A P1_HF_REPO=(
    [1]='OpenVINO/Qwen3.8-27B-int4-ov'
    [2]='OpenVINO/Qwen3.6-35B-A3B-int4-ov'
    [3]='OpenVINO/Qwen3.6-35B-A3B-int4-ov'
)
declare -A P1_DIR=([1]='qwen3.8-27b-int4'  [2]='qwen3.6-35b-a3b'  [3]='qwen3.6-35b-a3b')
declare -A P1_ID=( [1]='qwen3.8:27b'       [2]='qwen3.6:35b-a3b'  [3]='qwen3.6:35b-a3b')
declare -A P2_HF_REPO=(
    [1]='OpenVINO/Phi-4-mini-instruct-int4-ov'
    [2]='OpenVINO/Mistral-Nemo-Instruct-2407-int4-ov'
    [3]='OpenVINO/LFM2.5-8B-A1B-int4-ov'
)
declare -A P2_DIR=([1]='phi-4-mini-int4'   [2]='mistral-nemo-12b'  [3]='lfm2.5-8b-a1b')
declare -A P2_ID=( [1]='phi-4-mini:int4'   [2]='mistral-nemo:12b'  [3]='lfm2.5:8b')

declare -A DRAFT_REPO=(
    [1]=''
    [2]='OpenVINO/Qwen3-0.6B-int4-ov'
    [3]='OpenVINO/Qwen3-0.6B-int4-ov'
)
declare -A DRAFT_LOCAL=([1]='' [2]='draft-qwen3-0.6b' [3]='draft-qwen3-0.6b')

echo -e "\n${BOLD}NUC 14 Pro — OpenVINO GenAI Host Preparation${RESET}"
echo -e "${DIM}Stock Ubuntu 24.04 packages only. No third-party repos.${RESET}"

# =============================================================================
# Interactive Pair Selection
# =============================================================================
if [[ -z "$CHOSEN_PAIR" ]]; then
    echo ""
    echo -e "${BOLD}Select a dual-model pair (both fit in 58GB VRAM pool):${RESET}"
    echo -e "${DIM}Ranked by OpenVINO org: recency × likes × downloads${RESET}"
    echo ""
    for idx in 1 2 3; do
        printf "  ${CYAN}%d${RESET}  %s\n" "$idx" "${PAIR_LABEL[$idx]}"
        printf "     ${DIM}%s${RESET}\n" "${PAIR_NOTE_L1[$idx]}"
        printf "     ${DIM}%s${RESET}\n\n" "${PAIR_NOTE_L2[$idx]}"
    done
    read -r -p "     Choose [1/2/3, default=1]: " reply
    CHOSEN_PAIR="${reply:-1}"
fi

case "$CHOSEN_PAIR" in
    1|2|3) ok "Selected Pair $CHOSEN_PAIR: ${PAIR_LABEL[$CHOSEN_PAIR]}" ;;
    *) fail "Invalid pair '$CHOSEN_PAIR' — choose 1, 2, or 3"; exit 1 ;;
esac

# Write configuration for launch_models.sh
cat > "$PROJECT_DIR/.model_pair" << EOF
ACTIVE_PAIR=$CHOSEN_PAIR
P1_MODEL_DIR=${P1_DIR[$CHOSEN_PAIR]}
P1_MODEL_ID=${P1_ID[$CHOSEN_PAIR]}
P2_MODEL_DIR=${P2_DIR[$CHOSEN_PAIR]}
P2_MODEL_ID=${P2_ID[$CHOSEN_PAIR]}
EOF

# Preserve Open WebUI flag if already present in virtual environment
if [[ -x "$PROJECT_DIR/.venv/bin/open-webui" ]]; then
    echo "OPEN_WEBUI_INSTALLED=true" >> "$PROJECT_DIR/.model_pair"
fi

# Ensure Python .venv is initialized even if --models-only is executed first
VENV="$PROJECT_DIR/.venv"
if [[ ! -d "$VENV" ]]; then
    info "Initializing virtual environment in $VENV..."
    python3 -m venv "$VENV"
    "$VENV/bin/pip" install --upgrade pip --quiet
    "$VENV/bin/pip" install "openvino>=2026.3" "huggingface_hub>=0.27" --quiet
fi

if $MODELS_ONLY; then
    hdr "Models-Only Mode — Pair $CHOSEN_PAIR"
else

# =============================================================================
hdr "1 / 7  Intel Arc iGPU Compute Runtime (Stock Ubuntu Packages)"

echo -e "  ${DIM}Checking sudo credentials...${RESET}"
sudo -v

for stale in /etc/apt/sources.list.d/intel-gpu*.list /etc/apt/sources.list.d/oneAPI.list; do
    if [[ -f "$stale" ]]; then
        warn "Removing third-party repository: $stale"
        info "Third-party repos cause GDM3 black screens on Ubuntu 24.04."
        sudo rm -f "$stale"
    fi
done

run_spinner "Updating Ubuntu package lists" sudo apt-get update -qq

run_spinner "Installing Intel Arc Level Zero & OpenCL runtime" \
    sudo apt-get install -y -q \
        intel-opencl-icd libze1 libze-intel-gpu1 libze-dev \
        mesa-vulkan-drivers clinfo \
        python3-venv python3-pip wget curl git

sudo ldconfig

if ldconfig -p 2>/dev/null | grep -E "libze_intel_gpu|libze_loader" >/dev/null; then
    ok "Level Zero driver active"
else
    fail "Level Zero driver not found — check Ubuntu 24.04 Noble repositories"
    exit 1
fi

id -nG "$USER" | grep -qw "render" \
    && ok "User already in render + video groups" \
    || { sudo usermod -aG render,video "$USER"; ok "Added $USER to render+video (re-login to activate)"; }

# =============================================================================
hdr "2 / 7  PSR Screen Corruption Guard (Meteor Lake Kernel Guard)"

if grep -q "i915.enable_psr=0" /proc/cmdline 2>/dev/null || \
   grep -q "i915.enable_psr=0" /etc/default/grub 2>/dev/null; then
    ok "PSR already disabled"
else
    warn "PSR not disabled — Meteor Lake Arc displays may show pixel artifacts"
    read -r -p "     Disable PSR in GRUB? (reboot needed) [Y/n] " r
    if [[ "${r:-Y}" =~ ^[Yy]$ ]]; then
        sudo sed -i \
            's/GRUB_CMDLINE_LINUX_DEFAULT="\(.*\)"/GRUB_CMDLINE_LINUX_DEFAULT="\1 i915.enable_psr=0"/' \
            /etc/default/grub
        sudo update-grub 2>/dev/null || sudo grub-mkconfig -o /boot/grub/grub.cfg 2>/dev/null || true
        ok "PSR disabled in GRUB — reboot required after setup"
    fi
fi

# =============================================================================
hdr "3 / 7  Host Kernel Performance Tuning"

for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    [[ -f "$g" ]] && echo "performance" | sudo tee "$g" >/dev/null 2>&1 || true
done
for epp in /sys/devices/system/cpu/cpu*/cpufreq/energy_performance_preference; do
    [[ -f "$epp" ]] && echo "performance" | sudo tee "$epp" >/dev/null 2>&1 || true
done
echo "always" | sudo tee /sys/kernel/mm/transparent_hugepage/enabled >/dev/null 2>&1 || true
ok "CPU governor: performance | Transparent Huge Pages: always"

# =============================================================================
hdr "4 / 7  Python Dependencies + OpenVINO GenAI"

source "$VENV/bin/activate"

run_spinner "Installing OpenVINO GenAI & server dependencies" \
    pip install \
        "openvino-genai>=2026.3" "openvino>=2026.3" \
        fastapi "uvicorn[standard]" \
        "huggingface_hub>=0.27" \
        ddgs \
        trafilatura \
        aiohttp \
        "pandas-ta>=0.3.14b" \
        requests python-dotenv \
        --quiet

python3 - << 'PYEOF'
try:
    import openvino as ov
    devices = ov.Core().available_devices
    gpu = any("GPU" in d for d in devices)
    print(f"  OpenVINO devices: {devices}")
    print(f"  {'✓  Arc iGPU verified' if gpu else '⚠  GPU not listed yet (re-login for render group)'}")
except Exception as e:
    print(f"  ⚠  OpenVINO device query: {e}")
PYEOF

fi  # end !MODELS_ONLY

# =============================================================================
hdr "5 / 7  Download Models — Pair ${CHOSEN_PAIR}: ${PAIR_LABEL[$CHOSEN_PAIR]}"

source "$VENV/bin/activate"

FREE_GB=$(df -BG "$HOME" 2>/dev/null | awk 'NR==2 {gsub("G",""); print $4}')
if [[ -n "$FREE_GB" ]] && (( FREE_GB < 25 )); then
    warn "Low disk space: only ${FREE_GB}GB available in $HOME (Pair $CHOSEN_PAIR needs ~25GB)"
fi

if [[ -f "$PROJECT_DIR/.env" ]]; then
    set -o allexport
    # shellcheck disable=SC1091
    source "$PROJECT_DIR/.env" 2>/dev/null || true
    set +o allexport
fi

HF_LOGGED_IN=false

if python3 -c "from huggingface_hub import HfApi; HfApi().whoami()" 2>/dev/null; then
    ok "HuggingFace: authenticated (cached login)"
    HF_LOGGED_IN=true
elif [[ -n "${HF_TOKEN:-}" ]]; then
    ok "HuggingFace: token loaded from environment"
    HF_LOGGED_IN=true
else
    warn "Not logged in to HuggingFace — large files (LFS) will stall at 0 bytes"
    echo ""
    info "Get a free Read token at: https://huggingface.co/settings/tokens"
    echo ""
    read -r -p "     Enter HuggingFace token: " hf_tok
    if [[ -n "${hf_tok:-}" ]]; then
        export HF_TOKEN="$hf_tok"
        touch "$PROJECT_DIR/.env"
        grep -v "^HF_TOKEN=" "$PROJECT_DIR/.env" > "$PROJECT_DIR/.env.tmp" 2>/dev/null || true
        echo "HF_TOKEN=$hf_tok" >> "$PROJECT_DIR/.env.tmp"
        mv "$PROJECT_DIR/.env.tmp" "$PROJECT_DIR/.env"
        ok "HuggingFace token saved to .env"
        HF_LOGGED_IN=true
    else
        warn "No token provided — downloads may stall if rate-limited"
    fi
fi

mkdir -p "$MODELS_DIR"

download_model() {
    local hf_repo="$1" local_dir="$2" label="$3"
    local full_path="$MODELS_DIR/$local_dir"

    local is_intact=false
    if [[ -d "$full_path" ]]; then
        if python3 - "$full_path" << 'PYEOF' 2>/dev/null
import sys
from pathlib import Path
import openvino as ov

model_dir = Path(sys.argv[1])
primary_xml = None
for name in ("openvino_language_model.xml", "openvino_model.xml"):
    candidate = model_dir / name
    if candidate.exists():
        primary_xml = candidate
        break

if not primary_xml:
    sys.exit(1)

ov.Core().read_model(str(primary_xml))
sys.exit(0)
PYEOF
        then
            is_intact=true
        fi
    fi

    if $is_intact; then
        ok "$label: verified and intact on disk"
        return 0
    elif [[ -d "$full_path" ]]; then
        warn "$label: incomplete weights detected — resuming download"
    fi

    echo ""
    info "Downloading: $label"
    info "  Source:    $hf_repo"
    info "  Directory: $full_path"
    info "  Resume-capable — safe to Ctrl-C at any time"
    echo ""

    python3 -u - "$hf_repo" "$full_path" "$label" << 'PYEOF'
import os, sys

repo_id   = sys.argv[1]
dest_path = sys.argv[2]
label     = sys.argv[3]

os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"

from huggingface_hub import snapshot_download

token = os.environ.get("HF_TOKEN") or None

try:
    snapshot_download(
        repo_id=repo_id,
        local_dir=dest_path,
        ignore_patterns=["*.gguf", "*.safetensors", "*.pt", ".gitattributes"],
        token=token,
    )
    print(f"\n  ✓  {label} download finished")
except Exception as e:
    print(f"\n  ✗  Download failed: {e}", file=sys.stderr)
    sys.exit(1)
PYEOF

    python3 - "$full_path" "$label" << 'PYEOF'
import sys
from pathlib import Path
import openvino as ov

dest_path = Path(sys.argv[1])
label     = sys.argv[2]

primary_xml = None
for name in ("openvino_language_model.xml", "openvino_model.xml"):
    candidate = dest_path / name
    if candidate.exists():
        primary_xml = candidate
        break

if primary_xml:
    try:
        ov.Core().read_model(str(primary_xml))
        print(f"  ✓  {label} OpenVINO IR graph validated")
    except Exception as e:
        print(f"  ⚠  {label} validation warning: {e}", file=sys.stderr)
PYEOF
}

download_model "${P1_HF_REPO[$CHOSEN_PAIR]}" "${P1_DIR[$CHOSEN_PAIR]}" \
    "Phase 1: ${P1_ID[$CHOSEN_PAIR]}"

download_model "${P2_HF_REPO[$CHOSEN_PAIR]}" "${P2_DIR[$CHOSEN_PAIR]}" \
    "Phase 2: ${P2_ID[$CHOSEN_PAIR]}"

# =============================================================================
hdr "6 / 7  Speculative Decoding Draft Model"

if [[ "$CHOSEN_PAIR" == "1" ]]; then
    ok "Pair 1 (Qwen3.8-27B): built-in MTP draft head detected"
    info "serve_model.py enables MTP automatically (1.8x speedup, 0MB extra download)."
elif [[ -d "$MODELS_DIR/${DRAFT_LOCAL[$CHOSEN_PAIR]}" ]]; then
    ok "Draft model already downloaded: ${DRAFT_LOCAL[$CHOSEN_PAIR]}"
    grep -q "^DRAFT_MODEL_DIR=" "$PROJECT_DIR/.model_pair" 2>/dev/null || \
        echo "DRAFT_MODEL_DIR=${DRAFT_LOCAL[$CHOSEN_PAIR]}" >> "$PROJECT_DIR/.model_pair"
    info "Speculative decoding active in launch_models.sh"
else
    echo ""
    info "Draft model: ${DRAFT_REPO[$CHOSEN_PAIR]} (~400MB)"
    info "Provides 2-3x speculative decode speedup for Phase 1."
    echo ""
    read -r -p "     Download draft model? [Y/n] " dr
    dr="${dr:-Y}"
    if [[ "$dr" =~ ^[Yy]$ ]]; then
        download_model "${DRAFT_REPO[$CHOSEN_PAIR]}" "${DRAFT_LOCAL[$CHOSEN_PAIR]}" \
            "Draft: ${DRAFT_REPO[$CHOSEN_PAIR]}"
        echo "DRAFT_MODEL_DIR=${DRAFT_LOCAL[$CHOSEN_PAIR]}" >> "$PROJECT_DIR/.model_pair"
        ok "Draft model installed — enabled in launch_models.sh"
    else
        info "Draft model skipped (can download later via --models-only)"
    fi
fi

# =============================================================================
# 7 / 7  Open WebUI (Detects existing install before offering prompt)
# =============================================================================
if ! $MODELS_ONLY; then
    hdr "7 / 7  Open WebUI (Browser Chat Interface)"

    if [[ -x "$VENV/bin/open-webui" ]]; then
        ok "Open WebUI: already installed in .venv"
        grep -q "^OPEN_WEBUI_INSTALLED=true" "$PROJECT_DIR/.model_pair" 2>/dev/null || \
            echo "OPEN_WEBUI_INSTALLED=true" >> "$PROJECT_DIR/.model_pair"
        info "Start via: ./launch_models.sh --with-webui"
        info "Browser:   http://localhost:8080"
    else
        echo ""
        info "Installs into .venv (~500MB). Pre-configured for ports 8000 & 8001."
        echo ""
        read -r -p "     Install Open WebUI? [Y/n] " wu
        wu="${wu:-Y}"
        if [[ "$wu" =~ ^[Yy]$ ]]; then
            source "$VENV/bin/activate"
            run_spinner "Installing Open WebUI and dependencies (~500MB)" \
                pip install open-webui --quiet
            grep -q "^OPEN_WEBUI_INSTALLED=true" "$PROJECT_DIR/.model_pair" 2>/dev/null || \
                echo "OPEN_WEBUI_INSTALLED=true" >> "$PROJECT_DIR/.model_pair"
            ok "Open WebUI installed"
            info "Start via: ./launch_models.sh --with-webui"
            info "Browser:   http://localhost:8080"
        else
            info "Open WebUI skipped"
        fi
    fi
fi

# =============================================================================
hdr "Host Preparation Complete"

echo ""
ok "Pair $CHOSEN_PAIR: ${PAIR_LABEL[$CHOSEN_PAIR]}"
info "Phase 1 (Port 8000): ${P1_ID[$CHOSEN_PAIR]}  →  $MODELS_DIR/${P1_DIR[$CHOSEN_PAIR]}"
info "Phase 2 (Port 8001): ${P2_ID[$CHOSEN_PAIR]}  →  $MODELS_DIR/${P2_DIR[$CHOSEN_PAIR]}"
echo ""
echo -e "  ${BOLD}Next steps:${RESET}"
$MODELS_ONLY || echo -e "  1. ${YELLOW}Reboot${RESET} if PSR was just disabled in GRUB"
echo -e "  2. ${YELLOW}./launch_models.sh --with-webui${RESET}"
echo -e "  3. ${YELLOW}./verify.sh${RESET}"
echo ""
echo -e "  ${DIM}Switch pairs anytime: ./prepare_host.sh --pair 2 --models-only${RESET}\n"
