#!/usr/bin/env bash
# =============================================================================
# Trading Income Project — Host Preparation & Model Downloader
# NUC 14 Pro · Intel Core Ultra 5 125H · 58GB iGPU VRAM · Ubuntu 24.04
# =============================================================================
# Run ONCE on a fresh Ubuntu 24.04 install.
# Uses ONLY stock Ubuntu packages — no third-party Intel GPU repositories.
# (repositories.intel.com/gpu overwrites the Ubuntu display stack and causes
#  GDM3 black screens on boot — this script deliberately avoids that risk.)
#
# Usage:
#   ./prepare_host.sh                    # interactive pair selection
#   ./prepare_host.sh --pair 1           # Pair 1: DeepSeek-R1-32B + Mistral-Nemo
#   ./prepare_host.sh --pair 2           # Pair 2: Qwen3-30B-A3B + Mistral-Nemo
#   ./prepare_host.sh --pair 3           # Pair 3: Qwen2.5-32B + Mistral-Small-24B
#   ./prepare_host.sh --models-only      # skip drivers, download models only
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

MODELS_DIR="${MODELS_DIR:-$HOME/models}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_ONLY=false
CHOSEN_PAIR=""

for arg in "$@"; do
    case "$arg" in
        --pair)     shift; CHOSEN_PAIR="${1:-}"; shift ;;
        --pair=*)   CHOSEN_PAIR="${arg#--pair=}" ;;
        --models-only) MODELS_ONLY=true ;;
    esac
done

# =============================================================================
# Model pair definitions
# =============================================================================
declare -A PAIR_LABEL=(
    [1]='Thinking Quant      — DeepSeek-R1-32B + Mistral-Nemo-12B  (~26.3GB)'
    [2]='High-Speed Agentic  — Qwen3-30B-A3B   + Mistral-Nemo-12B  (~25.2GB)'
    [3]='Frontier Heavyweight— Qwen2.5-32B     + Mistral-Small-24B (~33.2GB)'
)
declare -A P1_HF_REPO=(
    [1]='OpenVINO/DeepSeek-R1-Distill-Qwen-32B-int4-ov'
    [2]='OpenVINO/Qwen3-30B-A3B-Instruct-int4-ov'
    [3]='OpenVINO/Qwen2.5-32B-Instruct-int4-ov'
)
declare -A P1_DIR=(
    [1]='deepseek-r1-32b'
    [2]='qwen3-30b-a3b'
    [3]='qwen2.5-32b'
)
declare -A P1_ID=(
    [1]='deepseek-r1:32b'
    [2]='qwen3:30b-a3b'
    [3]='qwen2.5:32b'
)
declare -A P2_HF_REPO=(
    [1]='OpenVINO/Mistral-Nemo-Instruct-2407-int4-ov'
    [2]='OpenVINO/Mistral-Nemo-Instruct-2407-int4-ov'
    [3]='OpenVINO/Mistral-Small-24B-Instruct-2501-int4-ov'
)
declare -A P2_DIR=(
    [1]='mistral-nemo-12b'
    [2]='mistral-nemo-12b'
    [3]='mistral-small-24b'
)
declare -A P2_ID=(
    [1]='mistral-nemo:12b'
    [2]='mistral-nemo:12b'
    [3]='mistral-small:24b'
)

echo -e "\n${BOLD}NUC 14 Pro — OpenVINO GenAI Host Preparation${RESET}"
echo -e "${DIM}Stock Ubuntu 24.04 packages only. No third-party repos.${RESET}"

# =============================================================================
# Pair selection
# =============================================================================
if [[ -z "$CHOSEN_PAIR" ]]; then
    echo ""
    echo -e "${BOLD}Select a dual-model pair (both reside in 58GB VRAM simultaneously):${RESET}"
    echo ""
    printf "  ${CYAN}1${RESET}  %s\n" "${PAIR_LABEL[1]}"
    printf "  ${DIM}   Best for: o1-class step-by-step reasoning on trade data${RESET}\n"
    printf "  ${CYAN}2${RESET}  %s\n" "${PAIR_LABEL[2]}"
    printf "  ${DIM}   Best for: fast multi-turn tool loops (30+ tok/s)${RESET}\n"
    printf "  ${CYAN}3${RESET}  %s\n" "${PAIR_LABEL[3]}"
    printf "  ${DIM}   Best for: peak JSON/tool precision + deep adversarial critique${RESET}\n"
    echo ""
    read -r -p "     Choose [1/2/3, default=1]: " reply
    CHOSEN_PAIR="${reply:-1}"
fi

case "$CHOSEN_PAIR" in
    1|2|3) ok "Selected Pair $CHOSEN_PAIR: ${PAIR_LABEL[$CHOSEN_PAIR]}" ;;
    *) fail "Invalid pair '$CHOSEN_PAIR' — choose 1, 2, or 3"; exit 1 ;;
esac

# Write selection to .env for launch_models.sh to read
echo "ACTIVE_PAIR=$CHOSEN_PAIR" > "$PROJECT_DIR/.model_pair"
echo "P1_MODEL_DIR=${P1_DIR[$CHOSEN_PAIR]}" >> "$PROJECT_DIR/.model_pair"
echo "P1_MODEL_ID=${P1_ID[$CHOSEN_PAIR]}" >> "$PROJECT_DIR/.model_pair"
echo "P2_MODEL_DIR=${P2_DIR[$CHOSEN_PAIR]}" >> "$PROJECT_DIR/.model_pair"
echo "P2_MODEL_ID=${P2_ID[$CHOSEN_PAIR]}" >> "$PROJECT_DIR/.model_pair"

if $MODELS_ONLY; then
    hdr "Downloading models for Pair $CHOSEN_PAIR"
else

# =============================================================================
hdr "1 / 6  Intel Arc iGPU Compute Runtime (stock Ubuntu only)"

# Remove third-party Intel GPU repo if present — it breaks GDM3 on boot
for stale in /etc/apt/sources.list.d/intel-gpu*.list /etc/apt/sources.list.d/oneAPI.list; do
    if [[ -f "$stale" ]]; then
        warn "Removing $stale (third-party Intel repo causes black screens)"
        sudo rm -f "$stale"
    fi
done

sudo apt-get update -qq
sudo apt-get install -y -qq \
    intel-opencl-icd libze1 libze-intel-gpu1 libze-dev \
    mesa-vulkan-drivers clinfo \
    python3-venv python3-pip wget curl git

sudo ldconfig

if ldconfig -p 2>/dev/null | grep -E "libze_intel_gpu|libze_loader" >/dev/null; then
    ok "Level Zero GPU driver active"
else
    fail "Level Zero driver not found — check Ubuntu 24.04 Noble repos"
    exit 1
fi

id -nG "$USER" | grep -qw "render" && ok "User in render group" \
    || { sudo usermod -aG render,video "$USER"; ok "Added to render+video (re-login to activate)"; }

# =============================================================================
hdr "2 / 6  PSR Screen Corruption Guard (Meteor Lake regression, kernel 6.10+)"

if grep -q "i915.enable_psr=0" /proc/cmdline 2>/dev/null || \
   grep -q "i915.enable_psr=0" /etc/default/grub 2>/dev/null; then
    ok "PSR already disabled"
else
    warn "PSR not disabled — may cause pixel artifacts on Arc iGPU"
    read -r -p "     Disable PSR to prevent display corruption? (reboot needed) [Y/n] " r
    if [[ "${r:-Y}" =~ ^[Yy]$ ]]; then
        sudo sed -i 's/GRUB_CMDLINE_LINUX_DEFAULT="\(.*\)"/GRUB_CMDLINE_LINUX_DEFAULT="\1 i915.enable_psr=0"/' \
            /etc/default/grub
        sudo update-grub 2>/dev/null || sudo grub-mkconfig -o /boot/grub/grub.cfg 2>/dev/null || true
        ok "PSR disabled in GRUB — reboot required"
    fi
fi

# =============================================================================
hdr "3 / 6  Host Kernel Performance Tuning"

for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    [[ -f "$g" ]] && echo "performance" | sudo tee "$g" >/dev/null 2>&1 || true
done
for epp in /sys/devices/system/cpu/cpu*/cpufreq/energy_performance_preference; do
    [[ -f "$epp" ]] && echo "performance" | sudo tee "$epp" >/dev/null 2>&1 || true
done
echo "always" | sudo tee /sys/kernel/mm/transparent_hugepage/enabled >/dev/null 2>&1 || true
ok "CPU governor: performance | THP: always"

# =============================================================================
hdr "4 / 6  Python Virtual Environment + OpenVINO GenAI"

VENV="$PROJECT_DIR/.venv"
[[ -d "$VENV" ]] || python3 -m venv "$VENV"
ok ".venv: $VENV"

source "$VENV/bin/activate"
pip install --upgrade pip --quiet
# openvino-genai 2026.3: Prompt Lookup Decoding on GPU, EAGLE-3 speculative decoding,
# 28-36% faster prefill than 2026.1 on Arc iGPU (benchmarked on Lunar Lake / Arc iGPU)
pip install "openvino-genai>=2026.3" "openvino>=2026.3" \
    fastapi "uvicorn[standard]" \
    huggingface_hub \
    "ddgs" \
    trafilatura \
    aiohttp \
    "pandas-ta>=0.3.14b" \
    requests python-dotenv --quiet
# ddgs: replaces deprecated duckduckgo_search package (renamed Oct 2025)
# trafilatura: used by tool_runner.py fetch_url for full article text extraction
ok "openvino-genai 2026.3+, fastapi, uvicorn, huggingface_hub, ddgs, trafilatura installed"

python3 - << 'PYEOF'
try:
    import openvino as ov
    devices = ov.Core().available_devices
    gpu = any("GPU" in d for d in devices)
    print(f"  OpenVINO devices: {devices}")
    print(f"  {'✓  GPU visible' if gpu else '⚠  GPU not yet listed (normal before reboot)'}")
except Exception as e:
    print(f"  ⚠  OpenVINO check: {e}")
PYEOF

fi  # end of !MODELS_ONLY block

# =============================================================================
hdr "5 / 6  Download Models — Pair ${CHOSEN_PAIR}: ${PAIR_LABEL[$CHOSEN_PAIR]}"

mkdir -p "$MODELS_DIR"

download_model() {
    local hf_repo="$1" local_dir="$2" label="$3"
    local full_path="$MODELS_DIR/$local_dir"
    if [[ -f "$full_path/openvino_model.xml" ]] || \
       [[ -f "$full_path/openvino_language_model.xml" ]]; then
        ok "$label: already downloaded"
        return
    fi
    echo ""
    info "Downloading $label (~$(du -sh "$full_path" 2>/dev/null | cut -f1 || echo '?'))"
    info "From: $hf_repo"
    info "To:   $full_path"
    huggingface-cli download "$hf_repo" \
        --local-dir "$full_path" \
        --exclude "*.gguf" "*.bin" ".gitattributes"
    ok "$label downloaded"
}

download_model "${P1_HF_REPO[$CHOSEN_PAIR]}" "${P1_DIR[$CHOSEN_PAIR]}" \
    "Phase 1: ${P1_ID[$CHOSEN_PAIR]}"
download_model "${P2_HF_REPO[$CHOSEN_PAIR]}" "${P2_DIR[$CHOSEN_PAIR]}" \
    "Phase 2: ${P2_ID[$CHOSEN_PAIR]}"

# =============================================================================
# =============================================================================
# Draft model for speculative decoding (optional, ~1GB)
# 2-3x decode speedup. Draft proposes tokens; main model verifies.
# Must share tokenizer family with Phase 1 model:
#   Pair 1 (DeepSeek-R1 = Qwen2.5 family) → Qwen2.5-1.5B draft
#   Pair 2 (Qwen3-30B-A3B = Qwen3 family) → Qwen3-0.6B draft
#   Pair 3 (Qwen2.5-32B  = Qwen2.5 family) → Qwen2.5-1.5B draft
declare -A DRAFT_REPO=(
    [1]="OpenVINO/Qwen2.5-1.5B-Instruct-int4-ov"
    [2]="OpenVINO/Qwen3-0.6B-int4-ov"
    [3]="OpenVINO/Qwen2.5-1.5B-Instruct-int4-ov"
)
declare -A DRAFT_LOCAL=(
    [1]="draft-qwen2.5-1.5b"
    [2]="draft-qwen3-0.6b"
    [3]="draft-qwen2.5-1.5b"
)
echo ""
echo -e "  ${BOLD}Speculative decoding draft model (~1GB, 2-3x decode speedup):${RESET}"
read -r -p "     Download draft model for Pair $CHOSEN_PAIR? [Y/n] " dr
dr="${dr:-Y}"
if [[ "$dr" =~ ^[Yy]$ ]]; then
    download_model "${DRAFT_REPO[$CHOSEN_PAIR]}" "${DRAFT_LOCAL[$CHOSEN_PAIR]}"         "Draft: ${DRAFT_REPO[$CHOSEN_PAIR]}"
    echo "DRAFT_MODEL_DIR=${DRAFT_LOCAL[$CHOSEN_PAIR]}" >> "$PROJECT_DIR/.model_pair"
    ok "Draft model ready — launch_models.sh will enable speculative decoding automatically"
else
    info "Skipped. Download later: ./prepare_host.sh --pair $CHOSEN_PAIR --models-only"
fi

# =============================================================================
hdr "6 / 6  Open WebUI (pip — ChatGPT-style desktop chat interface)"

# Open WebUI is a large install (~500MB of frontend assets).
# It runs as a local web app at http://localhost:8080, connecting
# to the OpenVINO model servers already configured on ports 8000/8001.
# WEBUI_AUTH=False: single-user local mode, no login required.
echo ""
read -r -p "     Install Open WebUI chat interface? (large download ~500MB) [Y/n] " wu
wu="${wu:-Y}"
if [[ "$wu" =~ ^[Yy]$ ]]; then
    source "$VENV/bin/activate"
    pip install open-webui --quiet
    ok "Open WebUI installed"
    info "Start:  ./launch_models.sh --with-webui"
    info "Access: http://localhost:8080"
    echo "OPEN_WEBUI_INSTALLED=true" >> "$PROJECT_DIR/.model_pair"
else
    info "Skipped. Install later: source .venv/bin/activate && pip install open-webui"
fi

hdr "6 / 6  Summary"

echo ""
ok "Pair $CHOSEN_PAIR: ${PAIR_LABEL[$CHOSEN_PAIR]}"
info "Phase 1 (Port 8000): ${P1_ID[$CHOSEN_PAIR]}  at $MODELS_DIR/${P1_DIR[$CHOSEN_PAIR]}"
info "Phase 2 (Port 8001): ${P2_ID[$CHOSEN_PAIR]}  at $MODELS_DIR/${P2_DIR[$CHOSEN_PAIR]}"
echo ""
info "Next steps:"
$MODELS_ONLY || info "  1. Reboot if PSR was just disabled"
info "  2. ./launch_models.sh     (start both servers)"
info "  3. ./setup.sh             (verify endpoints)"
info ""
info "To switch pairs:  ./prepare_host.sh --pair 2 --models-only"
echo ""
