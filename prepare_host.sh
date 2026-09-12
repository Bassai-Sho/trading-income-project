#!/usr/bin/env bash
# =============================================================================
# Trading Income Project — Host Preparation & Model Downloader
# NUC 14 Pro · Intel Core Ultra 5 125H · 58GB iGPU VRAM · Ubuntu 24.04
# =============================================================================
# Uses ONLY stock Ubuntu packages — no third-party Intel GPU repositories.
# Downloads via Python huggingface_hub.snapshot_download — immune to
# huggingface-cli / hf CLI renames and --exclude flag changes.
#
# Model pairs ranked by OpenVINO org: recency × likes × downloads (Sep 2026):
#   Pair 1: Qwen3.8-27B (♡17, 3.52k dl, 22 days) + Phi-4-mini (♡5, Jul 6)
#   Pair 2: Qwen3.6-35B-A3B MoE (♡11) + Mistral-Nemo (SWA, confirmed working)
#   Pair 3: Qwen3.6-35B-A3B MoE + LFM2.5-8B-A1B (Liquid AI — not a transformer)
#
# Usage:
#   ./prepare_host.sh                    # interactive pair selection
#   ./prepare_host.sh --pair 1           # Pair 1: Qwen3.8-27B + Phi-4-mini
#   ./prepare_host.sh --pair 2           # Pair 2: Qwen3.6-35B-A3B + Mistral-Nemo
#   ./prepare_host.sh --pair 3           # Pair 3: Qwen3.6-35B-A3B + LFM2.5-8B
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
        --pair)        shift; CHOSEN_PAIR="${1:-}"; shift ;;
        --pair=*)      CHOSEN_PAIR="${arg#--pair=}" ;;
        --models-only) MODELS_ONLY=true ;;
    esac
done

# =============================================================================
# Model pair definitions
# All confirmed present in OpenVINO HuggingFace org (verified Sep 2026).
# Ranked by: likes × recency × downloads from official OpenVINO model page.
#
# Qwen3.8-27B: listed as Image-Text-to-Text (VLM) but LLMPipeline uses
#   text component only — text-only chat works perfectly.
# LFM2.5-8B-A1B: Liquid Foundation Model (Liquid AI) — not a transformer.
#   Liquid state machine architecture. Maximum adversarial independence from Qwen.
# Qwen3-8B-int4-cw: channel-wise INT4 — better accuracy than standard sym INT4.
# =============================================================================
declare -A PAIR_LABEL=(
    [1]='Most Popular    — Qwen3.8-27B-int4 + Phi-4-mini-int4         (~19GB)'
    [2]='MoE Speed       — Qwen3.6-35B-A3B-int4 + Mistral-Nemo-int4   (~25GB)'
    [3]='Novel Adversary — Qwen3.6-35B-A3B-int4 + LFM2.5-8B-A1B-int4 (~23GB)'
)
declare -A PAIR_NOTE=(
    [1]='Phase 1: ♡17 most-liked, 3.52k downloads, 22 days old. Phase 2: Phi-4 (Microsoft, different family from Qwen)'
    [2]='Phase 1: MoE 35B/3.6B-active, 30+ tok/s. Phase 2: Mistral NeMo SWA confirmed working'
    [3]='Phase 1: MoE speed. Phase 2: Liquid AI (NOT a transformer — maximum adversarial independence)'
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

# Draft models for speculative decoding (share tokenizer family with Phase 1)
declare -A DRAFT_REPO=(
    [1]='OpenVINO/Qwen3-0.6B-int4-ov'
    [2]='OpenVINO/Qwen3-0.6B-int4-ov'
    [3]='OpenVINO/Qwen3-0.6B-int4-ov'
)
declare -A DRAFT_LOCAL=([1]='draft-qwen3-0.6b' [2]='draft-qwen3-0.6b' [3]='draft-qwen3-0.6b')

echo -e "\n${BOLD}NUC 14 Pro — OpenVINO GenAI Host Preparation${RESET}"
echo -e "${DIM}Stock Ubuntu 24.04 packages only. No third-party repos.${RESET}"

# =============================================================================
# Pair selection
# =============================================================================
if [[ -z "$CHOSEN_PAIR" ]]; then
    echo ""
    echo -e "${BOLD}Select a dual-model pair (both reside in 58GB VRAM simultaneously):${RESET}"
    echo -e "${DIM}Ranked by OpenVINO org: recency × likes × downloads${RESET}"
    echo ""
    printf "  ${CYAN}1${RESET}  %s\n" "${PAIR_LABEL[1]}"
    printf "  ${DIM}   %s${RESET}\n" "${PAIR_NOTE[1]}"
    echo ""
    printf "  ${CYAN}2${RESET}  %s\n" "${PAIR_LABEL[2]}"
    printf "  ${DIM}   %s${RESET}\n" "${PAIR_NOTE[2]}"
    echo ""
    printf "  ${CYAN}3${RESET}  %s\n" "${PAIR_LABEL[3]}"
    printf "  ${DIM}   %s${RESET}\n" "${PAIR_NOTE[3]}"
    echo ""
    read -r -p "     Choose [1/2/3, default=1]: " reply
    CHOSEN_PAIR="${reply:-1}"
fi

case "$CHOSEN_PAIR" in
    1|2|3) ok "Selected Pair $CHOSEN_PAIR: ${PAIR_LABEL[$CHOSEN_PAIR]}" ;;
    *) fail "Invalid pair '$CHOSEN_PAIR' — choose 1, 2, or 3"; exit 1 ;;
esac

# Write config for launch_models.sh
cat > "$PROJECT_DIR/.model_pair" << EOF
ACTIVE_PAIR=$CHOSEN_PAIR
P1_MODEL_DIR=${P1_DIR[$CHOSEN_PAIR]}
P1_MODEL_ID=${P1_ID[$CHOSEN_PAIR]}
P2_MODEL_DIR=${P2_DIR[$CHOSEN_PAIR]}
P2_MODEL_ID=${P2_ID[$CHOSEN_PAIR]}
EOF

if $MODELS_ONLY; then
    hdr "Models-only — downloading Pair $CHOSEN_PAIR"
else

# =============================================================================
hdr "1 / 7  Intel Arc iGPU Compute Runtime (stock Ubuntu packages only)"

for stale in /etc/apt/sources.list.d/intel-gpu*.list /etc/apt/sources.list.d/oneAPI.list; do
    if [[ -f "$stale" ]]; then
        warn "Removing $stale — third-party Intel repo causes GDM3 black screens"
        sudo rm -f "$stale"
    fi
done

echo -e "  ${DIM}Updating package lists...${RESET}"
sudo apt-get update -qq 2>/dev/null

echo -e "  ${DIM}Installing Intel Arc compute runtime...${RESET}"
DEBIAN_FRONTEND=noninteractive sudo apt-get install -y -q \
    intel-opencl-icd libze1 libze-intel-gpu1 libze-dev \
    mesa-vulkan-drivers clinfo \
    python3-venv python3-pip wget curl git \
    > /tmp/apt-install.log 2>&1 \
    && ok "Intel Arc compute packages installed" \
    || { warn "apt had warnings — see /tmp/apt-install.log"; }

sudo ldconfig

if ldconfig -p 2>/dev/null | grep -E "libze_intel_gpu|libze_loader" >/dev/null; then
    ok "Level Zero GPU driver active"
else
    fail "Level Zero driver not found — check Ubuntu 24.04 Noble repos"
    exit 1
fi

id -nG "$USER" | grep -qw "render" \
    && ok "User already in render+video groups" \
    || { sudo usermod -aG render,video "$USER"; ok "Added $USER to render+video (re-login to activate)"; }

# =============================================================================
hdr "2 / 7  PSR Screen Corruption Guard (Meteor Lake kernel 6.10+ regression)"

if grep -q "i915.enable_psr=0" /proc/cmdline 2>/dev/null || \
   grep -q "i915.enable_psr=0" /etc/default/grub 2>/dev/null; then
    ok "PSR already disabled"
else
    warn "PSR not disabled — Meteor Lake Arc may show pixel artifacts"
    read -r -p "     Disable PSR to prevent display corruption? (reboot needed) [Y/n] " r
    if [[ "${r:-Y}" =~ ^[Yy]$ ]]; then
        sudo sed -i \
            's/GRUB_CMDLINE_LINUX_DEFAULT="\(.*\)"/GRUB_CMDLINE_LINUX_DEFAULT="\1 i915.enable_psr=0"/' \
            /etc/default/grub
        sudo update-grub 2>/dev/null || sudo grub-mkconfig -o /boot/grub/grub.cfg 2>/dev/null || true
        ok "PSR disabled in GRUB — reboot required"
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
hdr "4 / 7  Python Virtual Environment + OpenVINO GenAI"

VENV="$PROJECT_DIR/.venv"
[[ -d "$VENV" ]] || python3 -m venv "$VENV"
ok ".venv: $VENV"

source "$VENV/bin/activate"
pip install --upgrade pip --quiet

echo -e "  ${DIM}Installing Python dependencies...${RESET}"
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

ok "openvino-genai 2026.3+, fastapi, uvicorn, huggingface_hub 0.27+, ddgs, trafilatura installed"

python3 - << 'PYEOF'
try:
    import openvino as ov
    devices = ov.Core().available_devices
    gpu = any("GPU" in d for d in devices)
    print(f"  OpenVINO devices: {devices}")
    print(f"  {'✓  GPU visible' if gpu else '⚠  GPU not listed yet (re-login for render group)'}")
except Exception as e:
    print(f"  ⚠  OpenVINO check: {e}")
PYEOF

fi  # end !MODELS_ONLY

# =============================================================================
hdr "5 / 7  Download Models — Pair ${CHOSEN_PAIR}: ${PAIR_LABEL[$CHOSEN_PAIR]}"

# HuggingFace authentication — required for large LFS files (model weights)
# Without a token, downloads stall at 0 bytes due to rate limiting.
# Free token: https://huggingface.co/settings/tokens  (Read permission only)
source "$PROJECT_DIR/.venv/bin/activate"

# Load .env so HF_TOKEN saved from a previous run is picked up automatically
if [[ -f "$PROJECT_DIR/.env" ]]; then
    set -o allexport
    # shellcheck disable=SC1091
    source "$PROJECT_DIR/.env" 2>/dev/null || true
    set +o allexport
fi

HF_LOGGED_IN=false

# Check cached login or env var
if python3 -c "from huggingface_hub import HfApi; HfApi().whoami()" 2>/dev/null; then
    ok "HuggingFace: authenticated (cached login)"
    HF_LOGGED_IN=true
elif [[ -n "${HF_TOKEN:-}" ]]; then
    ok "HuggingFace: token from environment"
    HF_LOGGED_IN=true
else
    warn "Not logged in to HuggingFace — large file (LFS) downloads will stall at 0 bytes"
    echo ""
    info "Get a free Read token at: https://huggingface.co/settings/tokens"
    echo ""
    read -r -p "     Enter HuggingFace token (paste and press Enter): " hf_tok
    if [[ -n "${hf_tok:-}" ]]; then
        export HF_TOKEN="$hf_tok"
        # Persist to .env so future runs and launch_models.sh can use it
        if [[ -f "$PROJECT_DIR/.env" ]]; then
            grep -q "^HF_TOKEN=" "$PROJECT_DIR/.env" \
                && sed -i "s|^HF_TOKEN=.*|HF_TOKEN=$hf_tok|" "$PROJECT_DIR/.env" \
                || echo "HF_TOKEN=$hf_tok" >> "$PROJECT_DIR/.env"
        else
            echo "HF_TOKEN=$hf_tok" > "$PROJECT_DIR/.env"
        fi
        ok "HuggingFace token set and saved to .env"
        HF_LOGGED_IN=true
    else
        warn "No token entered — proceeding but downloads may stall for large models"
    fi
fi


mkdir -p "$MODELS_DIR"

# Uses Python snapshot_download directly — avoids hf/huggingface-cli
# naming changes and --exclude flag behaviour differences across versions.
download_model() {
    local hf_repo="$1" local_dir="$2" label="$3"
    local full_path="$MODELS_DIR/$local_dir"

    # Check XML exists AND weights file exists and is >100MB (not a metadata-only download)
    local xml_ok=false bin_ok=false
    [[ -f "$full_path/openvino_model.xml" ]] || [[ -f "$full_path/openvino_language_model.xml" ]] && xml_ok=true
    for bin in "$full_path"/*.bin; do
        [[ -f "$bin" ]] && [[ $(stat -c%s "$bin" 2>/dev/null || echo 0) -gt 104857600 ]] && bin_ok=true && break
    done
    if $xml_ok && $bin_ok; then
        ok "$label: already downloaded"
        return 0
    elif $xml_ok && ! $bin_ok; then
        warn "$label: found metadata only (weights missing) — re-downloading"
    fi

    echo ""
    info "Downloading: $label"
    info "  From: $hf_repo"
    info "  To:   $full_path"
    info "  (resume-capable — safe to Ctrl-C and restart)"
    echo ""

    python3 - << PYEOF
from huggingface_hub import snapshot_download
import os, sys

# Use token from env if available (set by prepare_host.sh or .env)
token = os.environ.get("HF_TOKEN") or None

try:
    snapshot_download(
        repo_id="$hf_repo",
        local_dir="$full_path",
        ignore_patterns=["*.gguf", "*.safetensors", "*.pt", ".gitattributes"],
        token=token,
    )
    print("  ✓  $label downloaded")
except Exception as e:
    print(f"  ✗  Download failed: {e}", file=sys.stderr)
    sys.exit(1)
PYEOF
}

download_model "${P1_HF_REPO[$CHOSEN_PAIR]}" "${P1_DIR[$CHOSEN_PAIR]}" \
    "Phase 1: ${P1_ID[$CHOSEN_PAIR]}"

download_model "${P2_HF_REPO[$CHOSEN_PAIR]}" "${P2_DIR[$CHOSEN_PAIR]}" \
    "Phase 2: ${P2_ID[$CHOSEN_PAIR]}"

# =============================================================================
hdr "6 / 7  Speculative Decoding Draft Model (optional — ~400MB, 2-3x decode speedup)"

echo ""
echo -e "  ${DIM}Draft: ${DRAFT_REPO[$CHOSEN_PAIR]} (Qwen3-0.6B — 3.4k downloads, ♡3)${RESET}"
echo -e "  ${DIM}All three pairs use Qwen3.6/3.8 base — Qwen3-0.6B draft is tokenizer-compatible.${RESET}"
echo ""
read -r -p "     Download draft model? [Y/n] " dr
dr="${dr:-Y}"
if [[ "$dr" =~ ^[Yy]$ ]]; then
    download_model "${DRAFT_REPO[$CHOSEN_PAIR]}" "${DRAFT_LOCAL[$CHOSEN_PAIR]}" \
        "Draft: ${DRAFT_REPO[$CHOSEN_PAIR]}"
    echo "DRAFT_MODEL_DIR=${DRAFT_LOCAL[$CHOSEN_PAIR]}" >> "$PROJECT_DIR/.model_pair"
    ok "Draft model ready — launch_models.sh enables speculative decoding automatically"
else
    info "Skipped. Download later: ./prepare_host.sh --pair $CHOSEN_PAIR --models-only"
fi

# =============================================================================
hdr "7 / 7  Open WebUI (pip — chat interface at http://localhost:8080)"

echo ""
echo -e "  ${DIM}Installs into .venv (~500MB). Pre-configured for your OpenVINO servers.${RESET}"
echo ""
read -r -p "     Install Open WebUI? [Y/n] " wu
wu="${wu:-Y}"
if [[ "$wu" =~ ^[Yy]$ ]]; then
    source "$PROJECT_DIR/.venv/bin/activate"
    pip install open-webui --quiet
    echo "OPEN_WEBUI_INSTALLED=true" >> "$PROJECT_DIR/.model_pair"
    ok "Open WebUI installed"
    info "Start: ./launch_models.sh --with-webui"
    info "Open:  http://localhost:8080"
else
    info "Skipped. Install later: source .venv/bin/activate && pip install open-webui"
fi

# =============================================================================
hdr "Complete"

echo ""
ok "Pair $CHOSEN_PAIR: ${PAIR_LABEL[$CHOSEN_PAIR]}"
info "Phase 1 (Port 8000): ${P1_ID[$CHOSEN_PAIR]}  →  $MODELS_DIR/${P1_DIR[$CHOSEN_PAIR]}"
info "Phase 2 (Port 8001): ${P2_ID[$CHOSEN_PAIR]}  →  $MODELS_DIR/${P2_DIR[$CHOSEN_PAIR]}"
echo ""
echo -e "  ${BOLD}Next steps:${RESET}"
$MODELS_ONLY || echo -e "  1. ${YELLOW}Reboot${RESET} if PSR was just disabled"
echo -e "  2. ${YELLOW}./launch_models.sh --with-webui${RESET}"
echo -e "  3. ${YELLOW}./verify.sh${RESET}"
echo ""
echo -e "  ${DIM}Switch pairs: ./prepare_host.sh --pair 2 --models-only${RESET}"
echo ""
