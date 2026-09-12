#!/usr/bin/env bash
# =============================================================================
# Trading Income Project — NUC 14 Pro Setup & Lifecycle Manager
# Intel Core Ultra 5 125H (Meteor Lake Arc iGPU [8086:7d55]) · Ubuntu 24.04 · xe Driver
# =============================================================================
# Idempotent and interactive — safe to re-run at any time.
# Configures host hardware, Level Zero, IPEX-LLM runtime, and kernel optimizations.
# =============================================================================
set -euo pipefail

BOLD="\033[1m"; DIM="\033[2m"
GREEN="\033[32m"; YELLOW="\033[33m"; RED="\033[31m"; CYAN="\033[36m"
RESET="\033[0m"

ok()   { echo -e "  ${GREEN}✓${RESET}  $*"; }
warn() { echo -e "  ${YELLOW}⚠${RESET}  $*"; }
fail() { echo -e "  ${RED}✗${RESET}  $*"; }
skip() { echo -e "  ${DIM}○  $* — already satisfied${RESET}"; }
info() { echo -e "  ${DIM}    $*${RESET}"; }
hdr()  { echo -e "\n${BOLD}${CYAN}── $* ${RESET}"; }

ask() {
    local prompt="$1" default="${2:-Y}"
    local opts; [[ "$default" == "Y" ]] && opts="[Y/n]" || opts="[y/N]"
    read -r -p "$(echo -e "     ${prompt} ${DIM}${opts}${RESET} ")" reply
    reply="${reply:-$default}"
    [[ "$reply" =~ ^[Yy]$ ]]
}

IPEX_DIR="$HOME/.ipex-llm-ollama"
WRAPPER_BIN="$HOME/.local/bin/ollama-ipex"

# =============================================================================
# UNINSTALL / CLEANUP SUBROUTINE
# =============================================================================
do_uninstall() {
    echo -e "\n${BOLD}${RED}=== Uninstall & System Cleanup ===${RESET}\n"

    if ask "Stop running Ollama and IPEX processes?" "Y"; then
        sudo systemctl stop ollama.service ollama.socket 2>/dev/null || true
        pkill -9 -f "ollama" 2>/dev/null || true
        sudo fuser -k 11434/tcp 2>/dev/null || true
        ok "Stopped all Ollama processes and freed port 11434"
    fi

    if [[ -d "$IPEX_DIR" ]] || [[ -f "$WRAPPER_BIN" ]]; then
        if ask "Remove IPEX-LLM installation ($IPEX_DIR) and wrapper?" "Y"; then
            rm -rf "$IPEX_DIR"
            rm -f "$WRAPPER_BIN"
            rm -f "$HOME/.local/bin/ollama"
            sudo rm -f /etc/ld.so.conf.d/ipex-ollama.conf
            sudo ldconfig
            ok "Removed IPEX-LLM files, wrapper, and dynamic linker registration"
        fi
    fi

    if command -v ollama &>/dev/null || [[ -f /etc/systemd/system/ollama.service ]]; then
        if ask "Purge upstream CPU Ollama binary and systemd units?" "Y"; then
            sudo systemctl stop ollama.service ollama.socket 2>/dev/null || true
            sudo systemctl disable ollama.service ollama.socket 2>/dev/null || true
            sudo systemctl unmask ollama.service ollama.socket 2>/dev/null || true
            sudo rm -f /etc/systemd/system/ollama.service /etc/systemd/system/ollama.socket
            sudo rm -f /usr/local/bin/ollama /usr/bin/ollama
            sudo rm -rf /usr/local/share/ollama /usr/share/ollama
            sudo userdel ollama 2>/dev/null || true
            sudo groupdel ollama 2>/dev/null || true
            ok "Purged upstream Ollama"
        fi
    fi

    if ask "Remove PSR disable from GRUB kernel parameters?" "Y"; then
        sudo sed -i 's/ i915.enable_psr=0//g' /etc/default/grub 2>/dev/null || true
        sudo update-grub 2>/dev/null || sudo grub-mkconfig -o /boot/grub/grub.cfg 2>/dev/null || true
        ok "PSR parameter removed from GRUB (reboot required)"
    fi

    if ask "Clean Intel Arc & SYCL environment variables from ~/.bashrc?" "Y"; then
        sed -i '/# Intel Arc/d' "$HOME/.bashrc"
        sed -i '/ONEAPI_DEVICE_SELECTOR/d' "$HOME/.bashrc"
        sed -i '/SYCL_CACHE_PERSISTENT/d' "$HOME/.bashrc"
        sed -i '/SYCL_CACHE_DIR/d' "$HOME/.bashrc"
        sed -i '/SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS/d' "$HOME/.bashrc"
        sed -i '/ZES_ENABLE_SYSMAN/d' "$HOME/.bashrc"
        sed -i '/GGML_SYCL_/d' "$HOME/.bashrc"
        sed -i '/OLLAMA_NUM_GPU/d' "$HOME/.bashrc"
        sed -i '/OLLAMA_FLASH_ATTENTION/d' "$HOME/.bashrc"
        sed -i '/OLLAMA_CONTEXT_LENGTH/d' "$HOME/.bashrc"
        sed -i '/OLLAMA_NUM_BATCH/d' "$HOME/.bashrc"
        sed -i '/OLLAMA_KV_CACHE_TYPE/d' "$HOME/.bashrc"
        sed -i '/OLLAMA_NUM_PARALLEL/d' "$HOME/.bashrc"
        sed -i '/OLLAMA_KEEP_ALIVE/d' "$HOME/.bashrc"
        ok "Cleaned environment entries from ~/.bashrc"
    fi

    # Clean up third-party repos if previously installed
    if [[ -f /etc/apt/sources.list.d/intel-gpu-noble.list ]]; then
        sudo rm -f /etc/apt/sources.list.d/intel-gpu-noble.list /etc/apt/sources.list.d/oneAPI.list
        sudo rm -f /usr/share/keyrings/intel-graphics.gpg /usr/share/keyrings/intel-oneapi.gpg
        sudo apt-get update -qq
        ok "Removed third-party Intel graphics repo"
    fi

    echo -e "\n${BOLD}${GREEN}Cleanup finished.${RESET}\n"
    exit 0
}

if [[ "${1:-}" == "--uninstall" ]] || [[ "${1:-}" == "--remove" ]]; then
    do_uninstall
fi

# =============================================================================
# MAIN ENTRY
# =============================================================================
echo -e "\n${BOLD}NUC 14 Pro — Intel Arc iGPU Setup & Lifecycle Manager${RESET}"
echo -e "${DIM}Configures hardware and IPEX-LLM runtime safely (Stock Ubuntu 24.04).${RESET}\n"

echo "Select Action:"
echo "  1) Install / Optimize Arc iGPU + IPEX-LLM (Recommended)"
echo "  2) Remove / Uninstall Components"
read -r -p "     Choose [1/2, default=1]: " ACTION
ACTION="${ACTION:-1}"
if [[ "$ACTION" == "2" ]]; then
    do_uninstall
fi

# =============================================================================
# SECTION 1 — Hardware, Permissions & Kernel Performance
# =============================================================================
hdr "1 / 6  Hardware, Permissions & Kernel Performance"

UBUNTU=$(lsb_release -rs 2>/dev/null || echo "unknown")
KERNEL=$(uname -r)
RAM_GB=$(awk '/MemTotal/ {printf "%.0f", $2/1024/1024}' /proc/meminfo)

[[ "$UBUNTU" =~ ^24 ]] && ok "Ubuntu $UBUNTU" || warn "Ubuntu $UBUNTU — tested on 24.04 LTS"
ok "Kernel $KERNEL"
ok "System RAM: ${RAM_GB}GB"

if [ -d "/sys/bus/pci/drivers/xe" ]; then
    ok "Kernel driver: xe (modern Intel Xe driver active)"
else
    warn "Kernel driver: i915 or generic (xe recommended for Meteor Lake)"
fi

# PSR (Panel Self-Refresh) Screen Corruption Guard
# Affects: Meteor Lake Arc iGPU on both i915 and xe drivers
# Symptom: Random pixel artifacts, blocky corruption when screen refreshes
# Root cause: PSR causes display pipeline glitches on Meteor Lake (kernel 6.10+ regression)
# Fix: Disable PSR via kernel parameter. Safe for headless/inference systems.
# Reference: https://bugzilla.kernel.org/show_bug.cgi?id=219093
PSR_DISABLED=false
if grep -q "i915.enable_psr=0" /proc/cmdline 2>/dev/null || \
   grep -q "i915.enable_psr=0" /etc/default/grub 2>/dev/null; then
    skip "PSR already disabled (screen corruption protection active)"
    PSR_DISABLED=true
fi
if ! $PSR_DISABLED; then
    warn "PSR not disabled — Meteor Lake Arc may show screen corruption artifacts"
    info "This is a known kernel regression on Intel Arc (6.10+) — separate from driver repos"
    info "Symptom: random pixel lines/blocks when screen content changes"
    if ask "Disable PSR (Panel Self-Refresh) to prevent screen corruption? (sudo required)"; then
        GRUB_FILE="/etc/default/grub"
        if grep -q 'GRUB_CMDLINE_LINUX_DEFAULT=' "$GRUB_FILE"; then
            sudo sed -i 's/GRUB_CMDLINE_LINUX_DEFAULT="\(.*\)"/GRUB_CMDLINE_LINUX_DEFAULT="\1 i915.enable_psr=0"/' "$GRUB_FILE"
        else
            echo 'GRUB_CMDLINE_LINUX_DEFAULT="quiet splash i915.enable_psr=0"' | sudo tee -a "$GRUB_FILE"
        fi
        sudo update-grub 2>/dev/null || sudo grub-mkconfig -o /boot/grub/grub.cfg 2>/dev/null || true
        ok "PSR disabled — reboot required for this to take effect"
        warn "REBOOT REQUIRED before display corruption protection is active"
    else
        info "Skipped — you may experience screen artifacts. Re-run to apply later."
    fi
fi

# Check permissions for /dev/dri/renderD128
if [[ -w /dev/dri/renderD128 ]] || id -nG "$USER" 2>/dev/null | grep -qw "render"; then
    ok "Permissions: /dev/dri/renderD128 accessible by $USER"
else
    warn "Adding $USER to render and video groups..."
    sudo usermod -aG render,video "$USER"
    ok "Added to render group (logout/login may be needed for external terminals)"
fi

# CPU Governor & EPP Optimization
CURRENT_GOV=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo "unknown")
if [[ "$CURRENT_GOV" == "performance" ]]; then
    skip "CPU governor: performance"
else
    if ask "Set CPU governor and EPP to 'performance' for zero dispatch latency? (sudo required)"; then
        sudo apt-get install -y -qq cpupower-gui linux-tools-common linux-tools-generic 2>/dev/null || true
        for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
            echo "performance" | sudo tee "$g" >/dev/null 2>&1 || true
        done
        for epp in /sys/devices/system/cpu/cpu*/cpufreq/energy_performance_preference; do
            echo "performance" | sudo tee "$epp" >/dev/null 2>&1 || true
        done
        ok "CPU governor & EPP configured to performance"
    fi
fi

# Transparent Huge Pages
if grep -q "\[always\]" /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null; then
    skip "Transparent Huge Pages: always"
else
    if ask "Enable Transparent Huge Pages (THP) for faster tensor allocations? (sudo required)"; then
        echo "always" | sudo tee /sys/kernel/mm/transparent_hugepage/enabled >/dev/null 2>&1 || true
        echo "always" | sudo tee /sys/kernel/mm/transparent_hugepage/defrag >/dev/null 2>&1 || true
        ok "THP enabled"
    fi
fi

# =============================================================================
# SECTION 2 — Intel GPU Compute Runtime (Official Ubuntu Repos Only)
# =============================================================================
hdr "2 / 6  Intel GPU Compute Runtime (Stock Ubuntu Noble)"

# Mask legacy CPU Ollama systemd service to keep port 11434 free
sudo systemctl stop ollama.service ollama.socket 2>/dev/null || true
sudo systemctl disable ollama.service ollama.socket 2>/dev/null || true
sudo systemctl mask ollama.service ollama.socket 2>/dev/null || true

# Safety Guard: Ensure broken third-party intel-graphics repo is removed
if [[ -f /etc/apt/sources.list.d/intel-gpu-noble.list ]]; then
    warn "Detected dangerous third-party Intel graphics repo that causes black screens."
    sudo rm -f /etc/apt/sources.list.d/intel-gpu-noble.list /etc/apt/sources.list.d/oneAPI.list
    sudo rm -f /usr/share/keyrings/intel-graphics.gpg /usr/share/keyrings/intel-oneapi.gpg
    sudo apt-get update -qq
    ok "Safely removed third-party repo"
fi

# Ensure Ubuntu Universe repository is enabled
sudo add-apt-repository -y universe >/dev/null 2>&1 || true

PKGS_NEEDED=()
for pkg in intel-opencl-icd libze1 libze-intel-gpu1 clinfo; do
    dpkg -s "$pkg" &>/dev/null || PKGS_NEEDED+=("$pkg")
done

if (( ${#PKGS_NEEDED[@]} == 0 )); then
    skip "Intel Compute & Level Zero packages present (stock Ubuntu)"
else
    warn "Installing missing stock compute packages: ${PKGS_NEEDED[*]}"
    if ask "Install stock Ubuntu Level Zero & OpenCL packages? (sudo required)"; then
        sudo apt-get update -qq
        sudo apt-get install -y -qq "${PKGS_NEEDED[@]}"
        sudo ldconfig
        ok "Installed stock Ubuntu compute runtime"
    fi
fi

sudo ldconfig
if ldconfig -p 2>/dev/null | grep -E "libze_intel_gpu|libze_loader" >/dev/null 2>&1; then
    ok "Level Zero loader and GPU backend active in dynamic linker"
else
    fail "Level Zero driver not found in ldcache"
    exit 1
fi

# =============================================================================
# SECTION 3 — IPEX-LLM Deployment & Symlink Fix
# =============================================================================
hdr "3 / 6  IPEX-LLM Deployment & Symlink Fix"

if [[ -x "$IPEX_DIR/ollama" ]] && [[ -f "$IPEX_DIR/start-ollama.sh" ]] && [[ -L "$IPEX_DIR/libggml-cpu.so" ]]; then
    skip "IPEX-LLM portable release installed at $IPEX_DIR"
else
    info "Setting up IPEX-LLM for Meteor Lake Arc iGPU..."
    mkdir -p "$IPEX_DIR"

    ARCHIVE_FILE="$IPEX_DIR/ollama-ipex-llm.tgz"
    RELEASE_URL="https://github.com/ipex-llm/ipex-llm/releases/download/v2.3.0-nightly/ollama-ipex-llm-2.3.0b20250725-ubuntu.tgz"

    if [[ ! -f "$IPEX_DIR/ollama" ]]; then
        info "Downloading verified IPEX release (~140MB)..."
        wget -q --show-progress -O "$ARCHIVE_FILE" "$RELEASE_URL" || {
            fail "Download failed from $RELEASE_URL"
            exit 1
        }
        tar -xzf "$ARCHIVE_FILE" -C "$IPEX_DIR" --strip-components=1 2>/dev/null \
            || tar -xzf "$ARCHIVE_FILE" -C "$IPEX_DIR"
        rm -f "$ARCHIVE_FILE"

        if [[ -d "$IPEX_DIR/ollama-ipex-llm-2.3.0b20250725-ubuntu" ]]; then
            mv "$IPEX_DIR"/ollama-ipex-llm-2.3.0b20250725-ubuntu/* "$IPEX_DIR"/ 2>/dev/null || true
            rm -rf "$IPEX_DIR/ollama-ipex-llm-2.3.0b20250725-ubuntu"
        fi
    fi

    cd "$IPEX_DIR"
    ln -sf libggml-cpu-alderlake.so libggml-cpu.so
    chmod +x *.sh ollama ls-sycl-device 2>/dev/null || chmod +x ollama
    ok "IPEX-LLM core deployed and symlinked"
fi

if [[ ! -f /etc/ld.so.conf.d/ipex-ollama.conf ]]; then
    echo "$IPEX_DIR" | sudo tee /etc/ld.so.conf.d/ipex-ollama.conf > /dev/null
    sudo ldconfig
    ok "Registered $IPEX_DIR in /etc/ld.so.conf.d/ipex-ollama.conf"
fi

# =============================================================================
# SECTION 4 — Hardware Optimizations, Flash Attention & Context Sizing
# =============================================================================
hdr "4 / 6  Hardware Optimizations, Flash Attention & Context Sizing"

mkdir -p "$HOME/.sycl_cache"

cat << 'EOF' > "$IPEX_DIR/start-ollama.sh"
#!/bin/bash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LD_LIBRARY_PATH="$DIR:$LD_LIBRARY_PATH"

export ONEAPI_DEVICE_SELECTOR=level_zero:0
export ZES_ENABLE_SYSMAN=1
export SYCL_CACHE_PERSISTENT=1
export SYCL_CACHE_DIR="$HOME/.sycl_cache"
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1

export GGML_SYCL_DISABLE_OPT=0
export GGML_SYCL_DISABLE_GRAPH=0

export OLLAMA_FLASH_ATTENTION=1
export OLLAMA_CONTEXT_LENGTH=8192
export OLLAMA_NUM_BATCH=2048
export OLLAMA_KV_CACHE_TYPE=q8_0
export OLLAMA_NUM_GPU=999
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_KEEP_ALIVE=2h
export OLLAMA_HOST="127.0.0.1:11434"
export no_proxy=localhost,127.0.0.1

cd "$DIR"
exec ./ollama serve
EOF
chmod +x "$IPEX_DIR/start-ollama.sh"
ok "Configured optimized $IPEX_DIR/start-ollama.sh"

mkdir -p "$HOME/.local/bin"
cat << 'EOF' > "$WRAPPER_BIN"
#!/usr/bin/env bash
IPEX_DIR="$HOME/.ipex-llm-ollama"

export LD_LIBRARY_PATH="$IPEX_DIR:$LD_LIBRARY_PATH"
export ONEAPI_DEVICE_SELECTOR=level_zero:0
export ZES_ENABLE_SYSMAN=1
export SYCL_CACHE_PERSISTENT=1
export SYCL_CACHE_DIR="$HOME/.sycl_cache"
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
export GGML_SYCL_DISABLE_OPT=0
export GGML_SYCL_DISABLE_GRAPH=0
export OLLAMA_FLASH_ATTENTION=1
export OLLAMA_CONTEXT_LENGTH=8192
export OLLAMA_NUM_BATCH=2048
export OLLAMA_KV_CACHE_TYPE=q8_0
export OLLAMA_NUM_GPU=999
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_HOST="127.0.0.1:11434"
export no_proxy=localhost,127.0.0.1

if ! pgrep -f "ollama serve" &>/dev/null && ! curl -sf http://127.0.0.1:11434/api/tags &>/dev/null; then
    echo "Starting IPEX-LLM service on background port 11434..."
    nohup "$IPEX_DIR/start-ollama.sh" > /tmp/ipex-ollama.log 2>&1 &
    sleep 3
fi

exec "$IPEX_DIR/ollama" "$@"
EOF
chmod +x "$WRAPPER_BIN"
ln -sf "$WRAPPER_BIN" "$HOME/.local/bin/ollama"
ok "Created wrapper: $WRAPPER_BIN and linked to ~/.local/bin/ollama"

if ! grep -q "ONEAPI_DEVICE_SELECTOR=level_zero:0" "$HOME/.bashrc" 2>/dev/null; then
    cat << 'EOF' >> "$HOME/.bashrc"

# Intel Arc iGPU Inference Optimization (Meteor Lake)
export ONEAPI_DEVICE_SELECTOR=level_zero:0
export SYCL_CACHE_PERSISTENT=1
export SYCL_CACHE_DIR="$HOME/.sycl_cache"
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
export ZES_ENABLE_SYSMAN=1
export GGML_SYCL_DISABLE_OPT=0
export GGML_SYCL_DISABLE_GRAPH=0
export OLLAMA_FLASH_ATTENTION=1
export OLLAMA_CONTEXT_LENGTH=8192
export OLLAMA_NUM_BATCH=2048
export OLLAMA_KV_CACHE_TYPE=q8_0
export OLLAMA_NUM_GPU=999
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_KEEP_ALIVE=2h
export PATH="$HOME/.local/bin:$PATH"
EOF
    ok "Persistent environment variables added to ~/.bashrc"
fi

# =============================================================================
# SECTION 5 — GPU Telemetry Tooling (nvtop)
# =============================================================================
hdr "5 / 6  GPU Telemetry Tooling (nvtop)"

if command -v /snap/bin/nvtop &>/dev/null; then
    skip "nvtop installed"
else
    if ask "Install nvtop via Snap for Intel xe monitoring?"; then
        sudo apt-get remove -y nvtop 2>/dev/null || true
        sudo snap install nvtop
        sudo snap connect nvtop:hardware-observe 2>/dev/null || true
        ok "nvtop installed"
    fi
fi

# =============================================================================
# SECTION 6 — Daemon Health Verification
# =============================================================================
hdr "6 / 6  Daemon Health Verification"

if ! curl -sf http://localhost:11434/api/tags &>/dev/null; then
    info "Launching background IPEX server..."
    nohup "$IPEX_DIR/start-ollama.sh" > /tmp/ipex-ollama.log 2>&1 &
    sleep 4
fi

if curl -sf http://localhost:11434/api/tags &>/dev/null; then
    ok "IPEX Ollama server responding on port 11434"
else
    fail "IPEX Ollama server failed to start. Check /tmp/ipex-ollama.log"
    exit 1
fi

echo -e "\n${BOLD}${CYAN}══════════════════════════════════════════════════════════════${RESET}"
echo -e "${BOLD}${GREEN} NUC 14 Pro Hardware Setup Complete${RESET}"
echo -e "${BOLD}${CYAN}══════════════════════════════════════════════════════════════${RESET}"
echo -e "  • GPU Monitor:   /snap/bin/nvtop"
echo -e "  • Server Logs:   tail -f /tmp/ipex-ollama.log"
echo -e "  • Next Step:     Run ./setup.sh to configure Python (.venv) and install models.\n"
