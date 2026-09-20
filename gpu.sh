#!/usr/bin/env bash
# =============================================================================
# gpu.sh -- Intel Arc iGPU compute runtime: check it, install it, diagnose it.
#
#   bash gpu.sh                  # = status
#   bash gpu.sh status           # read-only health check          (exit 0 = healthy, 1 = not)
#   bash gpu.sh install          # install/repair the runtime if needed, then hold it.  No-op if healthy.
#   bash gpu.sh install --force  # reinstall even if it looks healthy
#   bash gpu.sh diag [label]     # read-only full report saved to LOGS/  (label e.g. before / after)
#   bash gpu.sh diff             # compare the latest 'before' and 'after' reports
#
# WHY THIS EXISTS (diagnosed 20 Sep 2026 from a diag report + a reproduction):
#   The old setup.sh installed intel-opencl-icd 24.52.32224.5 with `dpkg -i`, but
#   that package needs libigdgmm12 >= 22.5.5 and Ubuntu 24.04 ships 22.3.17.
#   setup.sh asked for libigdgmm12_22.5.2 -- a file that does not exist on that
#   release (404) -- and its spinner helper hid the failure.  intel-opencl-icd was
#   left "unpacked, unconfigured".  The files were on disk, so the GPU worked --
#   until the next routine apt run, whose "fix broken packages" step REMOVED
#   intel-opencl-icd.  GPU gone; re-running setup.sh put it back.  Repeat.
#   `install` fixes it properly (via apt, dependencies verified) and apt-mark holds
#   the packages.
#
# Deliberately NOT installed: intel-level-zero-gpu.  It cannot coexist with Ubuntu's
# libze-intel-gpu1 (both own libze_intel_gpu.so.1) and OpenVINO's GPU plugin uses
# OpenCL, not Level Zero.
# =============================================================================
set -uo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR" || exit 1

NEO_TAG="24.52.32224.5"; NEO_BASE="https://github.com/intel/compute-runtime/releases/download/${NEO_TAG}"
IGC_TAG="v2.5.6";        IGC_BASE="https://github.com/intel/intel-graphics-compiler/releases/download/${IGC_TAG}"
GMM_MIN="22.5.5"
DEB_GMM="libigdgmm12_${GMM_MIN}_amd64.deb"
DEB_ICD="intel-opencl-icd_${NEO_TAG}_amd64.deb"
DEB_IGC_CORE="intel-igc-core-2_2.5.6+18417_amd64.deb"
DEB_IGC_OCL="intel-igc-opencl-2_2.5.6+18417_amd64.deb"
HOLD_PKGS=(libigdgmm12 intel-opencl-icd intel-igc-core-2 intel-igc-opencl-2)
ICD_LIB=/usr/lib/x86_64-linux-gnu/intel-opencl/libigdrcl.so

G="\033[32m"; Y="\033[33m"; R="\033[31m"; D="\033[2m"; B="\033[1m"; Z="\033[0m"
ok()   { echo -e "  ${G}✓${Z}  $*"; }
warn() { echo -e "  ${Y}⚠${Z}  $*"; }
bad()  { echo -e "  ${R}✗${Z}  $*"; }
info() { echo -e "  ${D}    $*${Z}"; }

SUDO=""; [[ $EUID -ne 0 ]] && SUDO="sudo"
ME="${USER:-$(id -un)}"
PY="$ROOT_DIR/.venv/bin/python3"; [[ -x "$PY" ]] || PY="$(command -v python3 || echo python3)"
HAVE_VENV=false; [[ -x "$ROOT_DIR/.venv/bin/python3" ]] && HAVE_VENV=true

# actual dpkg status word: installed / unpacked / half-configured / half-installed / config-files / not-installed
# (NOT Status-Abbrev: its first letter is the *desired* state, so a held package reads "hi", not "ii")
pkg_state() { dpkg-query -W -f='${db:Status-Status}' "$1" 2>/dev/null; }
pkg_ver()   { dpkg-query -W -f='${Version}' "$1" 2>/dev/null; }
gpu_visible() { $HAVE_VENV && "$PY" -c "import sys, openvino as ov; sys.exit(0 if 'GPU' in ov.Core().available_devices else 1)" >/dev/null 2>&1; }

# ---- health report (used by status, install and diag) -----------------------
HEALTHY=true
report() {   # report [noov]  -- 'noov' skips the OpenVINO check (diag does its own, in more detail)
    HEALTHY=true
    local s v
    s=$(pkg_state intel-opencl-icd); v=$(pkg_ver intel-opencl-icd)
    if [[ "$s" == "installed" ]]; then ok "intel-opencl-icd $v installed and configured"
    else HEALTHY=false; bad "intel-opencl-icd is '${s:-not installed}'${s:+ -- $( [[ $s == config-files ]] && echo 'apt REMOVED it (only config left)' || echo 'half-installed: unpacked but dependencies unmet')}"; fi

    v=$(pkg_ver libigdgmm12)
    if [[ -n "$v" ]] && dpkg --compare-versions "$v" ge "$GMM_MIN"; then ok "libigdgmm12 $v (>= $GMM_MIN required by intel-opencl-icd)"
    else HEALTHY=false; bad "libigdgmm12 ${v:-missing} -- intel-opencl-icd needs >= $GMM_MIN (Ubuntu 24.04 ships 22.3.17)"; fi

    if [[ "$(pkg_state intel-igc-opencl-2)" == "installed" && "$(pkg_state intel-igc-core-2)" == "installed" ]]; then ok "Intel graphics compiler $(pkg_ver intel-igc-opencl-2)"
    else HEALTHY=false; bad "intel-igc-core-2 / intel-igc-opencl-2 not both installed"; fi

    if [[ -f "$ICD_LIB" ]]; then ok "OpenCL driver library present on disk"
    else HEALTHY=false; bad "libigdrcl.so missing -- /etc/OpenCL/vendors/intel.icd points at nothing"; fi

    local held; held=$(apt-mark showhold 2>/dev/null | grep -cE '^(libigdgmm12|intel-opencl-icd|intel-igc-core-2|intel-igc-opencl-2)$')
    if [[ "$held" -ge 4 ]]; then ok "packages held (apt cannot remove or replace them)"
    else warn "not all 4 packages are held ($held/4) -- a future apt run could disturb them"; fi

    if [[ "${1:-}" == noov ]]; then :
    elif $HAVE_VENV; then
        if gpu_visible; then ok "OpenVINO sees the GPU"; else HEALTHY=false; bad "OpenVINO does NOT see the GPU"; fi
    else info "no .venv yet -- skipped the OpenVINO GPU check"; fi
}

# =============================================================================
cmd_status() {
    echo -e "\n${B}Intel GPU compute runtime -- status${Z}"
    report
    $HEALTHY
}

# =============================================================================
cmd_install() {
    local force=false; [[ "${1:-}" == "--force" ]] && force=true
    echo -e "\n${B}Intel GPU compute runtime -- status${Z}"
    report
    if $HEALTHY && ! $force; then echo -e "\n  Nothing to do."; return 0; fi

    echo -e "\n${B}Installing${Z}"
    local TMP; TMP=$(mktemp -d /tmp/neo_XXXXXX); trap 'rm -rf "$TMP"' RETURN
    dl() {   # dl <base> <file>  -- fails loudly (404 = message + non-zero)
        local url="$1/$2" rc
        if command -v curl >/dev/null; then curl -fsSL --retry 3 -o "$TMP/$2" "$url" 2>"$TMP/err"
        else wget -q -O "$TMP/$2" "$url" 2>"$TMP/err"; fi
        rc=$?
        if [[ $rc -ne 0 || ! -s "$TMP/$2" ]]; then bad "download failed: $url"; sed 's/^/        /' "$TMP/err" | tail -3; return 1; fi
        dpkg-deb -I "$TMP/$2" >/dev/null 2>&1 || { bad "not a valid .deb: $2"; return 1; }
        ok "downloaded $2"
    }

    local DEBS=() v s
    v=$(pkg_ver libigdgmm12)
    if [[ -z "$v" ]] || ! dpkg --compare-versions "$v" ge "$GMM_MIN" || $force; then dl "$NEO_BASE" "$DEB_GMM" || return 1; DEBS+=("$TMP/$DEB_GMM"); fi
    if [[ "$(pkg_state intel-igc-core-2)" != installed || "$(pkg_state intel-igc-opencl-2)" != installed ]] || $force; then
        dl "$IGC_BASE" "$DEB_IGC_CORE" || return 1; dl "$IGC_BASE" "$DEB_IGC_OCL" || return 1
        DEBS+=("$TMP/$DEB_IGC_CORE" "$TMP/$DEB_IGC_OCL"); fi
    dl "$NEO_BASE" "$DEB_ICD" || return 1; DEBS+=("$TMP/$DEB_ICD")

    if [[ -n "$SUDO" ]]; then $SUDO -v || { bad "need sudo"; return 1; }; fi

    # a half-installed intel-opencl-icd (unpacked but unconfigured) makes apt refuse to
    # install anything -- clear it first (this is what apt itself would do to it)
    s=$(pkg_state intel-opencl-icd)
    if [[ -n "$s" && "$s" != installed && "$s" != config-files && "$s" != not-installed ]]; then
        warn "intel-opencl-icd is half-installed ($s) -- removing it so apt can proceed"
        $SUDO dpkg --remove --force-remove-reinstreq intel-opencl-icd >/dev/null 2>&1
    fi

    $SUDO apt-mark unhold "${HOLD_PKGS[@]}" >/dev/null 2>&1     # holds would block the install
    if $SUDO env DEBIAN_FRONTEND=noninteractive apt-get install -y -o Dpkg::Options::=--force-confold "${DEBS[@]}" >"$TMP/apt.log" 2>&1; then
        ok "installed through apt (dependencies verified)"
    else
        bad "apt could not install the packages:"; grep -E '^(E:|The following|  )' "$TMP/apt.log" | tail -8 | sed 's/^/        /'
        info "if it says 'Unmet dependencies', run:  sudo apt --fix-broken install   then re-run:  bash gpu.sh install"
        return 1
    fi
    $SUDO apt-mark hold "${HOLD_PKGS[@]}" >/dev/null && ok "held: ${HOLD_PKGS[*]}"
    $SUDO ldconfig 2>/dev/null

    echo -e "\n${B}Re-check${Z}"
    report
    if $HEALTHY; then echo -e "\n  ${G}Done.${Z} Launch as normal -- no need to re-run setup.sh."; return 0
    else echo -e "\n  ${R}Packages are fine but something else is wrong.${Z} Run:  bash gpu.sh diag   (render group? kernel driver? reboot needed?)"; return 1; fi
}

# =============================================================================
cmd_diff() {
    local Bf Af
    Bf=$(ls -t LOGS/gpu_diag_*_before.txt 2>/dev/null | head -1)
    Af=$(ls -t LOGS/gpu_diag_*_after.txt  2>/dev/null | head -1)
    [[ -z "$Bf" || -z "$Af" ]] && { echo "Need both:  bash gpu.sh diag before   and   bash gpu.sh diag after"; return 1; }
    echo "before: $Bf"; echo "after:  $Af"; echo "---- lines that CHANGED (- before / + after) ----"
    local changed   # (captured in a variable: under pipefail, diff's exit 1 = "files differ" must not look like "no changes")
    changed=$(diff -u "$Bf" "$Af" | grep -E '^[+-]' | grep -vE '^(\+\+\+|---|[+-]saved:)')
    if [[ -n "$changed" ]]; then echo "$changed"; else echo "(no differences -- whatever you ran did not change anything this report can see)"; fi
}

# =============================================================================
cmd_diag() {
    local LABEL="${1:-run}" OUT
    mkdir -p LOGS
    OUT="LOGS/gpu_diag_$(date +%Y%m%d_%H%M%S)_${LABEL}.txt"
    exec > >(tee "$OUT") 2>&1
    local RN ACCESS="n/a" DRIVER SESSION_HAS_RENDER=no DB_HAS_RENDER=no GPU_VISIBLE=no
    sec() { echo; echo "== $* =="; }
    RN=$(ls /dev/dri/renderD* 2>/dev/null | head -1)

    sec "1. Kernel driver bound to the iGPU"
    lspci -nnk 2>/dev/null | grep -A3 -Ei 'vga|display controller' | grep -Ei 'vga|display|driver in use' \
        || echo "(lspci unavailable or no display controller found)"
    DRIVER=$(lspci -k 2>/dev/null | grep -A3 -Ei 'vga|display controller' | grep -m1 'Kernel driver in use' | awk '{print $NF}')

    sec "2. Render node + who may use it"
    ls -l /dev/dri/ 2>/dev/null || echo "(no /dev/dri at all -- kernel driver did not create a GPU device)"
    if [[ -n "$RN" ]]; then
        if [[ -r "$RN" && -w "$RN" ]]; then echo "this shell CAN open $RN"; ACCESS=yes
        else echo "this shell can NOT open $RN"; ACCESS=no; fi
        command -v getfacl >/dev/null && getfacl -p "$RN" 2>/dev/null | grep -E 'user:|group:'
    fi
    echo "groups in THIS session : $(id -nG)"
    echo "groups in /etc/group   : $(id -nG "$ME")"
    id -nG | grep -qw render && SESSION_HAS_RENDER=yes
    id -nG "$ME" | grep -qw render && DB_HAS_RENDER=yes

    sec "3. Intel compute stack (packages, holds, ICD registration)"
    dpkg -l 2>/dev/null | grep -Ei 'intel-opencl|level-zero|libze|igdgmm|intel-igc|ocl-icd|intel-ocloc' | awk '{printf "%-3s %-34s %s\n",$1,$2,$3}'
    [[ -z "$(dpkg -l 2>/dev/null | grep -Ei 'intel-opencl|libze|level-zero')" ]] && echo "(no Intel OpenCL / Level Zero packages installed)"
    echo "apt holds: $(apt-mark showhold 2>/dev/null | tr '\n' ' ')"
    echo "ICD files: $(ls /etc/OpenCL/vendors/ 2>/dev/null | tr '\n' ' ')"
    local f; for f in /etc/OpenCL/vendors/*.icd; do [[ -f "$f" ]] && echo "  $f -> $(cat "$f")"; done
    env | grep -E '^(OCL_ICD|NEO|ZE_|SYCL_|OV_|OPENVINO)' | sort || true
    echo "OS: $(. /etc/os-release 2>/dev/null; echo "${PRETTY_NAME:-unknown}")   kernel: $(uname -r)"
    echo; echo "runtime health:"; report noov | sed 's/\x1b\[[0-9;]*m//g'

    sec "4. Recent package changes to that stack (what could have broken/reverted it)"
    local PKGLOG AH UU
    PKGLOG=$(grep -hE ' (install|upgrade|remove|purge) ' /var/log/dpkg.log* 2>/dev/null \
        | grep -Ei 'intel-opencl|level-zero|libze|igdgmm|intel-igc|ocl-icd|intel-ocloc' | sort | tail -15)
    [[ -n "$PKGLOG" ]] && echo "$PKGLOG" || echo "(none recorded)"
    AH=$(zcat -f /var/log/apt/history.log* 2>/dev/null | grep -B9 -A3 -E '^Remove:.*intel-opencl-icd' \
        | grep -E '^(Start-Date|Commandline|Requested-By|Remove:)' | cut -c1-220 | tail -16)
    if [[ -n "$AH" ]]; then echo "apt history -- who REMOVED intel-opencl-icd:"; echo "$AH"; else echo "apt history: no removal of intel-opencl-icd recorded"; fi
    UU=$(grep -hs 'Unattended-Upgrade' /etc/apt/apt.conf.d/20auto-upgrades | tr -d '\n')
    echo "unattended-upgrades: ${UU:-(no /etc/apt/apt.conf.d/20auto-upgrades)}"

    sec "5. OpenVINO inside the venv"
    "$PY" -m pip list 2>/dev/null | grep -Ei '^(openvino|numpy) ' || echo "(pip list unavailable)"
    timeout 90 "$PY" - <<'PYEOF' 2>&1 | tail -6
try:
    import openvino as ov
    c = ov.Core()
    devs = c.available_devices
    print("openvino", ov.get_version(), "| devices:", devs)
    if "GPU" in devs:
        print("GPU name:", c.get_property("GPU", "FULL_DEVICE_NAME"))
except Exception as e:
    print("openvino FAILED:", type(e).__name__, str(e).splitlines()[0][:160])
PYEOF
    gpu_visible && GPU_VISIBLE=yes
    if command -v clinfo >/dev/null; then echo "clinfo -l:"; clinfo -l 2>&1 | head -6
    else echo "(clinfo not installed: sudo apt install clinfo gives a second opinion independent of OpenVINO)"; fi

    sec "6. Kernel log for GPU trouble (this boot)"
    local K S
    K=$(journalctl -k -b --no-pager 2>/dev/null | grep -Ei 'xe |i915|drm|wedged|gpu hang|GuC|engine reset|timeout' | tail -12)
    [[ -n "$K" ]] && echo "$K" || echo "(nothing found, or no permission -- try: sudo dmesg | grep -Ei 'xe|i915|wedged|hang' | tail)"
    S=$(journalctl -b --no-pager -g 'PM: suspend (entry|exit)' 2>/dev/null | tail -4)
    if [[ -n "$S" ]]; then echo "suspend/resume events this boot:"; echo "$S"; else echo "no suspend/resume events this boot"; fi
    echo "booted: $(uptime -s 2>/dev/null)"

    sec "7. Who else is touching the GPU / stale servers"
    local PG; PG=$(pgrep -af 'serve_model|launch_models|chainlit' 2>/dev/null | sed 's/^[0-9]* //' | cut -c1-110)
    [[ -n "$PG" ]] && echo "$PG" || echo "(no serve_model / chainlit processes running)"
    [[ -n "$RN" ]] && echo "processes with $RN open: $(fuser "$RN" 2>/dev/null | wc -w)"
    echo "openvino-coder user service: $(systemctl --user is-active openvino-coder.service 2>/dev/null || echo not-installed)"

    sec "VERDICT (first matching cause)"
    local ICD_STATE GMM_VER
    ICD_STATE=$(pkg_state intel-opencl-icd); GMM_VER=$(pkg_ver libigdgmm12)
    if [[ "$GPU_VISIBLE" == "yes" ]]; then
        echo "GPU IS visible to OpenVINO right now -- capture this as 'after', or launch."
    elif [[ -z "$RN" ]]; then
        echo "NO RENDER NODE: the kernel driver (${DRIVER:-none bound}) created no GPU device. Check sections 1 & 6 (driver not bound, GPU wedged after a hang, or resume from suspend). Not fixable by userspace packages -- try a reboot."
    elif [[ "$ACCESS" == "no" && "$DB_HAS_RENDER" == "yes" && "$SESSION_HAS_RENDER" == "no" ]]; then
        echo "SESSION GROUPS STALE: you ARE in 'render' in /etc/group but this login session predates it. Log out/in (or 'newgrp render')."
    elif [[ "$ACCESS" == "no" ]]; then
        echo "PERMISSIONS: this shell cannot open $RN and you are not in 'render'."
    elif [[ "$ICD_STATE" != "installed" || ! -f "$ICD_LIB" ]] || ! dpkg --compare-versions "${GMM_VER:-0}" ge "$GMM_MIN"; then
        echo "INTEL OPENCL RUNTIME BROKEN (intel-opencl-icd: ${ICD_STATE:-not installed}, libigdgmm12: ${GMM_VER:-missing}, driver library: $([[ -f $ICD_LIB ]] && echo present || echo MISSING))."
        echo "   Fix:  bash gpu.sh install     (installs it via apt with dependencies satisfied, then holds it)"
        echo "   Why it recurs: see section 4 -- apt removes a half-installed package."
    else
        echo "UNDETERMINED: device node, permissions and packages look fine. Run  bash gpu.sh diag before / after  around whatever 'fixes' it, then  bash gpu.sh diff  -- the changed lines are the cause (often an OpenVINO nightly vs NEO version mismatch, or a wedged GPU)."
    fi
    echo; echo "saved: $OUT"
}

usage() { sed -n '3,10p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

case "${1:-status}" in
    status)         cmd_status; exit $? ;;
    install)        shift; cmd_install "$@"; exit $? ;;
    diag)           shift; cmd_diag "$@"; exit 0 ;;
    diff)           cmd_diff; exit $? ;;
    -h|--help|help) usage; exit 0 ;;
    *)              echo "unknown command: $1"; usage; exit 2 ;;
esac
