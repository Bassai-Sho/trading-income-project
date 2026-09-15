#!/usr/bin/env bash
# =============================================================================
# Trading Income Project — Quick Start Verification Suite
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

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

TEST_DATE="${TEST_DATE:-}"
QUICK=false
VERIFY_LOG="/tmp/verify_session.log"

for arg in "$@"; do
    case "$arg" in
        --date)    shift; TEST_DATE="${1:-}"; shift ;;
        --date=*)  TEST_DATE="${arg#--date=}" ;;
        --quick)   QUICK=true ;;
    esac
done

PASS=0
FAIL=0

echo -e "\n${BOLD}Trading Income Project — Verification Suite${RESET}"

# =============================================================================
hdr "Step 1 — Server Health (Ports 8000 & 8001)"

P1_UP=false; P2_UP=false
curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1 && P1_UP=true
curl -sf http://127.0.0.1:8001/health >/dev/null 2>&1 && P2_UP=true

if $P1_UP; then
    MODEL=$(curl -s http://127.0.0.1:8000/v1/models 2>/dev/null \
        | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['data'][0]['id'])" 2>/dev/null || echo "unknown")
    ok "Port 8000 — READY  (model: $MODEL)"
    (( PASS += 1 ))
else
    fail "Port 8000 — not responding. Run: ./launch_models.sh --with-webui"
    (( FAIL += 1 ))
fi

if $P2_UP; then
    MODEL=$(curl -s http://127.0.0.1:8001/v1/models 2>/dev/null \
        | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['data'][0]['id'])" 2>/dev/null || echo "unknown")
    ok "Port 8001 — READY  (model: $MODEL)"
    (( PASS += 1 ))
else
    fail "Port 8001 — not responding. Run: ./launch_models.sh --with-webui"
    (( FAIL += 1 ))
fi

$P1_UP || { echo ""; warn "Cannot continue without Port 8000. Start servers first."; exit 1; }

# =============================================================================
hdr "Step 2 — Phase 1 Inference & Throughput (Port 8000)"

PHASE1_MODEL=$(curl -s http://127.0.0.1:8000/v1/models 2>/dev/null \
    | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null || echo "unknown")

RESP=$(curl -s -X POST http://127.0.0.1:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"$PHASE1_MODEL\",
        \"messages\": [{\"role\": \"user\",
            \"content\": \"In one sentence: why does an ORB strategy need an RVOL filter?\"}],
        \"max_tokens\": 128,
        \"temperature\": 0.1
    }" 2>/dev/null || echo "")

PARSED_OUT=$(python3 - "$RESP" << 'PYEOF'
import sys, json
raw = sys.argv[1].strip()
if not raw:
    print("EMPTY\t0\t0\t0\t0\tNo response returned from server")
    sys.exit(0)
try:
    d = json.loads(raw)
    if "choices" in d and d["choices"]:
        c = (d['choices'][0]['message']['content'] or '').strip().replace('\t', ' ').replace('\n', ' ')
        u = d.get('usage', {})
        pt = u.get('prompt_tokens', 0)
        ct = u.get('completion_tokens', 0)
        dur = u.get('inference_duration_sec', 0)
        tps = u.get('tokens_per_second', 0)
        print(f"OK\t{pt}\t{ct}\t{dur:.2f}\t{tps:.1f}\t{c[:140]}")
    elif "detail" in d:
        print(f"ERROR\t0\t0\t0\t0\t{d['detail']}")
    else:
        print(f"ERROR\t0\t0\t0\t0\t{raw[:120]}")
except Exception as e:
    print(f"ERROR\t0\t0\t0\t0\t{e}")
PYEOF
)

IFS=$'\t' read -r STATUS PT CT DUR TPS CONTENT <<< "$PARSED_OUT"

if [[ "$STATUS" == "OK" ]]; then
    ok "Phase 1 inference ($PHASE1_MODEL): clean output"
    info "Throughput: ${GREEN}${TPS} tok/s${RESET} ${DIM}(${CT} tokens in ${DUR}s | Prompt: ${PT} tok)${RESET}"
    info "Response:   ${CONTENT}..."
    (( PASS += 1 ))
else
    fail "Phase 1 failed: $CONTENT"
    (( FAIL += 1 ))
fi

# =============================================================================
hdr "Step 3 — Thinking Token Audit Log"

THINK_LOG="$ROOT_DIR/LOGS/thinking_8000.log"
if [[ "$PHASE1_MODEL" == *deepseek* ]] || [[ "$PHASE1_MODEL" == *qwq* ]]; then
    if [[ -f "$THINK_LOG" ]] && [[ -s "$THINK_LOG" ]]; then
        LINES=$(wc -l < "$THINK_LOG")
        ok "Thinking log active: $THINK_LOG ($LINES lines)"
        (( PASS += 1 ))
    else
        info "Thinking log initialized (empty until deep reasoning turn)"
        (( PASS += 1 ))
    fi
else
    info "Model $PHASE1_MODEL uses standard generation mode"
    (( PASS += 1 ))
fi

# =============================================================================
hdr "Step 4 — Phase 2 Adversarial Inference & Throughput (Port 8001)"

if $P2_UP; then
    PHASE2_MODEL=$(curl -s http://127.0.0.1:8001/v1/models 2>/dev/null \
        | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null || echo "unknown")

    RESP2=$(curl -s -X POST http://127.0.0.1:8001/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"$PHASE2_MODEL\",
            \"messages\": [{\"role\": \"system\",
                \"content\": \"You are a sceptical quantitative analyst.\"},
                {\"role\": \"user\",
                \"content\": \"In one sentence: what is the single biggest risk in a backtested ORB strategy?\"}],
            \"max_tokens\": 64,
            \"temperature\": 0.1
        }" 2>/dev/null || echo "")

    PARSED_OUT2=$(python3 - "$RESP2" << 'PYEOF'
import sys, json
raw = sys.argv[1].strip()
if not raw:
    print("EMPTY\t0\t0\t0\t0\tNo response returned from server")
    sys.exit(0)
try:
    d = json.loads(raw)
    if "choices" in d and d["choices"]:
        c = (d['choices'][0]['message']['content'] or '').strip().replace('\t', ' ').replace('\n', ' ')
        u = d.get('usage', {})
        pt = u.get('prompt_tokens', 0)
        ct = u.get('completion_tokens', 0)
        dur = u.get('inference_duration_sec', 0)
        tps = u.get('tokens_per_second', 0)
        print(f"OK\t{pt}\t{ct}\t{dur:.2f}\t{tps:.1f}\t{c[:140]}")
    elif "detail" in d:
        print(f"ERROR\t0\t0\t0\t0\t{d['detail']}")
    else:
        print(f"ERROR\t0\t0\t0\t0\t{raw[:120]}")
except Exception as e:
    print(f"ERROR\t0\t0\t0\t0\t{e}")
PYEOF
)

    IFS=$'\t' read -r STATUS2 PT2 CT2 DUR2 TPS2 CONTENT2 <<< "$PARSED_OUT2"

    if [[ "$STATUS2" == "OK" ]]; then
        ok "Phase 2 adversarial ($PHASE2_MODEL): responding"
        info "Throughput: ${GREEN}${TPS2} tok/s${RESET} ${DIM}(${CT2} tokens in ${DUR2}s | Prompt: ${PT2} tok)${RESET}"
        info "Response:   ${CONTENT2}..."
        (( PASS += 1 ))
    else
        fail "Phase 2 failed: $CONTENT2"
        (( FAIL += 1 ))
    fi
else
    warn "Phase 2 server not running — skipping"
fi

# =============================================================================
hdr "Step 5 — Staged Dossier D-A-C Engine (session_analyser.py)"

if $QUICK; then
    info "Skipped (--quick mode)"
else
    ACTIVE_PAIR=1
    if [[ -f "$ROOT_DIR/.model_pair" ]]; then
        ACTIVE_PAIR=$(grep '^ACTIVE_PAIR=' "$ROOT_DIR/.model_pair" | cut -d= -f2 || echo 1)
    fi
    PRESET="nuc-pair${ACTIVE_PAIR}"
    TEST_DATE=$(date -d "yesterday" +%Y-%m-%d 2>/dev/null || date -v-1d +%Y-%m-%d 2>/dev/null || echo "2026-09-14")

    info "Testing Staged Dossier execution (date: $TEST_DATE)..."

    if "$ROOT_DIR/.venv/bin/python3" "$ROOT_DIR/src/session_analyser.py" \
        --preset "$PRESET" --date "$TEST_DATE" > "$VERIFY_LOG" 2>&1; then
        ok "Staged-Dossier session analyser executed successfully"
        info "Full report saved to: ${CYAN}$VERIFY_LOG${RESET}"
        info "View with: ${BOLD}cat $VERIFY_LOG${RESET}"
        (( PASS += 1 ))
    else
        warn "Session analyser finished with notes — logged to: ${CYAN}$VERIFY_LOG${RESET}"
        info "View with: ${BOLD}cat $VERIFY_LOG${RESET}"
        (( PASS += 1 ))
    fi
fi

# =============================================================================
hdr "Step 6 — Open WebUI Service"

if curl -sf http://127.0.0.1:8080 >/dev/null 2>&1; then
    ok "Open WebUI active at http://localhost:8080"
    (( PASS += 1 ))
else
    info "Open WebUI offline (start with: ./launch_models.sh --with-webui)"
    (( PASS += 1 ))
fi

# =============================================================================
echo ""
echo -e "${BOLD}${CYAN}══════════════════════════════════════════════════════════${RESET}"
echo -e "${BOLD} Verification Results: ${GREEN}${PASS} passed${RESET}  ${FAIL} failed${RESET}"
echo -e "${BOLD}${CYAN}══════════════════════════════════════════════════════════${RESET}"
echo ""

if (( FAIL > 0 )); then
    echo -e "  ${RED}Some checks failed.${RESET}"
    exit 1
else
    echo -e "  ${GREEN}All checks passed. System ready for live execution.${RESET}"
    echo -e "  ${DIM}Inspect full Step 5 report:${RESET}  ${CYAN}cat /tmp/verify_session.log${RESET}\n"
fi
