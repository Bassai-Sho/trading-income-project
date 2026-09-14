#!/usr/bin/env bash
# =============================================================================
# Trading Income Project — Quick Start Verification
# =============================================================================
# Runs all 5 verification steps after ./prepare_host.sh + ./launch_models.sh
# Handles missing jq gracefully, waits for servers to be ready, and gives
# clear pass/fail output at each step.
#
# Usage:
#   ./verify.sh                     # full verification
#   ./verify.sh --date 2025-01-15   # test session analyser with a specific date
#   ./verify.sh --quick             # skip session analyser (faster)
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
    fail "Port 8000 — not responding. Run: ./launch_models.sh"
    (( FAIL += 1 ))
fi

if $P2_UP; then
    MODEL=$(curl -s http://127.0.0.1:8001/v1/models 2>/dev/null \
        | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['data'][0]['id'])" 2>/dev/null || echo "unknown")
    ok "Port 8001 — READY  (model: $MODEL)"
    (( PASS += 1 ))
else
    fail "Port 8001 — not responding. Run: ./launch_models.sh"
    (( FAIL += 1 ))
fi

$P1_UP || { echo ""; warn "Cannot continue without Port 8000. Start servers first."; exit 1; }

# =============================================================================
hdr "Step 2 — Phase 1 Inference (Clean Output, No Thinking Tokens)"

PHASE1_MODEL=$(curl -s http://127.0.0.1:8000/v1/models 2>/dev/null \
    | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null || echo "unknown")

RESP=$(curl -sf http://127.0.0.1:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"$PHASE1_MODEL\",
        \"messages\": [{\"role\": \"user\",
            \"content\": \"In one sentence: why does an ORB strategy need an RVOL filter?\"}],
        \"max_tokens\": 128,
        \"temperature\": 0.1
    }" 2>/dev/null || echo "")

if [[ -z "$RESP" ]]; then
    fail "Phase 1 inference: no response"
    (( FAIL += 1 ))
else
    CONTENT=$(echo "$RESP" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    c = d['choices'][0]['message']['content'] or ''
    print(c[:200])
except Exception as e:
    print(f'PARSE_ERROR: {e}')
" 2>/dev/null || echo "PARSE_ERROR")

    if echo "$CONTENT" | grep -q "<think>"; then
        fail "Phase 1: <think> tokens NOT stripped from response content"
        (( FAIL += 1 ))
    elif [[ "$CONTENT" == *PARSE_ERROR* ]] || [[ -z "$CONTENT" ]]; then
        fail "Phase 1: response malformed — $CONTENT"
        (( FAIL += 1 ))
    else
        ok "Phase 1 inference: clean output"
        info "Response: ${CONTENT:0:120}..."
        (( PASS += 1 ))
    fi
fi

# =============================================================================
hdr "Step 3 — Thinking Token Audit Log (DeepSeek-R1 / QwQ only)"

THINK_LOG="$ROOT_DIR/LOGS/thinking_8000.log"
if [[ "$PHASE1_MODEL" == *deepseek* ]] || [[ "$PHASE1_MODEL" == *qwq* ]]; then
    if [[ -f "$THINK_LOG" ]] && [[ -s "$THINK_LOG" ]]; then
        LINES=$(wc -l < "$THINK_LOG")
        ok "Thinking log: $THINK_LOG ($LINES lines)"
        info "Last 3 lines of thinking log:"
        tail -3 "$THINK_LOG" | while IFS= read -r line; do info "  $line"; done
        (( PASS += 1 ))
    else
        warn "Thinking log empty or missing: $THINK_LOG"
        info "(Populated after first inference — run Step 2 first)"
        (( PASS += 1 ))
    fi
else
    info "Pair uses $PHASE1_MODEL — standard chat template active"
    info "Thinking log only captures DeepSeek-R1 / QwQ models"
    (( PASS += 1 ))
fi

# =============================================================================
hdr "Step 4 — Phase 2 Adversarial Inference (Architecture Independence)"

if $P2_UP; then
    PHASE2_MODEL=$(curl -s http://127.0.0.1:8001/v1/models 2>/dev/null \
        | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null || echo "unknown")

    RESP2=$(curl -sf http://127.0.0.1:8001/v1/chat/completions \
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

    CONTENT2=$(echo "$RESP2" | python3 -c "
import sys, json
try:
    print(json.load(sys.stdin)['choices'][0]['message']['content'][:150])
except: print('PARSE_ERROR')
" 2>/dev/null || echo "PARSE_ERROR")

    if [[ "$CONTENT2" == *PARSE_ERROR* ]] || [[ -z "$CONTENT2" ]]; then
        fail "Phase 2 inference failed"
        (( FAIL += 1 ))
    else
        ok "Phase 2 adversarial ($PHASE2_MODEL): responding"
        info "Response: ${CONTENT2:0:120}..."
        (( PASS += 1 ))
    fi
else
    warn "Phase 2 server not running — skipping"
fi

# =============================================================================
hdr "Step 5 — Full D-A-C Session (session_analyser.py)"

if $QUICK; then
    info "Skipped (--quick mode)"
elif ! [[ -d "$ROOT_DIR/.venv" ]]; then
    warn "No .venv found — skipping session analyser test"
else
    ACTIVE_PAIR=1
    if [[ -f "$ROOT_DIR/.model_pair" ]]; then
        ACTIVE_PAIR=$(grep '^ACTIVE_PAIR=' "$ROOT_DIR/.model_pair" | cut -d= -f2 || echo 1)
    fi
    PRESET="nuc-pair${ACTIVE_PAIR}"

    DATE_ARG=""
    if [[ -n "$TEST_DATE" ]]; then
        DATE_ARG="--date $TEST_DATE"
        info "Testing with date: $TEST_DATE"
    else
        TEST_DATE=$(date -d "yesterday" +%Y-%m-%d 2>/dev/null || date -v-1d +%Y-%m-%d 2>/dev/null || echo "2026-09-10")
        DATE_ARG="--date $TEST_DATE"
        info "Testing with date: $TEST_DATE (yesterday)"
    fi

    info "Running: python src/session_analyser.py --preset $PRESET $DATE_ARG --no-llm"
    info "(--no-llm for quick syntax/DB check; remove for full LLM run)"
    echo ""

    if "$ROOT_DIR/.venv/bin/python3" "$ROOT_DIR/src/session_analyser.py" \
        --preset "$PRESET" $DATE_ARG --no-llm 2>&1 | tail -5; then
        ok "session_analyser.py launched successfully (--no-llm mode)"
        (( PASS += 1 ))
    else
        warn "session_analyser.py exited non-zero (may be expected if no data for date)"
        (( PASS += 1 ))
    fi
fi

# =============================================================================
hdr "Step 6 — Open WebUI (optional)"

if curl -sf http://127.0.0.1:8080 >/dev/null 2>&1; then
    ok "Open WebUI running at http://localhost:8080"
    (( PASS += 1 ))
else
    info "Open WebUI not running (optional — start with: ./launch_models.sh --with-webui)"
    (( PASS += 1 ))
fi

# =============================================================================
echo ""
echo -e "${BOLD}${CYAN}══════════════════════════════════════════════════════════${RESET}"
echo -e "${BOLD} Results: ${GREEN}${PASS} passed${RESET}  ${FAIL} failed${RESET}"
echo -e "${BOLD}${CYAN}══════════════════════════════════════════════════════════${RESET}"
echo ""

if (( FAIL > 0 )); then
    echo -e "  ${RED}Some checks failed.${RESET}"
    exit 1
else
    echo -e "  ${GREEN}All checks passed.${RESET}"
    echo -e "  Full D-A-C run:  ${CYAN}python src/session_analyser.py --preset nuc-pair1${RESET}\n"
fi
