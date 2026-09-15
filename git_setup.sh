#!/usr/bin/env bash
# =============================================================================
# Trading Income Project — Automated Git Setup & Push
# =============================================================================
# Run once from project root to initialise git, apply security guardrails,
# create the initial baseline commit, and push to GitHub.
#
# Usage:
#   chmod +x git_setup.sh
#   ./git_setup.sh
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

[[ -f "README.md" && -d "src" ]] || {
    fail "Run this from the project root (where README.md lives)."
    exit 1
}

echo -e "\n${BOLD}Trading Income Project — GitHub Setup${RESET}"
echo -e "${DIM}Initialises git, verifies security boundaries, and pushes to GitHub.${RESET}"

# ── 1. Check Git Installation ────────────────────────────────────────────────
hdr "1 / 5  Checking Git Installation"
if ! command -v git &>/dev/null; then
    warn "Git not found"
    read -r -p "     Install git now? (sudo required) [Y/n] " reply
    reply="${reply:-Y}"
    if [[ "$reply" =~ ^[Yy]$ ]]; then
        sudo apt-get install -y git
    else
        fail "Git is required. Install with: sudo apt install git"
        exit 1
    fi
fi
ok "Git $(git --version | grep -oP '[\d.]+' | head -n1) available"

# ── 2. Configure Git Identity ─────────────────────────────────────────────────
hdr "2 / 5  Git Identity Configuration"
CURRENT_NAME=$(git config --global user.name 2>/dev/null || echo "")
CURRENT_EMAIL=$(git config --global user.email 2>/dev/null || echo "")

if [[ -n "$CURRENT_NAME" && -n "$CURRENT_EMAIL" ]]; then
    ok "Identity: $CURRENT_NAME <$CURRENT_EMAIL>"
else
    warn "Git identity not set"
    read -r -p "     Enter your Git Name (shown in commits): " GIT_NAME
    read -r -p "     Enter your Git Email: " GIT_EMAIL
    git config --global user.name "$GIT_NAME"
    git config --global user.email "$GIT_EMAIL"
    CURRENT_EMAIL="$GIT_EMAIL"
    ok "Identity set: $GIT_NAME <$GIT_EMAIL>"
fi

# ── 3. Hardened .gitignore Enforcement ────────────────────────────────────────
hdr "3 / 5  Enforcing Security Guardrails (.gitignore)"

if [[ ! -f ".gitignore" ]] || ! grep -q "DATA/" .gitignore; then
    cat << 'EOF' > .gitignore
# ── Environment & Credentials (NEVER COMMIT) ──────────────────────────────────
.env
.env.*
!.env.example
.webui_secret_key
kernel.errors.txt

# ── Generated Databases & Working Data ───────────────────────────────────────
DATA/
*.db
*.db-shm
*.db-wal

# ── Model Weights & OpenVINO Cache ───────────────────────────────────────────
*.bin
*.xml
.ov_cache/
models/
~/models/

# ── Runtime State & Logs ──────────────────────────────────────────────────────
LOGS/
*.log
*.tmp
*.bak
.model_pair
markov_state.json
candle_ngram.json
outcome_mc.json

# ── Open WebUI Local State ───────────────────────────────────────────────────
.webui/
webui/

# ── Python & Environments ────────────────────────────────────────────────────
__pycache__/
*.py[cod]
*.pyo
.pytest_cache/
*.egg-info/
dist/
build/
.venv/
venv/
env/

# ── Streamlit & IDEs ─────────────────────────────────────────────────────────
.streamlit/
.vscode/
.idea/
*.swp
.DS_Store
Thumbs.db
EOF
    ok "Created hardened .gitignore (excludes API keys, models, and databases)"
else
    ok ".gitignore security rules verified"
fi

# ── 4. Initialise & Stage ─────────────────────────────────────────────────────
hdr "4 / 5  Initialising & Committing Codebase"

if [[ ! -d ".git" ]]; then
    git init -b main
    ok "Git repository initialised on branch 'main'"
else
    ok "Git repository already initialised"
    git branch -M main 2>/dev/null || true
fi

# Untrack any accidentally cached sensitive files
for sensitive in .env DATA/ LOGS/; do
    if git ls-files --error-unmatch "$sensitive" >/dev/null 2>&1; then
        git rm -r --cached "$sensitive" >/dev/null 2>&1 || true
        warn "Removed $sensitive from git staging (protected by .gitignore)"
    fi
done

git add .
STAGED_COUNT=$(git diff --cached --name-only | wc -l)

if (( STAGED_COUNT > 0 )); then
    git commit -m "Initial commit — Trading Income Project v2.4 (OpenVINO GenAI, Staged Dossier D-A-C, 58GB VRAM)"
    ok "Committed $STAGED_COUNT files cleanly"
else
    info "No unstaged changes to commit"
fi

# ── 5. Remote Repository & Push ───────────────────────────────────────────────
hdr "5 / 5  GitHub Remote Setup & Push"

EXISTING_REMOTE=$(git remote get-url origin 2>/dev/null || echo "")

if [[ -z "$EXISTING_REMOTE" ]]; then
    echo ""
    info "Create an empty private repository at https://github.com/new"
    info "(Do NOT initialize with a README, license, or .gitignore)"
    echo ""
    read -r -p "     Enter GitHub Remote URL (HTTPS or SSH, or press Enter to skip): " REMOTE_URL

    if [[ -n "$REMOTE_URL" ]]; then
        git remote add origin "$REMOTE_URL"
        ok "Added remote origin: $REMOTE_URL"

        # Enable credential caching for HTTPS
        if [[ "$REMOTE_URL" == https://* ]]; then
            git config --global credential.helper store
            info "HTTPS credential storage enabled"
        fi

        read -r -p "     Push to origin/main now? [Y/n] " push_reply
        push_reply="${push_reply:-Y}"
        if [[ "$push_reply" =~ ^[Yy]$ ]]; then
            echo ""
            git push -u origin main && ok "Pushed to GitHub successfully" || {
                warn "Push failed. If using HTTPS, ensure you use a Personal Access Token (PAT) with 'repo' scope."
            }
        fi
    else
        info "Remote configuration skipped. Add anytime via: git remote add origin <url>"
    fi
else
    ok "Remote already linked: $EXISTING_REMOTE"
    read -r -p "     Push latest updates to GitHub now? [Y/n] " push_reply
    push_reply="${push_reply:-Y}"
    if [[ "$push_reply" =~ ^[Yy]$ ]]; then
        git push -u origin main && ok "Pushed updates to GitHub" || warn "Push failed — check authentication/permissions"
    fi
fi

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${CYAN}══════════════════════════════════════════════════════════════${RESET}"
echo -e "${BOLD}${GREEN} Git Repository Ready${RESET}"
echo -e "${BOLD}${CYAN}══════════════════════════════════════════════════════════════${RESET}"
echo ""
CURRENT_ORIGIN=$(git remote get-url origin 2>/dev/null || echo "Not configured")
echo -e "  ${BOLD}Remote URL:${RESET}  ${CYAN}$CURRENT_ORIGIN${RESET}"
echo -e "  ${BOLD}Branch:${RESET}      main"
echo ""
echo -e "  ${BOLD}Day-to-day workflow:${RESET}"
echo "    git status"
echo "    git add src/ docs/"
echo "    git commit -m 'Descriptive update message'"
echo "    git push"
echo ""
