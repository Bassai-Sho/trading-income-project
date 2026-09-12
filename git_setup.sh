#!/usr/bin/env bash
# =============================================================================
# Trading Income Project — GitHub Setup
# =============================================================================
# Run once, from the project root, after unzipping the package.
# Initialises git, creates the first commit, and pushes to GitHub.
#
# Prerequisites:
#   1. Create an empty GitHub repository at github.com/new (no README)
#   2. Have your GitHub Personal Access Token ready (or SSH key configured)
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
info() { echo -e "  ${DIM}    $*${RESET}"; }
hdr()  { echo -e "\n${BOLD}${CYAN}── $* ${RESET}"; }

# ── Must run from project root ─────────────────────────────────────────────
[[ -f "README.md" && -d "src" ]] || {
    echo -e "${RED}Run this from the project root (where README.md lives).${RESET}"
    exit 1
}

echo -e "\n${BOLD}Trading Income Project — GitHub Setup${RESET}"
echo -e "${DIM}Initialises git and pushes to your GitHub repository.${RESET}"

# ── Check git is installed ─────────────────────────────────────────────────
hdr "Checking git"
if ! command -v git &>/dev/null; then
    warn "Git not found"
    read -r -p "     Install git now? (sudo required) [Y/n] " reply
    reply="${reply:-Y}"
    if [[ "$reply" =~ ^[Yy]$ ]]; then
        sudo apt-get install -y git
    else
        echo "Install git manually: sudo apt install git"
        exit 1
    fi
fi
ok "Git $(git --version | grep -oP '[\d.]+')"

# ── Git identity ───────────────────────────────────────────────────────────
hdr "Git identity"
CURRENT_NAME=$(git config --global user.name  2>/dev/null || echo "")
CURRENT_EMAIL=$(git config --global user.email 2>/dev/null || echo "")

if [[ -n "$CURRENT_NAME" && -n "$CURRENT_EMAIL" ]]; then
    ok "Identity: $CURRENT_NAME <$CURRENT_EMAIL>"
else
    warn "Git identity not set"
    read -r -p "     Your name (shown in commits): "  GIT_NAME
    read -r -p "     Your email: "                    GIT_EMAIL
    git config --global user.name  "$GIT_NAME"
    git config --global user.email "$GIT_EMAIL"
    ok "Identity set: $GIT_NAME <$GIT_EMAIL>"
fi

# ── Initialise git repo ────────────────────────────────────────────────────
hdr "Initialising repository"

if [[ -d ".git" ]]; then
    ok "Git already initialised"
else
    git init
    ok "Git initialised"
fi

# Set default branch to main
git symbolic-ref HEAD refs/heads/main 2>/dev/null || true

# ── Safety check: nothing sensitive staged ─────────────────────────────────
hdr "Safety check"
if [[ -f ".env" ]] && git ls-files --error-unmatch .env &>/dev/null 2>&1; then
    warn ".env is tracked! Removing from git now..."
    git rm --cached .env
    ok ".env removed from tracking"
else
    ok ".env not tracked (good — it contains your API keys)"
fi

# Verify DATA/ is excluded
if git check-ignore -q DATA/ 2>/dev/null; then
    ok "DATA/ excluded by .gitignore (SQLite databases stay local)"
else
    warn "DATA/ is not excluded — checking .gitignore..."
    echo "DATA/" >> .gitignore
    ok "Added DATA/ to .gitignore"
fi

# ── First commit ───────────────────────────────────────────────────────────
hdr "Creating first commit"

git add .
STAGED=$(git diff --cached --name-only | wc -l)
ok "$STAGED files staged"

echo ""
info "Files that will be committed:"
git diff --cached --name-only | head -20 | while read f; do info "  $f"; done
echo ""

git commit -m "Initial commit — trading income project v1" 2>/dev/null || {
    warn "Nothing new to commit (already committed)"
}
ok "Commit created"

# ── Remote repository ──────────────────────────────────────────────────────
hdr "GitHub remote"

EXISTING_REMOTE=$(git remote get-url origin 2>/dev/null || echo "")

if [[ -n "$EXISTING_REMOTE" ]]; then
    ok "Remote already configured: $EXISTING_REMOTE"
else
    echo ""
    echo -e "  ${DIM}Create an empty repository at github.com/new first.${RESET}"
    echo -e "  ${DIM}Set visibility to Private. Do not add README or .gitignore.${RESET}"
    echo ""
    read -r -p "     Your GitHub username: "         GH_USER
    read -r -p "     Repository name [trading-income-project]: " GH_REPO
    GH_REPO="${GH_REPO:-trading-income-project}"

    echo ""
    echo -e "  ${BOLD}Authentication method:${RESET}"
    echo -e "  ${CYAN}1${RESET}  HTTPS with Personal Access Token ${DIM}(simpler)${RESET}"
    echo -e "  ${CYAN}2${RESET}  SSH key ${DIM}(more secure, no token expiry)${RESET}"
    echo ""
    read -r -p "     Choose [1/2, default=1]: " AUTH_CHOICE
    AUTH_CHOICE="${AUTH_CHOICE:-1}"

    if [[ "$AUTH_CHOICE" == "2" ]]; then
        REMOTE_URL="git@github.com:${GH_USER}/${GH_REPO}.git"
        echo ""
        info "Make sure your SSH key is added to GitHub:"
        info "  cat ~/.ssh/id_ed25519.pub   (then paste into GitHub → Settings → SSH keys)"
        info "  ssh-keygen -t ed25519 -C \"${CURRENT_EMAIL}\"  (if you need to create one)"
    else
        REMOTE_URL="https://github.com/${GH_USER}/${GH_REPO}.git"
        echo ""
        info "You'll need a Personal Access Token (not your password):"
        info "  GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)"
        info "  Scopes: tick 'repo'. Copy the token and use it as your password when prompted."
        echo ""
        # Enable credential storage so you don't have to enter it every push
        git config --global credential.helper store
        info "Credentials will be saved after first push (credential.helper=store)"
    fi

    git remote add origin "$REMOTE_URL"
    ok "Remote added: $REMOTE_URL"
fi

# ── Push ───────────────────────────────────────────────────────────────────
hdr "Pushing to GitHub"

git branch -M main
echo ""
read -r -p "     Push to GitHub now? [Y/n] " reply
reply="${reply:-Y}"

if [[ "$reply" =~ ^[Yy]$ ]]; then
    echo ""
    git push -u origin main && ok "Pushed to GitHub successfully" || {
        warn "Push failed. Common fixes:"
        info "  • Check your Personal Access Token has 'repo' scope"
        info "  • Verify the repository exists: $(git remote get-url origin 2>/dev/null)"
        info "  • For HTTPS: ensure credential.helper is set: git config --global credential.helper store"
        info "  • For SSH: test connection: ssh -T git@github.com"
        exit 1
    }
else
    warn "Skipped. Push manually when ready:"
    info "  git push -u origin main"
fi

# ── Summary ────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${CYAN}══════════════════════════════════════════${RESET}"
echo -e "${BOLD} Repository ready${RESET}"
echo -e "${BOLD}${CYAN}══════════════════════════════════════════${RESET}"
echo ""
REMOTE=$(git remote get-url origin 2>/dev/null || echo "not configured")
echo -e "  Remote: ${DIM}$REMOTE${RESET}"
echo ""
echo -e "${BOLD}  Day-to-day workflow:${RESET}"
echo "  git add src/changed_file.py"
echo "  git commit -m 'Brief description of change'"
echo "  git push"
echo ""
echo -e "  ${DIM}Full guide: docs/GITHUB_SETUP.md${RESET}"
echo ""
