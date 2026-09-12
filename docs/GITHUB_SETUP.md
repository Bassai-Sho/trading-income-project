# GitHub Setup Guide

## 1. Create the repository on GitHub

Go to [github.com/new](https://github.com/new) and fill in:

| Field | Value |
|---|---|
| Repository name | `trading-income-project` (or your preferred name) |
| Visibility | **Private** (recommended — trading strategies are sensitive) |
| Initialise with README | **No** — we already have one |
| Add .gitignore | **No** — we already have one |
| Choose a licence | Up to you |

Click **Create repository**. GitHub will show you the empty repo page — leave it open.

---

## 2. Set up Git on the NUC (first time only)

```bash
# Install git if needed
sudo apt install git -y

# Set your identity (shown in commit history)
git config --global user.name  "Your Name"
git config --global user.email "your@email.com"
```

---

## 3. Authenticate with GitHub

**Option A — Personal Access Token (simplest, works immediately)**

1. Go to GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Click **Generate new token (classic)**
3. Scopes: tick `repo` (full control of private repos)
4. Copy the token — you only see it once

When Git asks for a password, paste the token.

To avoid entering it every time:
```bash
git config --global credential.helper store
# After your first push, credentials are saved to ~/.git-credentials
```

**Option B — SSH key (more secure, no token expiry)**

```bash
# Generate a key
ssh-keygen -t ed25519 -C "your@email.com"

# Copy the public key
cat ~/.ssh/id_ed25519.pub

# Paste it into GitHub → Settings → SSH and GPG keys → New SSH key

# Test the connection
ssh -T git@github.com
```

If using SSH, replace the remote URL in Step 4 with the SSH form:
`git@github.com:YOUR_USERNAME/trading-income-project.git`

---

## 4. Initialise and push

Run these from inside the project directory (where `README.md` lives):

```bash
cd ~/trading/trading-income-project    # adjust path if different

git init
git add .
git commit -m "Initial commit — trading income project v1"

# Replace YOUR_USERNAME with your GitHub username
git remote add origin https://github.com/YOUR_USERNAME/trading-income-project.git

git branch -M main
git push -u origin main
```

On success you'll see something like:
```
Enumerating objects: 29, done.
Counting objects: 100% (29/29), done.
Branch 'main' set up to track remote branch 'main' from 'origin'.
```

---

## 5. What the .gitignore protects

The `.gitignore` already excludes everything sensitive:

| Excluded | Why |
|---|---|
| `.env` | Contains API keys — never commit this |
| `DATA/` | SQLite databases — large and contain private trade data |
| `*.db` | All database files |
| `LOGS/` | Log files |
| `markov_state.json` | Generated model state — rebuilt automatically |
| `candle_ngram.json` | Generated — rebuilt by historical_sim |
| `outcome_mc.json` | Generated — rebuilt by historical_sim |
| `__pycache__/` | Python bytecode |
| `venv/`, `.venv/` | Virtual environments |

**Always verify before pushing:**
```bash
git status    # should show only src/*.py, docs/, and config files
```

---

## 6. Day-to-day workflow

```bash
# After making changes to a file
git add src/trading_engine.py          # stage a specific file
git add src/                           # stage all changes in src/
git commit -m "Fix VWAP slope calculation in morning brief"
git push

# Pull changes from another machine
git pull
```

**Good commit message format:**
```
Add fred_store.py with FRED macro data download
Fix: correct ORB range calculation on early-close days
Update: wire sentiment_store into runner.py daily schedule
```

---

## 7. Useful Git commands

```bash
git status            # what's changed but not committed
git diff              # show exact changes line by line
git log --oneline     # compact commit history
git log --oneline -10 # last 10 commits

git stash             # temporarily shelve uncommitted changes
git stash pop         # restore shelved changes

# If you accidentally add something you shouldn't have
git reset HEAD .env   # unstage a file
git rm --cached DATA/ # stop tracking a directory (also add to .gitignore)
```

---

## 8. Pull the latest version to another machine

```bash
# On a second machine (e.g. your laptop)
git clone https://github.com/YOUR_USERNAME/trading-income-project.git
cd trading-income-project

# Run setup
./setup.sh

# Create and fill in your .env (this is machine-specific, never in git)
cp .env.example .env
nano .env
```

---

## 9. Branch strategy (optional, for disciplined development)

```bash
# Create a feature branch before making changes
git checkout -b feature/ladder-exits

# Work, commit, test
git add src/trading_engine.py
git commit -m "Add ladder exit mechanic at 1R/2R/3R"

# Merge back to main when ready
git checkout main
git merge feature/ladder-exits
git push
```

This keeps `main` always in a known-good state.
