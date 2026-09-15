# Trading Income Project — GitHub Setup & Deployment Guide

This guide details how to version-control the codebase, safeguard private credentials and databases, and deploy the system onto a fresh Intel NUC machine.

---

## 1. Create the Repository on GitHub

1. Go to [github.com/new](https://github.com/new).
2. Set the repository details:

| Field | Value |
|---|---|
| **Repository name** | `trading-income-project` |
| **Visibility** | **Private** (Recommended — trading strategies and logs contain proprietary alpha) |
| **Initialize with README** | **No** (The project already contains an updated `README.md`) |
| **Add .gitignore** | **No** (The project already contains a customized `.gitignore`) |
| **Choose a license** | None / Private |

3. Click **Create repository**. Keep the resulting GitHub page open.

---

## 2. Option A: Automated Git Setup (Recommended)

The project includes an interactive setup helper that validates security exclusions before pushing:

```bash
chmod +x git_setup.sh
./git_setup.sh
```

### What `git_setup.sh` Does:
1. Verifies that `.env`, `DATA/*.db`, `~/models/`, and `.venv/` are excluded by `.gitignore`.
2. Initializes Git and sets the default branch to `main`.
3. Creates the initial baseline commit.
4. Prompts for your GitHub repository URL (HTTPS or SSH) and pushes the codebase.

---

## 3. Option B: Manual Git Setup & First Push

If you prefer running Git commands manually:

### Step 1: Configure Git Identity (First time only)
```bash
git config --global user.name  "Your Name"
git config --global user.email "your@email.com"
```

### Step 2: Initialize, Commit, and Link Remote
Run from your project root:

```bash
cd ~/github/trading-income-project

git init
git branch -M main

# Verify sensitive files are ignored before staging
git status

# Stage source code, tests, and documentation
git add .
git commit -m "Initial commit — Trading Income Project v2.4"

# Link your remote repository (replace with your GitHub username)
git remote add origin https://github.com/<YOUR_USERNAME>/trading-income-project.git

# Push to GitHub
git push -u origin main
```

---

## 4. GitHub Authentication

### Option 1: Personal Access Token (HTTPS)
1. Go to **GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)**.
2. Click **Generate new token (classic)**.
3. Select scope: **`repo`** (Full control of private repositories).
4. Copy the generated token (`ghp_...`).
5. When Git prompts for your password in the terminal, **paste the token**.

To avoid typing the token on every push:
```bash
git config --global credential.helper store
```

### Option 2: SSH Key (Recommended & Permanent)
```bash
# 1. Generate SSH key
ssh-keygen -t ed25519 -C "your@email.com"

# 2. Copy the public key
cat ~/.ssh/id_ed25519.pub

# 3. Add to GitHub: Settings → SSH and GPG keys → New SSH key

# 4. Test connection
ssh -T git@github.com

# 5. Set origin to SSH URL
git remote set-url origin git@github.com:<YOUR_USERNAME>/trading-income-project.git
```

---

## 5. What `.gitignore` Protects (Security Boundaries)

The project's `.gitignore` enforces strict separation between application source code and private runtime data:

| Excluded Pattern | Why It Is Excluded |
|---|---|
| `.env`, `.env.*` | Contains API keys (`ALPACA_SECRET_KEY`, `FRED_API_KEY`, Discord webhooks) |
| `DATA/*.db`, `*.db-wal` | Local SQLite stores containing real execution records and trade journals |
| `~/models/`, `.ov_cache/` | Heavy model binaries (`.bin`, `.xml`) — downloaded directly from Hugging Face |
| `LOGS/`, `*.log` | Runtime execution logs and PID trackers |
| `.venv/`, `venv/` | Python virtual environment binaries |
| `__pycache__/` | Compiled Python bytecode |

**Sanity Check:**
Before pushing, run `git status`. It should only list files inside `src/`, `docs/`, shell scripts, `README.md`, `requirements.txt`, and `.env.example`.

---

## 6. Day-to-Day Development Workflow

```bash
# 1. Check modified files
git status

# 2. Stage changes
git add src/session_analyser.py
# Or stage all code updates:
git add src/ docs/

# 3. Commit with a descriptive message
git commit -m "Update Staged Dossier pipeline with live tok/s telemetry"

# 4. Push to GitHub
git push
```

---

## 7. Deploying to a Fresh NUC or Second Host

To replicate this environment on another machine:

```bash
# 1. Clone repository
git clone https://github.com/<YOUR_USERNAME>/trading-income-project.git
cd trading-income-project

# 2. Run the unified setup (handles drivers, .venv, keys, models, and data bootstrap)
chmod +x setup.sh launch_models.sh verify.sh
./setup.sh

# 3. Launch model servers & Open WebUI
./launch_models.sh --with-webui

# 4. Run verification suite
./verify.sh

# 5. Start automated master runner
python3 src/runner.py
```

---

## 8. Feature Branch Workflow

For disciplined strategy and quant development:

```bash
# 1. Create a feature branch
git checkout -b feature/vwap-trailing-stop

# 2. Develop, test, and commit
git add src/trading_engine.py
git commit -m "Enhance VWAP trailing stop with ladder exit levels"

# 3. Merge back to main when verified
git checkout main
git merge feature/vwap-trailing-stop
git push
```
