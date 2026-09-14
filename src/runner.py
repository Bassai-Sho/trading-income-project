"""
runner.py
=========
Single-command orchestrator for the entire Trading Income Project system.

USAGE
─────
  python runner.py                          # start everything, auto-schedule
  python runner.py --date 2026-09-08        # replay a specific session
  python runner.py --no-dashboard           # engine + analyser only (headless)
  python runner.py --weekend                # run weekly tournament now
  python runner.py --status                 # check what's running

ONE COMMAND REPLACES:
  python morning_brief.py
  python trading_engine.py
  streamlit run trading_dashboard.py
  python session_analyser.py   (auto-triggered at EOD)
  python tournament_evaluator.py  (auto-triggered Saturday)

SCHEDULE (all times EST)
────────────────────────
  Mon-Fri 09:00  → morning_brief.py (pre-market brief)
  Mon-Fri 09:28  → trading_engine.py starts (waits until 09:30)
  Mon-Fri 09:29  → trading_dashboard.py starts (streamlit)
  Mon-Fri 09:29  → markov_engine.py updated from overnight VIX
  Mon-Fri 15:35  → session_analyser.py (triggered by engine EOD)
  Mon-Fri 15:40  → monte_carlo forecast updated in DB
  Saturday 08:00 → tournament_evaluator.py (weekly Design Studio cycle)
  Every 60s      → health check (heartbeat table + process alive?)

DEPENDENCIES
────────────
  pip install apscheduler

PROCESS MANAGEMENT
──────────────────
  All child processes run as subprocesses of this runner.
  If a subprocess dies unexpectedly:
    - Runner logs the failure and sends Discord alert
    - Trading engine: restart attempted immediately
    - Dashboard: restart attempted after 30 seconds
    - Analyser: log the failure, skip (non-critical)
  Runner itself can be run as a systemd service on the NUC:
    sudo systemctl enable trading-runner
    sudo systemctl start  trading-runner
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import subprocess
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

log = logging.getLogger("runner")

# ---------------------------------------------------------------------------
# Try optional deps
# ---------------------------------------------------------------------------
try:
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.cron          import CronTrigger
    from apscheduler.triggers.interval      import IntervalTrigger
    HAS_SCHEDULER = True
except ImportError:
    HAS_SCHEDULER = False

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------
DEFAULT_CONFIG: dict[str, Any] = {
    "ticker":           "SPY",
    "orb_method":       "15min",
    "account":          10_000.0,
    "risk_pct":         0.01,
    "db_path":          "DATA/paper_account.db",
    "llm_preset":       os.environ.get("LLM_PRESET", "nuc-pair1"),
    "discord_webhook":  "",
    "telegram_token":   "",
    "telegram_chat_id": "",
    "streamlit_port":   8501,
    "tz_offset_hours":  -5,          # EST = UTC-5 (no DST adjust)
    "run_dashboard":    True,
    "run_monte_carlo":  True,
    "run_tournament":   True,
    "restart_engine_on_crash": True,
    "log_level":        "INFO",
}

HERE = Path(__file__).parent.resolve()

# ---------------------------------------------------------------------------
# Process registry
# ---------------------------------------------------------------------------
_procs: dict[str, subprocess.Popen] = {}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_est() -> datetime:
    return datetime.utcnow() + timedelta(hours=DEFAULT_CONFIG["tz_offset_hours"])

def _is_market_day() -> bool:
    """Rough NYSE market-day check (weekday only; does not check holidays)."""
    return _now_est().weekday() < 5

def _is_market_hours() -> bool:
    """True between 09:29 and 15:35 EST on weekdays."""
    if not _is_market_day():
        return False
    now = _now_est()
    open_t  = now.replace(hour=9,  minute=29, second=0)
    close_t = now.replace(hour=15, minute=35, second=0)
    return open_t <= now <= close_t

def _notify(message: str, level: str = "INFO") -> None:
    """Fire-and-forget notification to Discord and Telegram."""
    cfg = DEFAULT_CONFIG
    webhook  = cfg.get("discord_webhook", "")
    tg_token = cfg.get("telegram_token", "")
    tg_chat  = cfg.get("telegram_chat_id", "")

    emoji = {"INFO": "📡", "TRADE": "✅", "WARN": "⚠️", "HALT": "🛑",
             "START": "🟢", "STOP": "🔴"}.get(level, "📡")
    ts   = _now_est().strftime("%H:%M EST")
    text = f"{emoji} **RUNNER** [{ts}] {message}"

    import threading
    def _send():
        import urllib.request, json as _j
        if webhook:
            try:
                payload = _j.dumps({"content": text}).encode()
                req = urllib.request.Request(
                    webhook, data=payload,
                    headers={"Content-Type": "application/json"})
                urllib.request.urlopen(req, timeout=5)
            except Exception as e:
                log.warning("Discord failed: %s", e)
        if tg_token and tg_chat:
            try:
                payload = _j.dumps({"chat_id": tg_chat, "text": text,
                                     "parse_mode": "Markdown"}).encode()
                req = urllib.request.Request(
                    f"https://api.telegram.org/bot{tg_token}/sendMessage",
                    data=payload,
                    headers={"Content-Type": "application/json"})
                urllib.request.urlopen(req, timeout=5)
            except Exception as e:
                log.warning("Telegram failed: %s", e)

    threading.Thread(target=_send, daemon=True).start()

# ---------------------------------------------------------------------------
# Script launchers
# ---------------------------------------------------------------------------

def launch_morning_brief() -> None:
    """Run morning_brief.py and print output. Blocks until complete."""
    script = HERE / "morning_brief.py"
    if not script.exists():
        log.warning("morning_brief.py not found at %s", script)
        return
    log.info("▶ morning_brief.py")
    result = subprocess.run(
        [sys.executable, str(script),
         "--ticker", DEFAULT_CONFIG["ticker"],
         "--orb",    DEFAULT_CONFIG["orb_method"],
         "--account", str(DEFAULT_CONFIG["account"])],
        capture_output=False,
        timeout=60,
    )
    if result.returncode != 0:
        log.warning("morning_brief.py exited with code %d", result.returncode)


def launch_engine() -> subprocess.Popen:
    """Start trading_engine.py as a background subprocess."""
    script = HERE / "trading_engine.py"
    if not script.exists():
        log.error("trading_engine.py not found")
        return None
    cfg = DEFAULT_CONFIG
    cmd = [
        sys.executable, str(script),
        "--ticker",  cfg["ticker"],
        "--orb",     cfg["orb_method"],
        "--account", str(cfg["account"]),
        "--db",      cfg["db_path"],
    ]
    if cfg.get("llm_preset"):
        cmd += ["--preset", cfg["llm_preset"]]
    log.info("▶ trading_engine.py %s", " ".join(cmd[2:]))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, bufsize=1,
                             universal_newlines=True)
    _procs["engine"] = proc
    _notify(f"Engine started: {cfg['ticker']} {cfg['orb_method']} "
            f"account=${cfg['account']:,.0f}", "START")
    return proc


def launch_dashboard() -> subprocess.Popen | None:
    """Start Streamlit dashboard as background subprocess."""
    if not DEFAULT_CONFIG.get("run_dashboard", True):
        return None
    script = HERE / "trading_dashboard.py"
    if not script.exists():
        log.warning("trading_dashboard.py not found")
        return None
    port = DEFAULT_CONFIG.get("streamlit_port", 8501)
    cmd  = ["streamlit", "run", str(script),
             "--server.port", str(port),
             "--server.headless", "true"]
    log.info("▶ streamlit run trading_dashboard.py (port %d)", port)
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
    _procs["dashboard"] = proc
    _notify(f"Dashboard started at http://localhost:{port}", "START")
    return proc


def launch_sentiment_update(db: str = "DATA/market_data.db") -> None:
    """Update CBOE P/C + CFTC COT (called daily at 16:40 EST)."""
    try:
        result = subprocess.run(
            [sys.executable, "-c",
             f"import sys; sys.path.insert(0,'src'); "
             f"from sentiment_store import SentimentStore; "
             f"r=SentimentStore('{db}').update(); "
             f"print('Sentiment update:', r)"],
            capture_output=True, text=True, timeout=120
        )
        if result.stdout: log.info("sentiment: %s", result.stdout.strip())
    except Exception as e:
        log.warning("Sentiment update failed: %s", e)


def launch_fred_update(db: str = "DATA/market_data.db") -> None:
    """Update FRED macro series with last 30 days (called daily at 16:35 EST)."""
    try:
        log.info('Updating FRED macro data...')
        result = subprocess.run(
            [sys.executable, '-c',
             f"import sys; sys.path.insert(0,'src'); "
             f"from fred_store import FredDataStore; "
             f"s=FredDataStore('{db}'); r=s.update(); "
             f"print('FRED update:', sum(r.values()), 'observations')"],
            capture_output=True, text=True, timeout=120
        )
        if result.stdout: log.info('fred_update: %s', result.stdout.strip())
        if result.returncode != 0: log.warning('FRED update stderr: %s', result.stderr[:200])
    except Exception as e:
        log.warning('FRED update failed: %s', e)


def launch_store_update(ticker: str = "SPY", store_db: str = "DATA/market_data.db") -> None:
    """Update the SQLite market data store with yesterday's bars (called daily at 16:30 EST)."""
    try:
        log.info('▶ market_data_store: updating %s...', ticker)
        result = subprocess.run(
            [sys.executable, '-c',
             f"import sys; sys.path.insert(0,'{HERE}'); "
             f"from market_data_store import MarketDataStore; "
             f"s = MarketDataStore('{store_db}'); r = s.update('{ticker}'); "
             f"print('Store update:', r)"],
            capture_output=True, text=True, timeout=120
        )
        if result.stdout:
            log.info('store_update: %s', result.stdout.strip())
    except Exception as e:
        log.warning('Store update failed: %s', e)


def launch_markov_update() -> None:
    """Update the Markov engine transition matrix with fresh VIX data."""
    try:
        markov_script = HERE / "markov_engine.py"
        if not markov_script.exists():
            return
        log.info("▶ markov_engine: updating VIX transition matrix")
        result = subprocess.run(
            [sys.executable, "-c",
             f"import sys; sys.path.insert(0,'{HERE}'); "
             f"from markov_engine import RegimeMarkovChain; "
             f"rmc = RegimeMarkovChain(); "
             f"r = rmc.fit_from_yfinance(years=2); "
             f"rmc.save('markov_state.json'); "
             f"print('Markov updated:', r['n_obs'], 'days')"],
            capture_output=True, text=True, timeout=30
        )
        if result.stdout:
            log.info("markov_engine: %s", result.stdout.strip())
    except Exception as e:
        log.warning("Markov update failed: %s", e)


def launch_monte_carlo_update(db_path: str) -> None:
    """Compute and save Monte Carlo forecast to DB."""
    if not DEFAULT_CONFIG.get("run_monte_carlo", True):
        return
    try:
        log.info("▶ monte_carlo_extended: computing forecast")
        result = subprocess.run(
            [sys.executable, "-c",
             f"import sys; sys.path.insert(0,'{HERE}'); "
             f"from monte_carlo_extended import forecast_report; "
             f"import json, sqlite3; "
             f"r = forecast_report('{db_path}', current_equity={DEFAULT_CONFIG['account']}); "
             f"with sqlite3.connect('{db_path}') as c: "
             f"  c.execute('CREATE TABLE IF NOT EXISTS monte_carlo_forecasts "
             f"(id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, report_json TEXT)'); "
             f"  c.execute('INSERT INTO monte_carlo_forecasts (ts,report_json) VALUES (?,?)', "
             f"  (r[\"generated_at\"], json.dumps(r))); "
             f"print('MC:', r['headline'][:80])"],
            capture_output=True, text=True, timeout=60
        )
        if result.stdout:
            log.info("monte_carlo: %s", result.stdout.strip())
    except Exception as e:
        log.warning("Monte Carlo update failed: %s", e)


def launch_session_analyser(session_date: str = "") -> None:
    """Trigger the LLM session analyser (called at EOD by engine, or manually)."""
    script = HERE / "session_analyser.py"
    if not script.exists():
        log.warning("session_analyser.py not found")
        return
    date_str = session_date or str(date.today())
    cfg = DEFAULT_CONFIG
    cmd = [sys.executable, str(script),
           "--date",   date_str,
           "--db",     cfg["db_path"],
           "--preset", cfg.get("llm_preset", "nuc-pair1")]
    if cfg.get("discord_webhook"):
        cmd += ["--discord", cfg["discord_webhook"]]
    if cfg.get("telegram_token"):
        cmd += ["--telegram-token",   cfg["telegram_token"],
                "--telegram-chat-id", cfg["telegram_chat_id"]]
    log.info("▶ session_analyser.py --date %s", date_str)
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
    _procs["analyser"] = proc


def launch_tournament() -> None:
    """Run the weekly tournament + Design Studio gene mixing cycle."""
    if not DEFAULT_CONFIG.get("run_tournament", True):
        return
    script = HERE / "tournament_evaluator.py"
    if not script.exists():
        log.warning("tournament_evaluator.py not found")
        return
    cfg = DEFAULT_CONFIG
    cmd = [sys.executable, str(script)]
    if cfg.get("discord_webhook"):
        cmd += ["--discord", cfg["discord_webhook"]]
    log.info("▶ tournament_evaluator.py (weekly Design Studio cycle)")
    _notify("Weekly tournament + Design Studio gene-mixing cycle starting...", "INFO")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    log.info("tournament: %s", result.stdout[-500:] if result.stdout else "(no output)")
    _notify(f"Tournament complete. Check dashboard for results.", "INFO")

# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

def health_check(db_path: str) -> dict:
    """Check all running processes and DB heartbeat."""
    status: dict[str, Any] = {}

    # Check engine heartbeat from DB
    try:
        with sqlite3.connect(db_path, timeout=5) as conn:
            row = conn.execute(
                "SELECT status, ts, message FROM heartbeat WHERE id=1"
            ).fetchone()
        if row:
            status["engine_db_status"] = row[0]
            status["engine_last_tick"]  = row[1]
            # Stale if > 5 minutes old
            if row[1]:
                tick_dt = datetime.fromisoformat(row[1])
                age_min = (datetime.utcnow() - tick_dt).total_seconds() / 60
                status["engine_tick_age_min"] = round(age_min, 1)
                if age_min > 5 and _is_market_hours():
                    log.warning("Engine heartbeat stale (%d min)", age_min)
    except Exception:
        status["engine_db_status"] = "DB_ERROR"

    # Check subprocess liveness
    for name, proc in _procs.items():
        if proc is None:
            status[f"{name}_alive"] = False
        elif proc.poll() is None:
            status[f"{name}_alive"] = True
        else:
            status[f"{name}_alive"] = False
            log.warning("Process '%s' has died (code %s)", name, proc.returncode)
            # Restart engine if configured
            if name == "engine" and DEFAULT_CONFIG.get("restart_engine_on_crash"):
                log.warning("Restarting engine...")
                _notify("Engine crashed — attempting restart", "WARN")
                launch_engine()
            elif name == "dashboard":
                log.warning("Restarting dashboard...")
                launch_dashboard()

    return status


def print_status(db_path: str) -> None:
    """Print current system status to console."""
    status = health_check(db_path)
    print("\n" + "="*55)
    print("  TRADING SYSTEM STATUS")
    print("="*55)
    for k, v in status.items():
        icon = "✅" if v is True or v == "running" else "❌" if v is False else "—"
        print(f"  {icon}  {k}: {v}")
    print(f"\n  Time (EST): {_now_est().strftime('%H:%M')}")
    print(f"  Market hours: {'YES' if _is_market_hours() else 'NO'}")
    print(f"  Market day: {'YES' if _is_market_day() else 'NO'}")
    print("="*55 + "\n")

# ---------------------------------------------------------------------------
# APScheduler setup
# ---------------------------------------------------------------------------

def build_scheduler(db_path: str) -> "BackgroundScheduler | None":
    """Create and configure the APScheduler instance."""
    if not HAS_SCHEDULER:
        log.error("APScheduler not installed. Run: pip install apscheduler")
        return None

    scheduler = BackgroundScheduler(daemon=True)

    # Mon-Fri 09:00 → morning brief
    scheduler.add_job(
        launch_morning_brief,
        CronTrigger(day_of_week="mon-fri", hour=9, minute=0),
        id="morning_brief", replace_existing=True,
    )
    # Mon-Fri 09:25 → Markov engine update (before engine starts)
    scheduler.add_job(
        launch_markov_update,
        CronTrigger(day_of_week="mon-fri", hour=9, minute=25),
        id="markov_update", replace_existing=True,
    )
    # Mon-Fri 09:28 → Engine starts (waits for 09:30 internally)
    scheduler.add_job(
        launch_engine,
        CronTrigger(day_of_week="mon-fri", hour=9, minute=28),
        id="engine_start", replace_existing=True,
    )
    # Mon-Fri 09:29 → Dashboard starts
    scheduler.add_job(
        launch_dashboard,
        CronTrigger(day_of_week="mon-fri", hour=9, minute=29),
        id="dashboard_start", replace_existing=True,
    )
    # Mon-Fri 15:40 → Monte Carlo forecast update (after EOD close)
    scheduler.add_job(
        lambda: launch_monte_carlo_update(db_path),
        CronTrigger(day_of_week="mon-fri", hour=15, minute=40),
        id="monte_carlo_update", replace_existing=True,
    )
    # Mon-Fri 16:30 → Market data store update (after all bars settled)
    scheduler.add_job(
        lambda: launch_store_update(DEFAULT_CONFIG['ticker']),
        CronTrigger(day_of_week="mon-fri", hour=16, minute=30),
        id="store_update", replace_existing=True,
    )
    # Mon-Fri 16:35 → FRED macro data update
    scheduler.add_job(
        lambda: launch_fred_update(),
        CronTrigger(day_of_week="mon-fri", hour=16, minute=35),
        id="fred_update", replace_existing=True,
    )
    # Mon-Fri 16:40 → CBOE P/C scrape (today's sentiment)
    scheduler.add_job(
        lambda: launch_sentiment_update(),
        CronTrigger(day_of_week="mon-fri", hour=16, minute=40),
        id="sentiment_update", replace_existing=True,
    )
    # Saturday 08:00 → Weekly tournament + Design Studio
    scheduler.add_job(
        launch_tournament,
        CronTrigger(day_of_week="sat", hour=8, minute=0),
        id="tournament", replace_existing=True,
    )
    # Every 60 seconds → health check
    scheduler.add_job(
        lambda: health_check(db_path),
        IntervalTrigger(seconds=60),
        id="health_check", replace_existing=True,
    )
    return scheduler

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Trading system orchestrator")
    p.add_argument("--ticker",     default=DEFAULT_CONFIG["ticker"])
    p.add_argument("--orb",        default=DEFAULT_CONFIG["orb_method"])
    p.add_argument("--account",    type=float, default=DEFAULT_CONFIG["account"])
    p.add_argument("--db",         default=DEFAULT_CONFIG["db_path"])
    p.add_argument("--preset",     default=DEFAULT_CONFIG["llm_preset"])
    p.add_argument("--discord",    default=DEFAULT_CONFIG["discord_webhook"])
    p.add_argument("--tg-token",   default=DEFAULT_CONFIG["telegram_token"])
    p.add_argument("--tg-chat",    default=DEFAULT_CONFIG["telegram_chat_id"])
    p.add_argument("--no-dashboard", action="store_true")
    p.add_argument("--status",     action="store_true", help="Print status and exit")
    p.add_argument("--weekend",    action="store_true", help="Run tournament now")
    p.add_argument("--morning",    action="store_true", help="Run morning brief now")
    p.add_argument("--date",       default="", help="Session date for manual analyser")
    p.add_argument("--analyse",    action="store_true", help="Run session analyser now")
    p.add_argument("--monte-carlo", action="store_true", help="Run MC forecast now")
    args = p.parse_args()

    # Merge CLI into config
    DEFAULT_CONFIG.update({
        "ticker":         args.ticker.upper(),
        "orb_method":     args.orb,
        "account":        args.account,
        "db_path":        args.db,
        "llm_preset":     args.preset,
        "discord_webhook": args.discord,
        "telegram_token": args.tg_token,
        "telegram_chat_id": args.tg_chat,
        "run_dashboard":  not args.no_dashboard,
    })

    logging.basicConfig(
        level=getattr(logging, DEFAULT_CONFIG["log_level"]),
        format="%(asctime)s  %(name)-12s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )
    log.info("="*55)
    log.info("  TRADING SYSTEM RUNNER")
    log.info("  Ticker: %s  ORB: %s  Account: £%,.0f",
             DEFAULT_CONFIG["ticker"], DEFAULT_CONFIG["orb_method"],
             DEFAULT_CONFIG["account"])
    log.info("="*55)

    # ── One-shot commands ───────────────────────────────────────────────────
    if args.status:
        print_status(args.db); return
    if args.weekend:
        launch_tournament(); return
    if args.morning:
        launch_morning_brief(); return
    if args.analyse:
        launch_session_analyser(args.date); time.sleep(5); return
    if getattr(args, "monte_carlo", False):
        launch_monte_carlo_update(args.db); return

    # ── Full scheduled mode ─────────────────────────────────────────────────
    if not HAS_SCHEDULER:
        log.error("APScheduler not installed. Run: pip install apscheduler")
        sys.exit(1)

    scheduler = build_scheduler(args.db)
    scheduler.start()
    log.info("Scheduler started. Waiting for scheduled jobs...")
    log.info("Jobs scheduled:")
    for job in scheduler.get_jobs():
        log.info("  %-20s  %s", job.id, job.next_run_time)

    # If currently market hours on startup, launch immediately
    if _is_market_hours():
        log.info("Market hours detected on startup — launching engine + dashboard now")
        launch_markov_update()
        launch_engine()
        launch_dashboard()
        _notify(f"Runner started during market hours. Engine active for "
                f"{DEFAULT_CONFIG['ticker']}.", "START")
    else:
        _notify(f"Runner started. Next session: {DEFAULT_CONFIG['ticker']} "
                f"{DEFAULT_CONFIG['orb_method']} at 09:28 EST.", "START")

    # Keep alive — log heartbeat every 5 minutes
    try:
        while True:
            time.sleep(300)
            now = _now_est()
            status = health_check(args.db)
            alive = [k for k, v in status.items() if k.endswith("_alive") and v]
            log.info("Heartbeat %s | alive: %s",
                     now.strftime("%H:%M EST"), alive if alive else "idle")
    except (KeyboardInterrupt, SystemExit):
        log.info("Shutting down...")
        scheduler.shutdown(wait=False)
        for name, proc in _procs.items():
            if proc and proc.poll() is None:
                log.info("Terminating %s...", name)
                proc.terminate()
        _notify("Runner shut down cleanly.", "STOP")
        log.info("All processes terminated.")


if __name__ == "__main__":
    main()
