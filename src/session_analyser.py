"""
session_analyser.py
===================
Autonomous Staged-Dossier LLM Post-Session Analysis for the Trading Income Project.
Eliminates multi-turn ReAct loops in favor of deterministic pre-compilation,
anchored macro filtering, single-pass frontier reasoning, and adversarial audit.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sqlite3
import sys
import time
import urllib.request
from datetime import date, datetime, timezone
UTC = timezone.utc
from typing import Any

from openai import OpenAI

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

log = logging.getLogger("analyser")

CONFIG: dict[str, Any] = {
    "llm_base_url":       os.environ.get("LLM_BASE_URL", "http://127.0.0.1:8000/v1"),
    "llm_model":          os.environ.get("LLM_MODEL_PRIMARY", "qwen3.8:27b"),
    "llm_max_tokens":     1536,
    "llm_temperature":    0.1,
    "timeout_seconds":    600,

    "scout_base_url":     os.environ.get("DAC_BASE_URL", "http://127.0.0.1:8001/v1"),
    "scout_model":        os.environ.get("OLLAMA_MODEL_ADVERSARIAL", "phi-4-mini:int4"),

    "dac_base_url":       os.environ.get("DAC_BASE_URL", "http://127.0.0.1:8001/v1"),
    "dac_model":          os.environ.get("OLLAMA_MODEL_ADVERSARIAL", "phi-4-mini:int4"),

    "db_path":            "DATA/paper_account.db",
    "discord_webhook":    os.environ.get("DISCORD_WEBHOOK_URL", ""),
}

# ---------------------------------------------------------------------------
# Database & Ingestion Layer (Deterministic — < 0.05s)
# ---------------------------------------------------------------------------

def _ensure_dirs(db_path: str) -> None:
    dirname = os.path.dirname(db_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

def _conn(db_path: str) -> sqlite3.Connection:
    _ensure_dirs(db_path)
    c = sqlite3.connect(db_path, timeout=10)
    c.row_factory = sqlite3.Row
    return c

def _q(db_path: str, sql: str, params: tuple = ()) -> list[dict]:
    try:
        with _conn(db_path) as c:
            return [dict(r) for r in c.execute(sql, params).fetchall()]
    except Exception as e:
        log.warning("DB query failed: %s", e)
        return []

def _q1(db_path: str, sql: str, params: tuple = ()) -> dict | None:
    rows = _q(db_path, sql, params)
    return rows[0] if rows else None

def compile_session_dossier(session_date: str, db_path: str) -> dict[str, Any]:
    sess = _q1(db_path, "SELECT * FROM sessions WHERE session_date=?", (session_date,))
    trades = _q(db_path, "SELECT * FROM positions WHERE date(opened_at)=? ORDER BY id", (session_date,))
    decs = _q(db_path, "SELECT * FROM decisions WHERE session_date=? ORDER BY id", (session_date,))
    sl = _q1(db_path, "SELECT * FROM session_learning WHERE session_date=? ORDER BY id DESC LIMIT 1", (session_date,))
    wfa = _q(db_path, "SELECT * FROM wfa_results ORDER BY id DESC LIMIT 6")

    all_closed = _q(db_path, "SELECT actual_r FROM positions WHERE status='closed' AND actual_r IS NOT NULL")
    n_hist = len(all_closed)
    wins = sum(1 for r in all_closed if (r.get("actual_r") or 0) > 0)
    
    alpha_p = 1 + wins
    beta_p = 1 + (n_hist - wins)
    est_wr = round(alpha_p / (alpha_p + beta_p), 3)

    return {
        "session_date": session_date,
        "n_session_trades": len(trades),
        "session_pnl_r": sum(t.get("actual_r", 0) for t in trades),
        "trades": trades,
        "total_decisions": len(decs),
        "decision_sample": decs[:4],
        "session_learning": sl or {},
        "wfa_splits": wfa,
        "historical_trades_count": n_hist,
        "bayesian_win_rate": est_wr,
        "rule_17_sample_valid": (n_hist >= 20),
    }

# ---------------------------------------------------------------------------
# Anchored Macro Scout (Port 8001 · Phi-4 / Llama — ~2.0s)
# ---------------------------------------------------------------------------

SCOUT_PROMPT = """You are an economic news extractor.
Extract the key macroeconomic catalysts and market reactions from the text.

RULES:
1. Copy economic figures, percentages, and CPI/Fed numbers VERBATIM.
2. Ignore all ads, editorial opinions, trading advice, and clickbait.
3. Output EXACTLY 2 or 3 concise bullet points.
4. Maximum 100 words.
"""

def fetch_and_distill_macro(session_date: str, cfg: dict) -> str:
    """Extracts raw web news (Brave Search primary) and distills it via the small scout model."""
    try:
        from tool_runner import tool_web_search
        query = f"S&P 500 SPY stock market close {session_date}"
        raw_text = tool_web_search(query, n_results=3)
    except Exception as e:
        log.warning("Web search unavailable: %s", e)
        return "Macro search unavailable — relying strictly on local database."

    if not raw_text or "No results" in raw_text:
        return "No specific macroeconomic catalysts reported."

    # Extract verified numbers deterministically to prevent hallucination
    num_pattern = r"(?:[\+\-]?\d+(?:\.\d+)?%|\b\d+\s*bps\b|\$[\d,]+(?:\.\d+)?)"
    anchors = list(dict.fromkeys(re.findall(num_pattern, raw_text)))[:6]

    try:
        client = OpenAI(base_url=cfg["scout_base_url"], api_key="unused", timeout=30.0)
        resp = client.chat.completions.create(
            model=cfg["scout_model"],
            messages=[
                {"role": "system", "content": SCOUT_PROMPT},
                {"role": "user", "content": f"Verified Numbers: {anchors}\n\nRaw Context:\n{raw_text[:3000]}"}
            ],
            max_tokens=150,
            temperature=0.0,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        log.warning("Scout model distillation failed (%s) — using raw snippet", e)
        return raw_text[:300] + "..."

# ---------------------------------------------------------------------------
# Deep Quant Reasoning (Port 8000 · Single Pass — ~30s)
# ---------------------------------------------------------------------------

QUANT_SYSTEM_PROMPT = """You are the Senior Quantitative Strategist for the Trading Income Project.
You are evaluating a completed trading session dossier.

MANDATORY DIRECTIVES:
1. You NEVER suggest parameter changes automatically. All recommendations are proposals for human review.
2. If total historical trades < 20, you MUST explicitly flag findings as PREMATURE due to small sample size.
3. Formulate your findings into these exact Markdown sections:
   ## SESSION SUMMARY
   ## WHAT WORKED
   ## WHAT DIDN'T WORK
   ## EXECUTION & FILL QUALITY
   ## WFA REGIME HEALTH
   ## ACTION ITEMS FOR HUMAN REVIEW (Numbered, specific, with falsification criteria)
   ## CONFIDENCE TIER (HIGH / MEDIUM / LOW)
"""

def run_quant_analysis(dossier: dict, macro_summary: str, cfg: dict) -> tuple[str, dict]:
    client = OpenAI(base_url=cfg["llm_base_url"], api_key="unused", timeout=cfg["timeout_seconds"])

    prompt = f"""<think>
</think>Analyze the following pre-compiled trading session dossier:

### MACROECONOMIC CONTEXT
{macro_summary}

### EMPIRICAL SESSION DOSSIER
{json.dumps(dossier, indent=2, default=str)}

Synthesize the quantitative performance, identify execution drag, check WFA regime validity, and provide structured action items.
"""
    t0 = time.monotonic()
    log.info("Sending dossier to Quant Strategist (%s @ %s)...", cfg["llm_model"], cfg["llm_base_url"])
    resp = client.chat.completions.create(
        model=cfg["llm_model"],
        messages=[
            {"role": "system", "content": QUANT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        max_tokens=cfg["llm_max_tokens"],
        temperature=cfg["llm_temperature"],
    )
    elapsed = max(time.monotonic() - t0, 0.001)
    usage = getattr(resp, "usage", None)
    pt = getattr(usage, "prompt_tokens", 0)
    ct = getattr(usage, "completion_tokens", 0)
    tps = getattr(usage, "tokens_per_second", round(ct / elapsed, 1))

    log.info("Quant analysis complete: %d prompt tok | %d comp tok in %.1fs -> %.1f tok/s", pt, ct, elapsed, tps)
    return resp.choices[0].message.content.strip(), {"prompt_tokens": pt, "completion_tokens": ct, "duration": elapsed, "tps": tps}

# ---------------------------------------------------------------------------
# Adversarial D-A-C Auditor (Port 8001 · Single Pass — ~15s)
# ---------------------------------------------------------------------------

DAC_PROMPT = """You are the Adversarial Validator. Stress-test the following analysis.
D-A-C PROTOCOL:
Phase D — Divergent: 3 alternative structural or statistical explanations.
Phase A — Adversarial: Challenge each action item. (MANDATORY: If sample n < 20, reject parameter changes).
Phase C — Convergent: Classify recommendations into GROUNDED, PROVISIONAL, or REJECTED.
"""

def run_adversarial_audit(quant_report: str, dossier: dict, cfg: dict) -> str:
    client = OpenAI(base_url=cfg["dac_base_url"], api_key="unused", timeout=cfg["timeout_seconds"])
    user_msg = f"""=== INITIAL QUANT REPORT ===
{quant_report}

=== SAMPLE SIZE CONTEXT ===
Historical Trades: {dossier['historical_trades_count']}
Bayesian Win Rate: {dossier['bayesian_win_rate']}
Rule 17 Active: {not dossier['rule_17_sample_valid']}

Execute Phase D, Phase A, and Phase C audit.
"""
    t0 = time.monotonic()
    log.info("Sending to Adversarial Auditor (%s @ %s)...", cfg["dac_model"], cfg["dac_base_url"])
    resp = client.chat.completions.create(
        model=cfg["dac_model"],
        messages=[
            {"role": "system", "content": DAC_PROMPT},
            {"role": "user", "content": user_msg}
        ],
        max_tokens=1024,
        temperature=0.1,
    )
    elapsed = max(time.monotonic() - t0, 0.001)
    log.info("Adversarial audit complete in %.1fs", elapsed)
    return resp.choices[0].message.content.strip()

# ---------------------------------------------------------------------------
# Notifications & Persistence
# ---------------------------------------------------------------------------

def send_discord(report: str, session_date: str, webhook: str) -> None:
    if not webhook:
        return
    try:
        chunks = []
        body = f"📊 **Session Analysis — {session_date}**\n\n" + report
        while body:
            chunks.append(body[:1900])
            body = body[1900:]
        for chunk in chunks:
            data = json.dumps({"content": chunk}).encode()
            req = urllib.request.Request(webhook, data=data, headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=10)
        log.info("Sent analysis report to Discord (%d chunks)", len(chunks))
    except Exception as e:
        log.warning("Discord send failed: %s", e)

def save_analysis(db_path: str, session_date: str, full_report: str, model_used: str) -> None:
    _ensure_dirs(db_path)
    with _conn(db_path) as c:
        c.execute("""CREATE TABLE IF NOT EXISTS llm_analysis (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_date TEXT NOT NULL,
            ts TEXT NOT NULL,
            report TEXT NOT NULL,
            model_used TEXT
        )""")
        c.execute("INSERT INTO llm_analysis (session_date, ts, report, model_used) VALUES (?,?,?,?)",
                  (session_date, datetime.now(UTC).isoformat(), full_report, model_used))

def main(session_date: str, cfg: dict) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")

    log.info("Starting Staged-Dossier Analysis for %s", session_date)
    dossier = compile_session_dossier(session_date, cfg["db_path"])
    macro_summary = fetch_and_distill_macro(session_date, cfg)
    quant_report, stats = run_quant_analysis(dossier, macro_summary, cfg)
    audit_report = run_adversarial_audit(quant_report, dossier, cfg)

    full_report = f"{quant_report}\n\n{'═'*60}\n## D-A-C ADVERSARIAL STRESS TEST\n{'═'*60}\n\n{audit_report}"
    save_analysis(cfg["db_path"], session_date, full_report, cfg["llm_model"])
    
    if cfg.get("discord_webhook"):
        send_discord(full_report, session_date, cfg["discord_webhook"])

    print("\n" + "="*60 + f"\n  SESSION ANALYSIS DOSSIER — {session_date}\n" + "="*60)
    print(full_report)
    print("="*60 + "\n")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Staged-Dossier Session Analyser")
    p.add_argument("--date", default=str(date.today()))
    p.add_argument("--db", default=CONFIG["db_path"])
    p.add_argument("--preset", default="nuc-pair1")
    p.add_argument("--discord", default=CONFIG["discord_webhook"])
    args, _ = p.parse_known_args()

    cfg = {**CONFIG, "db_path": args.db, "discord_webhook": args.discord}
    main(args.date, cfg)
