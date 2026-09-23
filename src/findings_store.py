"""
findings_store.py
=================
Structured persistence layer for D-A-C session findings.

WHY THIS EXISTS
───────────────
The session_analyser.py produces free-text analysis that humans read.
This module extracts a structured JSON summary from that text (via a
second Ollama call) and stores it in a `dac_findings` table.

When the same (area, observation) pair appears in 3+ sessions within
20 trading days, it becomes a "confirmed pattern" that morning_brief.py
surfaces before the GO/NO-GO assessment.

This is the self-reinforcing loop:
  D-A-C analysis → structured findings → pattern accumulation
  → morning brief surfaces patterns → human-informed decision

WHAT IS NOT AUTOMATED
──────────────────────
Parameter changes, rule updates, and Notion entries remain manual.
The system surfaces confirmed patterns; the human decides what to do.

EXTRACTION SCHEMA
──────────────────
{
  "findings": [
    {
      "category":    "GROUNDED | PROVISIONAL | REJECTED | MONITOR",
      "area":        "signal | risk | regime | data | psychology",
      "observation": "one sentence, specific and falsifiable",
      "implication": "what to watch for or consider",
      "confidence":  1-5
    }
  ],
  "overall_verdict":    "PROCEED | CAUTION | HALT",
  "top_concern":        "one sentence",
  "tool_calls_made":    12,
  "analysis_quality":   "HIGH | MEDIUM | LOW"
}
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any

log = logging.getLogger("findings_store")

# A finding must appear in this many sessions within PATTERN_WINDOW days
PATTERN_THRESHOLD = 3
PATTERN_WINDOW    = 20   # trading days

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

FINDINGS_SCHEMA = """
CREATE TABLE IF NOT EXISTS dac_findings (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_date    TEXT    NOT NULL,
    extracted_at    TEXT    NOT NULL,
    category        TEXT    NOT NULL,   -- GROUNDED / PROVISIONAL / REJECTED / MONITOR
    area            TEXT    NOT NULL,   -- signal / risk / regime / data / psychology
    observation     TEXT    NOT NULL,   -- the finding text
    obs_hash        TEXT    NOT NULL,   -- hash(area + observation[:80]) for dedup
    implication     TEXT,
    confidence      INTEGER,            -- 1-5
    overall_verdict TEXT,               -- PROCEED / CAUTION / HALT
    top_concern     TEXT,
    tool_calls_made INTEGER,
    analysis_quality TEXT,
    raw_json        TEXT                -- full JSON for audit
);
CREATE INDEX IF NOT EXISTS idx_findings_date ON dac_findings(session_date);
CREATE INDEX IF NOT EXISTS idx_findings_hash ON dac_findings(obs_hash);

CREATE TABLE IF NOT EXISTS confirmed_patterns (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    obs_hash        TEXT    UNIQUE NOT NULL,
    area            TEXT    NOT NULL,
    observation     TEXT    NOT NULL,
    implication     TEXT,
    first_seen      TEXT,
    last_seen       TEXT,
    occurrence_count INTEGER DEFAULT 1,
    status          TEXT DEFAULT 'active',  -- active / dismissed
    dismissed_at    TEXT,
    dismiss_reason  TEXT
);
CREATE INDEX IF NOT EXISTS idx_patterns_hash ON confirmed_patterns(obs_hash);
"""

# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _conn(db_path: str) -> sqlite3.Connection:
    c = sqlite3.connect(db_path, timeout=10)
    c.row_factory = sqlite3.Row
    return c

def ensure_schema(db_path: str) -> None:
    with _conn(db_path) as c:
        c.executescript(FINDINGS_SCHEMA)

def _obs_hash(area: str, observation: str) -> str:
    key = f"{area.lower()}:{observation[:80].lower()}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]

# ---------------------------------------------------------------------------
# Store findings from one D-A-C session
# ---------------------------------------------------------------------------

def store_findings(
    db_path:         str,
    session_date:    str,
    findings_json:   dict,
    tool_calls_made: int = 0,
) -> int:
    """
    Parse a findings JSON dict and persist each finding.
    Returns the number of individual findings stored.
    """
    ensure_schema(db_path)
    findings = findings_json.get("findings", [])
    overall  = findings_json.get("overall_verdict", "")
    concern  = findings_json.get("top_concern", "")
    quality  = findings_json.get("analysis_quality", "")
    now_ts   = datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds")
    stored   = 0

    with _conn(db_path) as c:
        for f in findings:
            category = str(f.get("category", "MONITOR")).upper()
            area     = str(f.get("area", "signal")).lower()
            obs      = str(f.get("observation", "")).strip()
            impl     = str(f.get("implication", "")).strip()
            conf     = int(f.get("confidence", 3))
            if not obs:
                continue
            oh = _obs_hash(area, obs)
            c.execute(
                """INSERT INTO dac_findings
                   (session_date, extracted_at, category, area, observation,
                    obs_hash, implication, confidence, overall_verdict,
                    top_concern, tool_calls_made, analysis_quality, raw_json)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (session_date, now_ts, category, area, obs, oh,
                 impl, conf, overall, concern,
                 tool_calls_made, quality,
                 json.dumps(findings_json))
            )
            stored += 1

    # Update confirmed patterns
    _update_patterns(db_path, session_date)
    log.info("Stored %d findings for %s", stored, session_date)
    return stored


def _update_patterns(db_path: str, session_date: str) -> None:
    """
    After storing findings, check which observation hashes have appeared
    PATTERN_THRESHOLD times in the last PATTERN_WINDOW trading days.
    Upsert into confirmed_patterns.
    """
    cutoff = (date.fromisoformat(session_date) - timedelta(days=PATTERN_WINDOW * 1.5)).isoformat()

    with _conn(db_path) as c:
        # Count occurrences per obs_hash in the window, GROUNDED/PROVISIONAL only
        rows = c.execute(
            """SELECT obs_hash, area, observation, implication,
                      COUNT(*) as n, MIN(session_date) as first, MAX(session_date) as last
               FROM dac_findings
               WHERE session_date >= ?
                 AND category IN ('GROUNDED','PROVISIONAL')
               GROUP BY obs_hash
               HAVING n >= ?""",
            (cutoff, PATTERN_THRESHOLD)
        ).fetchall()

        for row in rows:
            existing = c.execute(
                "SELECT id, occurrence_count FROM confirmed_patterns WHERE obs_hash=?",
                (row["obs_hash"],)
            ).fetchone()
            if existing:
                c.execute(
                    "UPDATE confirmed_patterns SET occurrence_count=?, last_seen=? WHERE obs_hash=?",
                    (row["n"], row["last"], row["obs_hash"])
                )
            else:
                c.execute(
                    """INSERT INTO confirmed_patterns
                       (obs_hash, area, observation, implication, first_seen, last_seen, occurrence_count)
                       VALUES (?,?,?,?,?,?,?)""",
                    (row["obs_hash"], row["area"], row["observation"],
                     row["implication"], row["first"], row["last"], row["n"])
                )


# ---------------------------------------------------------------------------
# Query confirmed patterns (for morning_brief.py)
# ---------------------------------------------------------------------------

def get_confirmed_patterns(db_path: str) -> list[dict]:
    """
    Return all active confirmed patterns (appeared PATTERN_THRESHOLD+ times).
    Called by morning_brief.py.
    """
    try:
        ensure_schema(db_path)
        with _conn(db_path) as c:
            rows = c.execute(
                """SELECT area, observation, implication, occurrence_count, last_seen
                   FROM confirmed_patterns
                   WHERE status='active'
                   ORDER BY occurrence_count DESC, last_seen DESC"""
            ).fetchall()
        return [dict(r) for r in rows]
    except Exception as e:
        log.warning("Could not load confirmed patterns: %s", e)
        return []


def dismiss_pattern(db_path: str, obs_hash: str, reason: str) -> bool:
    """Human dismisses a false-positive pattern."""
    try:
        with _conn(db_path) as c:
            c.execute(
                "UPDATE confirmed_patterns SET status='dismissed', "
                "dismissed_at=?, dismiss_reason=? WHERE obs_hash=?",
                (datetime.now(timezone.utc).replace(tzinfo=None).isoformat(), reason, obs_hash)
            )
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Extraction prompt and Ollama call
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT = """\
You just completed a D-A-C trading session analysis. Based on your analysis above,
extract a structured JSON findings object. Be specific and conservative.

Rules:
- GROUNDED: finding is supported by data you retrieved from tools
- PROVISIONAL: plausible but needs more sessions to confirm
- REJECTED: challenged and failed adversarial scrutiny
- MONITOR: worth watching but insufficient evidence yet
- confidence 1=very low, 5=very high
- Keep observation to one sentence, falsifiable and specific
- Limit to the 5 most important findings

Respond ONLY with valid JSON matching this schema exactly:
{
  "findings": [
    {
      "category": "GROUNDED",
      "area": "signal",
      "observation": "...",
      "implication": "...",
      "confidence": 4
    }
  ],
  "overall_verdict": "PROCEED",
  "top_concern": "...",
  "analysis_quality": "HIGH"
}

Areas: signal, risk, regime, data, psychology
Verdicts: PROCEED, CAUTION, HALT
Quality: HIGH (5+ tool calls, full analysis), MEDIUM (2-4 calls), LOW (0-1 calls)\
"""

def extract_structured_findings(
    free_text_report: str,
    tool_calls_made:  int,
    cfg:              dict,
) -> dict | None:
    """
    Run a second, short Ollama call to extract structured findings from
    the free-text D-A-C report.

    Returns parsed JSON dict or None if extraction fails.
    Only runs if tool_calls_made >= 2 and report is >= 300 chars
    (ensures we don't extract findings from a failed/empty analysis).
    """
    if tool_calls_made < 2 or len(free_text_report) < 300:
        log.info("Skipping findings extraction: too few tool calls or short report")
        return None

    try:
        from openai import OpenAI
    except ImportError:
        return None

    client = OpenAI(
        base_url=cfg.get("llm_base_url", "http://localhost:11434/v1"),
        api_key=cfg.get("llm_api_key",  "ollama"),
    )

    # Use the faster model for extraction (it's a simple structured task)
    extraction_model = cfg.get("llm_model_fallback", cfg.get("llm_model", "qwen3:8b"))

    try:
        resp = client.chat.completions.create(
            model=extraction_model,
            messages=[
                {"role": "user",      "content": f"Here is a D-A-C trading analysis:\n\n{free_text_report[:6000]}"},
                {"role": "assistant", "content": "I will extract the structured findings."},
                {"role": "user",      "content": EXTRACTION_PROMPT},
            ],
            response_format={"type": "json_object"},
            max_tokens=800,
            temperature=0.05,   # near-deterministic for structured extraction
            timeout=120,
        )
        raw = resp.choices[0].message.content or ""
        parsed = json.loads(raw)
        log.info("Structured extraction: %d findings, verdict=%s",
                 len(parsed.get("findings", [])),
                 parsed.get("overall_verdict", "?"))
        return parsed
    except Exception as e:
        log.warning("Structured findings extraction failed: %s", e)
        return None
