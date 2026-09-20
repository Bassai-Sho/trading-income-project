"""
src/domain_telemetry.py
=======================
Adaptive Scraping Policy Ledger.
Tracks domain health, 403s, paywalls, and auto-quarantines dead sources.
Stores state in DATA/scraping_telemetry.db (WAL mode — safe alongside historical_sim.py).
"""

import sqlite3
import time
from urllib.parse import urlparse
from pathlib import Path

DB_PATH = Path("DATA/scraping_telemetry.db")

def init_telemetry_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("""
        CREATE TABLE IF NOT EXISTS domain_health (
            domain                TEXT PRIMARY KEY,
            success_count         INTEGER DEFAULT 0,
            fail_count            INTEGER DEFAULT 0,
            consecutive_failures  INTEGER DEFAULT 0,
            last_error_code       TEXT,
            preferred_strategy    TEXT DEFAULT 'HTTPX_TRAFILATURA',
            avg_latency_ms        INTEGER DEFAULT 0,
            quarantined_until     REAL    DEFAULT 0.0,
            updated_at            REAL    DEFAULT 0.0
        );
        """)
        conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_domain_strategy
            ON domain_health(domain, preferred_strategy);
        """)

init_telemetry_db()


def get_domain_strategy(url: str) -> str:
    """Returns 'HTTPX_TRAFILATURA', 'PLAYWRIGHT_JS', or 'SKIP'."""
    domain = urlparse(url).netloc.lower().replace("www.", "")
    now = time.time()
    try:
        with sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=0.5) as conn:
            row = conn.execute(
                "SELECT preferred_strategy, quarantined_until FROM domain_health WHERE domain = ?",
                (domain,)
            ).fetchone()
            if not row:
                return "HTTPX_TRAFILATURA"
            strategy, quarantined_until = row
            # Quarantine expired — allow one speculative re-attempt
            if strategy == "SKIP" and quarantined_until and now > quarantined_until:
                return "HTTPX_TRAFILATURA"
            return strategy
    except Exception:
        return "HTTPX_TRAFILATURA"  # Fail open — never block on telemetry error


def record_domain_result(url: str, success: bool,
                          error_type: str = None, latency_ms: int = 0):
    """Updates domain health and adapts future scraping strategy autonomously."""
    domain = urlparse(url).netloc.lower().replace("www.", "")
    now = time.time()
    try:
        with sqlite3.connect(DB_PATH, timeout=2.0) as conn:
            row = conn.execute(
                "SELECT success_count, fail_count, consecutive_failures, "
                "preferred_strategy, avg_latency_ms FROM domain_health WHERE domain = ?",
                (domain,)
            ).fetchone()

            if not row:
                succ, fail, consec, strat, avg_lat = 0, 0, 0, "HTTPX_TRAFILATURA", latency_ms
            else:
                succ, fail, consec, strat, avg_lat = row
                avg_lat = int((avg_lat + latency_ms) / 2) if avg_lat else latency_ms

            if success:
                succ   += 1
                consec  = 0
                strat   = "HTTPX_TRAFILATURA" if strat == "SKIP" else strat
                quarantine = 0.0
                last_err   = None
            else:
                fail   += 1
                consec += 1
                last_err = error_type

                if error_type in ("403_BOT", "CAPTCHA", "PAYWALL",
                                  "HTTP_401", "HTTP_403") and consec >= 2:
                    strat      = "SKIP"
                    quarantine = now + 3600 * 24 * 3   # 3-day quarantine
                elif error_type == "EMPTY_CONTENT" and consec >= 2:
                    strat      = "PLAYWRIGHT_JS"        # Escalate to JS rendering
                    quarantine = 0.0
                elif consec >= 4:
                    strat      = "SKIP"
                    quarantine = now + 3600 * 24 * 7   # 7-day quarantine
                else:
                    quarantine = 0.0

            conn.execute("""
            INSERT INTO domain_health
                (domain, success_count, fail_count, consecutive_failures,
                 preferred_strategy, avg_latency_ms, last_error_code,
                 quarantined_until, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(domain) DO UPDATE SET
                success_count        = excluded.success_count,
                fail_count           = excluded.fail_count,
                consecutive_failures = excluded.consecutive_failures,
                preferred_strategy   = excluded.preferred_strategy,
                avg_latency_ms       = excluded.avg_latency_ms,
                last_error_code      = excluded.last_error_code,
                quarantined_until    = excluded.quarantined_until,
                updated_at           = excluded.updated_at;
            """, (domain, succ, fail, consec, strat, avg_lat, last_err, quarantine, now))
    except Exception as e:
        print(f"[Telemetry DB Error] {e}")


def telemetry_report() -> str:
    """Quick summary of domain health for /status or debugging."""
    try:
        with sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=0.5) as conn:
            rows = conn.execute(
                "SELECT domain, preferred_strategy, success_count, fail_count, "
                "consecutive_failures, last_error_code "
                "FROM domain_health ORDER BY fail_count DESC LIMIT 20"
            ).fetchall()
            if not rows:
                return "No domain telemetry recorded yet."
            lines = ["Domain Health Report:", "─" * 60]
            for d, strat, s, f, consec, err in rows:
                flag = "🔴" if strat == "SKIP" else "🟡" if strat == "PLAYWRIGHT_JS" else "🟢"
                lines.append(f"{flag} {d:<35} {strat:<20} ✓{s} ✗{f} [{err or '—'}]")
            return "\n".join(lines)
    except Exception as e:
        return f"Telemetry read error: {e}"
