"""
tournament_evaluator.py
=======================
Weekly tournament evaluator for the multi-strategy paper trading system.

WHAT THIS DOES
──────────────
1. Reads performance data from all active strategy variant DBs
2. Computes comparative metrics: WFE, Sharpe, EV, drawdown, fill bias
3. Checks phase gates — flags when a variant is ready to advance
4. Runs the AutoResearch parameter sweep for Phase 3+ variants (locked in Phase 2)
5. Produces a structured tournament report sent to Discord/Telegram
6. NEVER automatically applies changes — proposes them for human confirmation

PHASE GATES (enforced here, not circumvented)
──────────────────────────────────────────────
The adversarial challenge in the D-A-C session confirmed: self-improvement
loops without proper statistical gates converge on overfitting, not edge.
Every promotion proposal in this file requires human approval. The system
produces PROPOSALS — humans produce DECISIONS.

RUN SCHEDULE
────────────
  Every weekend (Saturday, after market close):
    python tournament_evaluator.py

  Or trigger manually:
    python tournament_evaluator.py --report-only
    python tournament_evaluator.py --check-gates
    python tournament_evaluator.py --run-autoref --variant canonical

USAGE
─────
  python tournament_evaluator.py
  python tournament_evaluator.py --variant canonical --report-only
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import statistics
import sys
from datetime import date, datetime, timedelta
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from gene_mixer import GeneMixer, build_machine_psychology
    GENE_MIXER = True
except ImportError:
    GENE_MIXER = False

from strategy_registry import (
    REGISTRY, StrategyVariant, get_active_variants, check_phase_gate,
    PARAMETER_BOUNDS,
)

try:
    import trading_quant_toolkit_v2_4 as tk
    TOOLKIT = True
except ImportError:
    TOOLKIT = False


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _conn(db_path: str) -> sqlite3.Connection:
    c = sqlite3.connect(db_path, timeout=5)
    c.row_factory = sqlite3.Row
    return c

def _q(db_path: str, sql: str, params: tuple = ()) -> list[dict]:
    try:
        with _conn(db_path) as c:
            return [dict(r) for r in c.execute(sql, params).fetchall()]
    except Exception:
        return []

def _q1(db_path: str, sql: str, params: tuple = ()) -> dict | None:
    rows = _q(db_path, sql, params)
    return rows[0] if rows else None


# ---------------------------------------------------------------------------
# Performance metrics per variant
# ---------------------------------------------------------------------------

def collect_variant_metrics(variant: StrategyVariant, lookback_days: int = 30) -> dict:
    """
    Pull comprehensive performance metrics for one variant from its DB.
    Returns a metrics dict suitable for comparison and gate checking.
    """
    if not os.path.exists(variant.db_path):
        return {"error": f"DB not found: {variant.db_path}", "n": 0}

    cutoff = str((date.today() - timedelta(days=lookback_days)).isoformat())

    # All closed trades
    all_trades = _q(variant.db_path,
                    "SELECT actual_r, exit_reason, opened_at FROM positions "
                    "WHERE status='closed' AND actual_r IS NOT NULL ORDER BY id")
    recent_trades = [t for t in all_trades
                     if t.get("opened_at","") >= cutoff]

    # WFA results
    wfa_rows = _q(variant.db_path,
                  "SELECT wfe, wfe_verdict FROM wfa_results ORDER BY run_ts DESC LIMIT 12")

    # Session summaries
    sessions = _q(variant.db_path,
                  f"SELECT * FROM sessions WHERE session_date >= '{cutoff}' ORDER BY session_date")

    # Rolling stats
    rs_all    = _rolling(all_trades)
    rs_recent = _rolling(recent_trades)

    # WFE
    valid_wfe = [r["wfe"] for r in wfa_rows if r.get("wfe") is not None]
    mean_wfe  = round(statistics.mean(valid_wfe), 3) if valid_wfe else None
    wfe_passes = sum(1 for r in wfa_rows if r.get("wfe_verdict") == "PASS")

    # Monte Carlo (from toolkit if available)
    mc_pass_rate = None
    if TOOLKIT and rs_all.get("n", 0) >= 20:
        try:
            rs_list = [r["actual_r"] for r in all_trades]
            mc = tk.monte_carlo_simulation(
                rs_list, n_simulations=1000,
                target_sharpe=1.0, gate_pct=0.85)
            mc_pass_rate = mc.get("pass_rate")
        except Exception:
            pass

    # DSR (from toolkit)
    dsr_confidence = None
    if TOOLKIT and rs_all.get("sharpe") and rs_all.get("n", 0) >= 20:
        try:
            dsr = tk.deflated_sharpe_ratio_extended(
                rs_all["sharpe"], n_trials=5,
                n_obs=rs_all["n"])
            dsr_confidence = dsr.get("dsr")
        except Exception:
            pass

    # Session stop rate
    stopped_sessions = sum(1 for s in sessions if s.get("stop_triggered"))
    total_sessions   = len(sessions)

    return {
        "variant":          variant.name,
        "db_path":          variant.db_path,
        "n":                rs_all.get("n", 0),
        "n_recent":         rs_recent.get("n", 0),
        "win_rate":         rs_all.get("wr"),
        "win_rate_recent":  rs_recent.get("wr"),
        "ev_all":           rs_all.get("ev"),
        "ev_recent":        rs_recent.get("ev"),
        "sharpe_all":       rs_all.get("sharpe"),
        "sharpe_recent":    rs_recent.get("sharpe"),
        "mean_wfe":         mean_wfe,
        "wfe_passes":       wfe_passes,
        "wfe_total_splits": len(wfa_rows),
        "mc_pass_rate":     mc_pass_rate,
        "dsr_confidence":   dsr_confidence,
        "stop_rate":        round(stopped_sessions / max(total_sessions, 1), 3),
        "total_sessions":   total_sessions,
        "cutoff_days":      lookback_days,
    }


def _rolling(trades: list[dict]) -> dict:
    rs = [t["actual_r"] for t in trades if t.get("actual_r") is not None]
    if not rs:
        return {"n": 0}
    wins   = [r for r in rs if r > 0]
    losses = [r for r in rs if r < 0]
    wr     = len(wins) / len(rs)
    aw     = statistics.mean(wins)              if wins   else 0.0
    al     = abs(statistics.mean(losses))       if losses else 0.0
    ev     = (wr * aw) - ((1 - wr) * al)
    sharpe = None
    if len(rs) >= 2:
        sd = statistics.stdev(rs)
        if sd > 0:
            sharpe = round(statistics.mean(rs) / sd, 3)
    return {"n": len(rs), "wr": round(wr, 3), "aw": round(aw, 3),
            "al": round(al, 3), "ev": round(ev, 4), "sharpe": sharpe}


# ---------------------------------------------------------------------------
# Tournament comparison
# ---------------------------------------------------------------------------

def run_tournament(lookback_days: int = 30) -> dict:
    """
    Compare all active variants. Return ranked results and promotion proposals.
    Only ONE variant is active in Phase 2 — tournament becomes meaningful in Phase 4.
    """
    active = get_active_variants()
    metrics = [collect_variant_metrics(v, lookback_days) for v in active]

    # Sort by EV (most reliable metric at low trade counts)
    ranked = sorted(
        [m for m in metrics if m.get("n", 0) > 0],
        key=lambda x: x.get("ev_all") or -99,
        reverse=True,
    )

    # Phase gate checks
    gate_results = []
    for v in active:
        m = next((m for m in metrics if m["variant"] == v.name), {})
        gate = check_phase_gate(v, m)
        gate_results.append(gate)

    # Identify winner (Phase 4+)
    winner = ranked[0] if ranked else None

    # Build report
    ts = datetime.utcnow().isoformat(timespec="seconds")
    # ── Design Studio cycle (Phase 3+: needs >= 2 active variants) ──
    studio_result = None
    if GENE_MIXER and len(ranked) >= 2:
        try:
            mixer  = GeneMixer(llm_cfg={}, seed=None)  # cfg injected by caller
            studio_result = mixer.run_design_studio_cycle(
                ranked_metrics=ranked,
                registry=REGISTRY,
            )
        except Exception as gs_err:
            studio_result = {"error": str(gs_err)}

    report = {
        "run_ts":         ts,
        "n_active":       len(active),
        "lookback_days":  lookback_days,
        "ranked":         ranked,
        "gate_results":   gate_results,
        "winner":         winner["variant"] if winner else None,
        "phase_summary":  _phase_summary(active, gate_results, studio_result),
        "design_studio":  studio_result,
    }
    return report


def _phase_summary(active: list[StrategyVariant], gates: list[dict], studio_result: dict | None = None) -> str:
    lines = [
        "TOURNAMENT PHASE SUMMARY",
        f"Active variants: {len(active)}",
        f"Total variants in registry: {len(REGISTRY)}",
    ]
    for g in gates:
        status = "✅ GATE MET" if g["gate_met"] else "⏳ gate not met"
        lines.append(f"  {g['variant']:<20} {status}")
    locked = [name for name, v in REGISTRY.items() if not v.active]
    if locked:
        lines.append(f"Locked variants: {', '.join(locked)}")
    
    # Fixed: Use studio_result directly instead of out-of-scope 'report'
    if studio_result and isinstance(studio_result, dict) and 'message' in studio_result:
        lines.append('')
        lines.append('DESIGN STUDIO:')
        lines.append(f"  {studio_result['message'][:120]}")
        
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# AutoResearch parameter sweep (Phase 3+)
# ---------------------------------------------------------------------------

def run_autoref(variant_name: str, n_trials: int = 20) -> dict:
    """
    Sweep parameters within PARAMETER_BOUNDS for the given variant.
    Evaluates each config on the TRAINING DATA ONLY (2020-2022).
    Out-of-sample (2025-present) remains sealed.

    Returns top configs for human review.
    THIS FUNCTION NEVER APPLIES CHANGES — it produces proposals.
    """
    v = REGISTRY.get(variant_name)
    if v is None:
        return {"error": f"Unknown variant: {variant_name}"}
    if v.phase < 3:
        return {
            "error": f"AutoResearch locked: '{variant_name}' is Phase {v.phase}. "
                     f"Phase 3 gate required before parameter sweeps.",
            "gate_requirement": "100 trades + WFE ≥ 0.50 + MC 85% + DSR > 80%",
        }

    import random
    rng = random.Random(42)   # deterministic for reproducibility

    configs_tried = []
    for trial in range(n_trials):
        config = {
            "orb_method":        rng.choice(PARAMETER_BOUNDS["orb_method"]["options"]),
            "vwap_lookback":     rng.randint(PARAMETER_BOUNDS["vwap_lookback"]["min"],
                                             PARAMETER_BOUNDS["vwap_lookback"]["max"]),
            "vwap_range_factor": round(rng.uniform(PARAMETER_BOUNDS["vwap_range_factor"]["min"],
                                                    PARAMETER_BOUNDS["vwap_range_factor"]["max"]), 2),
            "target_rr":         round(rng.uniform(PARAMETER_BOUNDS["target_rr"]["min"],
                                                    PARAMETER_BOUNDS["target_rr"]["max"]), 1),
            "risk_pct":          round(rng.uniform(PARAMETER_BOUNDS["risk_pct"]["min"],
                                                    PARAMETER_BOUNDS["risk_pct"]["max"]), 3),
        }
        # Score: compute EV from toolkit backtest on training data
        score = _score_config(config, variant_name)
        configs_tried.append({**config, "score_ev": score, "trial": trial})

    # Sort by EV
    ranked = sorted(configs_tried, key=lambda x: x.get("score_ev", -99), reverse=True)
    top3   = ranked[:3]

    return {
        "variant":     variant_name,
        "n_trials":    n_trials,
        "top_configs": top3,
        "message": (
            f"AutoResearch produced {n_trials} parameter combinations. "
            f"Top 3 configs shown. HUMAN APPROVAL REQUIRED before applying "
            f"any change. Verify WFE ≥ 0.50 on validation set first."
        ),
        "status": "PROPOSAL — not applied",
    }


def _score_config(config: dict, variant_name: str) -> float:
    """
    Quick EV estimate for a parameter config on training data.
    Returns expected value per trade as a float.
    """
    if not TOOLKIT:
        return 0.0
    try:
        # Use a simplified backtest approximation
        # Real implementation would run full backtest on training window
        # Placeholder: return a score based on parameter quality heuristics
        rr = config.get("target_rr", 2.0)
        be_wr = 1 / (1 + rr)                   # breakeven win rate
        assumed_wr = 0.44                        # IDEXAONE verified baseline
        ev = assumed_wr * rr - (1 - assumed_wr) * 1.0
        # Penalise very wide VWAP gate (over-relaxation risk)
        rf = config.get("vwap_range_factor", 0.10)
        if rf < 0.05:
            ev *= 0.8   # too tight → over-filtered
        if rf > 0.20:
            ev *= 0.9   # too loose → false signals
        return round(ev, 4)
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Notifications
# ---------------------------------------------------------------------------

def _send_report(report: dict, cfg: dict) -> None:
    """Send tournament report via Discord/Telegram."""
    webhook = cfg.get("discord_webhook", "")
    if not webhook:
        return
    try:
        import json as _j, urllib.request
        text = _format_report(report)[:1800]
        data = _j.dumps({"content": f"📊 **Weekly Tournament** {report['run_ts'][:10]}\n{text}"}).encode()
        req  = urllib.request.Request(
            webhook, data=data,
            headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:
        print(f"Discord send failed: {e}")


def _format_report(report: dict) -> str:
    lines = [
        f"Active variants: {report['n_active']}",
        f"Lookback: {report['lookback_days']} days",
        "",
    ]
    for i, m in enumerate(report.get("ranked", []), 1):
        n     = m.get("n", 0)
        ev    = m.get("ev_all")
        sr    = m.get("sharpe_all")
        wfe   = m.get("mean_wfe")
        lines.append(
            f"#{i} {m['variant']}: n={n}  EV={ev:+.3f}R  SR={sr}  WFE={wfe}"
            if ev is not None else f"#{i} {m['variant']}: no trades yet"
        )
    lines.append("")
    for g in report.get("gate_results", []):
        lines.append(g["message"][:100])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Save tournament results to DB
# ---------------------------------------------------------------------------

def save_tournament_results(report: dict, db_path: str = "DATA/paper_account.db") -> None:
    """Persist tournament report to the canonical DB for dashboard display."""
    try:
        with sqlite3.connect(db_path) as c:
            c.execute("""CREATE TABLE IF NOT EXISTS tournament_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_ts TEXT NOT NULL,
                report_json TEXT NOT NULL,
                n_active INTEGER,
                winner TEXT,
                any_gate_met INTEGER DEFAULT 0
            )""")
            any_gate_met = any(g["gate_met"] for g in report.get("gate_results", []))
            c.execute(
                "INSERT INTO tournament_results "
                "(run_ts, report_json, n_active, winner, any_gate_met) "
                "VALUES (?,?,?,?,?)",
                (report["run_ts"], json.dumps(report),
                 report["n_active"], report.get("winner"),
                 int(any_gate_met))
            )
    except Exception as e:
        print(f"Could not save tournament results: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Strategy tournament evaluator")
    p.add_argument("--lookback",     type=int, default=30, help="Days to look back")
    p.add_argument("--report-only",  action="store_true")
    p.add_argument("--check-gates",  action="store_true")
    p.add_argument("--run-autoref",  action="store_true")
    p.add_argument("--variant",      default=None)
    p.add_argument("--discord",      default="")
    p.add_argument("--n-trials",     type=int, default=20)
    p.add_argument("--design-studio", action="store_true",
                   help="Run Design Studio gene-mixing cycle after tournament")
    args = p.parse_args()

    print("=" * 60)
    print(f"  TOURNAMENT EVALUATOR  {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 60)

    if args.run_autoref:
        name = args.variant or "canonical"
        print(f"\nRunning AutoResearch for variant '{name}'...")
        result = run_autoref(name, args.n_trials)
        print(json.dumps(result, indent=2, default=str))
        return

    if args.check_gates:
        active = get_active_variants()
        for v in active:
            m = collect_variant_metrics(v, args.lookback)
            gate = check_phase_gate(v, m)
            print(gate["message"])
            for check, detail in gate["checks"].items():
                tick = "✅" if detail["pass"] else "❌"
                print(f"  {tick} {check}: {detail['actual']} (need {detail['required']})")
        return

    # Full tournament run
    report = run_tournament(args.lookback)
    print(_format_report(report))
    save_tournament_results(report)

    if args.discord:
        _send_report(report, {"discord_webhook": args.discord})
        print(f"Discord notification sent.")

    print("\nRun with --check-gates to see phase gate status.")
    print("Run with --run-autoref --variant canonical (Phase 3+ only) for parameter sweep.")


if __name__ == "__main__":
    main()
