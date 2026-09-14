"""
strategy_registry.py
====================
Central registry of all strategy variants for the multi-strategy tournament.

DESIGN PHILOSOPHY (from D-A-C convergent phase, 08 Sep 2026)
──────────────────────────────────────────────────────────────
The goal is a self-improving, multi-strategy tournament system. The D-A-C
challenge established that this architecture is correct but must be
phase-gated. This file defines all variants and their unlock conditions.

PHASE GATES (all enforced by tournament_evaluator.py)
──────────────────────────────────────────────────────
Phase 2 → 3:  100 trades + WFE ≥ 0.50 + Monte Carlo 85% + DSR > 80%
Phase 3 → 4:  3 consecutive rolling 20-trade windows with positive EV
Phase 4 → 5:  Sharpe ≥ 1.0 + WFE ≥ 0.50 sustained over 3 months

Current status: PHASE 2 — single strategy, manual Gate 3, no auto-modification.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from datetime import time as Time


# ---------------------------------------------------------------------------
# Parameter bounds  (AutoResearch may sweep ONLY within these ranges)
# ---------------------------------------------------------------------------

PARAMETER_BOUNDS: dict[str, dict] = {
    "orb_method":        {"type": "choice", "options": ["5min", "15min", "30min"]},
    "vwap_lookback":     {"type": "int",    "min": 2,    "max": 8},
    "vwap_range_factor": {"type": "float",  "min": 0.05, "max": 0.25},
    "vwap_max_flips":    {"type": "int",    "min": 1,    "max": 4},
    "vwap_hold_bars":    {"type": "int",    "min": 1,    "max": 4},
    "target_rr":         {"type": "float",  "min": 1.5,  "max": 4.0},
    "risk_pct":          {"type": "float",  "min": 0.005, "max": 0.01},
    "session_end":       {"type": "time",   "min": Time(10, 0), "max": Time(11, 30)},
}

# Any parameter change outside these bounds REQUIRES HUMAN APPROVAL.
# Changes within bounds may be proposed by AutoResearch and applied
# automatically ONLY after WFE ≥ 0.50 is confirmed on the new config.


# ---------------------------------------------------------------------------
# Strategy variant definition
# ---------------------------------------------------------------------------

@dataclass
class StrategyVariant:
    """
    Defines a single strategy variant for the tournament.

    Each variant runs in its own isolated paper account DB.
    Parameters must be within PARAMETER_BOUNDS for auto-modification.
    Phase gates control when a variant is allowed to be promoted.
    """
    name:        str
    db_path:     str
    description: str
    phase:       int           # 2 = current phase, 3 = auto-sweep unlocked, 4 = tournament
    active:      bool = True

    # Core strategy parameters
    ticker:          str   = "SPY"
    orb_method:      str   = "15min"
    vwap_lookback:   int   = 5
    vwap_range_factor: float = 0.10
    vwap_max_flips:  int   = 2
    vwap_hold_bars:  int   = 2
    target_rr:       float = 2.0
    risk_pct:        float = 0.01
    session_end:     Time  = Time(11, 0)
    use_vwap_trailing: bool = True
    use_ladder_exit: bool  = True

    # Risk budget (per-variant daily limits)
    starting_balance:  float = 10_000.0
    account_balance:   float = 10_000.0
    max_daily_loss_pct: float = 0.03
    consec_loss_pause: int   = 3
    max_attempts_per_dir: int = 2

    # Phase gate requirements (what must be met before promotion)
    gate_min_trades:   int   = 100
    gate_min_wfe:      float = 0.50
    gate_min_mc_pass:  float = 0.85
    gate_min_dsr:      float = 0.80

    # Performance tracking
    notes: str = ""

    def to_engine_config(self, base_config: dict) -> dict:
        """Merge variant params into a copy of the engine CONFIG."""
        cfg = dict(base_config)
        cfg.update({
            "ticker":           self.ticker,
            "orb_method":       self.orb_method,
            "vwap_lookback":    self.vwap_lookback,
            "account_balance":  self.account_balance,
            "risk_pct":         self.risk_pct,
            "starting_balance": self.starting_balance,
            "target_rr":        self.target_rr,
            "db_path":          self.db_path,
            "max_daily_loss_pct": self.max_daily_loss_pct,
            "consec_loss_pause": self.consec_loss_pause,
            "session_end":      self.session_end,
            "use_vwap_trailing": self.use_vwap_trailing,
            "use_ladder_exit":  self.use_ladder_exit,
            "variant_name":     self.name,
        })
        return cfg


# ---------------------------------------------------------------------------
# Strategy variants registry
# ---------------------------------------------------------------------------

REGISTRY: dict[str, StrategyVariant] = {

    # ── Phase 2: The canonical strategy (currently active) ──────────────────
    # This is the only active variant. Do not modify parameters here manually —
    # any change must go through the WFA gate first.
    "canonical": StrategyVariant(
        name        = "canonical",
        db_path     = "DATA/paper_account.db",
        description = (
            "Hybrid ORB 15-min on SPY. Foundation: Zarattini 2024 RVOL filter, "
            "Gao 2018 30-min window (fallback), Maroy 2025 VWAP+Ladder exits. "
            "AND-gate: ORB break + VWAP slope gate + manual retest (Gate 3). "
            "1% risk, 2:1 R:R minimum, 3% daily stop, 11:00 EST cutoff."
        ),
        phase        = 2,
        active       = True,
        orb_method   = "15min",
        target_rr    = 2.0,
        risk_pct     = 0.01,
        notes        = "Phase 2 baseline. Gate 3 still manual. Zero self-modification.",
    ),

    # ── Phase 3: AutoResearch variants (LOCKED — phase gate not met) ─────────
    # These become active ONLY after canonical reaches Phase 3 gate.
    # Parameters swept by AutoResearch within PARAMETER_BOUNDS.
    "orb-30min": StrategyVariant(
        name        = "orb-30min",
        db_path     = "DATA/paper_account_orb30.db",
        description = (
            "30-min ORB window variant. Tests whether wider range definition "
            "improves edge on higher-VIX sessions. Gao 2018 supports 30-min "
            "as primary window when VIX > 20."
        ),
        phase        = 3,
        active       = False,    # LOCKED: activate only after Phase 3 gate
        orb_method   = "30min",
        target_rr    = 2.0,
        notes        = "LOCKED until canonical WFE ≥ 0.50 + 100 trades",
    ),

    "tight-rr": StrategyVariant(
        name        = "tight-rr",
        db_path     = "DATA/paper_account_tightrr.db",
        description = (
            "1.5:1 R:R variant. Tests whether tighter target improves win rate "
            "enough to offset smaller average winner. Breakeven win rate at 1.5:1 = 40%."
        ),
        phase        = 3,
        active       = False,
        target_rr    = 1.5,
        notes        = "LOCKED until Phase 3 gate",
    ),

    "wide-vwap": StrategyVariant(
        name        = "wide-vwap",
        db_path     = "DATA/paper_account_widevwap.db",
        description = (
            "Wider VWAP slope gate (range_factor 0.05). Tests whether the "
            "over-filtering flag from SessionLearner is correct — i.e., that "
            "lowering the VWAP slope threshold improves entry rate and EV."
        ),
        phase        = 3,
        active       = False,
        vwap_range_factor = 0.05,
        notes        = "LOCKED until Phase 3 gate. Only activate if D-A-C confirms over-filtering.",
    ),

    "commodity-add": StrategyVariant(
        name        = "commodity-add",
        db_path     = "DATA/paper_account_gold.db",
        description = (
            "Gold (GC=F) ORB variant with 0.5% risk cap (commodity rule). "
            "Tests whether the ORB strategy generalises to commodities. "
            "Gold/Silver demonstrated 30-50% weekly crashes (Jason Sen). "
            "Smaller risk, expected lower Sharpe but diversification value."
        ),
        phase        = 4,
        active       = False,
        ticker       = "GC=F",
        risk_pct     = 0.005,
        notes        = "LOCKED until Phase 4 gate. Requires P3-003 OOS validation first.",
    ),

    # ── Phase 4: Tournament variants (LOCKED — far gate) ─────────────────────
    "qqqq-momentum": StrategyVariant(
        name        = "qqqq-momentum",
        db_path     = "DATA/paper_account_qqq.db",
        description = (
            "QQQ intraday momentum variant. Zarattini 2024 SPY momentum "
            "paper showed IWM returned 11.72% vs SPY 6.67% annualised — "
            "QQQ is the tech-heavy alternative. Different correlation profile."
        ),
        phase        = 4,
        active       = False,
        ticker       = "QQQ",
        notes        = "LOCKED until Phase 4 gate. Different beta, needs own RVOL calibration.",
    ),
}


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------

def get_active_variants() -> list[StrategyVariant]:
    """Return all currently active variants (respects phase gates)."""
    return [v for v in REGISTRY.values() if v.active]


def get_variant(name: str) -> StrategyVariant | None:
    return REGISTRY.get(name)


def activate_variant(name: str, reason: str) -> bool:
    """
    Activate a locked variant. REQUIRES a reason string documenting
    which gate was met. This is the human approval step.
    """
    v = REGISTRY.get(name)
    if v is None:
        print(f"Unknown variant: {name}")
        return False
    if v.active:
        print(f"{name} is already active.")
        return True
    v.active = True
    print(f"ACTIVATED variant '{name}': {reason}")
    print(f"  DB: {v.db_path}")
    print(f"  Phase: {v.phase}")
    return True


def print_status() -> None:
    """Print the current registry status for human review."""
    print("=" * 65)
    print("  STRATEGY REGISTRY — Current Status")
    print("=" * 65)
    for name, v in REGISTRY.items():
        status = "✅ ACTIVE" if v.active else f"🔒 LOCKED (Phase {v.phase})"
        print(f"  {name:<20} {status}")
        print(f"    {v.description[:70]}")
        if not v.active:
            print(f"    Gate: {v.notes}")
        print()


# ---------------------------------------------------------------------------
# Phase gate checker (used by tournament_evaluator.py)
# ---------------------------------------------------------------------------

def check_phase_gate(variant: StrategyVariant, stats: dict) -> dict:
    """
    Check if a variant has met its phase gate requirements.
    stats: dict from PaperAccountDB.rolling_stats() + wfa_results

    Returns {'gate_met': bool, 'checks': dict, 'message': str}
    """
    checks = {
        "min_trades": {
            "required": variant.gate_min_trades,
            "actual":   stats.get("n", 0),
            "pass":     stats.get("n", 0) >= variant.gate_min_trades,
        },
        "wfe": {
            "required": variant.gate_min_wfe,
            "actual":   stats.get("mean_wfe"),
            "pass":     (stats.get("mean_wfe") or 0) >= variant.gate_min_wfe,
        },
        "monte_carlo": {
            "required": variant.gate_min_mc_pass,
            "actual":   stats.get("mc_pass_rate"),
            "pass":     (stats.get("mc_pass_rate") or 0) >= variant.gate_min_mc_pass,
        },
        "dsr": {
            "required": variant.gate_min_dsr,
            "actual":   stats.get("dsr_confidence"),
            "pass":     (stats.get("dsr_confidence") or 0) >= variant.gate_min_dsr,
        },
    }

    gate_met = all(c["pass"] for c in checks.values())
    failing  = [k for k, c in checks.items() if not c["pass"]]

    msg = (
        f"✅ ALL GATES MET for '{variant.name}'. Ready to advance to Phase {variant.phase}."
        if gate_met else
        f"❌ Gate not met for '{variant.name}'. Failing: {', '.join(failing)}. "
        f"Continue paper trading."
    )
    return {"gate_met": gate_met, "checks": checks, "message": msg, "variant": variant.name}


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print_status()

    canonical = get_variant("canonical")
    assert canonical is not None and canonical.active
    assert len(get_active_variants()) == 1

    # Phase gate check with insufficient data
    gate = check_phase_gate(canonical, {"n": 12, "mean_wfe": 0.3, "mc_pass_rate": 0.7, "dsr_confidence": 0.6})
    assert not gate["gate_met"]
    print(gate["message"])

    # Phase gate check with passing data
    gate2 = check_phase_gate(canonical, {"n": 150, "mean_wfe": 0.62, "mc_pass_rate": 0.88, "dsr_confidence": 0.85})
    assert gate2["gate_met"]
    print(gate2["message"])

    print("\n✅ strategy_registry.py verified")
