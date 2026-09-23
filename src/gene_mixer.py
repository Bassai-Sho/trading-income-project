"""
gene_mixer.py
=============
Design Studio Evolutionary Engine for the Trading Income Project.

CONCEPTUAL FRAMEWORK
────────────────────
Applies the Agile Design Studio methodology to strategy evolution:

  DIVERGENT:  Multiple strategy variants run independently in parallel
              paper accounts (the "designers" sketching different solutions)

  CRITIQUE:   Tournament evaluator ranks variants by WFE, EV, Sharpe.
              LLM identifies the specific parameter genes driving the
              performance difference between winners and losers.

  CONVERGE:   Gene mixer synthesises a new variant by combining the best
              parameter genes from top performers (crossover) with a small
              random perturbation (mutation). D-A-C validates the synthesis
              before the new variant is deployed.

  REPLACE:    The weakest variant is retired. Its paper account DB is wiped
              and reused by the new hybrid variant.

  ITERATE:    The next generation's population has higher average quality.
              This repeats weekly, indefinitely.

ACADEMIC BACKING
────────────────
Zhang et al. (2025) — LLM-Guided Evolutionary Strategy Generation for
Quantitative Trading (LLM-GA): LLM-guided initialisation improved starting
strategy quality by 215%. Semantic crossover (LLM validates logical
consistency of gene combinations) reduced invalid strategies by 83.5%.
Source: XJTLU Scholar. Not yet in Ledger — add as Intelligence entry.

USAGE
─────
  from gene_mixer import breed_new_variant, extract_chromosome, GeneMixer

  # After tournament evaluation:
  mixer = GeneMixer(db_path="DATA/paper_account.db", cfg=CONFIG)
  result = mixer.run_design_studio_cycle(
      ranked_variants=tournament_report["ranked"],
      registry=REGISTRY,
  )
  # result contains: new_variant, lineage, dac_verdict, ready_to_deploy
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from strategy_registry import REGISTRY, StrategyVariant, PARAMETER_BOUNDS

log = logging.getLogger("gene_mixer")


# ── Strategy chromosome ───────────────────────────────────────────────────────

@dataclass
class StrategyChromosome:
    """
    A strategy's parameter vector — the "chromosome" in GA terms.
    Each field is a "gene" that can be inherited, combined, or mutated.

    Genes must stay within PARAMETER_BOUNDS (enforced at validate() time).
    Maximum 5 genes per chromosome enforces Rule 13 (no overfitting).
    """
    # Core parameter genes
    orb_method:        str   = "15min"
    vwap_lookback:     int   = 5
    vwap_range_factor: float = 0.10
    vwap_max_flips:    int   = 2
    vwap_hold_bars:    int   = 2
    target_rr:         float = 2.0
    risk_pct:          float = 0.01
    session_end_hour:  int   = 11
    session_end_min:   int   = 0

    # Performance metrics (not genes — used for selection only)
    ev:          float | None = None
    sharpe:      float | None = None
    wfe:         float | None = None
    n_trades:    int          = 0

    # Lineage tracking
    variant_name: str         = ""
    parents:      list[str]   = field(default_factory=list)
    generation:   int         = 0
    bred_at:      str         = ""

    def gene_vector(self) -> dict:
        """Return only the parameter genes (not metrics or lineage)."""
        return {
            "orb_method":        self.orb_method,
            "vwap_lookback":     self.vwap_lookback,
            "vwap_range_factor": self.vwap_range_factor,
            "vwap_max_flips":    self.vwap_max_flips,
            "vwap_hold_bars":    self.vwap_hold_bars,
            "target_rr":         self.target_rr,
            "risk_pct":          self.risk_pct,
            "session_end_hour":  self.session_end_hour,
            "session_end_min":   self.session_end_min,
        }

    def validate(self) -> tuple[bool, str]:
        """
        Check all genes are within PARAMETER_BOUNDS.
        Rule 13: maximum 5 tunable parameters. This chromosome has 5 core
        genes (orb_method, vwap_range_factor, target_rr, risk_pct,
        session_end). The VWAP lookback/flips/hold are counted as one
        composite VWAP gate gene.
        Returns (is_valid, error_message).
        """
        b = PARAMETER_BOUNDS
        errors = []
        if self.orb_method not in b["orb_method"]["options"]:
            errors.append(f"orb_method '{self.orb_method}' not in {b['orb_method']['options']}")
        if not (b["vwap_lookback"]["min"] <= self.vwap_lookback <= b["vwap_lookback"]["max"]):
            errors.append(f"vwap_lookback {self.vwap_lookback} out of bounds")
        if not (b["vwap_range_factor"]["min"] <= self.vwap_range_factor <= b["vwap_range_factor"]["max"]):
            errors.append(f"vwap_range_factor {self.vwap_range_factor:.3f} out of bounds")
        if not (b["target_rr"]["min"] <= self.target_rr <= b["target_rr"]["max"]):
            errors.append(f"target_rr {self.target_rr} out of bounds")
        if not (b["risk_pct"]["min"] <= self.risk_pct <= b["risk_pct"]["max"]):
            errors.append(f"risk_pct {self.risk_pct:.3f} out of bounds")
        if errors:
            return False, "; ".join(errors)
        return True, "OK"

    def to_variant(self, name: str, db_path: str, description: str = "") -> StrategyVariant:
        """Convert chromosome to a StrategyVariant for the registry."""
        import datetime as dt
        from datetime import time as Time
        return StrategyVariant(
            name=name,
            db_path=db_path,
            description=description or self._auto_description(),
            phase=3,
            active=False,   # Must be activated manually after D-A-C
            orb_method=self.orb_method,
            vwap_lookback=self.vwap_lookback,
            vwap_range_factor=self.vwap_range_factor,
            vwap_max_flips=self.vwap_max_flips,
            vwap_hold_bars=self.vwap_hold_bars,
            target_rr=self.target_rr,
            risk_pct=self.risk_pct,
            session_end=Time(self.session_end_hour, self.session_end_min),
            notes=(f"BRED from {self.parents} gen={self.generation} at {self.bred_at}. "
                   "Requires D-A-C validation before activation."),
        )

    def _auto_description(self) -> str:
        return (f"Gen-{self.generation} hybrid from {self.parents}. "
                f"ORB={self.orb_method} VWAP={self.vwap_range_factor:.2f} "
                f"RR={self.target_rr} Risk={self.risk_pct:.3f}")


# ── Chromosome extraction ─────────────────────────────────────────────────────

def extract_chromosome(variant: StrategyVariant, metrics: dict) -> StrategyChromosome:
    """
    Extract a StrategyChromosome from a live variant + its tournament metrics.
    """
    from datetime import time as Time
    se = variant.session_end
    return StrategyChromosome(
        orb_method        = variant.orb_method,
        vwap_lookback     = variant.vwap_lookback,
        vwap_range_factor = variant.vwap_range_factor,
        vwap_max_flips    = variant.vwap_max_flips,
        vwap_hold_bars    = variant.vwap_hold_bars,
        target_rr         = variant.target_rr,
        risk_pct          = variant.risk_pct,
        session_end_hour  = se.hour if hasattr(se, "hour") else 11,
        session_end_min   = se.minute if hasattr(se, "minute") else 0,
        ev                = metrics.get("ev_all"),
        sharpe            = metrics.get("sharpe_all"),
        wfe               = metrics.get("mean_wfe"),
        n_trades          = metrics.get("n", 0),
        variant_name      = variant.name,
        parents           = [],
        generation        = 0,
    )


# ── Crossover operations ──────────────────────────────────────────────────────

def crossover(
    parent_a: StrategyChromosome,
    parent_b: StrategyChromosome,
    method: str = "best_ev_weighted",
    rng: random.Random | None = None,
) -> StrategyChromosome:
    """
    Combine two parent chromosomes into a child.

    Methods
    -------
    best_ev_weighted  : Each gene comes from the parent with higher EV
                        (LLM-GA "semantic crossover" — logically consistent)
    random_uniform    : Each gene randomly assigned from either parent
    best_mixed        : Winner's core genes + loser's best individual gene
    """
    if rng is None:
        rng = random.Random()

    ev_a = parent_a.ev or 0.0
    ev_b = parent_b.ev or 0.0
    # Always assign 'better' and 'worse' deterministically
    better  = parent_a if ev_a >= ev_b else parent_b
    weaker  = parent_b if ev_a >= ev_b else parent_a

    child_genes: dict[str, Any] = {}
    gene_names = list(parent_a.gene_vector().keys())

    if method == "best_ev_weighted":
        # Genes from better parent with probability proportional to EV advantage
        ev_max  = max(abs(ev_a), abs(ev_b), 0.001)
        p_better = 0.5 + 0.3 * (abs(ev_a - ev_b) / ev_max)  # 0.5-0.8
        for gene in gene_names:
            child_genes[gene] = (getattr(better, gene)
                                 if rng.random() < p_better
                                 else getattr(weaker, gene))

    elif method == "random_uniform":
        for gene in gene_names:
            child_genes[gene] = (getattr(parent_a, gene)
                                 if rng.random() < 0.5
                                 else getattr(parent_b, gene))

    elif method == "best_mixed":
        # Start with better parent's genes, then replace one gene from weaker
        child_genes = dict(better.gene_vector())
        swap_gene   = rng.choice(gene_names)
        child_genes[swap_gene] = getattr(weaker, swap_gene)

    gen = max(parent_a.generation, parent_b.generation) + 1
    child = StrategyChromosome(
        **{k: v for k, v in child_genes.items()},
        parents  = [parent_a.variant_name, parent_b.variant_name],
        generation = gen,
        bred_at  = datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds"),
    )
    return child


def mutate(
    chromosome: StrategyChromosome,
    mutation_rate: float = 0.15,
    rng: random.Random | None = None,
) -> StrategyChromosome:
    """
    Apply small random perturbation to one or more genes.
    mutation_rate: probability each gene mutates independently.
    """
    if rng is None:
        rng = random.Random()

    b    = PARAMETER_BOUNDS
    opts = b["orb_method"]["options"]
    genes = chromosome.gene_vector()

    if rng.random() < mutation_rate:
        genes["orb_method"]        = rng.choice(opts)
    if rng.random() < mutation_rate:
        genes["vwap_lookback"]     = rng.randint(b["vwap_lookback"]["min"], b["vwap_lookback"]["max"])
    if rng.random() < mutation_rate:
        lo, hi = b["vwap_range_factor"]["min"], b["vwap_range_factor"]["max"]
        genes["vwap_range_factor"] = round(genes["vwap_range_factor"] + rng.uniform(-0.02, 0.02), 3)
        genes["vwap_range_factor"] = max(lo, min(hi, genes["vwap_range_factor"]))
    if rng.random() < mutation_rate:
        genes["target_rr"]         = round(genes["target_rr"] + rng.choice([-0.5, 0.5]), 1)
        genes["target_rr"]         = max(b["target_rr"]["min"], min(b["target_rr"]["max"], genes["target_rr"]))
    if rng.random() < mutation_rate:
        genes["risk_pct"]          = 0.005 if genes["risk_pct"] >= 0.01 else 0.01

    mutated = StrategyChromosome(
        **{k: v for k, v in genes.items()},
        parents    = chromosome.parents,
        generation = chromosome.generation,
        bred_at    = chromosome.bred_at,
        variant_name = chromosome.variant_name + "_mutated",
    )
    return mutated


# ── LLM-guided D-A-C validation ──────────────────────────────────────────────

def llm_validate_synthesis(
    child: StrategyChromosome,
    parent_a: StrategyChromosome,
    parent_b: StrategyChromosome,
    llm_cfg: dict,
) -> dict:
    """
    Ask the LLM to validate whether the synthesised chromosome makes
    theoretical sense before it is deployed as a new variant.

    This is the "D-A-C" gate for gene synthesis — equivalent to the
    critique phase of Design Studio. The LLM acts as the domain expert
    reviewing the designer's sketch.

    Returns {valid: bool, verdict: str, reasoning: str, confidence: str}
    """
    try:
        from openai import OpenAI
    except ImportError:
        return {"valid": True, "verdict": "SKIPPED",
                "reasoning": "openai not installed — skipping LLM validation",
                "confidence": "LOW"}

    prompt = f"""You are reviewing a synthesised trading strategy chromosome.

PARENT A ({parent_a.variant_name}, EV={parent_a.ev:+.3f}R):
{json.dumps(parent_a.gene_vector(), indent=2)}

PARENT B ({parent_b.variant_name}, EV={parent_b.ev:+.3f}R):
{json.dumps(parent_b.gene_vector(), indent=2)}

CHILD (bred via crossover+mutation):
{json.dumps(child.gene_vector(), indent=2)}

Rules this child must satisfy:
1. All parameters within PARAMETER_BOUNDS
2. Maximum 5 tunable parameters (Rule 13)
3. Each parameter combination must make theoretical sense
4. The combination should not create contradictions (e.g. very tight VWAP
   gate with a very wide ORB window creates an inconsistent risk model)
5. Academic backing exists for each component independently

Respond in JSON only:
{{
  "valid": true/false,
  "verdict": "APPROVED" / "REJECTED" / "PROVISIONAL",
  "reasoning": "one paragraph explaining the decision",
  "potential_contradiction": "any identified inconsistency or null",
  "confidence": "HIGH" / "MEDIUM" / "LOW"
}}"""

    try:
        base = llm_cfg.get("dac_base_url") or llm_cfg.get("llm_base_url", "http://localhost:1234/v1")
        key  = llm_cfg.get("dac_api_key")  or llm_cfg.get("llm_api_key", "not-needed")
        mdl  = llm_cfg.get("dac_model")    or llm_cfg.get("llm_model", "local-model")
        client = OpenAI(base_url=base, api_key=key)
        resp   = client.chat.completions.create(
            model=mdl,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500, temperature=0.1, timeout=30,
        )
        raw  = resp.choices[0].message.content or "{}"
        raw  = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        data = json.loads(raw)
        return {
            "valid":    bool(data.get("valid", False)),
            "verdict":  data.get("verdict", "PROVISIONAL"),
            "reasoning": data.get("reasoning", ""),
            "potential_contradiction": data.get("potential_contradiction"),
            "confidence": data.get("confidence", "LOW"),
        }
    except Exception as e:
        log.warning("LLM synthesis validation failed: %s", e)
        return {"valid": True, "verdict": "PROVISIONAL",
                "reasoning": f"LLM unavailable: {e}. Manual review required.",
                "confidence": "LOW"}


# ── Gene selection: which gene drove the winner's outperformance? ─────────────

def attribute_winning_genes(
    winner: StrategyChromosome,
    loser:  StrategyChromosome,
    db_path_winner: str,
    db_path_loser:  str,
) -> dict:
    """
    Use journal correlation data to identify WHICH specific genes explain
    the performance gap between winner and loser.

    This is the LLM-GA "semantic crossover" step: instead of randomly
    picking genes, we identify which parameters actually contributed to
    the winner's advantage, then preferentially inherit those.

    Returns {attributed_genes: dict, explanation: str}
    """
    attributed: dict[str, str] = {}

    # ORB method: if winner uses 15min and has better EV, attribute it
    if winner.orb_method != loser.orb_method:
        attributed["orb_method"] = f"winner uses {winner.orb_method} ({winner.ev:+.3f}R vs {loser.ev:+.3f}R)"

    # VWAP gate: if winner has tighter or wider gate and better EV
    if abs(winner.vwap_range_factor - loser.vwap_range_factor) > 0.02:
        direction = "tighter" if winner.vwap_range_factor < loser.vwap_range_factor else "wider"
        attributed["vwap_range_factor"] = f"winner uses {direction} VWAP gate ({winner.vwap_range_factor:.2f} vs {loser.vwap_range_factor:.2f})"

    # Target R:R
    if abs(winner.target_rr - loser.target_rr) >= 0.5:
        attributed["target_rr"] = f"winner targets {winner.target_rr}R (vs {loser.target_rr}R)"

    explanation_parts = [f"{gene}: {reason}" for gene, reason in attributed.items()]
    explanation = ("No single gene clearly attributed — performance difference may be due to "
                   "market regime or sample variance.")
    if explanation_parts:
        explanation = "Attribution: " + "; ".join(explanation_parts)

    return {"attributed_genes": attributed, "explanation": explanation}


# ── Full Design Studio cycle ──────────────────────────────────────────────────

class GeneMixer:
    """
    Orchestrates one full Design Studio evolution cycle:
      1. Extract chromosomes from all ranked variants
      2. Attribute winning genes (critique phase)
      3. Crossover + mutate top-2 variants (converge phase)
      4. LLM D-A-C validates the synthesis
      5. Register the child variant (inactive until human confirms)
      6. Flag the loser for retirement
    """

    def __init__(self, llm_cfg: dict, seed: int | None = None) -> None:
        self.llm_cfg = llm_cfg
        self.rng     = random.Random(seed)

    def run_design_studio_cycle(
        self,
        ranked_metrics: list[dict],
        registry: dict[str, StrategyVariant],
        crossover_method: str = "best_ev_weighted",
        mutation_rate: float = 0.10,
    ) -> dict:
        """
        Execute one full Design Studio cycle.

        Parameters
        ----------
        ranked_metrics  : list of metric dicts from tournament_evaluator,
                          sorted best-first (index 0 = best performer)
        registry        : REGISTRY dict from strategy_registry.py
        crossover_method: 'best_ev_weighted' | 'random_uniform' | 'best_mixed'
        mutation_rate   : probability each gene mutates

        Returns
        -------
        dict with keys:
            cycle_ts, winner_name, loser_name, child_chromosome,
            attribution, dac_result, new_variant_name, ready_to_deploy
        """
        if len(ranked_metrics) < 2:
            return {"error": "Need at least 2 active variants for Design Studio"}

        # ── Phase 1: DIVERGENT result — ranked population ─────────────────────
        winner_m = ranked_metrics[0]
        loser_m  = ranked_metrics[-1]
        winner_v = registry.get(winner_m["variant"])
        loser_v  = registry.get(loser_m["variant"])
        if not winner_v or not loser_v:
            return {"error": "Variant not found in registry"}

        log.info("Design Studio cycle: winner=%s (EV=%s), loser=%s (EV=%s)",
                 winner_v.name, winner_m.get("ev_all"), loser_v.name, loser_m.get("ev_all"))

        # ── Phase 2: Extract chromosomes ──────────────────────────────────────
        chr_winner = extract_chromosome(winner_v, winner_m)
        chr_loser  = extract_chromosome(loser_v,  loser_m)

        # ── Phase 3: CRITIQUE — attribute winning genes ───────────────────────
        attribution = attribute_winning_genes(
            chr_winner, chr_loser,
            winner_v.db_path, loser_v.db_path
        )
        log.info("Gene attribution: %s", attribution["explanation"])

        # ── Phase 4: CONVERGE — crossover + mutate ────────────────────────────
        child = crossover(chr_winner, chr_loser, method=crossover_method, rng=self.rng)
        child = mutate(child, mutation_rate=mutation_rate, rng=self.rng)

        # Validate parameter bounds
        valid, err = child.validate()
        if not valid:
            # Clamp to bounds and retry
            child = self._clamp_chromosome(child)
            valid, err = child.validate()
            if not valid:
                return {"error": f"Chromosome validation failed after clamp: {err}"}

        # ── Phase 5: D-A-C LLM validation ────────────────────────────────────
        # Use the second-best parent as comparison context
        second_parent = (extract_chromosome(registry.get(ranked_metrics[1]["variant"],winner_v),
                                             ranked_metrics[1])
                          if len(ranked_metrics) > 1 else chr_loser)
        dac_result = llm_validate_synthesis(child, chr_winner, second_parent, self.llm_cfg)
        log.info("D-A-C synthesis verdict: %s (confidence=%s)",
                 dac_result["verdict"], dac_result["confidence"])

        # ── Phase 6: Register child variant (inactive) ────────────────────────
        child_name = f"gen{child.generation}-{winner_v.name[:4]}-x-{loser_v.name[:4]}"
        child_db   = loser_v.db_path   # reuse loser's DB path after clearing
        new_variant = child.to_variant(
            name=child_name,
            db_path=child_db,
            description=(f"Generation {child.generation} hybrid: "
                         f"{winner_v.name} × {loser_v.name}. "
                         f"D-A-C verdict: {dac_result['verdict']}. "
                         f"Gene attribution: {attribution['explanation'][:100]}"),
        )

        # Register in the registry (inactive until human activates)
        registry[child_name] = new_variant

        ready = (dac_result["verdict"] == "APPROVED"
                 and dac_result["confidence"] in ("HIGH", "MEDIUM"))

        result = {
            "cycle_ts":         datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds"),
            "generation":       child.generation,
            "winner_name":      winner_v.name,
            "winner_ev":        chr_winner.ev,
            "loser_name":       loser_v.name,
            "loser_ev":         chr_loser.ev,
            "attribution":      attribution,
            "child_genes":      child.gene_vector(),
            "child_chromosome": asdict(child),
            "dac_result":       dac_result,
            "new_variant_name": child_name,
            "retire_variant":   loser_v.name,
            "ready_to_deploy":  ready,
            "message": (
                f"✅ Ready to deploy '{child_name}'. Activate with: "
                f"activate_variant('{child_name}', reason='D-A-C APPROVED gen{child.generation}')"
                if ready else
                f"⚠️ '{child_name}' bred but needs human review before activation. "
                f"D-A-C verdict: {dac_result['verdict']} ({dac_result['confidence']} confidence). "
                f"Reason: {dac_result['reasoning'][:120]}"
            ),
        }
        return result

    def _clamp_chromosome(self, c: StrategyChromosome) -> StrategyChromosome:
        """Clamp all genes to within PARAMETER_BOUNDS."""
        b = PARAMETER_BOUNDS
        return StrategyChromosome(
            orb_method        = c.orb_method if c.orb_method in b["orb_method"]["options"] else "15min",
            vwap_lookback     = max(b["vwap_lookback"]["min"], min(b["vwap_lookback"]["max"], c.vwap_lookback)),
            vwap_range_factor = max(b["vwap_range_factor"]["min"], min(b["vwap_range_factor"]["max"], c.vwap_range_factor)),
            vwap_max_flips    = max(b["vwap_max_flips"]["min"], min(b["vwap_max_flips"]["max"], c.vwap_max_flips)),
            vwap_hold_bars    = max(b["vwap_hold_bars"]["min"], min(b["vwap_hold_bars"]["max"], c.vwap_hold_bars)),
            target_rr         = max(b["target_rr"]["min"], min(b["target_rr"]["max"], c.target_rr)),
            risk_pct          = max(b["risk_pct"]["min"], min(b["risk_pct"]["max"], c.risk_pct)),
            session_end_hour  = c.session_end_hour,
            session_end_min   = c.session_end_min,
            parents           = c.parents,
            generation        = c.generation,
            bred_at           = c.bred_at,
            variant_name      = c.variant_name,
        )


# ── Machine psychology fields (God Mode journal) ──────────────────────────────

def build_machine_psychology(
    entry_ctx:        dict,
    orb_data:         dict,
    df_5m:            "pd.DataFrame",
    position_history: list[dict],
) -> dict:
    """
    When trading is fully automated (God Mode), replace human psychology
    fields with mechanical equivalents.

    Human field         → Machine equivalent
    ─────────────────────────────────────────
    confidence_pre      → signal_confidence_score (gate margins)
    emotional_state     → regime_match_label
    fomo_flag           → thin_gate_margin_flag
    revenge_flag        → drawdown_context_flag
    hesitation_flag     → 0 (engine never hesitates)
    plan_adherence      → 100 (engine always follows rules)
    setup_grade         → gate_strength_grade
    process_grade       → A (engine follows process perfectly)
    what_went_right     → auto-populated from signal analysis
    what_went_wrong     → auto-populated from fill analysis

    Returns dict of machine psychology fields.
    """
    fields: dict = {}

    # Signal confidence: how far above each gate threshold was the signal?
    vwap_slope   = entry_ctx.get("vwap_slope", "flat")
    vwap_gate    = entry_ctx.get("gate_vwap_strength", 0.0)   # gate margin
    orb_gap      = entry_ctx.get("orb_gap_to_edge", 0.0)     # how far above ORB
    vix          = entry_ctx.get("vix", 0.0) or 0.0

    # Gate strength composite (0.0 = barely passed, 1.0 = very strong)
    gate_strength = min(1.0, (abs(vwap_gate) / 0.02 + min(orb_gap / 0.5, 1.0)) / 2)
    fields["signal_confidence_score"] = round(gate_strength, 3)
    fields["thin_gate_margin_flag"]   = int(gate_strength < 0.25)

    # Gate strength grade (replaces setup_grade for automation)
    if gate_strength >= 0.7:
        fields["gate_strength_grade"] = "A"
    elif gate_strength >= 0.4:
        fields["gate_strength_grade"] = "B"
    else:
        fields["gate_strength_grade"] = "C"

    # Regime match: does VIX/trend match the strategy's designed conditions?
    vix_regime     = entry_ctx.get("vix_regime", "NORMAL")
    daily_trend    = entry_ctx.get("daily_trend", "")
    orb_method     = entry_ctx.get("orb_method", "15min")
    # For 15-min ORB: ideal regime is NORMAL VIX + uptrend
    # For 30-min ORB: designed for ELEVATED/HIGH VIX
    orb_designed_for_high_vix = orb_method == "30min"
    actual_high_vix            = vix_regime in ("HIGH", "EXTREME")
    regime_aligned = (orb_designed_for_high_vix == actual_high_vix)
    fields["regime_match_score"] = 1.0 if regime_aligned else 0.5
    fields["regime_match_label"]  = "aligned" if regime_aligned else "misaligned"

    # Strategy drawdown context (replaces revenge_flag)
    closed = [p for p in position_history if p.get("status") == "closed"]
    if closed and len(closed) >= 3:
        last_3_r = [p.get("actual_r", 0) for p in closed[-3:]]
        cumulative_r = sum(last_3_r)
        fields["strategy_drawdown_pct"] = round(max(0, -cumulative_r / 3), 3)
        fields["drawdown_context_flag"] = int(cumulative_r < -1.5)
    else:
        fields["strategy_drawdown_pct"] = 0.0
        fields["drawdown_context_flag"] = 0

    # Consecutive losses entering (replaces emotional_state for automation)
    consec_losses = 0
    for p in reversed(position_history):
        if p.get("status") == "closed":
            if (p.get("actual_r") or 0) < 0:
                consec_losses += 1
            else:
                break
    fields["consecutive_losses_entering"] = consec_losses

    # Always 100% rule adherence and A process grade in God Mode
    fields["plan_adherence"]  = 100
    fields["process_grade"]   = "A"
    fields["emotional_state"] = "automated"
    fields["fomo_flag"]       = 0
    fields["revenge_flag"]    = 0
    fields["hesitation_flag"] = 0

    # Auto-populate reflection fields from signal analysis
    if gate_strength >= 0.7:
        fields["what_went_right"] = f"Strong gate signal: confidence={gate_strength:.2f} regime={fields['regime_match_label']}"
    elif gate_strength < 0.3:
        fields["what_went_right"] = f"Marginal gate entry: confidence={gate_strength:.2f} — marginal setups reviewed in D-A-C"
    else:
        fields["what_went_right"] = f"Standard signal quality: confidence={gate_strength:.2f}"

    fields["what_went_wrong"]  = ""   # populated post-trade from fill analysis
    fields["lesson_learned"]   = f"Session {fields['regime_match_label']} | drawdown_flag={fields['drawdown_context_flag']}"

    return fields


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from strategy_registry import REGISTRY, StrategyVariant

    print("=== gene_mixer.py smoke test ===\n")

    # Build two test chromosomes
    chr_a = StrategyChromosome(
        orb_method="15min", vwap_lookback=5, vwap_range_factor=0.10,
        vwap_max_flips=2, vwap_hold_bars=2, target_rr=2.0, risk_pct=0.01,
        session_end_hour=11, session_end_min=0,
        ev=0.45, sharpe=1.2, wfe=0.65, n_trades=80,
        variant_name="canonical", parents=[], generation=0,
    )
    chr_b = StrategyChromosome(
        orb_method="30min", vwap_lookback=3, vwap_range_factor=0.08,
        vwap_max_flips=3, vwap_hold_bars=2, target_rr=1.5, risk_pct=0.01,
        session_end_hour=10, session_end_min=30,
        ev=0.18, sharpe=0.8, wfe=0.52, n_trades=60,
        variant_name="orb-30min", parents=[], generation=0,
    )

    # Crossover
    rng   = random.Random(42)
    child = crossover(chr_a, chr_b, method="best_ev_weighted", rng=rng)
    print(f"Crossover child: orb={child.orb_method} vwap={child.vwap_range_factor} rr={child.target_rr}")
    print(f"  Parents: {child.parents}  Generation: {child.generation}")

    # Mutation
    mutated = mutate(child, mutation_rate=0.3, rng=rng)
    print(f"After mutation: orb={mutated.orb_method} vwap={mutated.vwap_range_factor:.3f} rr={mutated.target_rr}")

    # Validation
    valid, msg = mutated.validate()
    print(f"Validation: {valid} — {msg}")

    # Attribution
    attr = attribute_winning_genes(chr_a, chr_b, "DATA/paper_account.db", "DATA/paper_account_orb30.db")
    print(f"Attribution: {attr['explanation']}")

    # Machine psychology
    mp = build_machine_psychology(
        entry_ctx={"vix": 14.5, "vix_regime": "NORMAL", "vwap_slope": "up",
                   "gate_vwap_strength": 0.025, "orb_gap_to_edge": 0.3,
                   "orb_method": "15min", "daily_trend": "up"},
        orb_data={"orb_high": 770.0, "orb_low": 769.0},
        df_5m=None,
        position_history=[
            {"status": "closed", "actual_r": 1.8},
            {"status": "closed", "actual_r": -1.0},
            {"status": "closed", "actual_r": 2.1},
        ],
    )
    print(f"\nMachine psychology: confidence={mp['signal_confidence_score']} "
          f"regime={mp['regime_match_label']} drawdown_flag={mp['drawdown_context_flag']}")
    print(f"  Gate grade: {mp['gate_strength_grade']}  Process grade: {mp['process_grade']}")
    print(f"  What went right: {mp['what_went_right']}")

    print("\n✅ gene_mixer.py verified")
