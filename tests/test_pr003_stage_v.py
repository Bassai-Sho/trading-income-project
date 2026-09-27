"""PR-003 Stage V: MES cost conversion, exposure null-alpha test, correlation
guardrail, and the end-to-end wiring, on small/synthetic data."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluation.mes_costs import SHARES_PER_CONTRACT, breakeven_cost_per_contract_side, mes_fee_fn
from evaluation.null_exposure import null_alpha_exposure
from market_data_store import session_close
from synthetic_bars import noisy_days


def test_mes_fee_scales_with_contracts_not_notional():
    fee = mes_fee_fn(cost_per_contract_side=1.0)
    assert fee("SPY", "BUY", 50, 400.0, None) == pytest.approx(1.0)     # 1 contract
    assert fee("SPY", "BUY", 500, 100.0, None) == pytest.approx(10.0)   # 10 contracts, cheaper shares
    assert fee("SPY", "BUY", 25, 999.0, None) == pytest.approx(0.5)     # fractional contract, same rate


def test_breakeven_is_exact():
    rng = np.random.default_rng(0)
    gross = rng.normal(0.0003, 0.01, 500)
    cost = np.full(500, 0.0001)
    be = breakeven_cost_per_contract_side(gross.sum(), cost.sum(), current_rate=1.13)
    from evaluation.mes_costs import mes_fee_fn as _
    scaled_cost = cost * (be / 1.13)
    assert (gross - scaled_cost).sum() == pytest.approx(0, abs=1e-9)


# ── exposure null ─────────────────────────────────────────────────────────────

def test_null_exposure_direction_with_real_edge_passes():
    idx = pd.bdate_range("2019-01-02", periods=200)
    rng = np.random.default_rng(1)
    gross = pd.Series(rng.normal(0.001, 0.005, 200), index=idx)        # genuine positive drift
    cost = pd.Series(0.0002, index=idx)
    r = null_alpha_exposure(gross, cost, draws=500)
    assert r["passed"] and r["p_value"] < 0.05


def test_null_exposure_pure_noise_fails():
    idx = pd.bdate_range("2019-01-02", periods=200)
    rng = np.random.default_rng(2)
    gross = pd.Series(rng.normal(0.0, 0.005, 200), index=idx)          # no drift: direction is a coin flip
    cost = pd.Series(0.0002, index=idx)
    r = null_alpha_exposure(gross, cost, draws=500)
    assert not r["passed"] and r["p_value"] > 0.05


def test_null_exposure_cost_is_paid_regardless_of_direction_guessed():
    """A day with a real edge but tiny gross moves is dominated by cost either
    way, so an aggressive cost should push the p-value up (harder to pass)."""
    idx = pd.bdate_range("2019-01-02", periods=200)
    rng = np.random.default_rng(3)
    gross = pd.Series(rng.normal(0.0005, 0.003, 200), index=idx)
    cheap = null_alpha_exposure(gross, pd.Series(0.00001, index=idx), draws=500, seed=9)
    costly = null_alpha_exposure(gross, pd.Series(0.002, index=idx), draws=500, seed=9)
    assert costly["p_value"] >= cheap["p_value"]


def test_null_exposure_is_deterministic_given_a_seed():
    idx = pd.bdate_range("2019-01-02", periods=50)
    gross = pd.Series(0.0005, index=idx); cost = pd.Series(0.0001, index=idx)
    a = null_alpha_exposure(gross, cost, draws=300, seed=7)
    b = null_alpha_exposure(gross, cost, draws=300, seed=7)
    assert a == b


# ── end-to-end wiring (small synthetic data, patched cost/window) ──────────────

import pr003_stage_v as sv                                            # noqa: E402


def test_stage_v_end_to_end(tmp_path, monkeypatch):
    df = noisy_days("2019-01-02", "2019-09-30", seed=7, px=280.0)
    df = df[[session_close(d) is not None for d in df.index.date]]
    monkeypatch.setattr(sv.pr, "load_spy", lambda db: df)
    monkeypatch.setattr(sv.pr, "load_dividends", lambda: {})
    out = sv.run("unused.db", out_dir=tmp_path)
    assert set(out["gates"]) == {"null_alpha", "correlation", "expectancy"}
    assert -1.0 <= out["correlation_with_spy_buy_hold"] <= 1.0
    assert Path(out["path"]).exists()
    import json
    saved = json.loads(Path(out["path"]).read_text())
    assert saved["variant"] == list(sv.FINAL_VARIANT) or tuple(saved["variant"]) == sv.FINAL_VARIANT


def test_correlation_gate_actually_fails_when_correlation_is_high(monkeypatch, tmp_path):
    """Force net returns to equal SPY's own returns exactly (corr = 1.0):
    the correlation gate must fail. Caught a real gap: the end-to-end test
    above never exercised a high-correlation case, so a mutation that
    hard-coded the gate to True went undetected until this was added."""
    df = noisy_days("2019-01-02", "2019-09-30", seed=7, px=280.0)
    df = df[[session_close(d) is not None for d in df.index.date]]
    monkeypatch.setattr(sv.pr, "load_spy", lambda db: df)
    monkeypatch.setattr(sv.pr, "load_dividends", lambda: {})
    spy_r = sv.spy_buy_hold_daily(df)
    monkeypatch.setattr(sv.rf, "run_variant_full", lambda *a, **k: (
        spy_r, pd.Series(0.0001, index=spy_r.index), spy_r + 0.0001, None))
    out = sv.run("unused.db", out_dir=tmp_path)
    assert out["correlation_with_spy_buy_hold"] > 0.99
    assert not out["gates"]["correlation"] and not out["stage_v_pass"]


def test_stage_v_never_reads_sealed_data():
    assert sv.pr.WINDOW[1].year == 2024 and sv.pr.WINDOW[1].month == 12
