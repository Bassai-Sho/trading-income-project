"""Descriptive intraday adverse-excursion check: purely a data pull, no
fitting, no gate. Tests the calculation itself, not any decision threshold."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pr003_gap_tail as gt
from market_data_store import session_close
from synthetic_bars import noisy_days


def test_worst_intraday_excursion_picks_the_larger_side():
    idx = pd.bdate_range("2019-01-02", periods=3)
    daily = pd.DataFrame({"open": [100.0, 100.0, 100.0], "high": [101.0, 100.2, 108.0],
                         "low": [98.0, 99.9, 99.0]}, index=idx)
    df_dummy = pd.DataFrame()  # unused by this function; signature kept for interface symmetry
    out = gt.worst_intraday_excursion_per_day(df_dummy, daily)
    assert out.iloc[0] == pytest.approx(0.02)     # down move (100->98) bigger than up (100->101)
    assert out.iloc[1] == pytest.approx(0.002)    # both tiny; up move (0.2%) is the larger of the two
    assert out.iloc[2] == pytest.approx(0.08)      # up move (100->108) far bigger than down


def test_leverage_scaling_is_applied_not_skipped():
    """The whole point of this check: excursion x leverage, not raw
    excursion. A day with 2x the leverage of another but identical raw
    excursion must show 2x the scaled figure."""
    idx = pd.bdate_range("2019-01-02", periods=2)
    excursion = pd.Series([0.01, 0.01], index=idx)          # identical raw excursion
    lev = pd.Series([1.0, 2.0], index=idx)                  # different leverage
    scaled = (excursion * lev).dropna()
    assert scaled.iloc[1] == pytest.approx(2 * scaled.iloc[0])


def test_gap_tail_report_actually_applies_leverage_not_just_the_concept(monkeypatch, tmp_path):
    """The prior test only proved the ARITHMETIC concept in isolation, not
    that gap_tail_report's real code path applies it -- exercise the real
    function with a forced constant leverage and confirm the reported
    percentiles scale by exactly that factor vs. leverage=1."""
    df = noisy_days("2016-01-04", "2016-12-30", seed=6, px=200.0)
    df = df[[session_close(d) is not None for d in df.index.date]]
    gt.pr.load_spy = lambda db: df
    gt.pr.load_dividends = lambda: {}
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(gt.ed, "effective_leverage", lambda daily: pd.Series(1.0, index=daily.index))
    out_1x = gt.gap_tail_report("unused.db")
    monkeypatch.setattr(gt.ed, "effective_leverage", lambda daily: pd.Series(3.0, index=daily.index))
    out_3x = gt.gap_tail_report("unused.db")
    for p in (50, 90, 99):
        assert out_3x["percentiles_of_leverage_scaled_adverse_excursion"][p] == pytest.approx(
            3 * out_1x["percentiles_of_leverage_scaled_adverse_excursion"][p], rel=1e-6)


def test_gap_tail_report_end_to_end_on_synthetic_data(monkeypatch, tmp_path):
    df = noisy_days("2016-01-04", "2017-06-30", seed=6, px=200.0)
    df = df[[session_close(d) is not None for d in df.index.date]]
    gt.pr.load_spy = lambda db: df
    gt.pr.load_dividends = lambda: {}
    monkeypatch.chdir(tmp_path)
    out = gt.gap_tail_report("unused.db")
    assert out["n_days"] > 0
    pct = out["percentiles_of_leverage_scaled_adverse_excursion"]
    assert pct[50] <= pct[90] <= pct[95] <= pct[99] <= pct[100]     # percentiles must be monotone
    assert len(out["worst_10_days_reconciliation"]) <= 10
    for row in out["worst_10_days_reconciliation"]:
        assert row["leverage"] > 0 and row["raw_excursion_pct"] >= 0
        assert row["n_trade_fills"] is None or row["n_trade_fills"] >= 0
    assert "2018_volmageddon" in out["in_sample_analogues"]
    assert "2010-05-06" in out["caveat"] and "not examined" in out["caveat"]
    assert out["n_missing_leverage_warmup"] > 0    # the 14-day lookback always excludes early days
    assert "flat_overnight_citation" in out and len(out["flat_overnight_citation"]["tests"]) == 3
    assert "metric_definition" in out and "direction-agnostic" in out["metric_definition"].lower()
    assert Path(out["path"]).exists()


def test_direction_on_day_counts_fills_honestly_and_does_not_claim_a_side(monkeypatch, tmp_path):
    """direction (long/short) is deliberately NOT reported -- two derivation
    attempts were tried and found wrong by cross-checking against real data
    (see the function's docstring). Only n_trade_fills, which IS reliably
    derivable from run_box's existing API, is returned."""
    df = noisy_days("2018-01-02", "2018-06-29", seed=8, px=250.0)
    df = df[[session_close(d) is not None for d in df.index.date]]
    days = sorted(set(df.index.date))
    target = str(days[40])
    out = gt.direction_on_day(df, {}, target, warmup_days=20)
    assert "direction" not in out
    assert out["n_trade_fills"] is not None and out["n_trade_fills"] >= 0


def test_direction_on_day_handles_a_date_not_in_the_data():
    df = noisy_days("2018-01-02", "2018-03-30", seed=8, px=250.0)
    df = df[[session_close(d) is not None for d in df.index.date]]
    out = gt.direction_on_day(df, {}, "2019-01-02")
    assert out["n_trade_fills"] is None and "not in data" in out["note"]


def test_direction_on_day_fill_count_matches_the_full_multi_year_run():
    """Cross-check n_trade_fills (the one thing this function DOES claim)
    against an independent, already-tested full continuous run -- this is
    the real correctness bar, not just 'doesn't crash'."""
    df = noisy_days("2019-01-02", "2019-06-28", seed=5, px=280.0)
    df = df[[session_close(d) is not None for d in df.index.date]]
    from boxes.noise_area_momentum import NoiseAreaMomentumBox, Params as BoxParams
    from core.backtest_runner import run_box as _run_box
    from core.execution_core import AccountConfig as _AC
    import pr003_replication as pr
    full = _run_box(NoiseAreaMomentumBox(), BoxParams(), df, "SPY", _AC(cash=100_000.0, leverage=None),
                    fee_fn=pr.fee_fn("fidelity"), keep_reports=True)
    by_day = {}
    for r in full.reports:
        if r.status == "FILLED" and r.tag == "trade":
            by_day.setdefault(pd.Timestamp(r.timestamp).date(), 0)
            by_day[pd.Timestamp(r.timestamp).date()] += 1
    sample_dates = list(by_day.keys())[:5]
    for d in sample_dates:
        out = gt.direction_on_day(df, {}, str(d), warmup_days=20)
        assert out["n_trade_fills"] == by_day[d], f"{d}: mismatch vs the full continuous run"


def test_gap_tail_never_reads_sealed_data():
    assert gt.pr.WINDOW[1].year == 2024 and gt.pr.WINDOW[1].month == 12


def test_analogue_windows_are_actually_inside_the_data_range():
    """The two named in-sample analogues must themselves fall inside
    2016-2024 -- if someone edits IN_SAMPLE_ANALOGUES to a pre-2016 date by
    mistake, this must fail loudly, not silently report n_days=0 forever."""
    for name, (start, end) in gt.IN_SAMPLE_ANALOGUES.items():
        assert pd.Timestamp(start) >= pd.Timestamp(gt.pr.WINDOW[0]), name
        assert pd.Timestamp(end) <= pd.Timestamp(gt.pr.WINDOW[1]), name
