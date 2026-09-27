"""PR-003 Stage REFINE: grid wiring, cost-fraction computation, and the
adoption rule, on small synthetic data (fast — the real 48-variant grid runs
on the NUC)."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import json
from dataclasses import dataclass, field

import pr003_refine as rf
from evaluation.robustness import Grid
from market_data_store import session_close
from synthetic_bars import noisy_days


@pytest.fixture(scope="module")
def data():
    df = noisy_days("2019-01-02", "2019-09-30", seed=7, px=280.0)
    return df[[session_close(d) is not None for d in df.index.date]]


def test_run_variant_returns_aligned_series(data):
    r, c = rf.run_variant(data, {}, 14, 1.0, 30)
    assert (r.index == c.index).all() and len(r) == len(set(data.index.date))
    assert c.min() >= 0        # cost fraction is never negative


def test_grid_shape_and_published_point_present():
    assert len(rf.GRID.keys()) == 48
    assert rf.PUBLISHED in rf.GRID.keys()
    assert rf.GRID.axes["lookback"][1] == 14 and rf.GRID.axes["vm"][1] == 1.0 \
        and rf.GRID.axes["trade_every_min"][1] == 30


def test_adoption_rule_rejects_when_any_condition_fails(monkeypatch, tmp_path):
    """A tiny 3x1x1 grid (one real axis) with a fabricated report: adoption
    must require ALL FOUR conditions, not just beating the published mean."""
    small = Grid({"lookback": (7, 14, 28), "vm": (1.0,), "trade_every_min": (30,)})
    monkeypatch.setattr(rf, "GRID", small)
    monkeypatch.setattr(rf, "PUBLISHED", (14, 1.0, 30))

    @dataclass
    class FakeRep:
        best: tuple = (7, 1.0, 30)
        best_mean_daily_r: float = 0.001
        gates: dict = field(default_factory=lambda: {"dsr": True, "pbo": True, "walk_forward": True,
                                                      "plateau": True, "cost_stress": True})

    def fake_evaluate_grid(grid, daily_r, daily_cost, prior_trials=0):
        return FakeRep(gates=dict(FakeRep().gates, **getattr(fake_evaluate_grid, "_gates", {})))

    monkeypatch.setattr(rf, "evaluate_grid", fake_evaluate_grid)

    def fake_run_variant(df, divs, lb, vm, iv):
        idx = pd.bdate_range("2019-01-02", periods=30)
        base = 0.0005 if lb == 7 else 0.0002          # published (14) < best (7)
        return pd.Series(base, index=idx), pd.Series(0.0001, index=idx)

    monkeypatch.setattr(rf, "run_variant", fake_run_variant)
    monkeypatch.setattr(rf.pr, "load_spy", lambda db: pd.DataFrame(
        {"Open": [1.0], "High": [1.0], "Low": [1.0], "Close": [1.0], "Volume": [1.0]},
        index=pd.DatetimeIndex([rf.pr.WINDOW[0]])))
    monkeypatch.setattr(rf.pr, "load_dividends", lambda: {})
    out = rf.run("unused.db", out_dir=tmp_path)
    assert out["adopt_refinement"] and out["final_variant"] == (7, 1.0, 30)

    for flag in ("plateau", "pbo", "cost_stress"):
        fake_evaluate_grid._gates = {flag: False}
        out = rf.run("unused.db", out_dir=tmp_path)
        assert not out["adopt_refinement"] and out["final_variant"] == (14, 1.0, 30), flag


def test_adoption_rule_requires_beating_the_published_variant(monkeypatch, tmp_path):
    """All four gates pass, but the grid's 'best' variant does not actually
    beat the published mean — must still reject (caught a real gap: an
    earlier version of this test never covered this case)."""
    small = Grid({"lookback": (7, 14, 28), "vm": (1.0,), "trade_every_min": (30,)})
    monkeypatch.setattr(rf, "GRID", small)
    monkeypatch.setattr(rf, "PUBLISHED", (14, 1.0, 30))

    @dataclass
    class FakeRep:
        best: tuple = (7, 1.0, 30)
        best_mean_daily_r: float = 0.0001                 # WORSE than published below
        gates: dict = field(default_factory=lambda: {"dsr": True, "pbo": True,
                                                      "walk_forward": True, "plateau": True,
                                                      "cost_stress": True})

    monkeypatch.setattr(rf, "evaluate_grid", lambda *a, **k: FakeRep())

    def fake_run_variant(df, divs, lb, vm, iv):
        idx = pd.bdate_range("2019-01-02", periods=30)
        base = 0.0001 if lb == 7 else 0.0005               # published (14) > "best" (7)
        return pd.Series(base, index=idx), pd.Series(0.0001, index=idx)

    monkeypatch.setattr(rf, "run_variant", fake_run_variant)
    monkeypatch.setattr(rf.pr, "load_spy", lambda db: pd.DataFrame(
        {"Open": [1.0], "High": [1.0], "Low": [1.0], "Close": [1.0], "Volume": [1.0]},
        index=pd.DatetimeIndex([rf.pr.WINDOW[0]])))
    monkeypatch.setattr(rf.pr, "load_dividends", lambda: {})
    out = rf.run("unused.db", out_dir=tmp_path)
    assert not out["adopt_refinement"] and out["final_variant"] == (14, 1.0, 30)


def test_run_is_json_serialisable_with_tuple_keyed_fields(monkeypatch, tmp_path):
    """Reproduces the real 48-variant crash: GridReport.plateau_neighbours is
    keyed by variant tuples like (14, 1.0, 30), which json.dumps rejects as
    dict keys. The small fixture's default report never had this field
    populated, so the first version of this test suite missed it."""
    small = Grid({"lookback": (7, 14, 28), "vm": (1.0,), "trade_every_min": (30,)})
    monkeypatch.setattr(rf, "GRID", small)
    monkeypatch.setattr(rf, "PUBLISHED", (14, 1.0, 30))

    @dataclass
    class FakeRep:
        best: tuple = (14, 1.0, 30)
        best_mean_daily_r: float = 0.0002
        gates: dict = field(default_factory=lambda: {"dsr": True, "pbo": True,
                                                      "walk_forward": True, "plateau": True,
                                                      "cost_stress": True})
        plateau_neighbours: dict = field(default_factory=lambda: {(7, 1.0, 30): 0.0001,
                                                                   (28, 1.0, 30): 0.0001})

    monkeypatch.setattr(rf, "evaluate_grid", lambda *a, **k: FakeRep())

    def fake_run_variant(df, divs, lb, vm, iv):
        idx = pd.bdate_range("2019-01-02", periods=30)
        return pd.Series(0.0002, index=idx), pd.Series(0.0001, index=idx)

    monkeypatch.setattr(rf, "run_variant", fake_run_variant)
    monkeypatch.setattr(rf.pr, "load_spy", lambda db: pd.DataFrame(
        {"Open": [1.0], "High": [1.0], "Low": [1.0], "Close": [1.0], "Volume": [1.0]},
        index=pd.DatetimeIndex([rf.pr.WINDOW[0]])))
    monkeypatch.setattr(rf.pr, "load_dividends", lambda: {})
    out = rf.run("unused.db", out_dir=tmp_path)                 # must not raise
    saved = json.loads(Path(out["path"]).read_text())
    assert saved["report"]["plateau_neighbours"] == {"(7, 1.0, 30)": 0.0001, "(28, 1.0, 30)": 0.0001}


def test_never_reads_sealed_data():
    # refine.run asserts on pr.WINDOW, the same sealed boundary as replication
    assert rf.pr.WINDOW == (rf.pr.WINDOW[0], rf.pr.WINDOW[1]) and rf.pr.WINDOW[1].year == 2024 \
        and rf.pr.WINDOW[1].month == 12
