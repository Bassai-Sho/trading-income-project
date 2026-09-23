"""
pr001_instrument_test.py
========================
Runner for pre-registration PR-001 (Notion Action Item P2-121, frozen 23 Sep 2026).

    Stage 1 (in-sample 2016-2022, repeatable):
        python src/pr001_instrument_test.py --stage 1
    Stage 2 (out-of-sample 2023-2024, ONE SHOT, Stage-1 passers only):
        python src/pr001_instrument_test.py --stage 2 --confirm-one-shot

The strategy is NOT reimplemented here: every trade comes from
trading_engine._backtest_orb_full_gate() with the frozen engine CONFIG. This
file only (a) selects quality_ok sessions from market_data.db, (b) resamples to
5-minute bars, (c) supplies a per-ticker, per-quarter average daily volume to
the cost model, and (d) applies PR-001's pre-registered pass criteria.

Deliberately separate from historical_sim.run_simulation(), which also writes
sim_trades, refits markov_state.json and seeds learning components — side
effects a validation test must not have.

Results: DATA/pr001/stage1_<utc>.json and DATA/pr001/stage2.json (the latter
doubles as the one-shot lock: Stage 2 refuses to run if it exists).
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import statistics
import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trading_engine import _backtest_orb_full_gate, CONFIG as ENGINE_CONFIG  # noqa: E402
from historical_sim import _to_5min                                          # noqa: E402
from market_data_store import MarketDataStore, REGIMES                       # noqa: E402
import trading_quant_toolkit_v2_4 as tk                                      # noqa: E402

log = logging.getLogger("pr001")

# ── Frozen specification (P2-121). Changing anything here = PR-002. ─────────
SPEC = {
    "id": "PR-001",
    "frozen": "2026-09-23",
    "frozen_commit": "550ff4f",
    "tickers": ["SPY", "QQQ", "NVDA", "TSLA"],
    "is_window":  ("2016-01-01", "2022-12-31"),
    "oos_window": ("2023-01-01", "2024-12-31"),
    "engine": {"orb_method": "15min", "session_start": "09:30", "session_end": "11:00",
               "vwap_lookback": 3, "use_vwap_trailing": True},
    "exit_mode": "baseline",
    "risk_pct": 0.01,
    "account": 10_000.0,
    "dsr_n_trials": 10,
    "stage1": {"min_trades": 100, "min_expectancy": 0.0, "min_dsr": 0.95,
               "min_positive_regimes": 3},
    "stage2": {"min_wfe": 0.50, "min_expectancy": 0.0, "min_trades": 30},
}
IS_REGIMES = ["normal_bull", "covid_crash", "covid_recovery", "rate_hike_bear"]
OUT_DIR = Path("DATA/pr001")


class SpecViolation(RuntimeError):
    pass


def assert_frozen(cfg: dict) -> None:
    """Refuse to run if the engine configuration has drifted from the spec."""
    e = SPEC["engine"]
    actual = {
        "orb_method": cfg["orb_method"],
        "session_start": cfg["session_start"].strftime("%H:%M"),
        "session_end": cfg["session_end"].strftime("%H:%M"),
        "vwap_lookback": cfg.get("vwap_lookback", 3),
        "use_vwap_trailing": cfg.get("use_vwap_trailing", True),
    }
    diff = {k: (e[k], actual[k]) for k in e if e[k] != actual[k]}
    if diff:
        raise SpecViolation(f"Engine CONFIG differs from PR-001 (spec, actual): {diff}")


# ── Data ─────────────────────────────────────────────────────────────────────

def load_bars(store: MarketDataStore, ticker: str, start: str, end: str) -> tuple[pd.DataFrame, dict]:
    """1-min bars for quality_ok sessions only, capitalised; plus exclusion stats."""
    s, e = date.fromisoformat(start), date.fromisoformat(end)
    all_days  = store.get_date_range(ticker, s, e)
    good_days = set(store.get_date_range(ticker, s, e, quality_ok_only=True))
    df = store.get_bars_range(ticker, s, e)
    if df.empty:
        return df, {"sessions": 0, "quality_ok": 0, "excluded": 0}
    df = df[[d.isoformat() in good_days for d in df.index.date]]
    df = df.rename(columns={c: c.capitalize() for c in df.columns
                            if c in ("open", "high", "low", "close", "volume")})
    return df, {"sessions": len(all_days), "quality_ok": len(good_days),
                "excluded": len(all_days) - len(good_days)}


def quarterly_adv(df_1m: pd.DataFrame) -> dict[str, float]:
    """Median daily share volume per calendar quarter ('2019Q3' -> shares)."""
    daily = df_1m["Volume"].groupby(df_1m.index.date).sum()
    q = pd.PeriodIndex(pd.to_datetime(daily.index), freq="Q").astype(str)
    return daily.groupby(q).median().to_dict()


def run_trades(df_1m: pd.DataFrame, ticker: str) -> list[dict]:
    """Canonical backtest, one quarter at a time with that quarter's ADV.
    Sessions are independent inside the backtest (no position carries over
    days), so chunking changes only the cost model's ADV input."""
    adv = quarterly_adv(df_1m)
    df5 = _to_5min(df_1m)
    quarters = pd.PeriodIndex(df5.index.tz_localize(None), freq="Q").astype(str)
    trades: list[dict] = []
    for qtr in sorted(set(quarters)):
        chunk = df5[quarters == qtr]
        cfg = {**ENGINE_CONFIG, "ticker": ticker, "orb_method": SPEC["engine"]["orb_method"],
               "account_balance": SPEC["account"], "risk_pct": SPEC["risk_pct"],
               "avg_daily_volume": float(adv.get(qtr, ENGINE_CONFIG.get("avg_daily_volume", 150e6)))}
        res = _backtest_orb_full_gate(chunk, cfg, exit_mode=SPEC["exit_mode"])
        for t in res.get("trades", []):
            t["adv_used"] = cfg["avg_daily_volume"]
            t["chunk_avg_cost_r"] = res.get("avg_cost_r")   # per-quarter mean cost
        trades += res.get("trades", [])
    return trades


# ── Statistics ───────────────────────────────────────────────────────────────

def stats(rs: list[float], n_trials: int) -> dict:
    n = len(rs)
    out = {"n": n, "expectancy": None, "sharpe": None, "skew": None,
           "kurtosis": None, "dsr": None, "win_rate": None}
    if n == 0:
        return out
    mu = statistics.mean(rs)
    out["expectancy"] = mu
    out["win_rate"] = sum(r > 0 for r in rs) / n
    if n >= 3:
        sd = statistics.stdev(rs)
        if sd > 0:
            sr = mu / sd
            m2 = sum((r - mu) ** 2 for r in rs) / n
            skew = (sum((r - mu) ** 3 for r in rs) / n) / m2 ** 1.5
            kurt = (sum((r - mu) ** 4 for r in rs) / n) / m2 ** 2   # non-excess (normal = 3)
            out.update(sharpe=sr, skew=skew, kurtosis=kurt)
            try:
                out["dsr"] = tk.deflated_sharpe_ratio(sr, n, n_trials, skew=skew, kurtosis=kurt)
            except ValueError as e:
                out["dsr_error"] = str(e)
    return out


def regime_breakdown(trades: list[dict]) -> dict:
    res = {}
    for key in IS_REGIMES:
        r = REGIMES[key]
        rs = [t["actual_r"] for t in trades
              if str(r["start"]) <= t["session_date"] <= str(r["end"])]
        res[key] = {"label": r["label"], "n": len(rs),
                    "expectancy": statistics.mean(rs) if rs else None}
    return res


def stage1_verdict(st: dict, regimes: dict) -> dict:
    c = SPEC["stage1"]
    pos = sum(1 for v in regimes.values() if v["expectancy"] is not None and v["expectancy"] > 0)
    checks = {
        "min_trades":       st["n"] >= c["min_trades"],
        "expectancy":       st["expectancy"] is not None and st["expectancy"] > c["min_expectancy"],
        "dsr":              st["dsr"] is not None and st["dsr"] >= c["min_dsr"],
        "regimes_positive": pos >= c["min_positive_regimes"],
    }
    return {"checks": checks, "positive_regimes": pos, "pass": all(checks.values())}


def stage2_verdict(oos: dict, is_sharpe: float | None) -> dict:
    c = SPEC["stage2"]
    wfe = (oos["sharpe"] / is_sharpe) if (oos["sharpe"] is not None and is_sharpe and is_sharpe > 0) else None
    checks = {
        "min_trades": oos["n"] >= c["min_trades"],
        "expectancy": oos["expectancy"] is not None and oos["expectancy"] > c["min_expectancy"],
        "wfe":        wfe is not None and wfe >= c["min_wfe"],
    }
    return {"wfe": wfe, "checks": checks, "pass": all(checks.values())}


# ── Stages ───────────────────────────────────────────────────────────────────

def _commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, timeout=5).stdout.strip() or None
    except Exception:
        return None


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_stage1(store: MarketDataStore, out_dir: Path = OUT_DIR) -> dict:
    assert_frozen(ENGINE_CONFIG)
    start, end = SPEC["is_window"]
    report = {"spec": SPEC, "stage": 1, "run_at": _now(), "commit": _commit(), "tickers": {}}
    for tkr in SPEC["tickers"]:
        df, q = load_bars(store, tkr, start, end)
        trades = run_trades(df, tkr) if not df.empty else []
        st = stats([t["actual_r"] for t in trades], SPEC["dsr_n_trials"])
        rg = regime_breakdown(trades)
        report["tickers"][tkr] = {
            "sessions": q, "stats": st, "regimes": rg,
            "unassigned_regime_trades": st["n"] - sum(v["n"] for v in rg.values()),
            "avg_cost_r": (statistics.mean(t["chunk_avg_cost_r"] for t in trades)
                           if trades else None),
            "verdict": stage1_verdict(st, rg),
        }
        log.info("%s done: n=%d", tkr, st["n"])
    report["stage1_passers"] = [t for t, v in report["tickers"].items() if v["verdict"]["pass"]]
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"stage1_{report['run_at'].replace(':', '')}.json"
    path.write_text(json.dumps(report, indent=2, default=str))
    report["path"] = str(path)
    return report


def run_stage2(store: MarketDataStore, stage1_path: Path, confirm: bool,
               out_dir: Path = OUT_DIR) -> dict:
    lock = out_dir / "stage2.json"
    if lock.exists():
        raise SpecViolation(f"Stage 2 has already been run ({lock}). PR-001 allows ONE "
                            "out-of-sample run; a new test needs PR-002.")
    if not confirm:
        raise SpecViolation("Stage 2 uses the one-shot 2023-2024 window. Re-run with "
                            "--confirm-one-shot to proceed.")
    assert_frozen(ENGINE_CONFIG)
    s1 = json.loads(Path(stage1_path).read_text())
    if s1.get("spec") != json.loads(json.dumps(SPEC, default=str)):
        raise SpecViolation("Stage 1 results were produced under a different spec.")
    passers = s1["stage1_passers"]
    report = {"spec": SPEC, "stage": 2, "run_at": _now(), "commit": _commit(),
              "stage1_file": str(stage1_path), "tickers": {}}
    out_dir.mkdir(parents=True, exist_ok=True)
    if not passers:
        report["note"] = "No Stage-1 passers: OOS window not touched."
        lock.write_text(json.dumps(report, indent=2, default=str))
        return report
    lock.write_text(json.dumps({**report, "status": "started"}, indent=2))   # lock first
    start, end = SPEC["oos_window"]
    for tkr in passers:
        df, q = load_bars(store, tkr, start, end)
        trades = run_trades(df, tkr) if not df.empty else []
        oos = stats([t["actual_r"] for t in trades], SPEC["dsr_n_trials"])
        is_sharpe = s1["tickers"][tkr]["stats"]["sharpe"]
        report["tickers"][tkr] = {"sessions": q, "oos": oos, "is_sharpe": is_sharpe,
                                  "verdict": stage2_verdict(oos, is_sharpe)}
    report["status"] = "complete"
    lock.write_text(json.dumps(report, indent=2, default=str))
    return report


# ── Printing ─────────────────────────────────────────────────────────────────

def _f(x, fmt="{:+.3f}"):
    return "—" if x is None else fmt.format(x)


def print_stage1(r: dict) -> None:
    print(f"\n=== PR-001 STAGE 1 (IS {SPEC['is_window'][0]}..{SPEC['is_window'][1]}) "
          f"commit={r['commit']} ===")
    print(f"{'Ticker':<6}{'sess ok/all':>13}{'n':>6}{'Exp R':>9}{'Sharpe':>9}{'DSR':>7}"
          f"{'+regimes':>9}   PASS")
    for tkr, v in r["tickers"].items():
        st, q, vd = v["stats"], v["sessions"], v["verdict"]
        print(f"{tkr:<6}{q['quality_ok']:>7}/{q['sessions']:<5}{st['n']:>6}{_f(st['expectancy']):>9}"
              f"{_f(st['sharpe']):>9}{_f(st['dsr'], '{:.3f}'):>7}{vd['positive_regimes']:>6}/4"
              f"   {'✓' if vd['pass'] else '✗'}  "
              + " ".join(k for k, ok in vd["checks"].items() if not ok))
    print("\nRegime expectancy (n):")
    for tkr, v in r["tickers"].items():
        print(f"  {tkr:<5} " + "  ".join(
            f"{g['label'].split(' ', 1)[0]}: {_f(g['expectancy'])} ({g['n']})"
            for g in v["regimes"].values())
              + f"   [{v['unassigned_regime_trades']} trades in Jan 2020, no regime]")
    print(f"\nStage-1 passers: {r['stage1_passers'] or 'none'}")
    print(f"Saved: {r['path']}")


def print_stage2(r: dict) -> None:
    print(f"\n=== PR-001 STAGE 2 (OOS {SPEC['oos_window'][0]}..{SPEC['oos_window'][1]}, one shot) ===")
    if r.get("note"):
        print(r["note"]); return
    for tkr, v in r["tickers"].items():
        o, vd = v["oos"], v["verdict"]
        print(f"{tkr}: n={o['n']} exp={_f(o['expectancy'])} sharpe={_f(o['sharpe'])} "
              f"WFE={_f(vd['wfe'], '{:.2f}')}  {'PASS' if vd['pass'] else 'FAIL'}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")
    for noisy in ("trading_engine", "historical_sim", "market_data_store"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    p = argparse.ArgumentParser(description="PR-001 pre-registered instrument test")
    p.add_argument("--stage", type=int, choices=[1, 2], required=True)
    p.add_argument("--db", default="DATA/market_data.db")
    p.add_argument("--stage1-file", help="Stage 1 JSON to use for Stage 2 (default: latest)")
    p.add_argument("--confirm-one-shot", action="store_true")
    a = p.parse_args()
    store = MarketDataStore(a.db)
    try:
        if a.stage == 1:
            print_stage1(run_stage1(store))
        else:
            f = a.stage1_file or (sorted(OUT_DIR.glob("stage1_*.json")) or [None])[-1]
            if f is None:
                sys.exit("No Stage 1 results found — run --stage 1 first.")
            print_stage2(run_stage2(store, Path(f), a.confirm_one_shot))
    except SpecViolation as e:
        sys.exit(f"✗ {e}")
