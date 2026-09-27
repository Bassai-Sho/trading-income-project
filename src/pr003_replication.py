"""
pr003_replication.py — PR-003 Stage R: replication  (Notion P2-130, frozen 27 Sep 2026)
=====================================================================================
Does box #3 (the paper's rules, published parameters) reproduce the paper's
published monthly returns on our independent data (Alpaca SIP 1-min SPY)?

Window: 2016-01 .. 2024-12 (all pre-sealed data). 2025+ is NOT read.
Runs:   FIDELITY  = paper costs ($0.0035/sh, min $0.35/order, + $0.001/sh slippage)
        VIABILITY = commission + half a 1-cent spread + $0.001 slippage per share
                    per side [ASSUMED: SPY typically quotes a 1-cent spread]
Variants: vol_target (headline, as in the paper's table) and full (1x).
PASS (fidelity, headline variant), over months inside the paper's sample
(Feb 2016 .. Apr 2024 — Jan 2016 is partial for us because 14 sessions of
history are needed first and our data starts 4 Jan 2016):
  1) monthly-return correlation with the paper's table >= 0.70
  2) our monthly Sharpe >= 0.5 x the paper's over the same months
  3) same sign in >= 6 of 8 full years 2016-2023
  4) box #3 agrees with the authors' own code on the same data
     (daily-return correlation >= 0.95, after a 20-session warm-up)
Early signal only (not a gate): May-Dec 2024, post-publication (paper: +16.9%).

    python src/pr003_replication.py
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from boxes.noise_area_momentum import NoiseAreaMomentumBox, Params   # noqa: E402
from core.backtest_runner import run_box                             # noqa: E402
from core.execution_core import AccountConfig                        # noqa: E402
from evaluation.reference_noise_area import reference_backtest       # noqa: E402

log = logging.getLogger("pr003")
WINDOW = (date(2016, 1, 1), date(2024, 12, 31))
AUM0 = 100_000.0

# Paper, FAQ Q24 (version 3 Feb 2025): monthly % returns, vol-target version, IQFeed data
PAPER = {
    2016: [-1.0, -1.9, 1.3, -1.4, 0.4, -7.3, -3.1, -5.2, 8.4, -5.0, 1.4, 0.7],
    2017: [-3.3, 0.9, 1.7, -3.3, -2.4, 0.4, -1.8, 3.6, -1.1, 2.9, -2.2, -2.3],
    2018: [3.5, 8.9, 7.2, 3.5, -4.7, 2.0, 6.2, -1.7, -3.5, 13.4, 3.0, 12.5],
    2019: [0.9, -0.7, 2.9, 0.9, -5.1, -0.5, 1.4, 6.8, -1.4, 2.6, -1.7, 1.0],
    2020: [5.3, 1.9, -0.5, -1.8, -0.2, 8.0, 0.1, -1.8, 14.3, 2.8, -1.1, -1.9],
    2021: [7.8, 3.1, 0.7, 6.1, 2.6, -1.0, 2.8, 1.7, 2.8, 3.3, -3.2, 4.1],
    2022: [-5.5, 2.8, 0.2, 9.0, 2.1, 0.5, 0.2, 6.3, -1.0, 5.8, 1.6, 0.7],
    2023: [2.9, -1.3, 7.8, 1.8, 2.9, 4.1, 2.2, 6.0, 2.9, -1.1, 0.5, 3.8],
    2024: [8.8, -1.5, -0.4, 5.8, -4.3, 1.6, 8.2, -2.8, 4.1, 6.7, -2.6, 5.7],
}
PAPER_YEARLY = {2016: -12.8, 2017: -6.9, 2018: 61.1, 2019: 6.9, 2020: 26.8,
                2021: 34.8, 2022: 24.4, 2023: 37.2, 2024: 32.2}
GATE_MONTHS = ("2016-02", "2024-04")
POST_PUB = ("2024-05", "2024-12")


def paper_monthly() -> pd.Series:
    idx, vals = [], []
    for y, ms in PAPER.items():
        for m, v in enumerate(ms, 1):
            idx.append(f"{y}-{m:02d}"); vals.append(v / 100)
    return pd.Series(vals, index=idx)


def fee_fn(kind: str):
    def fidelity(sym, side, qty, px, ts):
        return max(0.35, 0.0035 * qty) + 0.001 * qty
    def viability(sym, side, qty, px, ts):
        return max(0.35, 0.0035 * qty) + (0.005 + 0.001) * qty
    return {"fidelity": fidelity, "viability": viability}[kind]


def load_spy(market_db: str) -> pd.DataFrame:
    from market_data_store import MarketDataStore
    df = MarketDataStore(market_db).get_bars_range("SPY", *WINDOW)
    df = df.rename(columns={c: c.capitalize() for c in df.columns
                            if c in ("open", "high", "low", "close", "volume")})
    return df[["Open", "High", "Low", "Close", "Volume"]]


def load_dividends(cache: Path = Path("DATA/spy_dividends.csv")) -> dict[str, float]:
    if not cache.exists():
        import yfinance as yf
        s = yf.Ticker("SPY").dividends
        cache.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"date": [str(d.date()) for d in s.index], "dividend": s.values}).to_csv(cache, index=False)
    d = pd.read_csv(cache)
    return {r.date: float(r.dividend) for r in d.itertuples()
            if str(WINDOW[0]) <= r.date <= str(WINDOW[1])}


def daily_returns(eq: dict, aum0: float = AUM0) -> pd.Series:
    s = pd.Series(eq).sort_index()
    prev = s.shift(1).fillna(aum0)
    return (s / prev - 1).rename("ret")


def monthly(r: pd.Series) -> pd.Series:
    r = r.copy(); r.index = pd.to_datetime(r.index)
    m = (1 + r).groupby(r.index.to_period("M")).prod() - 1
    m.index = m.index.astype(str)
    return m


def sharpe_m(m: pd.Series) -> float:
    return float(m.mean() / m.std(ddof=1) * np.sqrt(12)) if len(m) > 2 and m.std(ddof=1) > 0 else float("nan")


def stats_daily(r: pd.Series) -> dict:
    eq = (1 + r).cumprod()
    return {"daily_sharpe": float(r.mean() / r.std(ddof=1) * np.sqrt(252)),
            "ann_vol": float(r.std(ddof=1) * np.sqrt(252)),
            "cagr": float(eq.iloc[-1] ** (252 / len(r)) - 1),
            "max_dd": float((eq / eq.cummax() - 1).min())}


def evaluate(ours_m: pd.Series, paper_m: pd.Series) -> dict:
    a, b = GATE_MONTHS
    both = pd.concat([ours_m.rename("ours"), paper_m.rename("paper")], axis=1).loc[a:b].dropna()
    corr = float(both["ours"].corr(both["paper"]))
    s_o, s_p = sharpe_m(both["ours"]), sharpe_m(both["paper"])
    yrs = {}
    for y in range(2016, 2024):
        o = float((1 + ours_m[[k for k in ours_m.index if k.startswith(str(y))]]).prod() - 1)
        yrs[y] = {"ours": o, "paper": PAPER_YEARLY[y] / 100, "same_sign": (o > 0) == (PAPER_YEARLY[y] > 0)}
    same = sum(v["same_sign"] for v in yrs.values())
    pp = ours_m.loc[POST_PUB[0]:POST_PUB[1]]
    return {"months_compared": len(both), "monthly_corr": corr, "sharpe_ours": s_o,
            "sharpe_paper": s_p, "years": yrs, "years_same_sign": same,
            "post_pub_ours": float((1 + pp).prod() - 1), "post_pub_paper": 0.169,
            "gates": {"corr": corr >= 0.70, "sharpe": s_o >= 0.5 * s_p, "years": same >= 6}}


def run_variant(df, divs, sizing, costs):
    p = Params(sizing=sizing, dividends=divs)
    res = run_box(NoiseAreaMomentumBox(), p, df, "SPY",
                  AccountConfig(cash=AUM0, leverage=None), fee_fn=fee_fn(costs))
    return daily_returns(res.equity_by_day)


def run(market_db: str, out_dir: Path = Path("DATA/pr003")) -> dict:
    df = load_spy(market_db)
    assert df.index.max().date() <= WINDOW[1], "sealed window must not be read"
    divs = load_dividends()
    pm = paper_monthly()
    report = {"run_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
              "window": [str(WINDOW[0]), str(WINDOW[1])], "dividends": len(divs), "runs": {}}
    for sizing in ("vol_target", "full"):
        for costs in ("fidelity", "viability"):
            log.info("running %s / %s", sizing, costs)
            r = run_variant(df, divs, sizing, costs)
            m = monthly(r)
            entry = {"daily": stats_daily(r), "monthly": {k: float(v) for k, v in m.items()}}
            if sizing == "vol_target":
                entry["vs_paper"] = evaluate(m, pm)
            report["runs"][f"{sizing}/{costs}"] = entry
            if sizing == "vol_target" and costs == "fidelity":
                box_fid = r
    log.info("running the authors' reference code on the same data")
    ref = reference_backtest(df, divs).dropna()
    # compare after a 20-session warm-up: before its vol estimate exists, the
    # reference code sizes at max leverage by design, box #3 waits (documented quirk)
    j = pd.concat([box_fid.rename("box"), ref.rename("ref")], axis=1).dropna().iloc[20:]
    j = j[(j["box"] != 0) | (j["ref"] != 0)]
    report["reference"] = {"days": len(j), "daily_corr": float(j["box"].corr(j["ref"])),
                           "mean_abs_diff_bps": float((j["box"] - j["ref"]).abs().mean() * 1e4),
                           "ref_daily": stats_daily(ref), "agree": float(j["box"].corr(j["ref"])) >= 0.95}
    g = report["runs"]["vol_target/fidelity"]["vs_paper"]["gates"]
    report["stage_r_pass"] = all(g.values()) and report["reference"]["agree"]
    try:
        report["commit"] = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                                          text=True, timeout=5).stdout.strip()
    except Exception:
        report["commit"] = None
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"stage_r_{report['run_at'].replace(':', '')}.json"
    path.write_text(json.dumps(report, indent=2, default=str))
    report["path"] = str(path)
    return report


def show(r: dict) -> None:
    f = lambda x, s="{:+.2f}": "—" if x is None else s.format(x)
    print(f"\n=== PR-003 Stage R: replication of 'Beat the Market' on Alpaca SPY 2016-2024 (commit {r.get('commit')}) ===")
    print(f"{'run':<24}{'Sharpe':>8}{'vol':>8}{'CAGR':>9}{'maxDD':>9}")
    for k, v in r["runs"].items():
        d = v["daily"]
        print(f"{k:<24}{d['daily_sharpe']:>8.2f}{d['ann_vol']:>8.1%}{d['cagr']:>+9.1%}{d['max_dd']:>+9.1%}")
    e = r["runs"]["vol_target/fidelity"]["vs_paper"]
    print(f"\nvs paper (fidelity, vol-target), {e['months_compared']} months {GATE_MONTHS[0]}..{GATE_MONTHS[1]}:")
    print(f"  monthly correlation {e['monthly_corr']:.2f}  (gate >= 0.70)")
    print(f"  monthly Sharpe ours {e['sharpe_ours']:.2f} vs paper {e['sharpe_paper']:.2f}  (gate >= half)")
    print("  years  " + "  ".join(f"{y}: {v['ours']:+.0%}/{v['paper']:+.0%}{'' if v['same_sign'] else '*'}"
                              for y, v in e["years"].items()) + f"   same sign {e['years_same_sign']}/8 (gate >= 6)")
    print(f"  post-publication May-Dec 2024 (early signal only): ours {e['post_pub_ours']:+.1%} vs paper +16.9%")
    v = r["runs"]["vol_target/viability"]["vs_paper"]
    print(f"  viability costs: monthly corr {v['monthly_corr']:.2f}, Sharpe {v['sharpe_ours']:.2f}")
    x = r["reference"]
    print(f"\nauthors' code on our data: daily corr with box #3 {x['daily_corr']:.3f} (gate >= 0.95), "
          f"mean |diff| {x['mean_abs_diff_bps']:.1f} bps; reference Sharpe {x['ref_daily']['daily_sharpe']:.2f}")
    print(f"\nGATES {e['gates']} + reference agree={x['agree']}")
    print(f"STAGE R: {'PASS — replication succeeds' if r['stage_r_pass'] else 'FAIL — see gates'}")
    print(f"Saved: {r['path']}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    for noisy in ("market_data_store", "execution_core"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    a = argparse.ArgumentParser()
    a.add_argument("--db", default="DATA/market_data.db")
    args = a.parse_args()
    show(run(args.db))
