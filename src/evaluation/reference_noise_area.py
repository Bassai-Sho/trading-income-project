"""
evaluation/reference_noise_area.py  (PR-003 cross-check)
========================================================
A faithful port of the AUTHORS' OWN published Python backtest for "Beat the
Market" (Concretum Group, 'Backtesting 2 Years of FREE Data Using Python',
2024), which they state reproduces their MATLAB logic. It is NOT our strategy
implementation — box #3 is. It exists so that box #3 can be checked against
an independent implementation on the SAME data (PR-003, Stage R).

Deliberately kept quirks of the reference code (so differences can be named):
  * daily-vol window = returns of days d-15 .. d-2 (skips yesterday's return);
  * sigma needs >= 13 of 14 days (min_periods=13);
  * P&L = exposure x close-to-close changes (entry/exit at the check bar's close);
  * commission per exposure change = max($0.35, $0.0035 x shares); NO slippage;
  * if daily vol is unknown, the maximum leverage is used.
Input: 1-min bars 09:30-15:59, capitalised OHLCV, tz-aware New York index.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def reference_backtest(df_1m: pd.DataFrame, dividends: dict[str, float] | None = None,
                       aum0: float = 100_000.0, commission: float = 0.0035,
                       min_comm: float = 0.35, band_mult: float = 1.0, trade_freq: int = 30,
                       target_vol: float = 0.02, max_leverage: float = 4.0,
                       sizing: str = "vol_target") -> pd.Series:
    """Daily strategy returns indexed by session date (first days are NaN)."""
    dividends = dividends or {}
    df = pd.DataFrame({"open": df_1m["Open"].to_numpy(), "high": df_1m["High"].to_numpy(),
                       "low": df_1m["Low"].to_numpy(), "close": df_1m["Close"].to_numpy(),
                       "volume": df_1m["Volume"].to_numpy()},
                      index=df_1m.index.tz_localize(None) if df_1m.index.tz else df_1m.index)
    df["day"] = df.index.date
    groups = df.groupby("day")
    days = list(groups.groups.keys())

    df["move_open"] = np.nan
    df["vwap"] = np.nan
    df["spy_dvol"] = np.nan
    spy_ret = pd.Series(index=days, dtype=float)
    for d in range(1, len(days)):
        cur, prv = groups.get_group(days[d]), groups.get_group(days[d - 1])
        hlc = (cur["high"] + cur["low"] + cur["close"]) / 3
        df.loc[cur.index, "vwap"] = (cur["volume"] * hlc).cumsum() / cur["volume"].cumsum()
        df.loc[cur.index, "move_open"] = (cur["close"] / cur["open"].iloc[0] - 1).abs()
        spy_ret.loc[days[d]] = cur["close"].iloc[-1] / prv["close"].iloc[-1] - 1
        if d > 14:
            df.loc[cur.index, "spy_dvol"] = spy_ret.iloc[d - 15:d - 1].std(skipna=False)

    mins = (df.index - df.index.normalize()) / pd.Timedelta(minutes=1)
    df["min_from_open"] = np.asarray(mins) - (9 * 60 + 30) + 1
    df["minute_of_day"] = df["min_from_open"].round().astype(int)
    mg = df.groupby("minute_of_day")
    df["move_open_rolling_mean"] = mg["move_open"].transform(
        lambda x: x.rolling(window=14, min_periods=13).mean())
    df["sigma_open"] = df.groupby("minute_of_day")["move_open_rolling_mean"].transform(
        lambda x: x.shift(1))
    groups = df.groupby("day")

    out = pd.Series(index=days, dtype=float)
    aum = aum0
    for d in range(1, len(days)):
        cur, prv = groups.get_group(days[d]), groups.get_group(days[d - 1])
        if cur["sigma_open"].isna().all():
            continue
        prev_close_adj = prv["close"].iloc[-1] - dividends.get(str(days[d]), 0.0)
        open_price = cur["open"].iloc[0]
        closes = cur["close"]
        vol = cur["spy_dvol"].iloc[0]
        ub = max(open_price, prev_close_adj) * (1 + band_mult * cur["sigma_open"])
        lb = min(open_price, prev_close_adj) * (1 - band_mult * cur["sigma_open"])
        sig = np.zeros(len(cur))
        sig[((closes > ub) & (closes > cur["vwap"])).to_numpy()] = 1
        sig[((closes < lb) & (closes < cur["vwap"])).to_numpy()] = -1
        if sizing == "full":
            shares = round(aum / open_price)
        elif math.isnan(vol):
            shares = round(aum / open_price * max_leverage)
        else:
            shares = round(aum / open_price * min(target_vol / vol, max_leverage))
        idx = np.where(cur["min_from_open"].to_numpy() % trade_freq == 0)[0]
        exp = np.full(len(cur), np.nan)
        exp[idx] = sig[idx]
        filled, last = [], np.nan
        for v in exp:
            if not np.isnan(v):
                last = v
            if last == 0:
                last = np.nan
            filled.append(last)
        exposure = pd.Series(filled, index=cur.index).shift(1).fillna(0).to_numpy()
        trades = np.sum(np.abs(np.diff(np.append(exposure, 0))))
        gross = np.nansum(exposure * closes.diff().to_numpy()) * shares
        net = gross - trades * max(min_comm, commission * shares)
        out.loc[days[d]] = net / aum
        aum += net
    return out
