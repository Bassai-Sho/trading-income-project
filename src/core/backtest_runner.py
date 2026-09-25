"""
core/backtest_runner.py  (Phase A, P2-123 — slice 2)
====================================================
Drives ANY strategy box through the execution core over historical bars.

Per trading day:
  for each box-timeframe bar B (e.g. 5-min, built from the 1-min bars):
      1. every 1-min bar inside B goes to the core; each report goes to
         box.on_execution; follow-on orders it returns are submitted with the
         fill price as after_price and matched against the rest of that same
         minute (bracket behaviour)
      2. at B's close: box.on_bar(B) -> orders become active from B's end
  at the exchange-calendar close: core.end_session() (DAY orders expire)

The runner owns no strategy logic and no fill logic.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.execution_core import AccountConfig, ExecutionCore, FeeFn      # noqa: E402
from core.interfaces import Bar, ExecutionReport                        # noqa: E402
from historical_sim import _to_5min                                     # noqa: E402

MAX_FOLLOW_ON_DEPTH = 8


@dataclass
class RunResult:
    state: Any
    reports: list[ExecutionReport] = field(default_factory=list)


def run_box(box, params, df_1m: pd.DataFrame, symbol: str,
            account: AccountConfig | None = None, fee_fn: FeeFn | None = None,
            session_close=None, keep_reports: bool = False) -> RunResult:
    """df_1m: capitalised OHLCV, tz-aware America/New_York index, bars labelled
    by start. Only regular-session bars are expected (the data store's)."""
    if box.timeframe_minutes != 5:
        raise ValueError("runner currently builds 5-minute bars only")
    core = ExecutionCore(account or AccountConfig(), fee_fn=fee_fn, session_close=session_close)
    state = box.init_state(params)
    log: list[ExecutionReport] = []
    tf = timedelta(minutes=box.timeframe_minutes)
    df5 = _to_5min(df_1m)

    def dispatch(reports: list[ExecutionReport], bar: Bar | None, depth: int = 0) -> None:
        nonlocal state
        if keep_reports:
            log.extend(reports)
        for r in reports:
            if r.status == "ACCEPTED":
                continue
            state, acts = box.on_execution(r, state, params, core.view())
            if not acts:
                continue
            after = r.fill_price if (r.status == "FILLED" and bar is not None) else None
            now = bar.timestamp if after is not None else r.timestamp
            sub = core.submit(acts, now, after_price=after)
            dispatch(sub, bar, depth + 1)
            if after is not None:
                if depth >= MAX_FOLLOW_ON_DEPTH:
                    raise RuntimeError("follow-on order loop did not settle")
                dispatch(core.process_bar(bar), bar, depth + 1)

    days_1m = {d: g for d, g in df_1m.groupby(df_1m.index.date)}
    for day, bars5 in df5.groupby(df5.index.date):
        m1 = days_1m.get(day)
        if m1 is None or m1.empty:
            continue
        ts1 = m1.index
        o, h, l, c, v = (m1[k].to_numpy(float) for k in ("Open", "High", "Low", "Close", "Volume"))
        for b_ts, b in bars5.iterrows():
            lo, hi = ts1.searchsorted(b_ts), ts1.searchsorted(b_ts + tf)
            for j in range(lo, hi):
                bar1 = Bar(symbol, ts1[j].to_pydatetime(), o[j], h[j], l[j], c[j], v[j])
                dispatch(core.process_bar(bar1), bar1)
            bar5 = Bar(symbol, b_ts.to_pydatetime(), float(b["Open"]), float(b["High"]),
                       float(b["Low"]), float(b["Close"]), float(b["Volume"]))
            state, acts = box.on_bar(bar5, state, params, core.view())
            if acts:
                dispatch(core.submit(acts, (b_ts + tf).to_pydatetime()), None)
        close_t = core.session_close_for(day)
        end = datetime.combine(day, close_t) if close_t else ts1[-1].to_pydatetime()
        if ts1.tz is not None and end.tzinfo is None:
            end = pd.Timestamp(end).tz_localize(ts1.tz).to_pydatetime()
        dispatch(core.end_session(end), None)
    return RunResult(state, log)
