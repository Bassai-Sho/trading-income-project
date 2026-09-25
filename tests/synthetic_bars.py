"""Noisy synthetic 1-min sessions: fat-tailed returns, 8% of minutes gap from the prior close, random wicks and volume. For parity tests (P2-123)."""
import numpy as np, pandas as pd
def noisy_days(start, end, seed=0, px=100.0):
    rng = np.random.default_rng(seed); frames = []
    for d in pd.bdate_range(start, end):
        idx = pd.date_range(f"{d.date()} 09:30", f"{d.date()} 15:59", freq="1min", tz="America/New_York")
        n = len(idx); drift = rng.normal(0, 0.0004) * px
        ret = rng.standard_t(3, n) * px * 0.0008 + drift * (np.arange(n) < 120)
        c = px + np.cumsum(ret)
        gap = np.where(rng.random(n) < 0.08, rng.normal(0, px * 0.0015, n), 0.0)   # 8% of minutes open away from prior close
        o = np.r_[c[0] + rng.normal(0, px*0.002), c[:-1]] + gap
        wick = np.abs(rng.standard_t(3, (2, n))) * px * 0.0006
        h = np.maximum(o, c) + wick[0]; l = np.minimum(o, c) - wick[1]
        v = rng.lognormal(8, 1.0, n) * np.where(np.arange(n) < 30, 3, 1)
        frames.append(pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c, "Volume": v}, index=idx))
        px = c[-1] * (1 + rng.normal(0, 0.01))
    return pd.concat(frames)
