"""P2-125 universe stage 2: classification, opening-bar download (only what
RVOL needs; zero-filled), RVOL with no look-ahead, top-20 selection."""
import sqlite3, sys, types
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import universe_select as sel_mod
from universe_select import UniverseSelector, classify_name, rvol_frame, load_directory
from universe_store import UniverseStore


# ── Classification ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name, kind", [
    ("Agilent Technologies, Inc. Common Stock", "stock"),
    ("Amplify Energy Corp. Common Stock", "stock"),                 # brand word, not a fund
    ("Ark Restaurants Corp. - Common Stock", "stock"),
    ("Aardvark Therapeutics, Inc. - Common Stock", "stock"),
    ("Taiwan Semiconductor Manufacturing Company Ltd. American Depositary Shares", "stock"),
    ("ProShares Ultra Silver", "fund"),                             # pure issuer, no fund word
    ("SPDR S&P 500 ETF Trust", "fund"),
    ("Direxion Daily Semiconductor Bull 3X Shares", "fund"),
    ("iPath Bloomberg Commodity Index Total Return ETN", "fund"),
    ("abrdn Total Dynamic Dividend Fund Common Shares", "fund"),     # closed-end fund
    ("Cornerstone Strategic Value Fund, Inc.", "fund"),              # 'Fund, Inc.' leaked once
    ("DNP Select Income Fund Inc. Common Stock", "fund"),
    ("Kayne Anderson Energy Infrastructure Fund, Inc.", "fund"),
    ("Ares Acquisition Corporation III Redeemable warrants, each whole warrant", "noncommon"),
    ("Ares Acquisition Corporation III Units, each consisting of one share", "noncommon"),
    ("Arbor Realty Trust 6.375% Series D Cumulative Redeemable Preferred Stock", "noncommon"),
])
def test_classify_name(name, kind):
    assert classify_name(name) == kind


DIRECTORY = """Nasdaq Traded|Symbol|Security Name|Listing Exchange|Market Category|ETF|Round Lot Size|Test Issue|Financial Status|CQS Symbol|NASDAQ Symbol|NextShares
Y|AAA|Alpha Corp Common Stock|N| |N|100|N||AAA|AAA|N
Y|FFF|Some Obscure Tracker|P| |Y|100|N||FFF|FFF|N
Y|DSL|DOUBLELINE INCOME SOLUTIONS FUND|N| |N|100|N||DSL|DSL|N
Y|VNO|Vornado Realty Trust|N| |N|100|N||VNO|VNO|N
Y|TST|Test issue|Q| |N|100|Y||TST|TST|N
File Creation Time: 0925202621:32|||||
"""


def _store(tmp_path):
    u = UniverseStore(str(tmp_path / "u.db"))
    assets = [("AAA", "Alpha Corp Common Stock", "NYSE", "active"),
              ("FFF", "Some Obscure Tracker", "ARCA", "active"),        # ETF only per directory
              ("DSL", "DOUBLELINE INCOME SOLUTIONS FUND", "NYSE", "active"),   # closed-end fund, ETF flag N
              ("VNO", "Vornado Realty Trust", "NYSE", "active"),             # a REIT: a real company
              ("OLD", "Old Industries Inc Common Stock", "NYSE", "inactive"),
              ("UPRO", "ProShares UltraPro S&P500", "ARCA", "inactive"),
              ("ZZZ", "Zeta Corp Common Stock", "NASDAQ", "active")]
    with sqlite3.connect(u.db_path) as c:
        c.executemany("INSERT INTO universe_assets VALUES (?,?,?,?,1,'us_equity','')",
                      [(s, n, e, st) for s, n, e, st in assets])
    return u


def test_directory_flag_wins_for_current_listings_name_rules_otherwise(tmp_path):
    s = UniverseSelector(_store(tmp_path))
    s.classify(load_directory(text=DIRECTORY))
    with sqlite3.connect(s.store.db_path) as c:
        k = dict(c.execute("SELECT symbol, kind || '/' || source FROM universe_class"))
    assert k == {"AAA": "stock/directory", "FFF": "fund/directory", "DSL": "fund/directory",
                 "VNO": "stock/directory", "OLD": "stock/name_rules",
                 "UPRO": "fund/name_rules", "ZZZ": "stock/name_rules"}


# ── Download + RVOL + selection ───────────────────────────────────────────────

SESS = [str(d.date()) for d in pd.bdate_range("2019-01-02", "2019-03-29")]   # no holidays needed here


def _vol(sym, d):
    i = SESS.index(d)
    base = {"AAA": 1000, "OLD": 2000, "ZZZ": 3000, "UPRO": 50_000}[sym]
    return base * (5 if (sym == "OLD" and i == 30) else 1) * (3 if (sym == "ZZZ" and i == 30) else 1)


class _Client:
    def __init__(self, invalid=()):
        self.calls, self.invalid = [], set(invalid)

    def get_stock_bars(self, req):
        self.calls.append(req)
        bad = [s for s in req.symbol_or_symbols if s in self.invalid]
        if bad:
            from alpaca.common.exceptions import APIError
            raise APIError('{"message":"invalid symbol: %s"}' % bad[0])
        t = pd.Timestamp(req.start)
        t = (t.tz_localize("UTC") if t.tzinfo is None else t).tz_convert("America/New_York")
        d = str(t.date())
        rows, idx = [], []
        for s in req.symbol_or_symbols:
            if s == "ZZZ" and SESS.index(d) == 5:
                continue                                    # no trade in the first 5 minutes
            rows.append({"open": 10, "high": 11, "low": 9, "close": 10.5, "volume": _vol(s, d)})
            idx.append((s, t.tz_convert("UTC")))
        if not rows:
            return types.SimpleNamespace(df=pd.DataFrame())
        return types.SimpleNamespace(df=pd.DataFrame(rows, index=pd.MultiIndex.from_tuples(
            idx, names=["symbol", "timestamp"])))


@pytest.fixture
def ready(tmp_path, monkeypatch):
    monkeypatch.setattr(sel_mod._time, "sleep", lambda s: None)
    u = _store(tmp_path)
    s = UniverseSelector(u)
    s.classify(load_directory(text=DIRECTORY))
    with sqlite3.connect(u.db_path) as c:           # eligible: AAA/OLD/ZZZ/UPRO from session 20 on
        c.executemany("INSERT INTO universe_eligibility VALUES (?,?,20,2e6,1)",
                      [(d, sym) for d in SESS[20:] for sym in ("AAA", "OLD", "ZZZ", "UPRO")])
    import exchange_calendars as xc
    class _Cal:
        def sessions_in_range(self, a, b):
            return pd.DatetimeIndex([d for d in SESS if a <= d <= b])
    monkeypatch.setattr(xc, "get_calendar", lambda name: _Cal())
    return s


def test_downloads_only_what_rvol_needs_and_zero_fills(ready):
    cl = _Client()
    r = ready.download_open5(cl, date(2019, 1, 2), date(2019, 3, 29))
    assert r["days"] == len(SESS) and not r["failed_days"]
    first_needed = SESS.index(SESS[20]) - sel_mod.LOOKBACK            # 14 sessions before first eligible
    ny = lambda x: (pd.Timestamp(x).tz_localize("UTC") if pd.Timestamp(x).tzinfo is None
                    else pd.Timestamp(x)).tz_convert("America/New_York")
    requested = {ny(c.start).strftime("%Y-%m-%d") for c in cl.calls}
    assert all(ny(c.start).strftime("%H:%M") == "09:30" and ny(c.end).strftime("%H:%M") == "09:34"
               for c in cl.calls)
    assert min(requested) == SESS[first_needed]
    assert all(set(c.symbol_or_symbols) <= {"AAA", "OLD", "ZZZ"} for c in cl.calls)   # never the fund
    with sqlite3.connect(ready.store.db_path) as c:
        assert c.execute("SELECT volume FROM universe_open5 WHERE symbol='ZZZ' AND date=?",
                         (SESS[5],)).fetchone() in (None, (0.0,))
    cl2 = _Client()
    assert ready.download_open5(cl2, date(2019, 1, 2), date(2019, 3, 29))["days"] == 0 and not cl2.calls


def test_invalid_symbol_dropped_not_whole_day(ready):
    r = ready.download_open5(_Client(invalid={"OLD"}), date(2019, 1, 2), date(2019, 3, 29))
    assert not r["failed_days"]


def test_rvol_uses_only_prior_sessions():
    o = pd.DataFrame({"date": SESS[:20], "symbol": "X", "volume": [100.0] * 19 + [900.0]})
    rv = rvol_frame(o).set_index("date")
    assert rv.loc[SESS[19], "avg14"] == pytest.approx(100.0)       # today's 900 not in its own average
    assert rv.loc[SESS[13], "avg14"] != rv.loc[SESS[13], "avg14"]    # NaN: no full window yet


def test_selection_ranks_by_rvol_excludes_funds_and_low_rvol(ready):
    ready.download_open5(_Client(), date(2019, 1, 2), date(2019, 3, 29))
    r = ready.select(date(2019, 1, 2), date(2019, 3, 29), top=2, min_rvol=1.0)
    with sqlite3.connect(ready.store.db_path) as c:
        spike = c.execute("SELECT symbol, rank, ROUND(rvol, 3) FROM universe_selection WHERE date=? ORDER BY rank",
                          (SESS[30],)).fetchall()
        syms = {s for (s,) in c.execute("SELECT DISTINCT symbol FROM universe_selection")}
    assert spike[:2] == [("OLD", 1, 5.0), ("ZZZ", 2, 3.0)]            # 5x and 3x their averages
    assert "UPRO" not in syms                                         # leveraged fund never selected
    assert r["days"] > 0
