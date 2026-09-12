"""
broker_interface.py
===================
Abstract broker and market data provider interfaces (FinEx pattern, P2-026).

WHY THIS EXISTS
---------------
The trading_engine.py currently has yfinance hardcoded throughout Group E
functions. Swapping from paper trading (yfinance) to live trading (Alpaca
WebSocket) requires changing multiple call sites.

The FinEx architecture solves this cleanly: abstract base classes define
the interface; concrete implementations are injected at startup. Swapping
from paper to live is one config line change.

ARCHITECTURE
------------
DataProvider (abstract)
    ├── YFinanceDataProvider    — yfinance polling (current paper mode)
    └── AlpacaDataProvider      — Alpaca WebSocket (Phase 4 live)

BrokerClient (abstract)
    ├── PaperBrokerClient       — SQLite paper account (current)
    └── AlpacaBrokerClient      — Alpaca REST API (Phase 4 live)

USAGE IN TRADING ENGINE
-----------------------
# Paper trading (current default):
    data    = YFinanceDataProvider(ticker="SPY", interval="5m")
    broker  = PaperBrokerClient(db_path="paper_account.db",
                                 account=10_000, risk_pct=0.01)
    engine  = TradingEngine(data=data, broker=broker, cfg=CONFIG)
    engine.run()

# Live trading (Phase 4 — after WFE >= 0.50 and all gates pass):
    data    = AlpacaDataProvider(ticker="SPY", api_key=..., secret=...)
    broker  = AlpacaBrokerClient(api_key=..., secret=..., paper=True)
    engine  = TradingEngine(data=data, broker=broker, cfg=CONFIG)
    engine.run()

Only the data and broker lines change. ALL strategy logic, risk rules,
signal evaluation, and decision ledger remain identical.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)


# ===========================================================================
# ABSTRACT INTERFACES
# ===========================================================================

class DataProvider(ABC):
    """
    Abstract market data provider.
    Concrete implementations: YFinanceDataProvider, AlpacaDataProvider.
    """

    @abstractmethod
    def get_ohlcv(self, ticker: str, period: str = "2d",
                   interval: str = "5m") -> pd.DataFrame:
        """Return OHLCV DataFrame with DatetimeIndex in America/New_York."""
        ...

    @abstractmethod
    def get_daily(self, ticker: str, period: str = "1y") -> pd.DataFrame:
        """Return daily OHLCV DataFrame."""
        ...

    @abstractmethod
    def get_vix(self) -> float:
        """Return current VIX value."""
        ...

    @abstractmethod
    def source_name(self) -> str:
        """Human-readable name of the data source."""
        ...


class BrokerClient(ABC):
    """
    Abstract broker client.
    Concrete implementations: PaperBrokerClient, AlpacaBrokerClient.
    """

    @abstractmethod
    def get_account_balance(self) -> float:
        """Return current cash balance."""
        ...

    @abstractmethod
    def place_market_order(self, ticker: str, side: str,
                            units: float) -> dict:
        """
        Place a market order. side = 'buy' | 'sell'.
        Returns {'order_id', 'fill_price', 'units', 'status'}.
        """
        ...

    @abstractmethod
    def place_bracket_order(self, ticker: str, side: str, units: float,
                             stop_price: float,
                             target_price: float) -> dict:
        """
        Place a bracket order (entry + stop + target).
        Returns {'order_id', 'fill_price', 'units', 'status'}.
        """
        ...

    @abstractmethod
    def cancel_all_orders(self, ticker: str) -> None:
        """Cancel all open orders for the ticker."""
        ...

    @abstractmethod
    def get_open_positions(self) -> list[dict]:
        """Return list of open positions."""
        ...

    @abstractmethod
    def close_position(self, ticker: str) -> dict:
        """Close the position for ticker at market. Returns exit details."""
        ...

    @abstractmethod
    def is_paper(self) -> bool:
        """True if this is a paper trading account."""
        ...

    @abstractmethod
    def broker_name(self) -> str:
        """Human-readable broker name."""
        ...


# ===========================================================================
# YFINANCE DATA PROVIDER (current implementation)
# ===========================================================================

class YFinanceDataProvider(DataProvider):
    """
    yfinance-backed data provider. Polling model (not streaming).
    Used for paper trading. Replace with AlpacaDataProvider for Phase 4.

    Limitations vs AlpacaDataProvider:
      - Polls every poll_interval seconds rather than streaming bar closes
      - May have up to poll_interval delay from signal to evaluation
      - Rate-limited for large universe scans
      - No pre-market data without pre_post=True (which has issues)
    """

    def get_ohlcv(self, ticker: str, period: str = "2d",
                   interval: str = "5m") -> pd.DataFrame:
        import yfinance as yf
        df = yf.download(ticker, period=period, interval=interval,
                          auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df.index.name = "Datetime"
        return df

    def get_daily(self, ticker: str, period: str = "1y") -> pd.DataFrame:
        return self.get_ohlcv(ticker, period=period, interval="1d")

    def get_vix(self) -> float:
        df = self.get_ohlcv("^VIX", period="2d", interval="5m")
        return float(df["Close"].dropna().iloc[-1])

    def source_name(self) -> str:
        return "yfinance (polling)"


# ===========================================================================
# PAPER BROKER CLIENT (wraps trading_engine.py PaperAccountDB)
# ===========================================================================

class PaperBrokerClient(BrokerClient):
    """
    Paper broker backed by the SQLite PaperAccountDB.
    Zero latency, no commissions unless cost model applied explicitly.
    All fills simulated via FillSimulator in trading_engine._sim_trade().
    """

    def __init__(self, db_path: str, account: float, risk_pct: float) -> None:
        # Import lazily to avoid circular dependency
        from trading_engine import PaperAccountDB
        self.db      = PaperAccountDB(db_path)
        self._account = account
        self._risk   = risk_pct

    def get_account_balance(self) -> float:
        return self._account

    def place_market_order(self, ticker: str, side: str, units: float) -> dict:
        log.info("PaperBroker: market %s %s %.0f units (simulated)", side, ticker, units)
        return {"order_id": "paper", "fill_price": None, "units": units,
                "status": "filled", "paper": True}

    def place_bracket_order(self, ticker: str, side: str, units: float,
                             stop_price: float, target_price: float) -> dict:
        log.info("PaperBroker: bracket %s %s %.0f  stop=%.4f  target=%.4f",
                 side, ticker, units, stop_price, target_price)
        return {"order_id": "paper", "fill_price": None, "units": units,
                "status": "filled", "stop": stop_price, "target": target_price}

    def cancel_all_orders(self, ticker: str) -> None:
        log.info("PaperBroker: cancel_all_orders %s (no-op for paper)", ticker)

    def get_open_positions(self) -> list[dict]:
        pos = self.db.get_open_position()
        return [pos] if pos else []

    def close_position(self, ticker: str) -> dict:
        log.info("PaperBroker: close_position %s (engine handles this)", ticker)
        return {"status": "closed"}

    def is_paper(self) -> bool:
        return True

    def broker_name(self) -> str:
        return "SQLite Paper Account"


# ===========================================================================
# ALPACA BROKER CLIENT (Phase 4 — stub for now)
# ===========================================================================

class AlpacaDataProvider(DataProvider):
    """
    Phase 4: Alpaca Markets data provider.
    Requires: pip install alpaca-py

    When initialised, this connects to Alpaca's streaming data API
    and delivers 1-minute bars the moment they close (no polling delay).

    STATUS: Stub implementation — method bodies raise NotImplementedError.
    Complete after P4-006 (Alpaca API evaluation) is done and
    paper_account (P1-002) has been confirmed with Alpaca.
    """

    def __init__(self, ticker: str, api_key: str = "", secret_key: str = "",
                 paper: bool = True) -> None:
        self.ticker     = ticker
        self.api_key    = api_key
        self.secret_key = secret_key
        self.paper      = paper
        if not api_key:
            raise ValueError(
                "AlpacaDataProvider requires api_key and secret_key. "
                "Complete P4-006 first, then set these via environment variables."
            )

    def get_ohlcv(self, ticker: str, period: str = "2d",
                   interval: str = "5m") -> pd.DataFrame:
        raise NotImplementedError(
            "AlpacaDataProvider: implement using alpaca-py StockBarsRequest. "
            "See P4-007 for WebSocket streaming implementation."
        )

    def get_daily(self, ticker: str, period: str = "1y") -> pd.DataFrame:
        raise NotImplementedError("AlpacaDataProvider.get_daily() — implement in P4-007")

    def get_vix(self) -> float:
        raise NotImplementedError("Alpaca does not provide VIX directly. Use yfinance for VIX.")

    def source_name(self) -> str:
        env = "paper" if self.paper else "LIVE"
        return f"Alpaca Markets ({env})"


class AlpacaBrokerClient(BrokerClient):
    """
    Phase 4: Alpaca Markets broker client.
    Handles market orders, bracket orders, and position management via REST.

    STATUS: Stub — complete after P4-006 (Alpaca API evaluation).
    """

    def __init__(self, api_key: str = "", secret_key: str = "",
                 paper: bool = True) -> None:
        if not api_key:
            raise ValueError(
                "AlpacaBrokerClient requires api_key. "
                "Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables."
            )
        self._paper = paper

    def get_account_balance(self) -> float:
        raise NotImplementedError("AlpacaBrokerClient — implement in P4-006")

    def place_market_order(self, ticker: str, side: str, units: float) -> dict:
        raise NotImplementedError("AlpacaBrokerClient — implement in P4-006")

    def place_bracket_order(self, ticker: str, side: str, units: float,
                             stop_price: float, target_price: float) -> dict:
        raise NotImplementedError("AlpacaBrokerClient — implement in P4-006")

    def cancel_all_orders(self, ticker: str) -> None:
        raise NotImplementedError("AlpacaBrokerClient — implement in P4-006")

    def get_open_positions(self) -> list[dict]:
        raise NotImplementedError("AlpacaBrokerClient — implement in P4-006")

    def close_position(self, ticker: str) -> dict:
        raise NotImplementedError("AlpacaBrokerClient — implement in P4-006")

    def is_paper(self) -> bool:
        return self._paper

    def broker_name(self) -> str:
        return f"Alpaca Markets ({'paper' if self._paper else 'LIVE'})"


# ===========================================================================
# FACTORY
# ===========================================================================

def get_data_provider(source: str = "yfinance", **kwargs: Any) -> DataProvider:
    """
    Factory: return the correct DataProvider for the given source name.
    source: 'yfinance' | 'alpaca'
    """
    if source == "yfinance":
        return YFinanceDataProvider()
    elif source == "alpaca":
        return AlpacaDataProvider(**kwargs)
    raise ValueError(f"Unknown data source {source!r}. Valid: 'yfinance', 'alpaca'")


def get_broker_client(broker: str = "paper", **kwargs: Any) -> BrokerClient:
    """
    Factory: return the correct BrokerClient.
    broker: 'paper' | 'alpaca'
    """
    if broker == "paper":
        return PaperBrokerClient(**kwargs)
    elif broker == "alpaca":
        return AlpacaBrokerClient(**kwargs)
    raise ValueError(f"Unknown broker {broker!r}. Valid: 'paper', 'alpaca'")


# ===========================================================================
# Smoke test
# ===========================================================================

if __name__ == "__main__":
    import sys

    print("=== broker_interface.py smoke test ===\n")

    # YFinanceDataProvider
    dp = YFinanceDataProvider()
    print(f"DataProvider: {dp.source_name()}")
    df = dp.get_ohlcv("SPY", period="1d", interval="5m")
    assert not df.empty, "yfinance fetch returned empty DataFrame"
    print(f"  SPY 5m bars: {len(df)} rows  "
          f"close={float(df['Close'].iloc[-1]):.2f}")

    vix = dp.get_vix()
    print(f"  VIX: {vix:.2f}")

    # PaperBrokerClient
    import tempfile, os
    tmp = tempfile.mktemp(suffix=".db")
    pb  = PaperBrokerClient(tmp, account=10_000.0, risk_pct=0.01)
    print(f"\nBrokerClient: {pb.broker_name()}  paper={pb.is_paper()}")
    print(f"  Balance: £{pb.get_account_balance():,.0f}")
    r = pb.place_market_order("SPY", "buy", 50)
    assert r["status"] == "filled"
    print(f"  Market order: {r}")
    os.unlink(tmp)

    # Alpaca stubs raise NotImplementedError
    try:
        AlpacaDataProvider("SPY", api_key="")
        print("ERROR: should have raised")
        sys.exit(1)
    except ValueError:
        print("\nAlpaca stub raises ValueError without API key: ✅")

    # Factory
    dp2 = get_data_provider("yfinance")
    assert isinstance(dp2, YFinanceDataProvider)
    print("\nFactory get_data_provider('yfinance'): ✅")

    print("\n✅ broker_interface.py OK")
