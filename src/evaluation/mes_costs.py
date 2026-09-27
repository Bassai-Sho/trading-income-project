"""
evaluation/mes_costs.py  (PR-003 Stage V)
==========================================
Converts box #3's SPY-share trades into an MES-futures cost, without needing
real MES data: 1 MES contract's notional tracks the S&P 500 index at $5/point;
since SPX ~= 10 x SPY, that notional ~= SPY_price x 50 -- i.e. 1 MES contract
~= 50 SPY shares. Cost per contract is a flat $ amount (commission + exchange
fees are per-contract, not per-$-notional), so in share terms:
    fee($) = cost_per_contract_per_side x (shares / 50)
This also means the cost RATE in bps of notional falls as SPY's price rises
over 2016-2024 -- realistic, since exchange fees do not scale with price.

COST_PER_CONTRACT_SIDE = commission_fee + slippage, both [ASSUMED]:
  commission_fee = $0.50  (IBKR MES, all-in incl. exchange/regulatory fees;
      a worked example gives $0.49/contract/side round-trip-halved: $1.96 for
      2 contracts round trip = $0.49 each way, Sep 2026 pricing)
  slippage        = $0.625 (half of MES's typical 1-tick, $1.25, quoted spread;
      MES is highly liquid, so this is a deliberately cautious estimate, not
      a measurement)
These are placeholders for a broker-verified figure once an account exists;
Stage V reports the breakeven cost so the assumption's sensitivity is visible.
"""
from __future__ import annotations

SHARES_PER_CONTRACT = 50.0     # 1 MES contract notional ~= 50 SPY shares
COMMISSION_FEE = 0.50          # $/contract/side [ASSUMED]
SLIPPAGE = 0.625               # $/contract/side [ASSUMED]
COST_PER_CONTRACT_SIDE = COMMISSION_FEE + SLIPPAGE


def mes_fee_fn(cost_per_contract_side: float = COST_PER_CONTRACT_SIDE):
    def fee(sym, side, qty, px, ts):
        return cost_per_contract_side * abs(qty) / SHARES_PER_CONTRACT
    return fee


def breakeven_cost_per_contract_side(gross_frac_total: float, cost_frac_total: float,
                                     current_rate: float = COST_PER_CONTRACT_SIDE) -> float:
    """Cost rate at which total net P&L = 0, given trades are unaffected by
    cost (true here: box #3's signals don't depend on fees), so cost scales
    linearly with the rate: net(r) = gross - (r/r0) x cost(r0)."""
    if cost_frac_total <= 0:
        return float("inf")
    return current_rate * gross_frac_total / cost_frac_total
