"""Transaction cost model.

Costs applied:
  - Commission: 0.015% one-way (buy and sell)
  - Securities transaction tax: 0.25% sell-side only
  - Slippage: 5 bp (0.05%) per trade, both sides
"""

from config import COMMISSION_RATE, SECURITIES_TAX, SLIPPAGE_RATE


def calc_cost(trade_value: float, side: str) -> float:
    """Calculate total transaction cost for a single trade leg.

    Args:
        trade_value: Absolute KRW value of the trade (positive).
        side: "buy" or "sell".

    Returns:
        Total cost in KRW (positive).
    """
    trade_value = abs(trade_value)
    cost = trade_value * (COMMISSION_RATE + SLIPPAGE_RATE)
    if side == "sell":
        cost += trade_value * SECURITIES_TAX
    return cost


def calc_turnover(old_weights: dict[str, float], new_weights: dict[str, float]) -> float:
    """Compute one-way portfolio turnover between two weight dicts.

    Turnover = sum of absolute weight changes / 2  (one-way convention)
    """
    all_tickers = set(old_weights) | set(new_weights)
    total = sum(
        abs(new_weights.get(t, 0.0) - old_weights.get(t, 0.0))
        for t in all_tickers
    )
    return total / 2.0
