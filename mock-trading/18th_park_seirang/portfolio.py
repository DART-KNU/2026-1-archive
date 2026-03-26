"""Portfolio state management.

Tracks integer share counts, cash, and daily NAV.
All rebalancing is executed at opening prices on the rebalancing date.
Daily NAV is marked at closing prices.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from backtest.cost_model import calc_cost

logger = logging.getLogger(__name__)


@dataclass
class RebalLog:
    date: pd.Timestamp
    turnover: float
    total_cost: float
    n_buys: int
    n_sells: int
    cash_after: float


@dataclass
class Portfolio:
    initial_capital: float

    shares: Dict[str, int] = field(default_factory=dict)
    cash: float = field(init=False)
    nav_history: List[Tuple[pd.Timestamp, float]] = field(default_factory=list)
    rebal_logs: List[RebalLog] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.cash = self.initial_capital

    def current_nav(self, close_prices: pd.Series) -> float:
        """Compute NAV using provided close prices Series(ticker → price)."""
        equity = sum(
            self.shares.get(t, 0) * close_prices.get(t, np.nan)
            for t in self.shares
            if not np.isnan(close_prices.get(t, np.nan))
        )
        return equity + self.cash

    def update_nav(self, date: pd.Timestamp, close_prices: pd.Series) -> float:
        """Record today's NAV. Returns the NAV value."""
        nav = self.current_nav(close_prices)
        self.nav_history.append((date, nav))
        return nav

    def rebalance(
        self,
        target_weights: pd.Series,
        open_prices: pd.Series,
        date: pd.Timestamp,
    ) -> None:
        """Execute rebalancing at open prices.

        1. Compute target integer shares = floor(weight * nav / open_price)
        2. Sell excess positions first (frees cash)
        3. Buy new/increased positions
        4. Minimise residual cash by topping up top-weight stocks

        Args:
            target_weights: Series(ticker → weight), sums to 1.0.
            open_prices:    Series(ticker → open price) for execution day.
            date:           Execution date (for logging).
        """
        # Current NAV at last close (or initial capital on day 1)
        if self.nav_history:
            nav = self.nav_history[-1][1]
        else:
            nav = self.initial_capital

        # Resolve valid prices (drop tickers with missing open)
        valid_tickers = [
            t for t in target_weights.index
            if t in open_prices.index and not np.isnan(open_prices[t]) and open_prices[t] > 0
        ]
        weights = target_weights.reindex(valid_tickers).fillna(0.0)
        weights /= weights.sum()  # renormalize after dropping missing

        prices = open_prices.reindex(valid_tickers)

        # Target integer shares
        target_shares: Dict[str, int] = {
            t: int(np.floor(weights[t] * nav / prices[t]))
            for t in valid_tickers
            if prices[t] > 0
        }

        # Compute diff vs current holdings
        all_tickers = set(self.shares) | set(target_shares)
        total_cost = 0.0
        n_buys = n_sells = 0

        # --- Step 1: Sells ---
        for t in list(self.shares.keys()):
            current = self.shares.get(t, 0)
            target = target_shares.get(t, 0)
            diff = target - current
            if diff < 0:
                sell_qty = -diff
                sell_price = open_prices.get(t, None)
                if sell_price is None or np.isnan(sell_price) or sell_price <= 0:
                    # Cannot sell — keep position
                    logger.warning("Cannot sell %s at %s: invalid price", t, date)
                    continue
                sell_value = sell_qty * sell_price
                cost = calc_cost(sell_value, "sell")
                self.cash += sell_value - cost
                total_cost += cost
                self.shares[t] = target
                if target == 0:
                    del self.shares[t]
                n_sells += 1

        # --- Step 2: Buys ---
        for t in valid_tickers:
            current = self.shares.get(t, 0)
            target = target_shares.get(t, 0)
            diff = target - current
            if diff > 0:
                buy_value = diff * prices[t]
                if buy_value > self.cash:
                    # Reduce buy quantity to what cash allows
                    affordable = int(np.floor(self.cash / prices[t] / (1 + calc_cost(1.0, "buy"))))
                    diff = max(0, min(diff, affordable))
                    if diff == 0:
                        continue
                    buy_value = diff * prices[t]
                cost = calc_cost(buy_value, "buy")
                if buy_value + cost > self.cash:
                    continue  # Cannot afford even with reduced quantity
                self.cash -= buy_value + cost
                total_cost += cost
                self.shares[t] = current + diff
                n_buys += 1

        # --- Step 3: Minimise residual cash ---
        # Sort by target weight descending; try to buy 1 more share each
        sorted_tickers = sorted(valid_tickers, key=lambda t: weights.get(t, 0), reverse=True)
        for t in sorted_tickers:
            p = prices.get(t, 0.0)
            if p <= 0:
                continue
            cost_1 = calc_cost(p, "buy")
            if self.cash >= p + cost_1:
                self.cash -= p + cost_1
                total_cost += cost_1
                self.shares[t] = self.shares.get(t, 0) + 1

        # Compute turnover
        old_weights: Dict[str, float] = {}
        for t, cnt in {**self.shares, **{t: 0 for t in all_tickers}}.items():
            price = open_prices.get(t, 0.0)
            if price and nav > 0:
                old_weights[t] = cnt * price / nav

        turnover = sum(
            abs(weights.get(t, 0.0) - old_weights.get(t, 0.0))
            for t in all_tickers
        ) / 2.0

        self.rebal_logs.append(
            RebalLog(
                date=date,
                turnover=turnover,
                total_cost=total_cost,
                n_buys=n_buys,
                n_sells=n_sells,
                cash_after=self.cash,
            )
        )
        logger.info(
            "Rebalanced %s: buys=%d sells=%d cost=%.0f KRW cash=%.0f KRW",
            date.date(),
            n_buys,
            n_sells,
            total_cost,
            self.cash,
        )

    def get_nav_series(self) -> pd.Series:
        """Return NAV history as a pandas Series indexed by date."""
        if not self.nav_history:
            return pd.Series(dtype=float)
        dates, navs = zip(*self.nav_history)
        return pd.Series(navs, index=pd.DatetimeIndex(dates), name="nav")

    def get_weight_snapshot(self, close_prices: pd.Series) -> pd.Series:
        """Return current weight of each holding based on close prices."""
        nav = self.current_nav(close_prices)
        if nav <= 0:
            return pd.Series(dtype=float)
        weights = {
            t: cnt * close_prices.get(t, 0.0) / nav
            for t, cnt in self.shares.items()
        }
        return pd.Series(weights, name="weight")
