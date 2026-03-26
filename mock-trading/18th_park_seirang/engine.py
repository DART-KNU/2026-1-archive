"""Backtest engine — main loop.

Look-ahead bias prevention:
  - Weight computation: uses data up to t-1 (ref_date = prev trading day)
  - Trade execution: uses open prices on t (rebalancing date)
  - Daily NAV: uses close prices on t
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List

import pandas as pd

from backtest.portfolio import Portfolio, RebalLog
from strategy.rebalancer import get_rebalancing_dates, prev_trading_day
from strategy.universe import filter_universe
from strategy.weights import (
    compute_raw_weights,
    compute_benchmark_sector_weights,
    optimize_weights,
)
from config import INITIAL_CAPITAL

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    nav: pd.Series
    benchmark: pd.Series
    weight_history: Dict[pd.Timestamp, pd.Series] = field(default_factory=dict)
    sector_weights_history: Dict[pd.Timestamp, pd.Series] = field(default_factory=dict)
    bench_sector_weights_history: Dict[pd.Timestamp, pd.Series] = field(default_factory=dict)
    rebal_logs: List[RebalLog] = field(default_factory=list)


def run_backtest(
    close_prices: pd.DataFrame,
    open_prices: pd.DataFrame,
    adtv_30d: pd.DataFrame,
    mktcap_snaps: Dict[pd.Timestamp, pd.Series],
    sector_snaps: Dict[pd.Timestamp, pd.Series],
    benchmark: pd.Series,
    trading_days: pd.DatetimeIndex,
    start: str,
    end: str,
) -> BacktestResult:
    """Run the full backtest and return results.

    Args:
        close_prices:  (date × ticker) DataFrame of adjusted close prices.
        open_prices:   (date × ticker) DataFrame of adjusted open prices.
        adtv_30d:      (date × ticker) 30-day rolling ADTV.
        mktcap_snaps:  {ref_date → Series(ticker → mktcap)}.
        sector_snaps:  {ref_date → Series(ticker → sector_name)}.
        benchmark:     Series(date → index level).
        trading_days:  All trading days in the backtest window.
        start:         Backtest start date "YYYY-MM-DD".
        end:           Backtest end date "YYYY-MM-DD".

    Returns:
        BacktestResult with NAV, benchmark, weight history, and rebalancing logs.
    """
    rebal_dates = get_rebalancing_dates(start, end, trading_days)
    rebal_set = set(rebal_dates)
    logger.info("Rebalancing dates: %d scheduled", len(rebal_dates))

    portfolio = Portfolio(initial_capital=INITIAL_CAPITAL)

    weight_history: Dict[pd.Timestamp, pd.Series] = {}
    sector_weights_history: Dict[pd.Timestamp, pd.Series] = {}
    bench_sector_history: Dict[pd.Timestamp, pd.Series] = {}

    # Precompute ref_date → nearest available mktcap snapshot
    # (use the snapshot date closest to but not after ref_date)
    snap_dates = sorted(mktcap_snaps.keys())

    def nearest_snap(ref: pd.Timestamp) -> pd.Timestamp:
        candidates = [d for d in snap_dates if d <= ref]
        return candidates[-1] if candidates else snap_dates[0]

    for date in trading_days:
        # ----------------------------------------------------------------
        # Rebalancing day: compute weights and execute at open
        # ----------------------------------------------------------------
        if date in rebal_set:
            ref_date = prev_trading_day(date, trading_days)
            snap_date = nearest_snap(ref_date)

            mktcap_snap = mktcap_snaps[snap_date]
            sector_map = sector_snaps.get(snap_date, pd.Series(dtype=str))

            # KOSPI tickers as of ref_date
            kospi_tickers = mktcap_snap.index.tolist()

            # Universe filter (uses data ≤ ref_date)
            universe = filter_universe(ref_date, adtv_30d, kospi_tickers)

            if len(universe) == 0:
                logger.warning("Empty universe at %s — skipping rebalancing.", date)
            else:
                # Benchmark sector weights (full KOSPI)
                bench_sw = compute_benchmark_sector_weights(mktcap_snap, sector_map)

                # Raw weights for universe
                raw_w = compute_raw_weights(mktcap_snap, universe)

                # Sector map for universe tickers only
                universe_sector = sector_map.reindex(universe).fillna("기타")

                # Optimise
                final_w = optimize_weights(raw_w, universe_sector, bench_sw)

                # Execute at today's open prices
                open_row = open_prices.loc[date] if date in open_prices.index else pd.Series(dtype=float)
                portfolio.rebalance(final_w, open_row, date)

                # Record history
                weight_history[date] = final_w
                # Sector weights of portfolio
                port_sector = (
                    pd.DataFrame({"w": final_w, "sector": universe_sector})
                    .groupby("sector")["w"]
                    .sum()
                )
                sector_weights_history[date] = port_sector
                bench_sector_history[date] = bench_sw

        # ----------------------------------------------------------------
        # Daily NAV mark at close
        # ----------------------------------------------------------------
        close_row = close_prices.loc[date] if date in close_prices.index else pd.Series(dtype=float)
        portfolio.update_nav(date, close_row)

    nav_series = portfolio.get_nav_series()
    bench_aligned = benchmark.reindex(nav_series.index).ffill()

    return BacktestResult(
        nav=nav_series,
        benchmark=bench_aligned,
        weight_history=weight_history,
        sector_weights_history=sector_weights_history,
        bench_sector_weights_history=bench_sector_history,
        rebal_logs=portfolio.rebal_logs,
    )
