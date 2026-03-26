"""Quarterly rebalancing date calculator.

Rebalancing logic (PRD spec):
  - Months: March, June, September, December
  - Target day: third Friday of the month
  - Execution day: first trading day AFTER the third Friday (open price)
"""

from typing import List
import pandas as pd


def _third_friday(year: int, month: int) -> pd.Timestamp:
    """Return the third Friday of the given year/month."""
    first_day = pd.Timestamp(year, month, 1)
    # weekday(): Monday=0, Friday=4
    days_to_first_friday = (4 - first_day.weekday()) % 7
    first_friday = first_day + pd.Timedelta(days=days_to_first_friday)
    return first_friday + pd.Timedelta(weeks=2)


def get_rebalancing_dates(
    start: str,
    end: str,
    trading_days: pd.DatetimeIndex,
    rebal_months: List[int] = (3, 6, 9, 12),  # type: ignore[assignment]
) -> List[pd.Timestamp]:
    """Return sorted list of rebalancing execution dates (day-after-third-Friday).

    Args:
        start: Backtest start date string "YYYY-MM-DD".
        end:   Backtest end date string "YYYY-MM-DD".
        trading_days: All actual trading days in the backtest window.
        rebal_months: List of months to rebalance (default quarterly).

    Returns:
        List of Timestamps (execution dates = first trading day after third Friday).
    """
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    dates: List[pd.Timestamp] = []
    for year in range(start_ts.year, end_ts.year + 1):
        for month in rebal_months:
            third_fri = _third_friday(year, month)
            # First trading day strictly after third Friday
            candidates = trading_days[trading_days > third_fri]
            if len(candidates) == 0:
                continue
            exec_date = candidates[0]
            if start_ts <= exec_date <= end_ts:
                dates.append(exec_date)

    return sorted(dates)


def prev_trading_day(date: pd.Timestamp, trading_days: pd.DatetimeIndex) -> pd.Timestamp:
    """Return the trading day immediately before `date`."""
    candidates = trading_days[trading_days < date]
    if len(candidates) == 0:
        raise ValueError(f"No trading day before {date}")
    return candidates[-1]
