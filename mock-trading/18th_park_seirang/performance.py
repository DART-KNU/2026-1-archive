"""Performance metrics calculation.

Metrics:
  - Cumulative return (portfolio and benchmark)
  - Excess return
  - Annualised tracking error (TE)
  - Information ratio (IR)
  - Beta vs benchmark
  - R-squared
  - Max drawdown (portfolio and benchmark)
  - Annual breakdown of excess return and TE
"""

from typing import Dict, Any

import numpy as np
import pandas as pd


def _max_drawdown(nav: pd.Series) -> float:
    rolling_max = nav.cummax()
    drawdown = nav / rolling_max - 1.0
    return float(drawdown.min())


def _annualised_return(returns: pd.Series) -> float:
    n = len(returns)
    if n == 0:
        return 0.0
    return float((1.0 + returns).prod() ** (252.0 / n) - 1.0)


def compute_metrics(
    nav: pd.Series,
    benchmark: pd.Series,
) -> Dict[str, Any]:
    """Compute full performance metrics.

    Args:
        nav:       Portfolio NAV time series (levels).
        benchmark: Benchmark index level time series.

    Returns:
        Dict of named metrics.
    """
    port_ret = nav.pct_change().dropna()
    bench_ret = benchmark.pct_change().dropna()

    common = port_ret.index.intersection(bench_ret.index)
    p = port_ret.loc[common]
    b = bench_ret.loc[common]
    excess = p - b

    n = len(p)

    # Tracking error (annualised)
    te = float(excess.std() * np.sqrt(252)) if n > 1 else 0.0

    # IR
    ir = (float(excess.mean()) / float(excess.std()) * np.sqrt(252)) if excess.std() > 0 else 0.0

    # Beta
    cov = np.cov(p.values, b.values)
    beta = float(cov[0, 1] / cov[1, 1]) if cov[1, 1] > 0 else 1.0

    # R²
    corr = float(np.corrcoef(p.values, b.values)[0, 1])
    r_squared = corr ** 2

    # NAV aligned to common dates for drawdown
    nav_aligned = nav.reindex(common)
    bench_aligned = benchmark.reindex(common)

    metrics: Dict[str, Any] = {
        "cumulative_return": float((1 + p).prod() - 1),
        "benchmark_cumulative_return": float((1 + b).prod() - 1),
        "excess_cumulative_return": float((1 + p).prod() - (1 + b).prod()),
        "annualised_return": _annualised_return(p),
        "benchmark_annualised_return": _annualised_return(b),
        "annualised_te": te,
        "ir": ir,
        "beta": beta,
        "r_squared": r_squared,
        "max_drawdown": _max_drawdown(nav_aligned.dropna()),
        "benchmark_max_drawdown": _max_drawdown(bench_aligned.dropna()),
        "n_trading_days": n,
    }

    return metrics


def compute_annual_breakdown(
    nav: pd.Series,
    benchmark: pd.Series,
) -> pd.DataFrame:
    """Compute year-by-year excess return and tracking error.

    Returns:
        DataFrame with columns [port_return, bench_return, excess_return, te]
        indexed by year.
    """
    port_ret = nav.pct_change().dropna()
    bench_ret = benchmark.pct_change().dropna()
    common = port_ret.index.intersection(bench_ret.index)
    p = port_ret.loc[common]
    b = bench_ret.loc[common]
    excess = p - b

    rows = []
    for year in sorted(p.index.year.unique()):
        pm = p[p.index.year == year]
        bm = b[b.index.year == year]
        em = excess[excess.index.year == year]

        rows.append({
            "year": year,
            "port_return": float((1 + pm).prod() - 1),
            "bench_return": float((1 + bm).prod() - 1),
            "excess_return": float((1 + pm).prod() - (1 + bm).prod()),
            "te": float(em.std() * np.sqrt(252)) if len(em) > 1 else 0.0,
        })

    return pd.DataFrame(rows).set_index("year")


def compute_simple_mcap_te(
    nav: pd.Series,
    benchmark: pd.Series,
    close_prices: pd.DataFrame,
    mktcap_snaps: dict,
    trading_days: pd.DatetimeIndex,
) -> float:
    """Compute TE of a simple market-cap weighted portfolio (no cap, no filter).

    Used for comparison vs our modified strategy.
    This is an approximation: at each trading day, weights = mktcap proportional.
    We use available mktcap snapshots forward-filled.
    """
    # Use closest available mktcap snapshot for each trading day
    snap_dates = sorted(mktcap_snaps.keys())
    returns_list = []
    bench_ret = benchmark.pct_change().dropna()

    for i in range(1, len(trading_days)):
        date = trading_days[i]
        prev_date = trading_days[i - 1]

        # Get most recent snapshot ≤ prev_date
        avail = [d for d in snap_dates if d <= prev_date]
        if not avail:
            continue
        snap = mktcap_snaps[avail[-1]]

        # Simple mktcap weights over all stocks in snapshot
        total = snap.sum()
        if total <= 0:
            continue
        weights = snap / total

        # Portfolio return = weighted sum of stock returns
        tickers = weights.index.intersection(close_prices.columns)
        if len(tickers) == 0:
            continue

        if prev_date not in close_prices.index or date not in close_prices.index:
            continue

        stock_ret = close_prices.loc[date, tickers] / close_prices.loc[prev_date, tickers] - 1.0
        port_ret_val = float((weights.reindex(tickers) * stock_ret).sum())
        returns_list.append(port_ret_val)

    if not returns_list or date not in bench_ret.index:
        return float("nan")

    port_returns = pd.Series(returns_list)
    common_len = min(len(port_returns), len(bench_ret))
    excess = port_returns.values[:common_len] - bench_ret.values[:common_len]
    return float(np.std(excess) * np.sqrt(252))
