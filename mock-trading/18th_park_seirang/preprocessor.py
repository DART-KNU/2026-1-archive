"""Data preprocessing: builds clean, wide DataFrames from raw loader output.

Outputs consumed by strategy and backtest modules:
  - close_prices      : DataFrame (date × ticker) — adjusted close
  - open_prices       : DataFrame (date × ticker) — adjusted open
  - trading_values    : DataFrame (date × ticker) — daily KRW trading value (Close×Volume)
  - adtv_30d          : DataFrame (date × ticker) — 30-day rolling avg trading value
  - mktcap_snaps      : dict {rebal_ref_date → Series(ticker → market_cap)}
  - fundamental_snaps : dict {rebal_ref_date → DataFrame(ticker, [PBR, ROE])}
  - sector_snaps      : dict {rebal_ref_date → Series(ticker → sector_name)}
  - benchmark         : Series (date → index close level)
  - trading_days      : DatetimeIndex
"""

import logging
from typing import Dict, List, Tuple

import pandas as pd

from data.loader import (
    get_trading_days,
    get_kospi_tickers,
    load_ohlcv,
    load_market_cap_snapshot,
    load_fundamental_snapshot,
    load_sector_snapshot,
    load_benchmark,
)
from config import (
    START_DATE,
    END_DATE,
    ADTV_WINDOW,
    BENCHMARK_TICKERS,
    KRX_TO_SECTOR,
    DART_API_KEY,
    KSIC_TO_SECTOR,
)

logger = logging.getLogger(__name__)

# FDR uses English column names; pykrx used Korean.
# We handle both so switching back is trivial.
_CLOSE_COLS  = ["Close", "종가"]
_OPEN_COLS   = ["Open",  "시가"]
_TV_COLS     = ["Turnover", "거래대금"]  # loader adds Turnover = Close × Volume


def _pick_col(df: pd.DataFrame, candidates: list) -> str | None:
    return next((c for c in candidates if c in df.columns), None)


def build_all(
    rebal_ref_dates: List[pd.Timestamp],
    force_refresh: bool = False,
    collect_fundamentals: bool = False,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    Dict[pd.Timestamp, pd.Series],
    Dict[pd.Timestamp, pd.DataFrame],
    Dict[pd.Timestamp, pd.Series],
    pd.Series,
    pd.DatetimeIndex,
]:
    """Download (or restore from cache) all data needed for the backtest."""

    # ------------------------------------------------------------------
    # 1. Trading calendar
    # ------------------------------------------------------------------
    logger.info("Loading trading calendar...")
    trading_days = get_trading_days(START_DATE, END_DATE)
    logger.info("Trading days: %d", len(trading_days))

    # ------------------------------------------------------------------
    # 2. Universe: union of tickers across rebalancing dates
    # ------------------------------------------------------------------
    logger.info("Collecting universe tickers...")
    all_tickers: set[str] = set()
    for ref_date in rebal_ref_dates:
        all_tickers.update(get_kospi_tickers(ref_date))
    ticker_list = sorted(all_tickers)
    logger.info("Total unique tickers: %d", len(ticker_list))

    # ------------------------------------------------------------------
    # 3. OHLCV for all tickers (full period, cached per ticker)
    # ------------------------------------------------------------------
    logger.info("Loading OHLCV for %d tickers...", len(ticker_list))
    ohlcv_map = load_ohlcv(ticker_list, START_DATE, END_DATE, force_refresh)

    close_dict: Dict[str, pd.Series] = {}
    open_dict:  Dict[str, pd.Series] = {}
    tv_dict:    Dict[str, pd.Series] = {}

    for ticker, df in ohlcv_map.items():
        if df.empty:
            continue

        df_ri = df.reindex(trading_days)

        c_col = _pick_col(df, _CLOSE_COLS)
        o_col = _pick_col(df, _OPEN_COLS)
        t_col = _pick_col(df, _TV_COLS)

        if c_col:
            close_dict[ticker] = df_ri[c_col].ffill()
        if o_col:
            # Open on rebalancing day — no ffill (must be real open or NaN)
            open_dict[ticker] = df_ri[o_col]
        if t_col:
            tv_dict[ticker] = df_ri[t_col].fillna(0.0)

    close_prices  = pd.DataFrame(close_dict,  index=trading_days).sort_index()
    open_prices   = pd.DataFrame(open_dict,   index=trading_days).sort_index()
    trading_values = pd.DataFrame(tv_dict,   index=trading_days).sort_index()

    # Drop all-NaN tickers
    close_prices  = close_prices.dropna(axis=1, how="all")
    valid_tickers = close_prices.columns
    open_prices   = open_prices.reindex(columns=valid_tickers)
    trading_values = trading_values.reindex(columns=valid_tickers).fillna(0.0)

    logger.info(
        "Price matrices: %d dates × %d tickers",
        len(close_prices), len(valid_tickers),
    )

    # ------------------------------------------------------------------
    # 4. ADTV (30-day rolling average of Turnover = Close × Volume)
    # ------------------------------------------------------------------
    adtv_30d = trading_values.rolling(window=ADTV_WINDOW, min_periods=1).mean()

    # ------------------------------------------------------------------
    # 5. Market cap snapshots — shares × close_price (24 snapshots)
    # ------------------------------------------------------------------
    logger.info("Computing market cap snapshots (%d dates)...", len(rebal_ref_dates))
    mktcap_snaps: Dict[pd.Timestamp, pd.Series] = {}
    for ref_date in rebal_ref_dates:
        snap = load_market_cap_snapshot(ref_date, close_prices, force_refresh)
        if snap.empty:
            logger.warning("Empty market cap snapshot at %s", ref_date)
        mktcap_snaps[ref_date] = snap

    # ------------------------------------------------------------------
    # 6. Fundamental snapshots — PBR, ROE via OpenDartReader (optional)
    #    Only used for risk attribution (Value/Quality factors).
    #    Set collect_fundamentals=True to enable. Default: skip.
    # ------------------------------------------------------------------
    fundamental_snaps: Dict[pd.Timestamp, pd.DataFrame] = {}
    if collect_fundamentals:
        logger.info("Loading fundamental snapshots (%d dates)...", len(rebal_ref_dates))
        seen_years: dict[int, pd.DataFrame] = {}
        for ref_date in rebal_ref_dates:
            year = ref_date.year if ref_date.month > 3 else ref_date.year - 1
            if year not in seen_years:
                snap = load_fundamental_snapshot(ref_date, DART_API_KEY, force_refresh)
                seen_years[year] = snap
            fundamental_snaps[ref_date] = seen_years[year]
    else:
        logger.info("Fundamental snapshots skipped (collect_fundamentals=False).")

    # ------------------------------------------------------------------
    # 7. Sector snapshots (24 dates — FDR is static, same data each time)
    # ------------------------------------------------------------------
    logger.info("Loading sector snapshots (%d dates)...", len(rebal_ref_dates))
    sector_snaps: Dict[pd.Timestamp, pd.Series] = {}
    _sector_cache: pd.Series | None = None
    for ref_date in rebal_ref_dates:
        if _sector_cache is None:
            _sector_cache = load_sector_snapshot(
                ref_date, KRX_TO_SECTOR, force_refresh, DART_API_KEY
            )
        sector_snaps[ref_date] = _sector_cache  # static; DART/FDR data doesn't vary by date

    # ------------------------------------------------------------------
    # 8. Benchmark index (KOSPI KS11)
    # ------------------------------------------------------------------
    logger.info("Loading benchmark index (KOSPI KS11)...")
    benchmark_raw = load_benchmark(START_DATE, END_DATE, BENCHMARK_TICKERS, force_refresh)
    benchmark = benchmark_raw.reindex(trading_days).ffill()

    logger.info("Data build complete.")
    return (
        close_prices,
        open_prices,
        trading_values,
        adtv_30d,
        mktcap_snaps,
        fundamental_snaps,
        sector_snaps,
        benchmark,
        trading_days,
    )
