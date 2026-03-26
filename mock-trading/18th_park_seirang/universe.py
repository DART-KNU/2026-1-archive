"""Universe filter: removes illiquid and low-free-float stocks.

Filters applied (PRD spec):
  1. 30-day ADTV >= 50억 KRW
  2. Free-float ratio >= 15%  (approximated via pykrx foreign exhaustion data)
  3. No trading suspension in the last 30 days (zero trading value)
"""

import logging
from typing import List

import pandas as pd

from config import ADTV_THRESHOLD, FREE_FLOAT_THRESHOLD, ADTV_WINDOW

logger = logging.getLogger(__name__)


def _get_free_float_ratios(date: pd.Timestamp) -> pd.Series:
    """Approximate free-float ratios from KRX foreign investment limit data.

    pykrx provides the ratio of listed shares that foreigners can hold
    (외국인 한도비율). This is used as a proxy for free-float.

    Returns Series(ticker -> ratio in [0,1]).
    """
    from pykrx import stock  # type: ignore[import-untyped]

    date_str = date.strftime("%Y%m%d")
    try:
        df = stock.get_exhaustion_rates_of_foreign_investment_by_ticker(
            date_str, market="KOSPI"
        )
        if df is None or df.empty:
            return pd.Series(dtype=float)

        # Look for the limit ratio column; column names may vary
        ratio_col = next(
            (c for c in ["한도비율", "보유한도비율", "LimitRatio"] if c in df.columns),
            None,
        )
        if ratio_col is None:
            # Fallback: assume all pass (conservative — will rely on ADTV filter)
            logger.warning(
                "Free-float ratio column not found at %s. Columns: %s",
                date,
                df.columns.tolist(),
            )
            return pd.Series(dtype=float)

        return df[ratio_col] / 100.0  # Convert percentage to ratio

    except Exception as exc:
        logger.warning("Free-float data unavailable at %s: %s", date, exc)
        return pd.Series(dtype=float)


def filter_universe(
    ref_date: pd.Timestamp,
    adtv_30d: pd.DataFrame,
    kospi_tickers: List[str],
) -> List[str]:
    """Return tickers that pass all universe filters at ref_date.

    Args:
        ref_date:      t-1 reference date (data cut-off, no look-ahead).
        adtv_30d:      Pre-computed 30-day rolling ADTV DataFrame (date × ticker).
        kospi_tickers: All KOSPI ordinary share tickers as of ref_date.

    Returns:
        Sorted list of tickers that pass all filters.
    """
    if ref_date not in adtv_30d.index:
        logger.warning("ref_date %s not in adtv_30d index; using nearest.", ref_date)
        ref_date = adtv_30d.index[adtv_30d.index <= ref_date][-1]

    # --- Filter 1: ADTV >= 50억 ---
    adtv_row = adtv_30d.loc[ref_date].reindex(kospi_tickers).fillna(0.0)
    adtv_pass = adtv_row[adtv_row >= ADTV_THRESHOLD].index.tolist()

    # --- Filter 2: Free-float ratio >= 15% ---
    ff_ratios = _get_free_float_ratios(ref_date)
    if ff_ratios.empty:
        # Cannot get free-float data; skip this filter, log warning
        logger.warning(
            "Free-float filter skipped at %s (no data). ADTV filter only.", ref_date
        )
        ff_pass = adtv_pass
    else:
        ff_series = ff_ratios.reindex(adtv_pass).fillna(1.0)  # unknown = assume pass
        ff_pass = ff_series[ff_series >= FREE_FLOAT_THRESHOLD].index.tolist()

    # --- Filter 3: No suspension in last 30 days ---
    # A ticker is considered suspended if trading value was 0 on any day in last 30 days
    window_start_idx = max(0, adtv_30d.index.get_loc(ref_date) - ADTV_WINDOW + 1)  # type: ignore[arg-type]
    window = adtv_30d.iloc[window_start_idx : adtv_30d.index.get_loc(ref_date) + 1]  # type: ignore[misc]
    # Keep tickers that had at least ADTV_WINDOW/2 trading days with positive value
    min_trading_days = ADTV_WINDOW // 2
    active_tickers = (
        window[ff_pass]
        .gt(0)
        .sum()
        .pipe(lambda s: s[s >= min_trading_days].index.tolist())
    )

    logger.info(
        "Universe filter at %s: KOSPI=%d → ADTV=%d → FF=%d → active=%d",
        ref_date.date(),
        len(kospi_tickers),
        len(adtv_pass),
        len(ff_pass),
        len(active_tickers),
    )

    return sorted(active_tickers)
