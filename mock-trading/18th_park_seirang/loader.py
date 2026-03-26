"""Data loader — FinanceDataReader + OpenDartReader.

Replaces pykrx (broken due to KRX closed-API policy).

Sources:
  FinanceDataReader  → OHLCV, KOSPI ticker list, KOSPI index benchmark, sector
  OpenDartReader     → PBR / ROE fundamentals (requires DART_API_KEY)

Market cap proxy:
  Historical market cap = listed_shares × close_price
  (listed_shares from FDR StockListing — current snapshot, changes infrequently)

Known limitation:
  FDR StockListing is current-date only → no historical constituent changes.
  Delisted stocks are absent → minor survivorship bias.
"""

import logging
import os
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_path(name: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{name}.parquet")


def _save(df: pd.DataFrame, name: str) -> None:
    df.to_parquet(_cache_path(name))


def _load(name: str) -> Optional[pd.DataFrame]:
    path = _cache_path(name)
    if os.path.exists(path):
        try:
            return pd.read_parquet(path)
        except Exception:
            return None
    return None


# ---------------------------------------------------------------------------
# Internal: KOSPI listing (cached once — shares + sector)
# ---------------------------------------------------------------------------

def _get_kospi_listing(force_refresh: bool = False) -> pd.DataFrame:
    """Return current KOSPI listing with Stocks (shares) and Sector columns."""
    if not force_refresh:
        cached = _load("kospi_listing")
        if cached is not None:
            return cached

    import FinanceDataReader as fdr  # type: ignore[import-untyped]

    df = fdr.StockListing("KOSPI")
    if df is None or df.empty:
        logger.error("FDR StockListing('KOSPI') returned empty.")
        return pd.DataFrame()

    # Normalize symbol column
    sym_col = next((c for c in ["Symbol", "Code", "종목코드"] if c in df.columns), None)
    if sym_col and sym_col != "Symbol":
        df = df.rename(columns={sym_col: "Symbol"})

    df["Symbol"] = df["Symbol"].astype(str).str.zfill(6)
    _save(df, "kospi_listing")
    logger.info("KOSPI listing loaded: %d tickers", len(df))
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_trading_days(start: str, end: str) -> pd.DatetimeIndex:
    """Trading calendar from KOSPI index (KS11)."""
    import FinanceDataReader as fdr  # type: ignore[import-untyped]

    cache_key = f"trading_days_{start}_{end}"
    cached = _load(cache_key)
    if cached is not None:
        return pd.DatetimeIndex(cached["date"])

    df = fdr.DataReader("KS11", start, end)
    idx = pd.DatetimeIndex(df.index)
    pd.DataFrame({"date": idx}).to_parquet(_cache_path(cache_key))
    logger.info("Trading days: %d", len(idx))
    return idx


def get_kospi_tickers(date: pd.Timestamp) -> List[str]:
    """Current KOSPI ordinary share ticker list.

    Note: FDR does not support historical listings.
    The same current list is returned for all dates (minor survivorship bias).
    """
    listing = _get_kospi_listing()
    if listing.empty or "Symbol" not in listing.columns:
        return []
    return listing["Symbol"].tolist()


def load_ohlcv(
    tickers: List[str],
    start: str,
    end: str,
    force_refresh: bool = False,
) -> Dict[str, pd.DataFrame]:
    """Load OHLCV for each ticker via FDR. Adds Turnover = Close × Volume.

    Returns:
        dict {ticker → DataFrame(Open, High, Low, Close, Volume, Change, Turnover)}
    """
    import FinanceDataReader as fdr  # type: ignore[import-untyped]

    result: Dict[str, pd.DataFrame] = {}
    total = len(tickers)

    for i, ticker in enumerate(tickers, 1):
        cache_key = f"ohlcv_{ticker}_{start}_{end}"
        if not force_refresh:
            cached = _load(cache_key)
            if cached is not None:
                result[ticker] = cached
                continue

        try:
            df = fdr.DataReader(ticker, start, end)
            if df is not None and not df.empty:
                # Compute trading value (KRW proxy)
                if "Close" in df.columns and "Volume" in df.columns:
                    df["Turnover"] = df["Close"] * df["Volume"]
                _save(df, cache_key)
                result[ticker] = df
            else:
                logger.debug("No OHLCV data for %s", ticker)
        except Exception as exc:
            logger.debug("OHLCV failed for %s: %s", ticker, exc)

        if i % 100 == 0 or i == total:
            logger.info("OHLCV: %d / %d tickers loaded", i, total)

    return result


def load_market_cap_snapshot(
    date: pd.Timestamp,
    close_prices: Optional[pd.DataFrame] = None,
    force_refresh: bool = False,
) -> pd.Series:
    """Market cap proxy = listed_shares × close_price at given date.

    Args:
        date:         Snapshot date.
        close_prices: Full close price DataFrame (date × ticker). Used to pick
                      the price at `date`. If None, falls back to current Marcap.
        force_refresh: Re-download listing.
    """
    listing = _get_kospi_listing(force_refresh)
    if listing.empty:
        return pd.Series(dtype=float)

    # Find shares column
    shares_col = next(
        (c for c in ["Stocks", "SharesOutstanding", "상장주식수"] if c in listing.columns),
        None,
    )
    marcap_col = next(
        (c for c in ["Marcap", "시가총액"] if c in listing.columns), None
    )

    listing = listing.copy()
    listing["Symbol"] = listing["Symbol"].astype(str).str.zfill(6)
    listing = listing.set_index("Symbol")

    if shares_col and close_prices is not None:
        # Best path: shares × historical close
        if date in close_prices.index:
            price_row = close_prices.loc[date]
        else:
            # Use nearest available date
            avail = close_prices.index[close_prices.index <= date]
            if len(avail) == 0:
                logger.warning("No close price available at or before %s", date)
                price_row = pd.Series(dtype=float)
            else:
                price_row = close_prices.loc[avail[-1]]

        shares = listing[shares_col].reindex(price_row.index).fillna(0)
        mktcap = shares * price_row
        return mktcap.dropna().astype(float)

    elif marcap_col:
        # Fallback: use current Marcap (look-ahead bias caveat)
        logger.warning(
            "Using current Marcap for %s (no shares × price available). "
            "Minor look-ahead bias introduced.",
            date,
        )
        return listing[marcap_col].dropna().astype(float)

    logger.warning("Cannot compute market cap snapshot for %s", date)
    return pd.Series(dtype=float)


def load_fundamental_snapshot(
    date: pd.Timestamp,
    dart_api_key: str = "",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """PBR and ROE for KOSPI stocks at a rebalancing snapshot date.

    Uses OpenDartReader to fetch annual financial statements.
    For each date, uses the most recent annual report year.

    Returns:
        DataFrame with columns [PBR, ROE], indexed by ticker (Symbol).
        Returns empty DataFrame if dart_api_key is not set.
    """
    if not dart_api_key:
        logger.warning(
            "DART_API_KEY not set — skipping fundamentals. "
            "Value/Quality factors will be zero."
        )
        return pd.DataFrame(columns=["PBR", "ROE"])

    # Use most recent closed fiscal year before this date
    year = date.year if date.month > 3 else date.year - 1
    cache_key = f"fundamentals_{year}"
    if not force_refresh:
        cached = _load(cache_key)
        if cached is not None:
            return cached

    try:
        import OpenDartReader  # type: ignore[import-untyped]
        dart = OpenDartReader(dart_api_key)  # module IS the class
        records = _fetch_dart_fundamentals(dart, year)
        if records:
            df = pd.DataFrame(records).set_index("ticker")
            _save(df.reset_index(), cache_key)
            return df
    except Exception as exc:
        logger.error("OpenDartReader error: %s", exc)

    return pd.DataFrame(columns=["PBR", "ROE"])


def _fetch_dart_fundamentals(dart: object, year: int) -> list:  # type: ignore[return]
    """Fetch PBR and ROE for all KOSPI stocks for a given fiscal year."""
    import OpenDartReader  # type: ignore[import-untyped]

    dart_obj = dart  # type: ignore[assignment]

    listing = _get_kospi_listing()
    if listing.empty:
        return []

    tickers = listing["Symbol"].astype(str).str.zfill(6).tolist()

    # Build stock_code → corp_code map
    try:
        corp_df = dart_obj.corp_codes
        if corp_df is None or corp_df.empty:
            return []
        stock_to_corp: Dict[str, str] = {}
        for _, row in corp_df.iterrows():
            sc = str(row.get("stock_code", "")).zfill(6)
            cc = str(row.get("corp_code", ""))
            if sc and cc:
                stock_to_corp[sc] = cc
    except Exception as exc:
        logger.error("Failed to load DART corp codes: %s", exc)
        return []

    records = []
    ANNUAL_CODE = "11011"  # 사업보고서

    for i, ticker in enumerate(tickers):
        corp_code = stock_to_corp.get(ticker)
        if not corp_code:
            continue
        try:
            fs = dart_obj.finstate_all(corp_code, year, ANNUAL_CODE)
            if fs is None or (hasattr(fs, "empty") and fs.empty):
                continue

            pbr = _extract_dart_metric(fs, ["PBR", "주가순자산비율"])
            roe = _extract_dart_metric(
                fs, ["ROE", "자기자본이익률", "자기자본순이익률"]
            )

            # If PBR not directly available, compute from BPS and price
            if pbr is None:
                bps = _extract_dart_metric(
                    fs, ["BPS", "주당순자산가치", "1주당 순자산가치"]
                )
                # Price will be applied later using close prices
                # Store BPS for now; conversion happens in risk_attribution
                if bps is not None and bps > 0:
                    pbr = bps  # placeholder — needs price to convert

            if roe is not None or pbr is not None:
                records.append(
                    {
                        "ticker": ticker,
                        "PBR": pbr if pbr is not None else float("nan"),
                        "ROE": roe if roe is not None else float("nan"),
                    }
                )
        except Exception:
            continue

        if (i + 1) % 50 == 0:
            logger.info("DART fundamentals: %d / %d tickers", i + 1, len(tickers))

    logger.info("DART fundamentals loaded: %d records for year %d", len(records), year)
    return records


def _extract_dart_metric(fs: pd.DataFrame, keywords: List[str]) -> Optional[float]:
    """Extract a numeric value from a DART finstate DataFrame by keyword."""
    acnt_col = next(
        (c for c in ["account_nm", "account_id", "계정명", "항목명"] if c in fs.columns),
        None,
    )
    val_col = next(
        (
            c
            for c in ["thstrm_amount", "당기", "thstrm_dt", "현재가치"]
            if c in fs.columns
        ),
        None,
    )
    if not acnt_col or not val_col:
        return None

    for kw in keywords:
        mask = fs[acnt_col].astype(str).str.contains(kw, na=False, case=False)
        if mask.any():
            raw = fs.loc[mask, val_col].iloc[0]
            try:
                return float(str(raw).replace(",", "").replace(" ", ""))
            except (ValueError, TypeError):
                continue
    return None


def load_sector_snapshot(
    date: pd.Timestamp,
    krx_to_sector: Dict[str, str],
    force_refresh: bool = False,
    dart_api_key: str = "",
) -> pd.Series:
    """Sector group for each KOSPI ticker.

    Strategy:
      1. If DART API key is available: call dart.company() for each ticker
         to get induty_code (KSIC code), map via KSIC_TO_SECTOR.
      2. Fallback: assign all tickers to "기타".

    The sector map is computed once and cached regardless of `date`
    (FDR/DART data does not vary by historical date in free tier).
    """
    cache_key = "sector_map"
    if not force_refresh:
        cached = _load(cache_key)
        if cached is not None:
            return cached["sector"]

    listing = _get_kospi_listing(force_refresh)
    if listing.empty:
        return pd.Series(dtype=str)

    tickers = listing["Symbol"].astype(str).str.zfill(6).tolist()

    if dart_api_key:
        sector_series = _dart_sector_map(tickers, dart_api_key, krx_to_sector)
    else:
        logger.warning(
            "DART_API_KEY not set — all tickers assigned to '기타'. "
            "Sector neutrality constraint will be approximate."
        )
        sector_series = pd.Series("기타", index=tickers, name="sector")

    out = sector_series.rename("sector").to_frame()
    _save(out, cache_key)
    return sector_series


def _dart_sector_map(
    tickers: List[str],
    dart_api_key: str,
    krx_to_sector: Dict[str, str],
) -> pd.Series:
    """Build ticker → sector mapping via DART company API.

    Uses induty_code (KSIC code) from dart.company() and maps to sector groups
    via KSIC_TO_SECTOR config. Falls back to KRX_TO_SECTOR for Korean names.
    """
    try:
        import OpenDartReader as odr  # type: ignore[import-untyped]
        from config import KSIC_TO_SECTOR
    except ImportError:
        logger.error("OpenDartReader not installed.")
        return pd.Series("기타", index=tickers, name="sector")

    dart = odr(dart_api_key)  # OpenDartReader module IS the class

    # Build stock_code → corp_code lookup
    try:
        corp_df = dart.corp_codes
        stock_to_corp: Dict[str, str] = {}
        if corp_df is not None and not corp_df.empty:
            sc_col = next((c for c in ["stock_code", "종목코드"] if c in corp_df.columns), None)
            cc_col = next((c for c in ["corp_code", "고유번호"] if c in corp_df.columns), None)
            if sc_col and cc_col:
                for _, row in corp_df.iterrows():
                    sc = str(row[sc_col]).zfill(6)
                    cc = str(row[cc_col])
                    if sc and cc:
                        stock_to_corp[sc] = cc
    except Exception as exc:
        logger.error("Failed to load DART corp codes: %s", exc)
        return pd.Series("기타", index=tickers, name="sector")

    result: Dict[str, str] = {}
    total = len(tickers)

    for i, ticker in enumerate(tickers, 1):
        corp_code = stock_to_corp.get(ticker)
        if not corp_code:
            result[ticker] = "기타"
            continue
        try:
            info = dart.company(corp_code)
            if info is None or (hasattr(info, "empty") and info.empty):
                result[ticker] = "기타"
                continue

            induty_raw = ""
            if isinstance(info, pd.DataFrame):
                icol = next((c for c in ["induty_code", "업종코드"] if c in info.columns), None)
                if icol:
                    induty_raw = str(info[icol].iloc[0]).strip()
            elif isinstance(info, dict):
                induty_raw = str(info.get("induty_code", "")).strip()

            # Match KSIC code by prefix (longest match first)
            sector = "기타"
            for prefix in sorted(KSIC_TO_SECTOR.keys(), key=len, reverse=True):
                if induty_raw.upper().startswith(prefix.upper()):
                    sector = KSIC_TO_SECTOR[prefix]
                    break
            # Fallback: try KRX Korean name mapping
            if sector == "기타" and induty_raw in krx_to_sector:
                sector = krx_to_sector[induty_raw]

            result[ticker] = sector

        except Exception:
            result[ticker] = "기타"

        if i % 100 == 0 or i == total:
            logger.info("DART sector map: %d / %d tickers", i, total)

    logger.info(
        "Sector map built via DART: %d tickers, %d non-기타",
        len(result),
        sum(1 for v in result.values() if v != "기타"),
    )
    return pd.Series(result, name="sector")


def load_benchmark(
    start: str,
    end: str,
    candidate_tickers: List[str],  # kept for interface compatibility, unused
    force_refresh: bool = False,
) -> pd.Series:
    """Load KOSPI index (KS11) as benchmark via FDR.

    candidate_tickers is ignored — FDR's KS11 is used directly.
    """
    import FinanceDataReader as fdr  # type: ignore[import-untyped]

    cache_key = f"benchmark_ks11_{start}_{end}"
    if not force_refresh:
        cached = _load(cache_key)
        if cached is not None:
            logger.info("Benchmark: KOSPI (KS11) loaded from cache")
            return cached["Close"]

    try:
        df = fdr.DataReader("KS11", start, end)
        if df is not None and not df.empty and "Close" in df.columns:
            _save(df[["Close"]], cache_key)
            logger.info(
                "Benchmark: KOSPI index (KS11) via FinanceDataReader. "
                "Note: price return only (dividends NOT reinvested)."
            )
            return df["Close"]
    except Exception as exc:
        logger.error("FDR KS11 benchmark failed: %s", exc)

    raise RuntimeError("Cannot load KOSPI benchmark via FDR.")
