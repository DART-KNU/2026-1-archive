"""Barra-style factor risk decomposition.

Factors (5 total):
  1. Size       — ln(market_cap) cross-sectional z-score
  2. Momentum   — 12-1 month return z-score
  3. Low Vol    — 60-day realised volatility (inverse z-score)
  4. Value      — PBR inverse z-score  (from fundamental snapshot)
  5. Quality    — ROE z-score          (from fundamental snapshot)

Decomposition at the LAST rebalancing date:
  total_variance   = w' Σ w
  factor_variance  = w' (B F B') w
  specific_variance= w' D w
  pct_factor       = factor_variance / total_variance
"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _zscore(s: pd.Series) -> pd.Series:
    mu = s.mean()
    sigma = s.std()
    if sigma == 0:
        return pd.Series(0.0, index=s.index)
    return (s - mu) / sigma


def compute_factor_exposures(
    tickers: list,
    ref_date: pd.Timestamp,
    close_prices: pd.DataFrame,
    mktcap_snap: pd.Series,
    fundamental_snap: pd.DataFrame,
    trading_days: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Compute cross-sectional factor exposures (z-scores) for a list of tickers.

    Args:
        tickers:         Universe tickers to compute exposures for.
        ref_date:        Reference date for factor computation.
        close_prices:    Full close price DataFrame.
        mktcap_snap:     Market cap Series at ref_date.
        fundamental_snap: DataFrame with PBR and ROE columns at ref_date.
        trading_days:    Full trading calendar.

    Returns:
        DataFrame (tickers × [size, momentum, low_vol, value, quality])
    """
    exposures: Dict[str, pd.Series] = {}

    # ----------------------------------------------------------------
    # 1. Size — ln(market_cap)
    # ----------------------------------------------------------------
    caps = mktcap_snap.reindex(tickers).fillna(0.0)
    caps = caps.clip(lower=1.0)  # avoid log(0)
    exposures["size"] = _zscore(np.log(caps))

    # ----------------------------------------------------------------
    # 2. Momentum — 12-month return minus last month (skip-1-month)
    # ----------------------------------------------------------------
    if ref_date in close_prices.index:
        ref_pos = trading_days.get_loc(ref_date)

        # ~252 trading days ago (12 months)
        start_12m_pos = max(0, ref_pos - 252)
        # ~21 trading days ago (1 month)
        start_1m_pos = max(0, ref_pos - 21)

        close_now = close_prices.loc[ref_date, tickers]
        close_12m = close_prices.iloc[start_12m_pos][tickers] if start_12m_pos >= 0 else close_now
        close_1m = close_prices.iloc[start_1m_pos][tickers] if start_1m_pos >= 0 else close_now

        ret_12_1 = (close_1m / close_12m - 1.0).fillna(0.0)
        exposures["momentum"] = _zscore(ret_12_1)
    else:
        exposures["momentum"] = pd.Series(0.0, index=tickers)

    # ----------------------------------------------------------------
    # 3. Low Volatility — 60-day realised vol (lower vol = higher exposure)
    # ----------------------------------------------------------------
    if ref_date in close_prices.index:
        ref_pos = trading_days.get_loc(ref_date)
        start_60d_pos = max(0, ref_pos - 60)
        window = close_prices.iloc[start_60d_pos : ref_pos + 1][tickers]
        daily_ret = window.pct_change().dropna()
        vol_60d = daily_ret.std() * np.sqrt(252)
        vol_60d = vol_60d.fillna(vol_60d.median())
        # Invert: low vol → high exposure
        exposures["low_vol"] = _zscore(-vol_60d)
    else:
        exposures["low_vol"] = pd.Series(0.0, index=tickers)

    # ----------------------------------------------------------------
    # 4. Value — PBR inverse (lower PBR = higher value exposure)
    # ----------------------------------------------------------------
    if "PBR" in fundamental_snap.columns:
        pbr = fundamental_snap["PBR"].reindex(tickers).fillna(fundamental_snap["PBR"].median())
        pbr = pbr.clip(lower=0.01)
        exposures["value"] = _zscore(1.0 / pbr)
    else:
        logger.warning("PBR not available in fundamental snapshot; value factor = 0")
        exposures["value"] = pd.Series(0.0, index=tickers)

    # ----------------------------------------------------------------
    # 5. Quality — ROE
    # ----------------------------------------------------------------
    if "ROE" in fundamental_snap.columns:
        roe = fundamental_snap["ROE"].reindex(tickers).fillna(0.0)
        exposures["quality"] = _zscore(roe)
    else:
        logger.warning("ROE not available in fundamental snapshot; quality factor = 0")
        exposures["quality"] = pd.Series(0.0, index=tickers)

    return pd.DataFrame(exposures, index=tickers)


def decompose_risk(
    portfolio_weights: pd.Series,
    factor_exposures: pd.DataFrame,
    close_prices: pd.DataFrame,
    ref_date: pd.Timestamp,
    trading_days: pd.DatetimeIndex,
    lookback_days: int = 252,
) -> Dict[str, float]:
    """Decompose portfolio variance into factor and specific components.

    Uses realised covariance of factor returns estimated via cross-sectional
    regression of daily stock returns on factor exposures.

    Args:
        portfolio_weights:  Series(ticker → weight).
        factor_exposures:   DataFrame(ticker × factor) — z-scores.
        close_prices:       Full price DataFrame.
        ref_date:           End date for the estimation window.
        trading_days:       Full trading calendar.
        lookback_days:      Days to estimate factor covariance.

    Returns:
        Dict with keys: factor_variance, specific_variance, total_variance,
                        factor_pct, specific_pct, factor_exposures_summary.
    """
    tickers = portfolio_weights.index.tolist()
    B = factor_exposures.reindex(tickers).fillna(0.0).values  # (n × 5)
    w = portfolio_weights.reindex(tickers).fillna(0.0).values  # (n,)

    # Estimate stock return covariance from recent history
    if ref_date not in close_prices.index:
        logger.warning("ref_date %s not in close_prices; cannot decompose risk.", ref_date)
        return {}

    ref_pos = trading_days.get_loc(ref_date)
    start_pos = max(0, ref_pos - lookback_days)
    window_prices = close_prices.iloc[start_pos : ref_pos + 1][tickers].dropna(axis=1, how="any")

    if window_prices.shape[1] < 2 or window_prices.shape[0] < 30:
        logger.warning("Insufficient data for risk decomposition.")
        return {}

    valid_tickers = window_prices.columns.tolist()
    w_valid = portfolio_weights.reindex(valid_tickers).fillna(0.0)
    w_valid = w_valid / w_valid.sum()
    B_valid = factor_exposures.reindex(valid_tickers).fillna(0.0).values

    daily_ret = window_prices.pct_change().dropna()

    # Estimate factor returns via OLS at each date: r_t = B * f_t + e_t
    factor_returns_list = []
    residuals_list = []

    for _, row in daily_ret.iterrows():
        r = row.values  # (n_valid,)
        # OLS: f = (B'B)^{-1} B' r
        try:
            f, resid, _, _ = np.linalg.lstsq(B_valid, r, rcond=None)
            e = r - B_valid @ f
        except np.linalg.LinAlgError:
            continue
        factor_returns_list.append(f)
        residuals_list.append(e)

    if not factor_returns_list:
        logger.warning("No factor returns estimated.")
        return {}

    F_returns = np.array(factor_returns_list)  # (T × k)
    E_returns = np.array(residuals_list)        # (T × n_valid)

    # Factor covariance (annualised)
    F_cov = np.cov(F_returns.T) * 252.0  # (k × k)
    # Specific (idiosyncratic) variance (annualised)
    D_diag = np.var(E_returns, axis=0) * 252.0  # (n_valid,)

    # Decompose portfolio variance
    w_arr = w_valid.values
    factor_var = float(w_arr @ B_valid @ F_cov @ B_valid.T @ w_arr)
    specific_var = float(w_arr @ np.diag(D_diag) @ w_arr)
    total_var = factor_var + specific_var

    if total_var <= 0:
        return {}

    factor_pct = factor_var / total_var
    specific_pct = specific_var / total_var

    # Factor exposure summary (z-scores weighted by portfolio weights)
    factor_names = factor_exposures.columns.tolist()
    weighted_exposures = {
        fname: float(w_valid.values @ factor_exposures.reindex(valid_tickers)[fname].values)
        for fname in factor_names
    }

    result: Dict[str, float] = {
        "factor_variance": factor_var,
        "specific_variance": specific_var,
        "total_variance": total_var,
        "factor_pct": factor_pct,
        "specific_pct": specific_pct,
        "annualised_factor_vol": float(np.sqrt(factor_var)) * 100,
        "annualised_specific_vol": float(np.sqrt(specific_var)) * 100,
    }
    result.update(weighted_exposures)
    return result
