"""
QuantAlpha Backtest Runner
===========================
Evaluates a list of FactorSpec objects over KOSPI OHLCV data and
returns IC / Rank IC / ICIR / annualised return metrics.

Does NOT depend on Qlib — uses pure pandas/numpy, so it runs anywhere.

Flow:
  1. For each FactorSpec, compute factor values via FactorEngineMulti
  2. Compute forward returns (pred_horizon days ahead)
  3. Compute cross-sectional IC and Rank IC per bar
  4. Compute portfolio performance: TopK long-only, equal-weighted

Output:
  BacktestResult dataclass with metrics + StrategyTrajectory for the loop.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from quantalpha_factor_engine import FactorEngineMulti
from quantalpha_hypothesis import FactorSpec, Hypothesis

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Result structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class FactorMetrics:
    """IC metrics for a single factor."""
    factor_name:  str
    expression:   str
    ic:           float = 0.0       # Pearson IC mean
    icir:         float = 0.0       # IC / IC.std
    rank_ic:      float = 0.0       # Spearman Rank IC mean
    rank_icir:    float = 0.0
    coverage:     float = 0.0       # fraction of dates with valid IC
    quality:      str   = "low"     # high / medium / low

    def as_dict(self) -> dict:
        return {
            "factor_name": self.factor_name,
            "expression":  self.expression,
            "IC":          round(self.ic, 6),
            "ICIR":        round(self.icir, 6),
            "Rank IC":     round(self.rank_ic, 6),
            "Rank ICIR":   round(self.rank_icir, 6),
            "coverage":    round(self.coverage, 3),
            "quality":     self.quality,
        }


@dataclass
class BacktestResult:
    """Full backtest result for one loop iteration (= one set of factors)."""
    hypothesis:       str
    factor_metrics:   list[FactorMetrics] = field(default_factory=list)
    # portfolio metrics (simple TopK strategy)
    annualized_return: float = 0.0
    max_drawdown:      float = 0.0
    information_ratio: float = 0.0
    calmar_ratio:      float = 0.0
    # aggregate IC over all factors
    ensemble_rank_ic:  float = 0.0
    n_dates:           int   = 0

    def best_factor(self) -> Optional[FactorMetrics]:
        if not self.factor_metrics:
            return None
        return max(self.factor_metrics, key=lambda m: m.rank_ic)

    def promoted_factors(self, rank_ic_threshold: float = 0.02) -> list[FactorMetrics]:
        return [m for m in self.factor_metrics if m.rank_ic >= rank_ic_threshold]

    def summary(self) -> dict:
        return {
            "hypothesis":        self.hypothesis,
            "ensemble_rank_ic":  round(self.ensemble_rank_ic, 6),
            "annualized_return": round(self.annualized_return, 4),
            "max_drawdown":      round(self.max_drawdown, 4),
            "information_ratio": round(self.information_ratio, 4),
            "n_promoted":        len(self.promoted_factors()),
            "factors":           [m.as_dict() for m in self.factor_metrics],
        }


# ──────────────────────────────────────────────────────────────────────────────
# IC computation helpers
# ──────────────────────────────────────────────────────────────────────────────

def _ic_series(factor_df: pd.DataFrame, fwd_df: pd.DataFrame) -> pd.Series:
    """Pearson IC per date (cross-sectional correlation)."""
    common_dates = factor_df.index.intersection(fwd_df.index)
    records = []
    for dt in common_dates:
        f = factor_df.loc[dt].dropna()
        r = fwd_df.loc[dt].dropna()
        common = f.index.intersection(r.index)
        if len(common) < 5:
            continue
        ic = f[common].corr(r[common])
        if not np.isnan(ic):
            records.append((dt, ic))
    if not records:
        return pd.Series(dtype=float)
    dates, vals = zip(*records)
    return pd.Series(vals, index=pd.DatetimeIndex(dates))


def _rank_ic_series(factor_df: pd.DataFrame, fwd_df: pd.DataFrame) -> pd.Series:
    """Spearman Rank IC per date."""
    common_dates = factor_df.index.intersection(fwd_df.index)
    records = []
    for dt in common_dates:
        f = factor_df.loc[dt].dropna()
        r = fwd_df.loc[dt].dropna()
        common = f.index.intersection(r.index)
        if len(common) < 5:
            continue
        ric, _ = spearmanr(f[common], r[common])
        if not np.isnan(ric):
            records.append((dt, float(ric)))
    if not records:
        return pd.Series(dtype=float)
    dates, vals = zip(*records)
    return pd.Series(vals, index=pd.DatetimeIndex(dates))


def _compute_factor_metrics(name: str, expr: str,
                             factor_df: pd.DataFrame,
                             fwd_df: pd.DataFrame) -> FactorMetrics:
    ic_s   = _ic_series(factor_df, fwd_df)
    ric_s  = _rank_ic_series(factor_df, fwd_df)

    def _safe(s: pd.Series):
        if s.empty:
            return 0.0, 0.0
        mu  = float(s.mean())
        std = float(s.std())
        ir  = mu / std if std > 1e-8 else 0.0
        return round(mu, 6), round(ir, 6)

    ic, icir      = _safe(ic_s)
    ric, ricir    = _safe(ric_s)
    coverage      = len(ic_s) / max(len(fwd_df), 1)

    if abs(ric) >= 0.05:
        quality = "high"
    elif abs(ric) >= 0.02:
        quality = "medium"
    else:
        quality = "low"

    return FactorMetrics(
        factor_name=name, expression=expr,
        ic=ic, icir=icir,
        rank_ic=ric, rank_icir=ricir,
        coverage=round(coverage, 3), quality=quality,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Portfolio simulation (TopK equal-weight)
# ──────────────────────────────────────────────────────────────────────────────

def _portfolio_metrics(ensemble_signal: pd.DataFrame,
                       close_prices:   pd.DataFrame,
                       topk: int = 3,
                       rebal_freq: int = 21,
                       cost_bps: float = 15.0) -> dict:
    """
    Simple TopK equal-weight long-only portfolio.
    Returns annualized_return, max_drawdown, information_ratio, calmar_ratio.
    """
    # Align
    common_tickers = ensemble_signal.columns.intersection(close_prices.columns)
    common_dates   = ensemble_signal.index.intersection(close_prices.index)
    if len(common_dates) < 10 or len(common_tickers) < topk:
        return {"annualized_return": 0.0, "max_drawdown": 0.0,
                "information_ratio": 0.0, "calmar_ratio": 0.0}

    sig  = ensemble_signal[common_tickers].reindex(common_dates)
    px   = close_prices[common_tickers].reindex(common_dates)

    nav     = 1.0
    navs    = [nav]
    prev_holdings: set = set()
    rebal_dates = common_dates[::rebal_freq]

    daily_rets = px.pct_change().fillna(0.0)

    portfolio_rets: list[float] = []
    holdings: set = set()

    for i, dt in enumerate(common_dates):
        if i == 0:
            continue
        if dt in rebal_dates:
            row = sig.loc[dt].dropna()
            if len(row) >= topk:
                holdings = set(row.nlargest(topk).index)
                # transaction cost
                turnover = len(holdings.symmetric_difference(prev_holdings))
                nav *= (1 - turnover * cost_bps / 10_000)
                prev_holdings = holdings

        if holdings:
            day_ret = daily_rets.loc[dt][list(holdings)].mean()
        else:
            day_ret = 0.0

        portfolio_rets.append(day_ret)
        nav *= (1 + day_ret)
        navs.append(nav)

    if not portfolio_rets:
        return {"annualized_return": 0.0, "max_drawdown": 0.0,
                "information_ratio": 0.0, "calmar_ratio": 0.0}

    rets = pd.Series(portfolio_rets)
    ann_ret   = float((1 + rets.mean()) ** 252 - 1)
    ann_vol   = float(rets.std() * np.sqrt(252))
    info_rat  = ann_ret / ann_vol if ann_vol > 1e-8 else 0.0

    nav_s = pd.Series(navs)
    roll_max = nav_s.cummax()
    dd = (nav_s - roll_max) / roll_max
    max_dd = float(dd.min())

    calmar = ann_ret / abs(max_dd) if abs(max_dd) > 1e-8 else 0.0

    return {
        "annualized_return": round(ann_ret, 6),
        "max_drawdown":      round(max_dd, 6),
        "information_ratio": round(info_rat, 6),
        "calmar_ratio":      round(calmar, 6),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main runner
# ──────────────────────────────────────────────────────────────────────────────

class QuantAlphaRunner:
    """
    Runs backtest for a list of FactorSpec objects.

    Parameters
    ----------
    price_data    : {ticker: df_ohlcv}  — indexed by date
    pred_horizon  : forward-return horizon in trading days (default 21)
    topk          : portfolio construction — hold top-K tickers by signal
    cost_bps      : round-trip transaction cost in basis points
    """

    def __init__(
        self,
        price_data:   dict[str, pd.DataFrame],
        pred_horizon: int   = 21,
        topk:         int   = 3,
        cost_bps:     float = 15.0,
    ) -> None:
        self._price_data  = price_data
        self._pred_horizon = pred_horizon
        self._topk        = topk
        self._cost_bps    = cost_bps
        self._multi       = FactorEngineMulti(price_data)
        self._fwd_returns = self._build_forward_returns()
        self._close_prices = self._build_close_prices()

    def _build_forward_returns(self) -> pd.DataFrame:
        """Build forward return DataFrame: (date × ticker)."""
        cols = {}
        for ticker, df in self._price_data.items():
            engine_df = self._multi._engines[ticker]._df
            fwd = engine_df["close"].pct_change(self._pred_horizon).shift(-self._pred_horizon)
            cols[ticker] = fwd
        return pd.DataFrame(cols)

    def _build_close_prices(self) -> pd.DataFrame:
        """Close price panel for portfolio simulation."""
        cols = {}
        for ticker, _ in self._price_data.items():
            engine_df = self._multi._engines[ticker]._df
            cols[ticker] = engine_df["close"]
        return pd.DataFrame(cols)

    def run(self, hypothesis: Hypothesis, factors: list[FactorSpec]) -> BacktestResult:
        """
        Evaluate all factors, compute IC metrics, simulate portfolio.
        Returns BacktestResult.
        """
        logger.info(f"Running backtest for {len(factors)} factors | {hypothesis.hypothesis[:60]}...")

        factor_metrics_list: list[FactorMetrics] = []
        factor_dfs: list[pd.DataFrame] = []

        for spec in factors:
            logger.info(f"  Evaluating: {spec.factor_name} | {spec.factor_expression[:60]}")
            try:
                fdf = self._multi.eval(spec.factor_expression)
                metrics = _compute_factor_metrics(
                    spec.factor_name, spec.factor_expression,
                    fdf, self._fwd_returns,
                )
                factor_metrics_list.append(metrics)
                factor_dfs.append(fdf)
                logger.info(
                    f"    IC={metrics.ic:+.4f}  RankIC={metrics.rank_ic:+.4f}  "
                    f"ICIR={metrics.icir:+.4f}  quality={metrics.quality}"
                )
            except Exception as e:
                logger.warning(f"    Failed to evaluate {spec.factor_name}: {e}")
                factor_metrics_list.append(FactorMetrics(
                    factor_name=spec.factor_name,
                    expression=spec.factor_expression,
                ))

        # Ensemble signal = mean of all valid factor signals
        port_metrics = {"annualized_return": 0.0, "max_drawdown": 0.0,
                        "information_ratio": 0.0, "calmar_ratio": 0.0}
        ensemble_rank_ic = 0.0

        if factor_dfs:
            # Align and average
            ensemble = factor_dfs[0].copy()
            for fdf in factor_dfs[1:]:
                ensemble = ensemble.add(fdf, fill_value=0.0)
            ensemble = ensemble / len(factor_dfs)

            # Ensemble IC
            ric_s = _rank_ic_series(ensemble, self._fwd_returns)
            ensemble_rank_ic = float(ric_s.mean()) if not ric_s.empty else 0.0

            # Portfolio simulation
            port_metrics = _portfolio_metrics(
                ensemble, self._close_prices,
                topk=self._topk, cost_bps=self._cost_bps,
            )

        return BacktestResult(
            hypothesis=str(hypothesis),
            factor_metrics=factor_metrics_list,
            ensemble_rank_ic=round(ensemble_rank_ic, 6),
            n_dates=len(self._fwd_returns),
            **port_metrics,
        )
