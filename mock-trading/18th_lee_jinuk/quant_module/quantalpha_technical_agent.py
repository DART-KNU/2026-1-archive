"""
QuantAlpha Technical Agent  (backwards-compatible drop-in)
===========================================================
Fully backwards-compatible replacement for TechnicalAgent that silently
augments MACD/RSI/BB scoring with IC-weighted alpha factors from the
QuantAlpha factor library — without breaking any existing H-MAS call site.

Existing call patterns — all still work unchanged:
  ① TechnicalAgent()                         ← zero-arg construction
  ② agent.score_series(price_list)[-1]       ← list[float] → float
  ③ agent.process_series(price_list)         ← unchanged → TechnicalFactors

New API (when OHLCV DataFrame available):
  ④ agent.score_ticker(ticker, price_df)     ← full blended score
  ⑤ agent.score_universe({ticker: df})       ← batch scoring

One-line drop-in (hmas_agent_backtest.py, kospi_pipeline.py):
  # Before:
  from technical_agent import TechnicalAgent
  # After:
  from quantalpha_technical_agent import QuantAlphaTechnicalAgent as TechnicalAgent
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

# ── import base TechnicalAgent from project ──────────────────────────────────
_PROJECT_DIR = str(Path(__file__).resolve().parent)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

try:
    from technical_agent import (
        TechnicalAgent, TechnicalFactors, ScoringWeights, compute_technical_score,
    )
except ImportError:
    # Minimal stubs so this file loads standalone
    class TechnicalAgent:                                       # type: ignore
        def process_series(self, prices): return []
        def score_series(self, prices):   return [50.0] * len(prices)
    class TechnicalFactors:                                     # type: ignore
        technical_score: float = 50.0
    class ScoringWeights: pass                                  # type: ignore
    def compute_technical_score(f, w=None): return 50.0        # type: ignore

from quantalpha_library       import FactorLibrary, build_technical_weights
from quantalpha_factor_engine import FactorEngine

logger = logging.getLogger(__name__)

DEFAULT_LIBRARY_PATH = Path(__file__).parent / "quantalpha_factor_library.json"


# ──────────────────────────────────────────────────────────────────────────────
# Drop-in agent
# ──────────────────────────────────────────────────────────────────────────────

class QuantAlphaTechnicalAgent(TechnicalAgent):
    """
    Drop-in replacement for TechnicalAgent — fully backwards compatible.

    When the factor library is empty or missing → 100% legacy behaviour.
    When factors are present → final = base_w * legacy + alpha_w * alpha_signal.
    Weights grow automatically as the library accumulates higher-IC factors.
    """

    def __init__(
        self,
        library:          Optional[FactorLibrary] = None,
        macd_fast:        int   = 12,
        macd_slow:        int   = 26,
        macd_signal:      int   = 9,
        rsi_period:       int   = 14,
        roc_short:        int   = 5,
        roc_long:         int   = 21,
        bb_num_std:       float = 2.0,
        bb_bbw_threshold: float = 0.15,
        weights:          Optional[ScoringWeights] = None,
    ) -> None:
        super().__init__(
            macd_fast=macd_fast, macd_slow=macd_slow, macd_signal=macd_signal,
            rsi_period=rsi_period, roc_short=roc_short, roc_long=roc_long,
            bb_num_std=bb_num_std, bb_bbw_threshold=bb_bbw_threshold,
            weights=weights,
        )

        # Auto-load from default path when no library passed
        if library is None and DEFAULT_LIBRARY_PATH.exists():
            try:
                library = FactorLibrary(DEFAULT_LIBRARY_PATH)
            except Exception as e:
                logger.debug(f"Auto-load library failed: {e}")
                library = None

        self.library = library
        self._weight_config: dict = (
            build_technical_weights(library)
            if library and len(library) > 0
            else {"base_weight": 1.0, "alpha_weight": 0.0, "alpha_factors": []}
        )

        n = len(self._weight_config["alpha_factors"])
        aw = self._weight_config["alpha_weight"]
        if n > 0:
            logger.info(f"QuantAlphaTechnicalAgent | alpha_w={aw:.3f} n_factors={n}")
        else:
            logger.debug("QuantAlphaTechnicalAgent | no alpha factors → pure legacy")

    def refresh_weights(self) -> None:
        """Recompute weights after new factors are promoted to library."""
        if self.library:
            self._weight_config = build_technical_weights(self.library)

    # ── BACKWARDS-COMPATIBLE: score_series(list[float]) ──────────────────────

    def score_series(self, prices: Sequence[float]) -> list[float]:
        """
        Same signature as TechnicalAgent.score_series().
        Returns list of scores — length equals len(prices).
        Blends alpha signal into the final bar only; earlier bars unchanged.
        """
        legacy = super().score_series(prices)

        alpha_w = self._weight_config.get("alpha_weight", 0.0)
        if alpha_w < 1e-6 or not self._weight_config.get("alpha_factors"):
            return legacy    # ← zero overhead, identical to old behaviour

        try:
            # Build minimal DataFrame from close-price list
            df = pd.DataFrame({
                "close":  list(prices),
                "high":   list(prices),
                "low":    list(prices),
                "open":   list(prices),
                "volume": [1.0] * len(prices),
            })
            alpha_s = self._alpha_score_from_df(df)
            base_w  = self._weight_config["base_weight"]
            blended = round(max(0.0, min(100.0, base_w * legacy[-1] + alpha_w * alpha_s)), 2)
            return legacy[:-1] + [blended]
        except Exception as e:
            logger.debug(f"score_series blend failed: {e}")
            return legacy

    # ── NEW: score_ticker with full OHLCV ────────────────────────────────────

    def score_ticker(self, ticker: str, price_df: pd.DataFrame) -> float:
        """Score a single ticker using OHLCV DataFrame. Returns 0-100."""
        close_col = next(
            (c for c in price_df.columns if c.lower() == "close"), None
        )
        if price_df.empty or close_col is None:
            return 50.0

        prices = price_df[close_col].dropna().tolist()
        if len(prices) < 30:
            return 50.0

        legacy_last = float(super().score_series(prices)[-1])

        alpha_w = self._weight_config.get("alpha_weight", 0.0)
        if alpha_w < 1e-6 or not self._weight_config.get("alpha_factors"):
            return legacy_last

        try:
            alpha_s = self._alpha_score_from_df(price_df)
            base_w  = self._weight_config["base_weight"]
            return round(max(0.0, min(100.0, base_w * legacy_last + alpha_w * alpha_s)), 2)
        except Exception as e:
            logger.debug(f"{ticker} score_ticker failed: {e}")
            return legacy_last

    # ── NEW: batch scoring ───────────────────────────────────────────────────

    def score_universe(self, price_data: dict[str, pd.DataFrame]) -> dict[str, float]:
        """Score all tickers. Returns {ticker: score_0_100}."""
        return {t: self.score_ticker(t, df) for t, df in price_data.items()}

    def audit_report(self, ticker: str, price_df: pd.DataFrame) -> str:
        """Human-readable score breakdown for a ticker."""
        close_col = next((c for c in price_df.columns if c.lower() == "close"), None)
        prices    = price_df[close_col].dropna().tolist() if close_col else []
        legacy    = float(super().score_series(prices)[-1]) if len(prices) >= 30 else 50.0
        final     = self.score_ticker(ticker, price_df)
        wc        = self._weight_config
        lines = [
            f"[{ticker}] QuantAlpha Technical Score: {final:.1f}/100",
            f"  Legacy MACD/RSI/BB : {legacy:.1f}  (weight={wc['base_weight']:.2f})",
            f"  Alpha factors      : {len(wc['alpha_factors'])} active  "
            f"(weight={wc['alpha_weight']:.2f})",
        ]
        for fa in wc["alpha_factors"]:
            lines.append(
                f"    {fa['factor_name']:40s}  "
                f"RankIC={fa['rank_ic']:.4f}  w={fa['weight']:.4f}"
            )
        return "\n".join(lines)

    # ── internal ─────────────────────────────────────────────────────────────

    def _alpha_score_from_df(self, df: pd.DataFrame) -> float:
        """Evaluate alpha expressions and return weighted percentile score 0-100."""
        engine = FactorEngine(df)
        scores:  list[float] = []
        weights: list[float] = []

        for fa in self._weight_config["alpha_factors"]:
            try:
                vals = engine.eval(fa["factor_expression"])
                if vals.empty or vals.isna().all():
                    continue
                arr  = vals.dropna().values
                pct  = float(np.mean(arr <= float(arr[-1])))
                scores.append(pct * 100.0)
                weights.append(fa["weight"])
            except Exception as e:
                logger.debug(f"alpha factor {fa['factor_name']} eval failed: {e}")

        if not scores:
            return 50.0
        total_w = sum(weights) or 1.0
        return sum(s * w for s, w in zip(scores, weights)) / total_w
