"""
QuantAlpha Scenario — KOSPI
============================
Defines the market scenario description passed to the LLM during hypothesis
generation. Mirrors quantaalpha/factors/experiment.py :: QlibAlphaAgentScenario
but tailored to KOSPI OHLCV data available via yfinance / kospi_collectors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


# ──────────────────────────────────────────────────────────────────────────────
# Available base features (OHLCV derived, same as Qlib Alpha158 naming style)
# ──────────────────────────────────────────────────────────────────────────────

KOSPI_BASE_FEATURES = [
    "$open",   "$high",   "$low",    "$close",  "$volume",
    "$return",  # daily return = $close / Ref($close,1) - 1
    "$vwap",    # (high+low+close)/3  approximation
    "$turnover",  # volume / shares_outstanding  (if available, else volume proxy)
]

SUPPORTED_OPS = {
    # Time-series
    "TS_MEAN(A,n)", "TS_STD(A,n)", "TS_MAX(A,n)", "TS_MIN(A,n)",
    "TS_RANK(A,n)", "TS_ZSCORE(A,n)", "TS_CORR(A,B,n)", "TS_SUM(A,n)",
    "DELTA(A,n)", "DELAY(A,n)",
    # Cross-section
    "RANK(A)", "ZSCORE(A)", "MEAN(A)", "STD(A)",
    # Math
    "LOG(A)", "ABS(A)", "SIGN(A)", "SQRT(A)",
    # Technical built-ins
    "EMA(A,n)", "SMA(A,n)",
    # Conditional
    "MAX(A,B)", "MIN(A,B)",
}


@dataclass
class KOSPIScenario:
    """
    Scenario descriptor for the KOSPI alpha-mining loop.

    Attributes
    ----------
    universe       : list of KOSPI tickers in scope
    start_date     : historical data start
    end_date       : historical data end
    pred_horizon   : forward-return horizon in trading days (default 21 ≈ 1 month)
    """

    universe: List[str] = field(default_factory=lambda: [
        "005930", "000660", "005380", "035420", "373220",
        "105560", "051910", "207940", "035720", "006400",
    ])
    start_date: str = "2019-01-01"
    end_date:   str = "2024-12-31"
    pred_horizon: int = 21   # 1-month forward return

    # ── text descriptions for LLM ────────────────────────────────────────────

    def get_scenario_all_desc(self, filtered_tag: str = "") -> str:
        """Return full scenario description for LLM system prompt."""
        return f"""
=== KOSPI Alpha Factor Mining Scenario ===

Market         : Korea Stock Price Index (KOSPI)
Universe       : KOSPI large-caps ({len(self.universe)} tickers in scope)
Data frequency : Daily OHLCV
Data range     : {self.start_date} to {self.end_date}
Prediction     : {self.pred_horizon}-day forward return (≈ 1 month)
Objective      : Maximize Rank IC between factor signal and forward returns

=== Available base features ===
{', '.join(KOSPI_BASE_FEATURES)}

=== Supported operations ===
Time-series : TS_MEAN, TS_STD, TS_MAX, TS_MIN, TS_RANK, TS_ZSCORE,
              TS_CORR, TS_SUM, DELTA, DELAY, EMA, SMA
Cross-section: RANK, ZSCORE, MEAN, STD
Math         : LOG, ABS, SIGN, SQRT, MAX, MIN

=== Expression constraints ===
- Symbol length ≤ 250 characters
- At most 5 distinct base features per expression
- No free variables — every parameter must be a literal integer or float
- Each factor must be self-contained (no references to other factors)
- Avoid expressions with look-ahead bias (DELAY(A,n) refers to PAST n bars)

=== Evaluation metrics ===
Primary  : Rank IC  (Rank correlation of signal vs. forward return)
Secondary: IC, ICIR, annualized return, max drawdown
Promote threshold : Rank IC > 0.02  (weak signal),  > 0.05 (strong signal)

=== H-MAS integration ===
Promoted factors feed into:
  - TechnicalAgent  : replace/augment static MACD/RSI/BB scoring
  - QuantAgent      : additional momentum/value signals
  - PMAgent         : regime-conditional weight adjustment
""".strip()

    def get_data_desc(self) -> str:
        return (
            f"KOSPI OHLCV daily data, {self.start_date}~{self.end_date}, "
            f"universe size={len(self.universe)}, pred_horizon={self.pred_horizon}d"
        )
