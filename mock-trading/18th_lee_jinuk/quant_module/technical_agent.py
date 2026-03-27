"""
H-MAS Technical Agent — Factor Computation & Scoring
=====================================================
Implements all O(n) / O(k) indicators described in the paper:
  • EMA (streaming, O(1) space)
  • MACD (12/26/9 EMA parameters)
  • RSI-14 (Wilder's modified EMA)
  • Bollinger Band Width (Welford's online variance, O(1) space)
  • Rate-of-Change RoC-5 / RoC-21 (circular buffer, O(k) space)

Scoring policy (fine-grained, as specified in the paper):
  • Score > 70 ONLY when MACD crossover AND RSI momentum are concordant.
  • If RSI > 70 (overbought), cap technical score at 60 regardless of MACD.
  • Signals are combined with configurable sub-weights.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Sequence


# ──────────────────────────────────────────────────────────────────────────────
# Streaming primitives (O(1) space each)
# ──────────────────────────────────────────────────────────────────────────────


class EMA:
    """Exponential Moving Average — single-pass, O(1) space."""

    def __init__(self, span: int) -> None:
        if span < 1:
            raise ValueError("span must be >= 1")
        self.alpha: float = 2.0 / (span + 1)
        self.value: float | None = None

    def update(self, price: float) -> float:
        if self.value is None:
            self.value = price
        else:
            self.value = self.alpha * price + (1.0 - self.alpha) * self.value
        return self.value

    @property
    def ready(self) -> bool:
        return self.value is not None


class WilderEMA:
    """Wilder's smoothed moving average: alpha = 1 / period (used in RSI)."""

    def __init__(self, period: int) -> None:
        self.alpha: float = 1.0 / period
        self.value: float | None = None

    def update(self, x: float) -> float:
        if self.value is None:
            self.value = x
        else:
            self.value = self.alpha * x + (1.0 - self.alpha) * self.value
        return self.value


class WelfordVariance:
    """
    Welford's online algorithm for running mean + variance — O(1) space.
    Numerically stable; safe for financial price series.
    """

    def __init__(self) -> None:
        self.n: int = 0
        self.mean: float = 0.0
        self._m2: float = 0.0

    def update(self, x: float) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self._m2 += delta * (x - self.mean)

    @property
    def variance(self) -> float:
        return self._m2 / self.n if self.n > 1 else 0.0

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)


class CircularBuffer:
    """Fixed-size circular buffer — O(k) space."""

    def __init__(self, maxlen: int) -> None:
        self._buf: deque[float] = deque(maxlen=maxlen)
        self._maxlen = maxlen

    def push(self, value: float) -> None:
        self._buf.append(value)

    @property
    def oldest(self) -> float | None:
        return self._buf[0] if self.full else None

    @property
    def full(self) -> bool:
        return len(self._buf) == self._maxlen


# ──────────────────────────────────────────────────────────────────────────────
# MACD indicator
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class MACDState:
    value: float = 0.0          # MACD line = EMA12 - EMA26
    signal: float = 0.0         # 9-period EMA of MACD
    histogram: float = 0.0      # MACD - signal
    crossover: bool = False      # True when MACD crossed above signal this bar
    bullish: bool = False        # True when MACD > signal


class MACDIndicator:
    """MACD (12/26/9) — three EMA passes, constant memory."""

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9) -> None:
        self._ema_fast = EMA(fast)
        self._ema_slow = EMA(slow)
        self._ema_signal = EMA(signal)
        self._prev_macd: float | None = None
        self._prev_signal: float | None = None

    def update(self, price: float) -> MACDState:
        fast_val = self._ema_fast.update(price)
        slow_val = self._ema_slow.update(price)
        macd_val = fast_val - slow_val
        signal_val = self._ema_signal.update(macd_val)
        histogram = macd_val - signal_val

        # Detect bullish crossover (MACD crossed above signal this bar)
        crossover = False
        if self._prev_macd is not None and self._prev_signal is not None:
            if self._prev_macd <= self._prev_signal and macd_val > signal_val:
                crossover = True

        self._prev_macd = macd_val
        self._prev_signal = signal_val

        return MACDState(
            value=macd_val,
            signal=signal_val,
            histogram=histogram,
            crossover=crossover,
            bullish=macd_val > signal_val,
        )


# ──────────────────────────────────────────────────────────────────────────────
# RSI indicator
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class RSIState:
    value: float = 50.0
    overbought: bool = False   # RSI > 70
    oversold: bool = False     # RSI < 30
    bullish_momentum: bool = False  # RSI rising and between 40-70


class RSIIndicator:
    """RSI-14 using Wilder's smoothed moving average — O(1) space."""

    def __init__(self, period: int = 14, overbought: float = 70.0, oversold: float = 30.0) -> None:
        self._avg_gain = WilderEMA(period)
        self._avg_loss = WilderEMA(period)
        self._prev_price: float | None = None
        self._overbought = overbought
        self._oversold = oversold
        self._prev_rsi: float | None = None

    def update(self, price: float) -> RSIState:
        if self._prev_price is None:
            self._prev_price = price
            return RSIState()

        change = price - self._prev_price
        gain = max(change, 0.0)
        loss = max(-change, 0.0)

        avg_gain = self._avg_gain.update(gain)
        avg_loss = self._avg_loss.update(loss)
        self._prev_price = price

        if avg_loss == 0.0:
            rsi = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi = 100.0 - (100.0 / (1.0 + rs))

        # Bullish momentum: RSI is rising and in healthy territory (40–70)
        bullish_momentum = False
        if self._prev_rsi is not None:
            bullish_momentum = rsi > self._prev_rsi and 40.0 <= rsi <= 70.0
        self._prev_rsi = rsi

        return RSIState(
            value=rsi,
            overbought=rsi > self._overbought,
            oversold=rsi < self._oversold,
            bullish_momentum=bullish_momentum,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Bollinger Band Width (Welford-based)
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class BollingerState:
    upper: float = 0.0
    lower: float = 0.0
    width: float = 0.0          # BBW = 4σ / mean
    price_above_upper: bool = False
    price_below_lower: bool = False
    trend_strong: bool = False  # high BBW → strong trend


class BollingerIndicator:
    """Bollinger Bands via Welford online variance — O(1) space."""

    def __init__(self, num_std: float = 2.0, bbw_high_threshold: float = 0.15) -> None:
        self._welford = WelfordVariance()
        self._num_std = num_std
        self._bbw_threshold = bbw_high_threshold

    def update(self, price: float) -> BollingerState:
        self._welford.update(price)
        mean = self._welford.mean
        std = self._welford.std
        upper = mean + self._num_std * std
        lower = mean - self._num_std * std
        bbw = (4.0 * std / mean) if mean != 0.0 else 0.0

        return BollingerState(
            upper=upper,
            lower=lower,
            width=bbw,
            price_above_upper=price > upper,
            price_below_lower=price < lower,
            trend_strong=bbw > self._bbw_threshold,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Rate of Change (circular buffer)
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class RoCState:
    value: float = 0.0
    positive: bool = False


class RoCIndicator:
    """Rate-of-Change with circular buffer — O(k) space."""

    def __init__(self, period: int) -> None:
        self._buf = CircularBuffer(period)

    def update(self, price: float) -> RoCState:
        roc = 0.0
        if self._buf.full:
            past = self._buf.oldest
            roc = ((price - past) / past) * 100.0 if past else 0.0
        self._buf.push(price)
        return RoCState(value=roc, positive=roc > 0.0)


# ──────────────────────────────────────────────────────────────────────────────
# Composite bar output
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class TechnicalFactors:
    """All raw factor values for a single bar/tick."""
    price: float = 0.0
    ema_12: float = 0.0
    ema_26: float = 0.0
    macd: MACDState = field(default_factory=MACDState)
    rsi: RSIState = field(default_factory=RSIState)
    bollinger: BollingerState = field(default_factory=BollingerState)
    roc5: RoCState = field(default_factory=RoCState)
    roc21: RoCState = field(default_factory=RoCState)
    technical_score: float = 50.0   # 0–100 composite score


# ──────────────────────────────────────────────────────────────────────────────
# Scoring policy (fine-grained, from paper Section 5)
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class ScoringWeights:
    """Configurable sub-weights for each signal component."""
    macd_weight: float = 0.30
    rsi_weight: float = 0.25
    bollinger_weight: float = 0.20
    roc5_weight: float = 0.15
    roc21_weight: float = 0.10

    def __post_init__(self) -> None:
        total = (
            self.macd_weight + self.rsi_weight + self.bollinger_weight
            + self.roc5_weight + self.roc21_weight
        )
        if not math.isclose(total, 1.0, rel_tol=1e-6):
            raise ValueError(f"Weights must sum to 1.0, got {total:.4f}")


def compute_technical_score(
    factors: TechnicalFactors,
    weights: ScoringWeights | None = None,
) -> float:
    """
    Fine-grained scoring policy (paper Section 5):

    1. Derive a 0–100 sub-score for each indicator.
    2. Compute weighted composite.
    3. Apply policy constraints:
       a. Score > 70 only when MACD crossover/bullish AND RSI bullish momentum are *both* true.
       b. If RSI > 70 (overbought), cap the score at 60 regardless of MACD.

    Returns a score in [0, 100].
    """
    if weights is None:
        weights = ScoringWeights()

    # ── 1. MACD sub-score (0–100) ──────────────────────────────────────────
    macd = factors.macd
    if macd.crossover:
        macd_score = 85.0           # fresh bullish crossover → strong signal
    elif macd.bullish:
        macd_score = 65.0           # above signal, no fresh cross → moderate
    elif not macd.bullish and macd.histogram < 0:
        macd_score = 30.0           # bearish
    else:
        macd_score = 50.0           # neutral

    # ── 2. RSI sub-score (0–100) ───────────────────────────────────────────
    rsi_val = factors.rsi.value
    if factors.rsi.overbought:
        rsi_score = 40.0            # overbought → diminished score
    elif factors.rsi.oversold:
        rsi_score = 60.0            # oversold → contrarian mild positive
    elif factors.rsi.bullish_momentum:
        rsi_score = 75.0            # healthy uptrend
    else:
        # Linear mapping from RSI [30, 70] → [40, 65]
        rsi_score = 40.0 + (rsi_val - 30.0) * (65.0 - 40.0) / 40.0
        rsi_score = max(20.0, min(rsi_score, 65.0))

    # ── 3. Bollinger Band sub-score ────────────────────────────────────────
    bb = factors.bollinger
    if bb.price_above_upper:
        bb_score = 35.0             # extended, risk of mean reversion
    elif bb.price_below_lower:
        bb_score = 65.0             # oversold / mean reversion potential
    elif bb.trend_strong:
        bb_score = 70.0             # strong trend momentum
    else:
        bb_score = 50.0             # consolidation / neutral

    # ── 4. RoC-5 sub-score (short-term momentum) ──────────────────────────
    roc5_val = factors.roc5.value
    # Map [-5%, +5%] → [20, 80]; clip outside range
    roc5_score = 50.0 + roc5_val * 6.0
    roc5_score = max(10.0, min(roc5_score, 90.0))

    # ── 5. RoC-21 sub-score (medium-term momentum) ────────────────────────
    roc21_val = factors.roc21.value
    roc21_score = 50.0 + roc21_val * 3.0
    roc21_score = max(10.0, min(roc21_score, 90.0))

    # ── Weighted composite ─────────────────────────────────────────────────
    composite = (
        weights.macd_weight      * macd_score
        + weights.rsi_weight     * rsi_score
        + weights.bollinger_weight * bb_score
        + weights.roc5_weight    * roc5_score
        + weights.roc21_weight   * roc21_score
    )

    # ── Policy constraints (paper Section 5) ──────────────────────────────
    # Rule A: score > 70 only when MACD and RSI momentum are *concordant*
    macd_concordant = macd.crossover or macd.bullish
    rsi_concordant  = factors.rsi.bullish_momentum
    if composite > 70.0 and not (macd_concordant and rsi_concordant):
        composite = 70.0

    # Rule B: if RSI > 70 (overbought), cap at 60
    if factors.rsi.overbought:
        composite = min(composite, 60.0)

    return round(max(0.0, min(composite, 100.0)), 2)


# ──────────────────────────────────────────────────────────────────────────────
# Technical Agent
# ──────────────────────────────────────────────────────────────────────────────


class TechnicalAgent:
    """
    H-MAS Level-1 Technical Agent.

    Processes a streaming price series and emits a TechnicalFactors snapshot
    (including the fine-grained composite score) for each new price.

    Parameters
    ----------
    macd_fast, macd_slow, macd_signal : MACD parameters (default 12/26/9).
    rsi_period                         : RSI lookback (default 14).
    roc_short, roc_long                : RoC periods (default 5, 21).
    bb_num_std                         : Bollinger Band width in σ (default 2).
    bb_bbw_threshold                   : BBW level considered "strong trend".
    weights                            : Sub-signal weights (must sum to 1).
    """

    def __init__(
        self,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        rsi_period: int = 14,
        roc_short: int = 5,
        roc_long: int = 21,
        bb_num_std: float = 2.0,
        bb_bbw_threshold: float = 0.15,
        weights: ScoringWeights | None = None,
    ) -> None:
        self._ema12 = EMA(macd_fast)
        self._ema26 = EMA(macd_slow)
        self._macd  = MACDIndicator(macd_fast, macd_slow, macd_signal)
        self._rsi   = RSIIndicator(rsi_period)
        self._bb    = BollingerIndicator(bb_num_std, bb_bbw_threshold)
        self._roc5  = RoCIndicator(roc_short)
        self._roc21 = RoCIndicator(roc_long)
        self._weights = weights or ScoringWeights()

    def update(self, price: float) -> TechnicalFactors:
        """Process one price tick; returns full TechnicalFactors snapshot."""
        macd_state = self._macd.update(price)
        rsi_state  = self._rsi.update(price)
        bb_state   = self._bb.update(price)
        roc5_state = self._roc5.update(price)
        roc21_state = self._roc21.update(price)

        factors = TechnicalFactors(
            price=price,
            ema_12=self._ema12.value or price,
            ema_26=self._ema26.value or price,
            macd=macd_state,
            rsi=rsi_state,
            bollinger=bb_state,
            roc5=roc5_state,
            roc21=roc21_state,
        )
        factors.technical_score = compute_technical_score(factors, self._weights)
        return factors

    def process_series(self, prices: Sequence[float]) -> list[TechnicalFactors]:
        """Batch-process a price series. Returns one snapshot per price."""
        return [self.update(p) for p in prices]

    def score_series(self, prices: Sequence[float]) -> list[float]:
        """Convenience: return only the technical score for each price."""
        return [f.technical_score for f in self.process_series(prices)]


# ──────────────────────────────────────────────────────────────────────────────
# Audit / explainability helper
# ──────────────────────────────────────────────────────────────────────────────


def generate_audit_report(ticker: str, factors: TechnicalFactors) -> str:
    """
    Generate a natural-language audit report for a single bar snapshot,
    matching the audit-trail format described in paper Section 6.
    """
    macd = factors.macd
    rsi  = factors.rsi
    bb   = factors.bollinger
    score = factors.technical_score

    macd_desc = (
        "bullish crossover confirmed" if macd.crossover
        else "above signal (bullish)" if macd.bullish
        else "below signal (bearish)"
    )
    rsi_desc = (
        f"RSI={rsi.value:.1f} (OVERBOUGHT — score capped at 60)" if rsi.overbought
        else f"RSI={rsi.value:.1f} (oversold)" if rsi.oversold
        else f"RSI={rsi.value:.1f} (bullish momentum)" if rsi.bullish_momentum
        else f"RSI={rsi.value:.1f} (neutral)"
    )

    concordant = (macd.crossover or macd.bullish) and rsi.bullish_momentum
    concordance_note = (
        "MACD + RSI concordant: high-confidence signal allowed above 70."
        if concordant
        else "MACD/RSI not fully concordant: score capped at 70 if composite exceeded."
    )

    report = (
        f"[{ticker}] Technical Score: {score}/100\n"
        f"  MACD(12/26/9): {macd_desc}  |  histogram={macd.histogram:+.4f}\n"
        f"  {rsi_desc}\n"
        f"  Bollinger BBW={bb.width:.4f} ({'strong trend' if bb.trend_strong else 'consolidation'})"
        f" | price {'ABOVE upper band' if bb.price_above_upper else 'BELOW lower band' if bb.price_below_lower else 'within bands'}\n"
        f"  RoC-5={factors.roc5.value:+.2f}%  |  RoC-21={factors.roc21.value:+.2f}%\n"
        f"  Policy: {concordance_note}\n"
    )
    return report


# ──────────────────────────────────────────────────────────────────────────────
# Quick demo
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import random

    random.seed(42)

    # Simulate a 60-bar price series with a gentle uptrend
    prices: list[float] = [100.0]
    for _ in range(59):
        prices.append(prices[-1] * (1 + random.gauss(0.001, 0.012)))

    agent = TechnicalAgent()
    snapshots = agent.process_series(prices)

    print("=" * 65)
    print("H-MAS Technical Agent — Factor Output (last 5 bars)")
    print("=" * 65)
    for snap in snapshots[-5:]:
        print(generate_audit_report("DEMO", snap))

    print("All technical scores (last 10):")
    print([s.technical_score for s in snapshots[-10:]])
