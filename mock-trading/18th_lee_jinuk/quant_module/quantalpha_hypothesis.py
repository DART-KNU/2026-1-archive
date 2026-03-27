"""
QuantAlpha Hypothesis & Factor Proposal
========================================
Ports quantaalpha/factors/proposal.py behaviour into a self-contained module
that works with the existing H-MAS LLMClient (hmas_base.py).

Pipeline:
  1. AlphaHypothesisGen.gen(trace)        → Hypothesis dataclass
  2. AlphaHypothesis2Factor.convert(hyp, trace) → list[FactorSpec]

The LLM is instructed to return clean JSON with market hypothesis text plus
Qlib-style factor expressions compatible with quantalpha_factor_engine.py.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from textwrap import dedent
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Hypothesis:
    """Single market hypothesis produced by the LLM."""
    hypothesis: str
    concise_knowledge: str       = ""
    concise_observation: str     = ""
    concise_justification: str   = ""
    concise_specification: str   = ""
    reason: str                  = ""

    def __str__(self) -> str:
        return self.hypothesis


@dataclass
class FactorSpec:
    """
    A single alpha factor specification produced from a hypothesis.
    Maps to quantaalpha FactorTask.
    """
    factor_name:        str
    factor_description: str
    factor_expression:  str                    # e.g. "RANK(TS_MEAN($return,20)/TS_STD($return,20))"
    variables:          dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.factor_name}: {self.factor_expression}"


@dataclass
class Trace:
    """
    Rolling history of (Hypothesis, list[FactorSpec], feedback_dict) tuples.
    Mirrors quantaalpha Trace — passed to gen() so the LLM can build on prior rounds.
    """
    hist: list[tuple[Hypothesis, list[FactorSpec], dict]] = field(default_factory=list)
    scen: Any = None   # KOSPIScenario


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

_SYS_HYPO = dedent("""
You are a quantitative researcher specialising in Korean equity markets (KOSPI).
Your task is to generate a precise, testable market hypothesis that can be
implemented as a mathematical alpha factor expression.

The hypothesis must:
- Be grounded in market microstructure, momentum, mean-reversion, or
  fundamental/technical interaction effects specific to KOSPI.
- Be novel relative to prior hypotheses listed (do not repeat them).
- Lead naturally to a computable factor expression using OHLCV data.

Respond ONLY with a valid JSON object — no markdown, no preamble:
{
  "hypothesis": "<one clear sentence>",
  "concise_knowledge": "<transferable principle, conditional grammar>",
  "concise_observation": "<data or market observation>",
  "concise_justification": "<theoretical basis>",
  "concise_specification": "<scope, thresholds, expected direction>"
}
""").strip()

_SYS_FACTOR = dedent("""
You are a quantitative factor engineer for KOSPI equity alpha research.
Given a market hypothesis, produce 2–3 alpha factor expressions.

Rules:
1. Use ONLY these base features: $open, $high, $low, $close, $volume, $return, $vwap
2. Allowed ops: TS_MEAN(A,n), TS_STD(A,n), TS_MAX(A,n), TS_MIN(A,n),
   TS_RANK(A,n), TS_CORR(A,B,n), DELTA(A,n), DELAY(A,n),
   RANK(A), ZSCORE(A), LOG(A), ABS(A), SIGN(A), EMA(A,n), SMA(A,n),
   MAX(A,B), MIN(A,B)
3. Symbol length ≤ 250 characters per expression.
4. Use at most 5 distinct base features.
5. No look-ahead bias — DELAY(A,n) looks n bars BACK (past).
6. Each factor must be independent (no cross-references).
7. Factor names: snake_case, descriptive, ≤ 40 chars.

Respond ONLY with a valid JSON object:
{
  "Factor_Name_1": {
    "description": "<what this factor captures>",
    "formulation": "<mathematical formula in plain text>",
    "expression": "<Qlib-style expression string>",
    "variables": {}
  },
  "Factor_Name_2": { ... },
  "Factor_Name_3": { ... }
}
""").strip()


def _robust_json(text: str) -> dict:
    """Parse JSON, stripping markdown fences if present."""
    text = text.strip()
    # strip ```json ... ``` fences
    text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
    text = re.sub(r"\n?```$", "", text)
    return json.loads(text.strip())


def _build_hist_summary(trace: Trace) -> str:
    if not trace.hist:
        return "No prior rounds — this is round 1."
    lines = []
    for i, (hyp, factors, fb) in enumerate(trace.hist[-6:], 1):
        rank_ic = fb.get("rank_ic", "N/A")
        lines.append(
            f"Round {i}: {hyp.hypothesis}\n"
            f"  Factors: {', '.join(f.factor_name for f in factors)}\n"
            f"  Rank IC: {rank_ic}"
        )
    return "\n\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Hypothesis Generator
# ──────────────────────────────────────────────────────────────────────────────

class AlphaHypothesisGen:
    """
    LLM-powered hypothesis generator.
    Mirrors quantaalpha AlphaAgentHypothesisGen.
    Works with any callable: llm_fn(system_prompt, user_prompt) -> str
    """

    def __init__(self, scenario, llm_fn) -> None:
        self.scenario = scenario
        self._llm = llm_fn          # callable: (system, user) -> str

    def gen(self, trace: Trace) -> Hypothesis:
        hist_summary = _build_hist_summary(trace)
        scenario_desc = self.scenario.get_scenario_all_desc()

        user = dedent(f"""
        === Scenario ===
        {scenario_desc}

        === Prior rounds (avoid repeating) ===
        {hist_summary}

        Generate a NEW, testable market hypothesis for KOSPI alpha factor mining.
        """).strip()

        try:
            raw = self._llm(_SYS_HYPO, user)
            d = _robust_json(raw)
        except Exception as e:
            logger.warning(f"HypothesisGen LLM failed: {e}; using fallback")
            d = _fallback_hypothesis(len(trace.hist))

        return Hypothesis(
            hypothesis=d.get("hypothesis", "Momentum persistence in KOSPI mid-caps."),
            concise_knowledge=d.get("concise_knowledge", ""),
            concise_observation=d.get("concise_observation", ""),
            concise_justification=d.get("concise_justification", ""),
            concise_specification=d.get("concise_specification", ""),
        )


# ──────────────────────────────────────────────────────────────────────────────
# Factor Constructor
# ──────────────────────────────────────────────────────────────────────────────

class AlphaHypothesis2Factor:
    """
    Converts a Hypothesis into 2-3 FactorSpec objects.
    Mirrors quantaalpha AlphaAgentHypothesis2FactorExpression.
    """

    MAX_SYM_LEN = 250

    def __init__(self, scenario, llm_fn) -> None:
        self.scenario = scenario
        self._llm = llm_fn

    def convert(self, hypothesis: Hypothesis, trace: Trace) -> list[FactorSpec]:
        hist_summary = _build_hist_summary(trace)

        user = dedent(f"""
        === Target hypothesis ===
        {hypothesis.hypothesis}

        === Justification ===
        {hypothesis.concise_justification}
        Specification: {hypothesis.concise_specification}

        === Prior factors (avoid duplicates) ===
        {_prior_factors_summary(trace)}

        === Prior rounds summary ===
        {hist_summary}

        Generate 2–3 alpha factor expressions for this hypothesis.
        """).strip()

        try:
            raw = self._llm(_SYS_FACTOR, user)
            d = _robust_json(raw)
        except Exception as e:
            logger.warning(f"Hypothesis2Factor LLM failed: {e}; using fallback factors")
            return _fallback_factors(hypothesis)

        specs: list[FactorSpec] = []
        for name, info in d.items():
            if not isinstance(info, dict):
                continue
            expr = str(info.get("expression", "")).strip()
            if not expr:
                continue
            # enforce symbol length
            if len(expr) > self.MAX_SYM_LEN:
                logger.warning(f"Factor {name} expression too long ({len(expr)} chars), truncated concept skipped")
                continue
            specs.append(FactorSpec(
                factor_name=name,
                factor_description=info.get("description", ""),
                factor_expression=expr,
                variables=info.get("variables", {}),
            ))

        if not specs:
            logger.warning("No valid factors from LLM, using fallback")
            specs = _fallback_factors(hypothesis)

        return specs[:3]   # cap at 3


# ──────────────────────────────────────────────────────────────────────────────
# Fallbacks (no LLM / LLM failure)
# ──────────────────────────────────────────────────────────────────────────────

_FALLBACK_HYPOTHESES = [
    {
        "hypothesis": "Short-term price momentum in KOSPI is persistent over 20 trading days, as institutional herding amplifies trending behaviour.",
        "concise_knowledge": "If 20-day return is positive and ranked high cross-sectionally, forward returns tend to continue.",
        "concise_observation": "KOSPI exhibits strong institutional momentum clustering.",
        "concise_justification": "Momentum effect documented in emerging markets; Korea shows herding.",
        "concise_specification": "20-day lookback, cross-sectional rank normalisation, long top-decile.",
    },
    {
        "hypothesis": "Volatility-adjusted momentum (Sharpe-style) better predicts KOSPI returns than raw momentum by filtering noise-driven moves.",
        "concise_knowledge": "When momentum is scaled by volatility, signal-to-noise improves.",
        "concise_observation": "Raw momentum in KOSPI suffers from high volatility noise.",
        "concise_justification": "Risk-adjusted return metrics have higher IC in noisy markets.",
        "concise_specification": "20-day return / 20-day std, cross-sectional rank.",
    },
    {
        "hypothesis": "Volume-price divergence (high volume with flat price) predicts KOSPI reversals at the 5-day horizon.",
        "concise_knowledge": "If volume spikes but price does not follow, distribution is likely.",
        "concise_observation": "KOSPI retail participation creates volume spikes before reversals.",
        "concise_justification": "Smart money distribution identified by volume-price divergence.",
        "concise_specification": "Correlation between price change and volume over 5 days, low = reversal signal.",
    },
    {
        "hypothesis": "Mean-reversion after RSI extremes in KOSPI large-caps produces reliable short-term alpha.",
        "concise_knowledge": "If RSI exceeds 70 or drops below 30, mean-reversion tends to follow.",
        "concise_observation": "Institutional rebalancing creates reversals at RSI extremes.",
        "concise_justification": "Overbought/oversold signals have documented predictive power in KOSPI.",
        "concise_specification": "14-day RSI, extreme thresholds 70/30, 5-day forward return.",
    },
    {
        "hypothesis": "High-low range expansion relative to recent average signals KOSPI breakouts with continued momentum.",
        "concise_knowledge": "When daily range expands above 2x recent average, breakout often follows.",
        "concise_observation": "Volatility expansion in KOSPI precedes directional moves.",
        "concise_justification": "Range breakout is a classic technical signal with empirical support.",
        "concise_specification": "5-day high-low range / 20-day average range, rank cross-sectionally.",
    },
]

_FALLBACK_FACTOR_SETS = [
    [
        FactorSpec("momentum_20d", "20-day cross-sectional momentum rank",
                   "RANK(DELTA($close,20)/$close)"),
        FactorSpec("sharpe_momentum_20d", "Sharpe-style momentum factor",
                   "RANK(TS_MEAN($return,20)/(TS_STD($return,20)+1e-8))"),
    ],
    [
        FactorSpec("vol_adj_momentum", "Volatility-adjusted 20-day momentum",
                   "RANK(TS_MEAN($return,20)/(TS_STD($return,20)+1e-8))"),
        FactorSpec("vol_adj_momentum_10d", "Volatility-adjusted 10-day momentum",
                   "RANK(TS_MEAN($return,10)/(TS_STD($return,10)+1e-8))"),
    ],
    [
        FactorSpec("vol_price_divergence", "Volume-price correlation (low=reversal)",
                   "RANK(-TS_CORR($return,LOG($volume+1),5))"),
        FactorSpec("vol_price_divergence_10d", "Volume-price divergence 10d",
                   "RANK(-TS_CORR($return,LOG($volume+1),10))"),
    ],
    [
        FactorSpec("rsi_reversion", "RSI distance from 50 (mean-reversion)",
                   "RANK(-(TS_MEAN($return,14)/(TS_STD($return,14)+1e-8)-50))"),
        FactorSpec("short_reversion", "5-day return reversal",
                   "RANK(-DELTA($close,5)/$close)"),
    ],
    [
        FactorSpec("range_expansion", "High-low range expansion vs 20d avg",
                   "RANK(($high-$low)/(TS_MEAN($high-$low,20)+1e-8))"),
        FactorSpec("breakout_strength", "Close position within range * volume",
                   "RANK((($close-$low)/($high-$low+1e-8))*LOG($volume+1))"),
    ],
]


def _fallback_hypothesis(round_idx: int) -> dict:
    idx = round_idx % len(_FALLBACK_HYPOTHESES)
    return _FALLBACK_HYPOTHESES[idx]


def _fallback_factors(hypothesis: Hypothesis) -> list[FactorSpec]:
    # pick factor set based on keyword matching
    h = hypothesis.hypothesis.lower()
    if "momentum" in h and "volatil" in h:
        return _FALLBACK_FACTOR_SETS[1]
    elif "volume" in h or "divergen" in h:
        return _FALLBACK_FACTOR_SETS[2]
    elif "reversi" in h or "rsi" in h or "mean" in h:
        return _FALLBACK_FACTOR_SETS[3]
    elif "range" in h or "breakout" in h:
        return _FALLBACK_FACTOR_SETS[4]
    else:
        return _FALLBACK_FACTOR_SETS[0]


def _prior_factors_summary(trace: Trace) -> str:
    if not trace.hist:
        return "None"
    names = []
    for _, factors, _ in trace.hist:
        names.extend(f.factor_name for f in factors)
    return ", ".join(names) if names else "None"
