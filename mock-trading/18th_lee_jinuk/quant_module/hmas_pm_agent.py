"""
H-MAS — Portfolio Manager (PM) Agent  [B.7]
============================================
Chief Portfolio Manager: integrates Top-Down Macro analysis with
Bottom-Up Sector/Stock analysis to produce the definitive investment score.

Decision logic (fine-grained policy from paper):
  • Risk-Off → apply conservative discount unless exceptional defensive quality.
  • High Inflation → favor fundamental pricing-power momentum over technicals.
  • Tie-breaking uses macro context to resolve Tech vs. Fundamental conflicts.
  • Output: Final Score (0-100) + decisive Korean rationale.
"""

from __future__ import annotations

from dataclasses import dataclass
from textwrap import dedent
import json

from hmas_base import BaseAgent, LLMClient, MacroAgentOutput, SectorAgentOutput, PMAgentOutput


# ──────────────────────────────────────────────────────────────────────────────
# Regime-conditional factor weights (from paper Table 2)
# ──────────────────────────────────────────────────────────────────────────────

REGIME_WEIGHTS: dict[str, dict[str, float]] = {
    "Risk-On":      {"technical": 0.40, "fundamental": 0.35, "qualitative": 0.25},
    "Risk-Off":     {"technical": 0.15, "fundamental": 0.60, "qualitative": 0.25},
    "Transitional": {"technical": 0.30, "fundamental": 0.45, "qualitative": 0.25},
}


# ──────────────────────────────────────────────────────────────────────────────
# Agent
# ──────────────────────────────────────────────────────────────────────────────

class PMAgent(BaseAgent):
    """
    Chief Portfolio Manager — Final Integration Layer (B.7).

    Harmonization strategy:
      1. Receives Macro Environment Report (MacroAgentOutput) and
         Sector Specialist Report (SectorAgentOutput).
      2. Classifies macro regime → selects factor weight table.
      3. Optionally adjusts with a rule-based pre-score before LLM.
      4. LLM produces final score + Korean rationale.
    """

    # ── B.7.1 System Prompt ──────────────────────────────────────────────────
    @property
    def system_prompt(self) -> str:
        return dedent("""
        Role: You are the Chief Portfolio Manager (PM). Your task is to
        determine the definitive investment score by integrating Top-Down
        Macro analysis with Bottom-Up Sector/Stock analysis.

        Decision Logic (Integration Strategy):
        - Goal: Maximize alpha (1-month horizon) while strictly managing risk.
        - Harmonization: Balance "Market Context" (Macro) vs "Stock Specifics."
        - Macro Alignment:
            * If Macro is "Risk-Off", apply a conservative discount (lower scores)
              UNLESS the stock has exceptional defensive qualities (high fundamental
              score, stable cash flow, low debt).
            * If Inflation is elevated (rates score < 40), favor stocks with
              demonstrated pricing-power momentum in fundamentals.
        - Tie-Breaking: Use Macro context to resolve conflicts between Technicals
          and Fundamentals (e.g., High Inflation → favor fundamental pricing power
          momentum; Risk-On → trust technical momentum signals more).

        Output Requirements:
        - Final Score: 0-100 (Definitive conviction, integer).
        - Rationale: Decisive professional reasoning justifying the integration.
          Written in Korean, 150-200 characters.

        JSON schema (return exactly this structure):
        {
          "final_score": <int 0-100>,
          "reason": "<Korean string 150-200 chars>"
        }
        """).strip()

    # ── B.7.2 User Prompt builder ────────────────────────────────────────────
    @staticmethod
    def _build_user_prompt(
        ticker: str,
        macro_output: MacroAgentOutput,
        sector_output: SectorAgentOutput,
        macro_regime: str,
        regime_weights: dict[str, float],
    ) -> str:
        macro_json = json.dumps({
            "composite_score": macro_output.composite_score,
            "regime": macro_regime,
            "metrics": {
                "market_trend": macro_output.market_trend,
                "risk":         macro_output.risk,
                "economy":      macro_output.economy,
                "rates":        macro_output.rates,
                "inflation":    macro_output.inflation,
            },
            "summary": macro_output.summary,
        }, ensure_ascii=False, indent=2)

        sector_json = json.dumps({
            "conviction_score": sector_output.conviction_score,
            "investment_thesis": sector_output.investment_thesis,
        }, ensure_ascii=False, indent=2)

        weights_str = (
            f"Technical={regime_weights['technical']:.0%}  "
            f"Fundamental={regime_weights['fundamental']:.0%}  "
            f"Qualitative={regime_weights['qualitative']:.0%}"
        )

        return dedent(f"""
        Instruction: As the Portfolio Manager, review the Macro and Sector-level
        inputs to provide your final investment decision for the next 1 month.

        Ticker: {ticker}
        Macro Regime: {macro_regime}
        Active Factor Weights ({macro_regime}): {weights_str}

        [1. Macro Environment Report]
        {macro_json}

        [2. Sector Specialist Report]
        {sector_json}

        Final Decision Tasks:
        1. Final Investment Score (0-100):
           - 100: Maximum Overweight (High conviction, all signals aligned)
           -  50: Neutral / Hold (Market-weight, no clear edge)
           -   0: Strong Underweight / Avoid (High risk, negative signals)

        2. Final Rationale (Korean, 150-200 chars):
           - Explain how macro environment specifically influences this stock.
           - Summarize key reasons & risk-reward balance for the next 30 days.
           - State the regime-conditional weight adjustment if applied.

        Return ONLY the JSON object as specified in the system prompt.
        """).strip()

    # ── Rule-based pre-score (fast path, regime-conditional) ─────────────────
    @staticmethod
    def _compute_rule_based_score(
        macro_output: MacroAgentOutput,
        sector_output: SectorAgentOutput,
        macro_regime: str,
        regime_weights: dict[str, float],
    ) -> float:
        """
        Transparent pre-score used for audit and as LLM anchor.
        Combines Macro composite + Sector conviction with regime weights.
        """
        macro_score = macro_output.composite_score          # 0-100
        sector_score = float(sector_output.conviction_score)  # 0-100

        # Blend: macro provides the market context, sector provides stock signal
        # Weight: 40% macro top-down, 60% sector bottom-up (standard PM practice)
        blended = 0.40 * macro_score + 0.60 * sector_score

        # Risk-Off discount: if regime is Risk-Off and sector score < 60,
        # apply additional 10-point conservative discount
        if macro_regime == "Risk-Off" and sector_score < 60.0:
            blended -= 10.0

        # Risk-On boost: if Risk-On and both scores align (>65), small boost
        if macro_regime == "Risk-On" and macro_score > 65 and sector_score > 65:
            blended += 5.0

        return round(max(0.0, min(blended, 100.0)), 1)

    # ── Public interface ─────────────────────────────────────────────────────
    def run(
        self,
        ticker: str,
        macro_output: MacroAgentOutput,
        sector_output_or_regime=None,
        macro_regime: str = "Transitional",
        # ── 구버전 / 대체 호출 시그니처 ─────────────────────────────────
        tech_score: float = 50.0,
        quant_output=None,
        qual_output=None,
        news_output=None,
        **kwargs,
    ) -> PMAgentOutput:
        """
        Produce the final investment decision.

        호환 지원 시그니처:
          신버전: run(ticker, macro_out, sector_out, regime)
          구버전: run(ticker, macro_out, regime, tech_score=X, ...)
          혼합:   run(ticker, macro_out, sector_out, regime, tech_score=X, ...)
        """
        # ── 세 번째 인자가 문자열이면 regime, SectorAgentOutput이면 sector ──
        if isinstance(sector_output_or_regime, str):
            # 구버전: run(ticker, macro, "Risk-On", tech_score=...)
            macro_regime   = sector_output_or_regime
            sector_output  = None
        elif isinstance(sector_output_or_regime, SectorAgentOutput):
            sector_output  = sector_output_or_regime
        elif sector_output_or_regime is None:
            sector_output  = None
        else:
            sector_output  = None

        # ── sector_output이 없으면 서브 점수로 조립 ───────────────────────
        if sector_output is None:
            scores = [float(tech_score)]
            if quant_output is not None:
                scores.append(float(getattr(quant_output, "composite_score", 50)))
            if qual_output is not None:
                scores.append(float(getattr(qual_output, "composite_score", 50)))
            if news_output is not None:
                scores.append(float(getattr(news_output, "net_score", 50)))
            avg = sum(scores) / len(scores)
            sector_output = SectorAgentOutput(
                conviction_score=int(avg),
                investment_thesis=f"서브 에이전트 평균 {avg:.1f}/100",
                raw={},
            )

        regime_weights = REGIME_WEIGHTS.get(macro_regime, REGIME_WEIGHTS["Transitional"])

        user_prompt = self._build_user_prompt(
            ticker, macro_output, sector_output, macro_regime, regime_weights
        )
        data = self._call(user_prompt)

        return PMAgentOutput(
            final_score=int(data.get("final_score", 50)),
            reason=data.get("reason", ""),
            raw=data,
        )

    # ── Stub ─────────────────────────────────────────────────────────────────
    @staticmethod
    def stub_output(
        ticker: str = "7203",
        macro_regime: str = "Risk-On",
        macro_output: MacroAgentOutput | None = None,
        sector_output: SectorAgentOutput | None = None,
    ) -> PMAgentOutput:
        """Return a synthetic PM decision for offline testing."""
        from base import MacroAgentOutput, SectorAgentOutput

        if macro_output is None:
            macro_output = MacroAgentOutput(
                market_trend={"label": "Bullish", "score": 72},
                risk={"label": "Risk-On", "score": 68},
                economy={"label": "Expanding", "score": 65},
                rates={"label": "Accommodative", "score": 60},
                inflation={"label": "Disinflating", "score": 63},
                summary="리스크온 환경 지속. KOSPI 상승 바이어스 지지.",
            )
        if sector_output is None:
            sector_output = SectorAgentOutput(
                conviction_score=72,
                investment_thesis="セクターオーバーウェイトを推奨。",
            )

        regime_weights = REGIME_WEIGHTS.get(macro_regime, REGIME_WEIGHTS["Transitional"])
        rule_score = PMAgent._compute_rule_based_score(
            macro_output, sector_output, macro_regime, regime_weights
        )

        reason = (
            f"リスクオン環境下（VIX低位、米雇用堅調）においてテクニカル比重を高め（α=40%）統合スコアを算出。"
            f"セクター内競争力と固体電池触媒を評価し、オーバーウェイトを決定。最終スコア{int(rule_score)}。"
        )
        return PMAgentOutput(
            final_score=int(rule_score),
            reason=reason,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Smoke test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    for regime in ["Risk-On", "Transitional", "Risk-Off"]:
        out = PMAgent.stub_output(ticker="7203", macro_regime=regime)
        print(f"=== PM Agent ({regime}) ===")
        print(f"  Final Score : {out.final_score}/100")
        print(f"  Reason (JP) : {out.reason[:80]}...")
        print(f"  Weights     : {REGIME_WEIGHTS[regime]}")
        print()
