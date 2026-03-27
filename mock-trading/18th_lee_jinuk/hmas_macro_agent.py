"""
H-MAS — Macro Agent  [B.6]
===========================
Analyzes JP/US macroeconomic indicators (levels + Rate of Change)
to classify the market regime and score five macro dimensions:

  Market Direction  ·  Risk  ·  Economy  ·  Rates  ·  Inflation

Outputs: five labeled dimension scores (0-100) + Korean summary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from textwrap import dedent
from typing import Optional

from hmas_base import BaseAgent, LLMClient, MacroAgentOutput


# ──────────────────────────────────────────────────────────────────────────────
# Input data structure
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class MacroIndicator:
    """Single macro data point: current level + period-over-period RoC %."""
    name: str
    value: float
    roc_pct: float          # Rate of Change (%)
    unit: str = ""

    def format(self) -> str:
        return f"{self.name}: {self.value}{self.unit} (RoC: {self.roc_pct:+.2f}%)"


@dataclass
class MacroIndicators:
    """
    Full macro input panel — JP/US indicators across four categories.
    All fields are MacroIndicator instances.
    """
    # ── 1. Rates & Policy ────────────────────────────────────────────────────
    us_fed_rate: MacroIndicator = field(
        default_factory=lambda: MacroIndicator("US Fed Rate", 5.25, 0.0, "%"))
    us_10y_yield: MacroIndicator = field(
        default_factory=lambda: MacroIndicator("US 10Y Yield", 4.20, -0.5, "%"))
    jp_policy_rate: MacroIndicator = field(
        default_factory=lambda: MacroIndicator("JP Policy Rate", 0.10, 0.0, "%"))
    jp_10y_yield: MacroIndicator = field(
        default_factory=lambda: MacroIndicator("JP 10Y Yield", 0.75, +2.0, "%"))

    # ── 2. Inflation & Commodities ───────────────────────────────────────────
    us_cpi: MacroIndicator = field(
        default_factory=lambda: MacroIndicator("US CPI YoY", 3.1, -0.3, "%"))
    jp_cpi: MacroIndicator = field(
        default_factory=lambda: MacroIndicator("JP CPI YoY", 2.6, -0.1, "%"))
    gold: MacroIndicator = field(
        default_factory=lambda: MacroIndicator("Gold", 2050.0, +1.2, " USD/oz"))
    crude_oil: MacroIndicator = field(
        default_factory=lambda: MacroIndicator("Crude Oil (WTI)", 72.5, -3.4, " USD/bbl"))

    # ── 3. Growth & Economy ──────────────────────────────────────────────────
    us_payrolls: MacroIndicator = field(
        default_factory=lambda: MacroIndicator("US Non-Farm Payrolls", 216.0, +5.0, "K"))
    industrial_production: MacroIndicator = field(
        default_factory=lambda: MacroIndicator("US Industrial Production", 103.2, +0.3, ""))
    housing_starts: MacroIndicator = field(
        default_factory=lambda: MacroIndicator("US Housing Starts", 1.46, +4.2, "M"))
    unemployment_rate: MacroIndicator = field(
        default_factory=lambda: MacroIndicator("US Unemployment Rate", 3.7, +0.1, "%"))
    jp_business_index: MacroIndicator = field(
        default_factory=lambda: MacroIndicator("JP Business Conditions Index", 107.5, +1.5, ""))

    # ── 4. Market & Risk ─────────────────────────────────────────────────────
    usd_jpy: MacroIndicator = field(
        default_factory=lambda: MacroIndicator("USD/JPY", 148.5, +1.2, ""))
    nikkei_225: MacroIndicator = field(
        default_factory=lambda: MacroIndicator("Nikkei 225", 36500.0, +3.5, ""))
    sp500: MacroIndicator = field(
        default_factory=lambda: MacroIndicator("S&P 500", 4750.0, +2.1, ""))
    us_vix: MacroIndicator = field(
        default_factory=lambda: MacroIndicator("US VIX", 14.2, -8.5, ""))
    nikkei_vi: MacroIndicator = field(
        default_factory=lambda: MacroIndicator("Nikkei VI", 18.5, -5.2, ""))


# ──────────────────────────────────────────────────────────────────────────────
# Agent
# ──────────────────────────────────────────────────────────────────────────────

class MacroAgent(BaseAgent):
    """
    Macro Analyst — JP/US Macro Environment (B.6).

    Fine-grained policy:
      • Uses ONLY provided indicator data (no external news interpretation).
      • Scores based on both Levels and Rate of Change of each indicator.
      • Five labeled dimensions: market_trend, risk, economy, rates, inflation.
      • Score 100=Strong Buy / 50=Neutral / 0=Strong Sell.
      • Inflation scoring: High=Stable/Disinflation, Low=Stagflation/Deflation.
      • Summary in Korean (~200 chars) for active management implications.
    """

    # ── B.6.1 System Prompt ──────────────────────────────────────────────────
    @property
    def system_prompt(self) -> str:
        return dedent("""
        Role: You are a Macro Analyst on the trading team. Your task is to
        analyze JP/US macro indicators to identify factors influencing the
        1-month return of Korean equities (KOSPI).

        Evaluation Areas (Label & Score 0-100):
        - market_trend : Overall stock market direction (Bullish/Bearish).
        - risk         : Risk sentiment and potential volatility.
        - economy      : Economic growth trends.
        - rates        : Interest rate environment.
        - inflation    : Price level trends.

        Policy & Constraints:
        - Input Scope: Use ONLY the provided indicators. Do NOT interpret news.
        - Scoring Logic: Evaluate based on both "Levels" AND "Rate of Change."
        - Score 100: Strong Buy (Bullish), 50: Neutral, 0: Strong Sell (Bearish).
        - Inflation special rule: High score = Stable/Disinflation (good);
          Low score = Stagflation or Deflation (bad).
        - Summary: Concise macro implications for active management in Korean
          (~200 chars).
        - Output: Strictly JSON format only.

        JSON schema (return exactly this structure):
        {
          "metrics": {
            "market_trend": {"label": "<1-3 word descriptor>", "score": <0-100>},
            "risk":         {"label": "<1-3 word descriptor>", "score": <0-100>},
            "economy":      {"label": "<1-3 word descriptor>", "score": <0-100>},
            "rates":        {"label": "<1-3 word descriptor>", "score": <0-100>},
            "inflation":    {"label": "<1-3 word descriptor>", "score": <0-100>}
          },
          "summary": "<Korean string ~200 chars>"
        }
        """).strip()

    # ── B.6.2 User Prompt builder ────────────────────────────────────────────
    @staticmethod
    def _build_user_prompt(indicators: MacroIndicators) -> str:
        i = indicators
        return dedent(f"""
        Instruction: Evaluate the current macroeconomic environment to assess
        impact on the 1-month forward return of Korean stocks (KOSPI), based on
        "Levels" and "RoC" (Rate of Change) of each indicator.

        Scoring Rules (0-100):
        - market_trend  : Stock indices momentum. High score = upward trend.
        - risk          : VIX & safe assets. High score = Low VIX / Risk-on stable.
        - economy       : Employment, production. High score = expansion.
        - rates         : Levels & direction. High score = accommodative/falling.
        - inflation     : High score = Stable/Disinflation. Low = Stagflation/Deflation.

        Macro Indicators [Format: Value (RoC: Value%)]:

        [1. Rates & Policy]
        {i.us_fed_rate.format()}
        {i.us_10y_yield.format()}
        {i.jp_policy_rate.format()}
        {i.jp_10y_yield.format()}

        [2. Inflation & Commodities]
        {i.us_cpi.format()}
        {i.jp_cpi.format()}
        {i.gold.format()}
        {i.crude_oil.format()}

        [3. Growth & Economy]
        {i.us_payrolls.format()}
        {i.industrial_production.format()}
        {i.housing_starts.format()}
        {i.unemployment_rate.format()}
        {i.jp_business_index.format()}

        [4. Market & Risk]
        {i.usd_jpy.format()}
        {i.nikkei_225.format()}
        {i.sp500.format()}
        {i.us_vix.format()}
        {i.nikkei_vi.format()}

        Return ONLY the JSON object as specified in the system prompt.
        """).strip()

    # ── Regime classification helper ─────────────────────────────────────────
    @staticmethod
    def classify_regime(output: MacroAgentOutput) -> str:
        """
        Classify macro regime from agent output scores.
        Matches the Risk-On / Risk-Off / Transitional framework in the paper.
        """
        composite = output.composite_score
        risk_score = output.risk.get("score", 50)
        rates_score = output.rates.get("score", 50)

        if composite >= 60 and risk_score >= 60 and rates_score >= 55:
            return "Risk-On"
        elif composite <= 40 or risk_score <= 35:
            return "Risk-Off"
        else:
            return "Transitional"

    # ── Public interface ─────────────────────────────────────────────────────
    def run(self, indicators: MacroIndicators) -> MacroAgentOutput:
        """
        Analyze macroeconomic indicators and return MacroAgentOutput.

        Parameters
        ----------
        indicators : MacroIndicators
            Full macro data panel with levels and RoC values.

        Returns
        -------
        MacroAgentOutput
            Five dimension scores + Korean summary + composite score.
        """
        user_prompt = self._build_user_prompt(indicators)
        data = self._call(user_prompt)
        metrics = data.get("metrics", {})

        return MacroAgentOutput(
            market_trend=metrics.get("market_trend", {"label": "Neutral", "score": 50}),
            risk=metrics.get("risk",         {"label": "Neutral", "score": 50}),
            economy=metrics.get("economy",   {"label": "Neutral", "score": 50}),
            rates=metrics.get("rates",       {"label": "Neutral", "score": 50}),
            inflation=metrics.get("inflation", {"label": "Stable", "score": 50}),
            summary=data.get("summary", ""),
            raw=data,
        )

    # ── Stub ─────────────────────────────────────────────────────────────────
    @staticmethod
    def stub_output(regime: str = "Risk-On") -> MacroAgentOutput:
        """Return a synthetic output for the specified regime."""
        if regime == "Risk-On":
            out = MacroAgentOutput(
                market_trend={"label": "Bullish Momentum", "score": 72},
                risk={"label": "Risk-On", "score": 68},
                economy={"label": "Expanding", "score": 65},
                rates={"label": "Accommodative", "score": 60},
                inflation={"label": "Disinflating", "score": 63},
                summary="미국 고용 견조·VIX 저위 안정으로 리스크온 환경 지속. 한국 기준금리 동결 예상, KOSPI 단기 상승 바이어스 지지.",
            )
        elif regime == "Risk-Off":
            out = MacroAgentOutput(
                market_trend={"label": "Bearish Pressure", "score": 30},
                risk={"label": "Risk-Off", "score": 28},
                economy={"label": "Slowing", "score": 38},
                rates={"label": "Tightening", "score": 35},
                inflation={"label": "Elevated", "score": 32},
                summary="VIX 급등·미국 금리인상 지속 리스크로 리스크오프 압력 증가. KOSPI 하방 리스크 주의. 방어주 선별 전략 필요.",
            )
        else:  # Transitional
            out = MacroAgentOutput(
                market_trend={"label": "Mixed Signals", "score": 52},
                risk={"label": "Transitional", "score": 48},
                economy={"label": "Moderate", "score": 53},
                rates={"label": "Neutral", "score": 50},
                inflation={"label": "Moderating", "score": 55},
                summary="マクロ環境は移行期。Fed政策の方向性が不透明で株式市場はレンジ内推移を予想。銘柄固有要因の重要性が高まる局面。",
            )
        return out


# ──────────────────────────────────────────────────────────────────────────────
# Smoke test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    for regime in ["Risk-On", "Transitional", "Risk-Off"]:
        out = MacroAgent.stub_output(regime)
        classified = MacroAgent.classify_regime(out)
        print(f"=== Macro Agent ({regime}) ===")
        print(f"  Composite Score  : {out.composite_score}/100")
        print(f"  Classified Regime: {classified}")
        print(f"  Summary (JP)     : {out.summary[:60]}...")
        print()
