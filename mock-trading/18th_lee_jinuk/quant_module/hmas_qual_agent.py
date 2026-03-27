"""
H-MAS — Qualitative (Qual) Agent  [B.3]
========================================
Analyzes qualitative corporate disclosures:
  Business Overview · Risks · MD&A · Governance

Outputs three scores (1–5) + a Korean strategic insight.
"""

from __future__ import annotations

from textwrap import dedent
from dataclasses import dataclass
from typing import Optional

from hmas_base import BaseAgent, LLMClient, QualAgentOutput


# ──────────────────────────────────────────────────────────────────────────────
# Input data structure
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SecuritiesReportExcerpt:
    """Structured excerpts from a Korean Securities Report (사업보고서)."""
    ticker: str
    company_name: str
    info_updated: bool          # True = material changes since last filing
    business_overview: str      # Section 1: Business Description
    business_risks: str         # Section 2: Risk Factors
    mda: str                    # Section 3: MD&A (Management Discussion & Analysis)
    governance: str             # Section 4: Officers / Board / Governance


# ──────────────────────────────────────────────────────────────────────────────
# Agent
# ──────────────────────────────────────────────────────────────────────────────

class QualAgent(BaseAgent):
    """
    Strategic Analyst — Qualitative Corporate Disclosures (B.3).

    Fine-grained policy (from paper):
      • Distinguishes boilerplate text from meaningful strategic shifts.
      • Focuses on operational momentum, management credibility, hidden risks.
      • Returns scores for Business Momentum, Risk Severity, Management Trust.
      • Insight written in Korean (~150 chars).
    """

    # ── B.3.1 System Prompt ──────────────────────────────────────────────────
    @property
    def system_prompt(self) -> str:
        return dedent("""
        Role: You are a Strategic Analyst reporting to the Portfolio Manager.
        Your mission is to analyze qualitative corporate disclosures and provide
        a "Fundamental Risk & Catalyst Report" for the upcoming 1-month horizon.

        Perspective & Analysis Logic:
        - Filter: Distinguish between "stagnant boilerplate text" and
          "meaningful strategic shifts."
        - Focus: Identify qualitative triggers (catalysts or red flags) rather
          than just long-term value.
        - Target: Operational momentum, management credibility, and hidden
          structural risks.

        Guidelines:
        - Inputs: Excerpts from Securities Reports (Business Overview, Risks,
          MD&A, Governance).
        - Outputs: Three specific scores (1-5) and a strategic summary ("insight").
        - Format: Return ONLY a valid JSON object. The "insight" field MUST be
          written in Korean (~150 characters).

        JSON schema (return exactly this structure):
        {
          "business_momentum": <int 1-5>,
          "immediate_risk_severity": <int 1-5>,
          "management_trust": <int 1-5>,
          "insight": "<Korean string ~150 chars>"
        }

        Scoring reference:
          business_momentum     : 1=Deteriorating/Vague → 5=Strong tailwinds/Clear execution
          immediate_risk_severity: 1=High risk/Urgent   → 5=Low risk/Stable
          management_trust      : 1=Untrustworthy       → 5=Transparent/Aligned
        """).strip()

    # ── B.3.2 User Prompt builder ────────────────────────────────────────────
    @staticmethod
    def _build_user_prompt(report: SecuritiesReportExcerpt) -> str:
        info_flag = "Yes" if report.info_updated else "No"
        return dedent(f"""
        Instruction: Evaluate qualitative corporate data to advise the PM on
        stock attractiveness and potential risks for the next 1 month.

        Evaluation Items (Score 1-5):
        1. Business Momentum: Strength of cycle/strategy.
           (1: Deteriorating/Vague → 5: Strong tailwinds/Clear execution)
        2. Immediate Risk Severity: Probability of risks manifesting.
           (1: High risk/Urgent → 5: Low risk/Stable)
        3. Management Trust: Credibility & oversight structure.
           (1: Untrustworthy → 5: Transparent/Aligned)

        Rules & Output:
        - Focus: Look for changes in tone or NEW risk factors vs. prior filings.
        - Insight: Professional briefing in Korean (~150 chars).
        - Format: JSON only with the four keys defined in the system prompt.

        Input Data (Text Excerpts):
        Info Update: {info_flag}
        [1. Overview]    {report.business_overview}
        [2. Risks]       {report.business_risks}
        [3. MD&A]        {report.mda}
        [4. Governance]  {report.governance}
        """).strip()

    # ── Public interface ─────────────────────────────────────────────────────
    def run(self, report: SecuritiesReportExcerpt) -> QualAgentOutput:
        """
        Analyze a securities report excerpt and return QualAgentOutput.

        Parameters
        ----------
        report : SecuritiesReportExcerpt
            Structured text excerpts from the company filing.

        Returns
        -------
        QualAgentOutput
            Three 1-5 scores + Korean insight + composite 0-100 score.
        """
        user_prompt = self._build_user_prompt(report)
        data = self._call(user_prompt)

        return QualAgentOutput(
            business_momentum=int(data.get("business_momentum", 3)),
            immediate_risk_severity=int(data.get("immediate_risk_severity", 3)),
            management_trust=int(data.get("management_trust", 3)),
            insight=data.get("insight", ""),
            raw=data,
        )

    # ── Stub for offline testing ─────────────────────────────────────────────
    @staticmethod
    def stub_output(
        business_momentum: int = 4,
        immediate_risk_severity: int = 4,
        management_trust: int = 4,
        insight: str = "실적 모멘텀 양호, 리스크 요인 제한적. 경영진 투명성 높으며 단기 강세 스탠스 적절.",
    ) -> QualAgentOutput:
        """Return a synthetic output for unit-testing without an LLM key."""
        return QualAgentOutput(
            business_momentum=business_momentum,
            immediate_risk_severity=immediate_risk_severity,
            management_trust=management_trust,
            insight=insight,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Smoke test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sample_report = SecuritiesReportExcerpt(
        ticker="7203",
        company_name="Toyota Motor Corporation",
        info_updated=True,
        business_overview=(
            "Toyota operates in the automobile manufacturing segment globally. "
            "The company has accelerated its EV transition roadmap, targeting 3.5M BEV "
            "units by 2030 with a new solid-state battery pilot announced this quarter."
        ),
        business_risks=(
            "Key risks include FX exposure (USD/JPY sensitivity ~JPY 45B per yen move), "
            "semiconductor supply chain constraints, and intensifying competition in BEV from "
            "Chinese OEMs. Regulatory emission standards tightening in EU from 2025."
        ),
        mda=(
            "Operating income increased 18% YoY driven by favorable FX and cost reduction. "
            "North America margins improved to 9.2%. Management raised FY guidance by JPY 200B. "
            "Inventory normalization progressing; dealer stock at healthy 45-day supply."
        ),
        governance=(
            "Board refreshed with two independent outside directors added in June. "
            "Akio Toyoda transitioned to Chairman; new CEO Koji Sato leading operational focus. "
            "Remuneration aligned to ROIC and carbon reduction KPIs from FY2025."
        ),
    )

    # Offline stub demo (no API key needed)
    out = QualAgent.stub_output()
    print("=== Qual Agent Output (stub) ===")
    print(f"  Business Momentum    : {out.business_momentum}/5")
    print(f"  Immediate Risk Severity: {out.immediate_risk_severity}/5")
    print(f"  Management Trust     : {out.management_trust}/5")
    print(f"  Composite Score      : {out.composite_score}/100")
    print(f"  Insight (JP)         : {out.insight}")
