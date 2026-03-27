"""
H-MAS — News Agent  [B.4]
==========================
Analyzes news headlines/summaries from the past month.

Outputs Return Outlook (1-5) + Risk Outlook (1-5) + Korean reason.
"""

from __future__ import annotations

from dataclasses import dataclass
from textwrap import dedent
from typing import List

from hmas_base import BaseAgent, LLMClient, NewsAgentOutput


# ──────────────────────────────────────────────────────────────────────────────
# Input data structure
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class NewsItem:
    date: str       # e.g. "2024-01-15"
    headline: str
    summary: str = ""

    def format(self) -> str:
        text = f"{self.date}: {self.headline}"
        if self.summary:
            text += f" / {self.summary}"
        return text


# ──────────────────────────────────────────────────────────────────────────────
# Agent
# ──────────────────────────────────────────────────────────────────────────────

class NewsAgent(BaseAgent):
    """
    Senior News Analyst — Monthly news signal (B.4).

    Fine-grained policy:
      • Distinguishes temporary noise from structural changes
        (product launches, regulations, ESG, lawsuits, management changes).
      • Evaluates Return Outlook (upside) and Risk Outlook (downside) separately.
      • Empty-news case: both scores → 1, reason → 'No News'.
      • Reason written in Korean (~100 chars).
    """

    # ── B.4.1 System Prompt ──────────────────────────────────────────────────
    @property
    def system_prompt(self) -> str:
        return dedent("""
        Role: You are a Senior News Analyst specializing in the stock market.
        Your task is to analyze news headlines and summaries from the past month
        to provide qualitative insights that complement fundamental scores.

        Evaluation Guidelines:
        - Perspectives: Evaluate impact on "Return Outlook" (Upside) and
          "Risk Outlook" (Downside) separately.
        - Scoring Scale (1-5): 1=Minimal/None → 3=Moderate → 5=Extreme.
        - Analysis Logic: Distinguish between temporary noise and structural
          changes (e.g., product launches, regulations, ESG events, lawsuits,
          CEO changes).
        - Empty Case: If no news is provided, set both scores to 1 and reason
          to "No News".
        - Output: JSON format ONLY. Reason must be a concise Korean summary
          (~100 chars).

        JSON schema (return exactly this structure):
        {
          "return_outlook": <int 1-5>,
          "risk_outlook": <int 1-5>,
          "reason": "<Korean string ~100 chars>"
        }

        Scoring reference:
          return_outlook : 1=No positive catalyst → 5=Major positive catalyst
          risk_outlook   : 1=No material risk     → 5=Severe/urgent downside risk
        """).strip()

    # ── B.4.2 User Prompt builder ────────────────────────────────────────────
    @staticmethod
    def _build_user_prompt(news_items: List[NewsItem], ticker: str = "") -> str:
        header = f"Ticker: {ticker}\n" if ticker else ""

        if not news_items:
            news_block = "(No news articles available this month)"
        else:
            news_block = "\n".join(item.format() for item in news_items)

        return dedent(f"""
        Instruction: Evaluate "Return Outlook" and "Risk Outlook" (1-3 months)
        based on the provided news articles.

        {header}
        Evaluation Criteria (Score 1-5):
        - Return Outlook: Positive momentum (new products, contracts, expansion,
          earnings beats, management upgrades).
        - Risk Outlook: Potential downside/uncertainty (supply chain, lawsuits,
          regulatory actions, earnings misses, CEO departure).

        Rules & Output:
        - Balance: Identify risks even if news is generally positive.
        - Empty Case: If no news, set both scores to 1 and reason to "No News".
        - Format: JSON with scores (1-5) and Korean reason (~100 chars).

        News List for the Month:
        {news_block}
        """).strip()

    # ── Public interface ─────────────────────────────────────────────────────
    def run(
        self,
        news_items: List[NewsItem],
        ticker: str = "",
    ) -> NewsAgentOutput:
        """
        Analyze monthly news and return NewsAgentOutput.

        Parameters
        ----------
        news_items : List[NewsItem]
            News from the past ~30 days. Pass empty list for no-news case.
        ticker : str, optional
            Stock ticker for context.

        Returns
        -------
        NewsAgentOutput
            return_outlook (1-5), risk_outlook (1-5), Korean reason,
            net_score (0-100).
        """
        user_prompt = self._build_user_prompt(news_items, ticker)
        data = self._call(user_prompt)

        return NewsAgentOutput(
            return_outlook=int(data.get("return_outlook", 1)),
            risk_outlook=int(data.get("risk_outlook", 1)),
            reason=data.get("reason", "No News"),
            raw=data,
        )

    # ── Stub ─────────────────────────────────────────────────────────────────
    @staticmethod
    def stub_output(
        return_outlook: int = 4,
        risk_outlook: int = 2,
        reason: str = "긍정적 뉴스 우세. 상승 재료가 하락 요인을 압도하는 구도.",
    ) -> NewsAgentOutput:
        return NewsAgentOutput(
            return_outlook=return_outlook,
            risk_outlook=risk_outlook,
            reason=reason,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Smoke test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sample_news = [
        NewsItem("2024-01-15", "Toyota launches next-gen solid-state battery pilot plant",
                 "Commercial production targeting 2027; analyst upgrades follow."),
        NewsItem("2024-01-20", "CEO Koji Sato raises FY guidance by JPY 200B",
                 "Driven by North America margin expansion and favorable FX."),
        NewsItem("2024-01-28", "EU tightens 2025 emission standards; compliance cost risk flagged",
                 "Toyota estimates incremental CAPEX of JPY 80B for EU compliance."),
    ]

    out = NewsAgent.stub_output()
    print("=== News Agent Output (stub) ===")
    print(f"  Return Outlook : {out.return_outlook}/5")
    print(f"  Risk Outlook   : {out.risk_outlook}/5")
    print(f"  Net Score      : {out.net_score}/100")
    print(f"  Reason (JP)    : {out.reason}")
