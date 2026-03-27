"""
H-MAS Sector / L1 Integration Agent
=====================================
L1 에이전트(Technical, Quant, Qual, News) 점수를 통합해
섹터/종목 레벨의 conviction score를 산출합니다.

hmas_agent_backtest.py 와 kospi_pipeline.py 에서 import:
  from hmas_sector_agent import SectorAgent, AnalystReport,
                                GranularitySetting, SectorRoCData, RoCItem
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from textwrap import dedent
from typing import Optional, List

from hmas_base import LLMClient, SectorAgentOutput


# ──────────────────────────────────────────────────────────────────────────────
# Supporting data classes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class AnalystReport:
    """단일 L1 에이전트 리포트."""
    analyst: str       # e.g. "Technical Analyst"
    score:   float     # 0-100
    comment: str       # 한 줄 코멘트


@dataclass
class RoCItem:
    """단일 지표의 전년 대비 증감률(RoC) 비교."""
    name:           str
    target_roc:     float
    sector_avg_roc: float


@dataclass
class SectorRoCData:
    """재무 지표 전체 RoC 비교 패키지."""
    sales:                 RoCItem
    op_profit:             RoCItem
    net_income:            RoCItem
    total_assets:          RoCItem
    equity:                RoCItem
    cash:                  RoCItem
    receivables:           RoCItem
    inventory:             RoCItem
    financial_assets:      RoCItem
    interest_bearing_debt: RoCItem
    current_liabilities:   RoCItem
    issued_shares:         RoCItem
    op_cf:                 RoCItem
    investing_cf:          RoCItem
    dividends:             RoCItem
    monthly_close:         RoCItem


class GranularitySetting(str, Enum):
    FINE   = "fine"
    COARSE = "coarse"


# ──────────────────────────────────────────────────────────────────────────────
# Sector Agent
# ──────────────────────────────────────────────────────────────────────────────

class SectorAgent:
    """
    L1 에이전트 리포트를 통합해 conviction score(0-100)를 산출.
    LLM 없이도 규칙 기반으로 동작 (fallback).
    """

    SYSTEM_PROMPT = dedent("""
    Role: You are a Senior Equity Analyst integrating multiple sub-analyst
    reports into a single conviction score for a KOSPI stock.

    Sub-analysts: Technical, Quantitative Fundamental, Qualitative Strategic,
                  News & Sentiment.

    Output JSON only:
    {
      "conviction_score": <int 0-100>,
      "investment_thesis": "<Korean ~100 chars>"
    }
    """).strip()

    def __init__(
        self,
        llm:         Optional[LLMClient]     = None,
        granularity: GranularitySetting      = GranularitySetting.FINE,
    ) -> None:
        self.llm         = llm
        self.granularity = granularity

    # ── public ───────────────────────────────────────────────────────────────

    def run(
        self,
        ticker:  str                       = "",
        reports: Optional[List[AnalystReport]] = None,
        regime:  str                       = "Transitional",
    ) -> SectorAgentOutput:
        reports = reports or []
        rule_score  = int(sum(r.score for r in reports) / len(reports)) if reports else 50
        rule_thesis = (
            f"L1 평균 {rule_score}/100 ({regime} 환경). "
            + (f"최고 신호: {max(reports, key=lambda r: r.score).analyst}" if reports else "데이터 없음")
        )

        if self.llm is None:
            return SectorAgentOutput(
                conviction_score=rule_score,
                investment_thesis=rule_thesis,
                raw={},
            )

        try:
            import json
            user = self._build_user_prompt(ticker, reports, regime)
            raw  = self.llm.chat(self.SYSTEM_PROMPT, user)
            data = json.loads(raw)
            return SectorAgentOutput(
                conviction_score=int(data.get("conviction_score", rule_score)),
                investment_thesis=data.get("investment_thesis", rule_thesis),
                raw=data,
            )
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"SectorAgent LLM failed: {e}")
            return SectorAgentOutput(
                conviction_score=rule_score,
                investment_thesis=rule_thesis,
                raw={},
            )

    # ── internal ─────────────────────────────────────────────────────────────

    def _build_user_prompt(self, ticker, reports, regime):
        lines = [f"Ticker: {ticker}  |  Regime: {regime}\n", "Sub-analyst reports:"]
        for r in reports:
            lines.append(f"  [{r.analyst}]  Score={r.score:.1f}/100  |  {r.comment}")
        lines.append("\nReturn ONLY JSON.")
        return "\n".join(lines)
