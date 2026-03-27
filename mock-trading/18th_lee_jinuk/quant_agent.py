"""
H-MAS — Quantitative Fundamental Agent
=======================================
역할: 기업의 재무제표 '숫자'를 바탕으로 중장기적 투자 가치를 정량적으로 평가.
데이터: EDINET API에서 수집한 분기/반기/연간 재무제표
출력: 수익성·가치·현금흐름·재무건전성·성장성 5개 차원 점수 + 최종 종합 점수(0-100)

Fine-grained 평가 정책:
  • ROE > 15% → 높은 가중치 부여 (기관투자자 기준)
  • Debt/Equity > 2.0 → "Hard Sell" 후보 플래그 (리스크 게이트)
  • FCF 마진 > 10% → 현금창출력 프리미엄
  • 전년 대비 영업이익 증감률(RoC)을 모멘텀 보정치로 활용
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from textwrap import dedent
from typing import Optional

from hmas_base import BaseAgent, LLMClient


# ──────────────────────────────────────────────────────────────────────────────
# Output 데이터 구조
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class QuantAgentOutput:
    """정량 에이전트 출력."""
    # 5개 차원 점수 (1-5)
    profitability_score: int        # 수익성 (ROE, ROA)
    valuation_score: int            # 가치 (PER, PBR)
    cashflow_score: int             # 현금흐름 (FCF, 영업CF)
    financial_health_score: int     # 재무건전성 (부채비율, 자기자본비율)
    growth_score: int               # 성장성 (매출/EPS YoY)

    # 플래그
    hard_sell_flag: bool = False    # D/E > 2.0 또는 FCF 음수 연속
    highlight: str = ""            # 핵심 투자 포인트 (한국어)
    raw: dict = field(default_factory=dict)

    @property
    def composite_score(self) -> float:
        """5개 차원 가중 평균 → 0-100 변환."""
        # 가중치: 수익성 30%, 가치 20%, 현금흐름 20%, 건전성 15%, 성장 15%
        weighted = (
            self.profitability_score    * 0.30
            + self.valuation_score      * 0.20
            + self.cashflow_score       * 0.20
            + self.financial_health_score * 0.15
            + self.growth_score         * 0.15
        )
        # 1-5 → 0-100 변환
        score = (weighted - 1) / 4 * 100
        # Hard Sell 플래그 → 최대 40점으로 제한
        if self.hard_sell_flag:
            score = min(score, 40.0)
        return round(score, 1)

    def to_analyst_report_comment(self) -> str:
        flag_str = " ⚠️ HARD_SELL_FLAG" if self.hard_sell_flag else ""
        return (
            f"Quant: P={self.profitability_score} V={self.valuation_score} "
            f"CF={self.cashflow_score} H={self.financial_health_score} "
            f"G={self.growth_score} → {self.composite_score:.1f}/100{flag_str}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Rule-based pre-scorer (LLM 없이도 동작하는 투명한 규칙)
# ──────────────────────────────────────────────────────────────────────────────

def _rule_based_score(metrics) -> QuantAgentOutput:
    """
    재무 지표를 규칙 기반으로 1-5점 평가.
    LLM 점수의 앵커 역할 + LLM 없이도 동작하는 fallback.
    """
    m = metrics

    # ── 수익성 ──────────────────────────────────────────────────────────────
    roe = m.roe or 0
    roa = m.roa or 0
    if roe >= 20 and roa >= 8:      prof = 5
    elif roe >= 15 and roa >= 5:    prof = 4
    elif roe >= 10 and roa >= 3:    prof = 3
    elif roe >= 5:                  prof = 2
    else:                           prof = 1

    # ── 밸류에이션 (낮은 PER = 저평가 = 높은 점수) ──────────────────────────
    per = m.per or 999
    if per <= 10:                   val = 5
    elif per <= 15:                 val = 4
    elif per <= 25:                 val = 3
    elif per <= 35:                 val = 2
    else:                           val = 1

    # ── 현금흐름 ────────────────────────────────────────────────────────────
    fcf = m.free_cf or 0
    op_cf = m.operating_cf or 0
    net_sales = m.net_sales or 1
    fcf_margin = fcf / net_sales * 100 if net_sales else 0
    if fcf > 0 and fcf_margin >= 10:    cf = 5
    elif fcf > 0 and fcf_margin >= 5:   cf = 4
    elif fcf > 0:                        cf = 3
    elif op_cf > 0:                      cf = 2
    else:                                cf = 1

    # ── 재무 건전성 ─────────────────────────────────────────────────────────
    eq_ratio = m.equity_ratio or 0
    de = m.de_ratio or 999
    if eq_ratio >= 50 and de <= 0.5:    health = 5
    elif eq_ratio >= 40 and de <= 1.0:  health = 4
    elif eq_ratio >= 30 and de <= 1.5:  health = 3
    elif de <= 2.0:                     health = 2
    else:                               health = 1

    # ── 성장성 ──────────────────────────────────────────────────────────────
    sales_roc = m.net_sales_roc or 0
    op_roc    = m.operating_income_roc or 0
    eps_roc   = m.eps_roc or 0
    avg_roc   = (sales_roc + op_roc + eps_roc) / 3
    if avg_roc >= 20:                   growth = 5
    elif avg_roc >= 10:                 growth = 4
    elif avg_roc >= 3:                  growth = 3
    elif avg_roc >= -5:                 growth = 2
    else:                               growth = 1

    # Hard Sell 플래그
    hard_sell = (de > 2.0) or (fcf < 0 and op_cf < 0)

    highlight = (
        f"ROE={roe:.1f}% PER={per:.1f}x FCF_margin={fcf_margin:.1f}% "
        f"D/E={de:.2f}x 영업이익RoC={op_roc:+.1f}%"
    )

    return QuantAgentOutput(
        profitability_score=prof,
        valuation_score=val,
        cashflow_score=cf,
        financial_health_score=health,
        growth_score=growth,
        hard_sell_flag=hard_sell,
        highlight=highlight,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Agent
# ──────────────────────────────────────────────────────────────────────────────

class QuantAgent(BaseAgent):
    """
    정량적 펀더멘털 에이전트 (Quantitative Fundamental Agent).

    평가 로직:
      1. 규칙 기반으로 5개 차원 점수를 먼저 산출 (투명성·감사 가능성 확보)
      2. LLM에게 재무 지표 전문을 전달하여 점수 검토 + 하이라이트 보완
      3. LLM이 규칙 점수를 ±1점 범위에서 조정하고 한국어 코멘트 추가

    Fine-grained 평가 규칙:
      • ROE ≥ 15% + ROA ≥ 5%  → 수익성 4점 이상
      • D/E > 2.0              → Hard Sell 플래그 + 최종 점수 ≤ 40점 제한
      • FCF마진 ≥ 10%          → 현금흐름 프리미엄 (5점)
      • 영업이익 RoC ≥ 20%    → 성장성 모멘텀 보정
    """

    @property
    def system_prompt(self) -> str:
        return dedent("""
        Role: You are a Quantitative Fundamental Analyst on the investment committee.
        Your mission is to evaluate a company's investment value based EXCLUSIVELY
        on quantitative financial data (numbers from financial statements).

        Evaluation Framework (Score 1-5 for each dimension):
        1. Profitability   : ROE, ROA, Operating Margin
        2. Valuation       : PER, PBR (lower PER/PBR = better value)
        3. Cash Flow       : Free CF, Operating CF, FCF Margin
        4. Financial Health: Equity Ratio, D/E Ratio (higher equity = safer)
        5. Growth          : Revenue YoY, Operating Income YoY, EPS YoY (RoC)

        Fine-grained Policy (MUST follow):
        - ROE ≥ 15% AND ROA ≥ 5%: Assign Profitability ≥ 4
        - D/E Ratio > 2.0: Set hard_sell_flag=true; final composite MUST be ≤ 40
        - FCF Margin ≥ 10% (FCF/Revenue): Assign Cash Flow = 5
        - Operating Income RoC ≥ 20%: Add +1 momentum bonus to Growth (max 5)
        - You may adjust the rule-based anchor scores by ±1 only

        Output: JSON with scores (1-5), hard_sell_flag (bool), highlight (Korean ~100 chars).

        JSON schema:
        {
          "profitability_score": <1-5>,
          "valuation_score": <1-5>,
          "cashflow_score": <1-5>,
          "financial_health_score": <1-5>,
          "growth_score": <1-5>,
          "hard_sell_flag": <bool>,
          "highlight": "<Korean ~100 chars>"
        }
        """).strip()

    def _build_user_prompt(self, metrics, rule_scores: QuantAgentOutput) -> str:
        return dedent(f"""
        Instruction: Review the financial metrics and the rule-based anchor scores.
        Adjust each score by ±1 if justified. Follow all fine-grained policies.

        === Financial Metrics ===
        {metrics.to_prompt_block()}

        === Rule-Based Anchor Scores (your starting point) ===
        Profitability    : {rule_scores.profitability_score}/5
        Valuation        : {rule_scores.valuation_score}/5
        Cash Flow        : {rule_scores.cashflow_score}/5
        Financial Health : {rule_scores.financial_health_score}/5
        Growth           : {rule_scores.growth_score}/5
        Hard Sell Flag   : {rule_scores.hard_sell_flag}
        Anchor Highlight : {rule_scores.highlight}

        Provide your final scores, hard_sell_flag, and a Korean highlight (~100 chars).
        Return ONLY the JSON object.
        """).strip()

    def run(self, metrics) -> QuantAgentOutput:
        """
        재무 지표를 받아 QuantAgentOutput 반환.

        Parameters
        ----------
        metrics : FinancialMetrics (edinet_collector)

        Returns
        -------
        QuantAgentOutput
        """
        # 1. 규칙 기반 anchor 점수 계산
        rule_scores = _rule_based_score(metrics)

        if self.llm is None:
            # LLM 없이 규칙 점수만 반환
            return rule_scores

        # 2. LLM에게 검토 요청
        try:
            user_prompt = self._build_user_prompt(metrics, rule_scores)
            data = self._call(user_prompt)
            return QuantAgentOutput(
                profitability_score=int(data.get("profitability_score",
                                                  rule_scores.profitability_score)),
                valuation_score=int(data.get("valuation_score",
                                             rule_scores.valuation_score)),
                cashflow_score=int(data.get("cashflow_score",
                                            rule_scores.cashflow_score)),
                financial_health_score=int(data.get("financial_health_score",
                                                     rule_scores.financial_health_score)),
                growth_score=int(data.get("growth_score", rule_scores.growth_score)),
                hard_sell_flag=bool(data.get("hard_sell_flag", rule_scores.hard_sell_flag)),
                highlight=data.get("highlight", rule_scores.highlight),
                raw=data,
            )
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"QuantAgent LLM failed, using rule scores: {e}")
            return rule_scores

    @staticmethod
    def stub_output(
        profitability_score: int = 4,
        valuation_score: int = 4,
        cashflow_score: int = 4,
        financial_health_score: int = 4,
        growth_score: int = 5,
        hard_sell_flag: bool = False,
        highlight: str = "ROE16.8%・営業利益96%増と高収益。PER9.8倍で割安。FCFプラス継続。自己資本比率35.9%は業界標準。",
    ) -> QuantAgentOutput:
        return QuantAgentOutput(
            profitability_score=profitability_score,
            valuation_score=valuation_score,
            cashflow_score=cashflow_score,
            financial_health_score=financial_health_score,
            growth_score=growth_score,
            hard_sell_flag=hard_sell_flag,
            highlight=highlight,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Smoke test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))

    from dart_collector import FinancialMetrics

    metrics = make_stub_financial_metrics("7203")

    # Rule-based only (LLM 없이)
    agent = QuantAgent(llm=None)
    out = agent.run(metrics)

    print("=== Quantitative Agent Output (rule-based) ===")
    print(f"  Profitability    : {out.profitability_score}/5")
    print(f"  Valuation        : {out.valuation_score}/5")
    print(f"  Cash Flow        : {out.cashflow_score}/5")
    print(f"  Financial Health : {out.financial_health_score}/5")
    print(f"  Growth           : {out.growth_score}/5")
    print(f"  Hard Sell Flag   : {out.hard_sell_flag}")
    print(f"  Composite Score  : {out.composite_score}/100")
    print(f"  Highlight        : {out.highlight}")
    print(f"  Analyst Comment  : {out.to_analyst_report_comment()}")
