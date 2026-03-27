"""
H-MAS KOSPI Pipeline
=====================
실제 API 기반 KOSPI 종목 분석 파이프라인.

데이터 소스:
  DART API     → Quant (재무제표 수치) + Qual (사업보고서 텍스트)
  Yahoo Finance → Technical (주가) + News (뉴스) + Quant 보완 (PER/PBR)
  FRED          → Macro (한국·미국 거시지표)
  OpenAI        → Qual·News·Sector·Macro·PM Agent LLM 추론

환경변수 설정:
  export OPENAI_API_KEY="sk-..."
  export DART_API_KEY="..."
  export FRED_API_KEY="..."

실행:
  # 단일 종목
  python kospi_pipeline.py --ticker 005930

  # KOSPI 배치
  python kospi_pipeline.py --mode batch --tickers 005930 000660 005380 035420

  # JSON 출력
  python kospi_pipeline.py --ticker 005930 --json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import date
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

from hmas_base        import LLMClient, QualAgentOutput, NewsAgentOutput, SectorAgentOutput, MacroAgentOutput, PMAgentOutput
from hmas_qual_agent  import QualAgent, SecuritiesReportExcerpt
from hmas_news_agent  import NewsAgent, NewsItem
from hmas_sector_agent import SectorAgent, AnalystReport, GranularitySetting, SectorRoCData, RoCItem
from hmas_macro_agent import MacroAgent, MacroIndicators
from hmas_pm_agent    import PMAgent, REGIME_WEIGHTS
from quant_agent      import QuantAgent, QuantAgentOutput
from dart_collector   import DartClient, FinancialMetrics, SecuritiesReportText, KOSPI100_MASTER, KOSPI_SECTOR_PEERS
from kospi_collectors import KospiYFinanceCollector, KoreaMarcoCollector

try:
    from technical_agent import TechnicalAgent
    _TECH_AVAILABLE = True
except ImportError:
    _TECH_AVAILABLE = False


# ──────────────────────────────────────────────────────────────────────────────
# Result
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class KospiHMASResult:
    ticker: str
    company_name: str
    sector: str
    as_of_date: str

    technical_score: float
    quant_output: QuantAgentOutput
    qual_output: QualAgentOutput
    news_output: NewsAgentOutput
    sector_output: SectorAgentOutput
    macro_output: MacroAgentOutput
    macro_regime: str
    pm_output: PMAgentOutput
    regime_weights: dict

    def summary(self) -> str:
        w = self.regime_weights
        hs = "⚠️ HARD_SELL " if self.quant_output.hard_sell_flag else ""
        lines = [
            f"{'='*68}",
            f"  H-MAS KOSPI │ {self.company_name} ({self.ticker}) │ {self.sector}",
            f"  기준일: {self.as_of_date}",
            f"{'='*68}",
            f"",
            f"  ── L1: Specialist Agents ──────────────────────────────────",
            f"  Technical  {self.technical_score:6.1f}/100",
            f"  Quant      {self.quant_output.composite_score:6.1f}/100  {hs}"
            f"(P={self.quant_output.profitability_score} V={self.quant_output.valuation_score} "
            f"CF={self.quant_output.cashflow_score} H={self.quant_output.financial_health_score} "
            f"G={self.quant_output.growth_score})",
            f"  Qual       {self.qual_output.composite_score:6.1f}/100  "
            f"(BM={self.qual_output.business_momentum} RS={self.qual_output.immediate_risk_severity} "
            f"MT={self.qual_output.management_trust})",
            f"  News       {self.news_output.net_score:6.1f}/100  "
            f"(Ret={self.news_output.return_outlook} Risk={self.news_output.risk_outlook})",
            f"",
            f"  ── L2: Adjudicators ────────────────────────────────────────",
            f"  Sector     {self.sector_output.conviction_score:6d}/100",
            f"  Macro      {self.macro_output.composite_score:6.1f}/100  Regime: {self.macro_regime}",
            f"  Weights    Tech={w['technical']:.0%}  Fund={w['fundamental']:.0%}  Qual={w['qualitative']:.0%}",
            f"",
            f"  ── L3: Portfolio Manager ────────────────────────────────────",
            f"  ★ FINAL    {self.pm_output.final_score:6d}/100  →  {_label(self.pm_output.final_score)}",
            f"",
            f"  Quant  : {self.quant_output.highlight}",
            f"  Qual   : {self.qual_output.insight}",
            f"  News   : {self.news_output.reason[:80]}",
            f"  Macro  : {self.macro_output.summary[:80]}",
            f"  PM     : {self.pm_output.reason[:80]}",
            f"{'='*68}",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "company": self.company_name,
            "sector": self.sector,
            "as_of_date": self.as_of_date,
            "final_score": self.pm_output.final_score,
            "recommendation": _label(self.pm_output.final_score),
            "macro_regime": self.macro_regime,
            "regime_weights": self.regime_weights,
            "scores": {
                "technical":        round(self.technical_score, 1),
                "quant":            self.quant_output.composite_score,
                "quant_hard_sell":  self.quant_output.hard_sell_flag,
                "qual":             self.qual_output.composite_score,
                "news":             self.news_output.net_score,
                "sector":           self.sector_output.conviction_score,
                "macro":            self.macro_output.composite_score,
            },
            "details": {
                "quant_highlight": self.quant_output.highlight,
                "qual_insight":    self.qual_output.insight,
                "news_reason":     self.news_output.reason,
                "macro_summary":   self.macro_output.summary,
                "pm_reason":       self.pm_output.reason,
                "investment_thesis": self.sector_output.investment_thesis,
            },
        }


def _label(score: int) -> str:
    if score >= 75: return "Strong Overweight ★★★"
    if score >= 60: return "Overweight ★★"
    if score >= 45: return "Neutral ★"
    if score >= 30: return "Underweight"
    return "Strong Underweight"


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────────────────────────────────────

class KospiHMASPipeline:
    """
    KOSPI H-MAS 파이프라인.

    Parameters
    ----------
    llm          : LLMClient (None → Quant만 규칙 기반, 나머지 stub)
    granularity  : Sector Agent 모드 (FINE 권장)
    dart_key     : DART API 키
    fred_key     : FRED API 키
    """

    def __init__(
        self,
        llm: Optional[LLMClient] = None,
        granularity: GranularitySetting = GranularitySetting.FINE,
        dart_key: Optional[str] = None,
        fred_key: Optional[str] = None,
    ) -> None:
        self.llm  = llm
        self.dart = DartClient(api_key=dart_key or os.getenv("DART_API_KEY"))
        self.yf   = KospiYFinanceCollector()
        self.macro_collector = KoreaMarcoCollector(fred_api_key=fred_key or os.getenv("FRED_API_KEY"))

        self.quant_agent  = QuantAgent(llm)
        self.qual_agent   = QualAgent(llm)   if llm else None
        self.news_agent   = NewsAgent(llm)   if llm else None
        self.sector_agent = SectorAgent(llm, granularity) if llm else None
        self.macro_agent  = MacroAgent(llm)  if llm else None
        self.pm_agent     = PMAgent(llm)     if llm else None

    def run(
        self,
        ticker: str,
        as_of_date: Optional[date] = None,
        shared_macro: Optional[MacroIndicators] = None,
    ) -> KospiHMASResult:
        """
        단일 KOSPI 종목 분석.

        Parameters
        ----------
        ticker       : 6자리 종목코드 (e.g. "005930")
        as_of_date   : 분석 기준일 (None = 오늘)
        shared_macro : 배치 실행 시 공유 MacroIndicators
        """
        ref_date = as_of_date or date.today()
        info = KOSPI100_MASTER.get(ticker, {"name": ticker, "sector": "기타"})
        company_name = info["name"]
        sector       = info["sector"]

        logger.info(f"{'='*60}")
        logger.info(f"[{ticker}] {company_name} | {sector} | {ref_date}")
        logger.info(f"{'='*60}")

        # ── 1. DART 재무제표 수집 ──────────────────────────────────────────

        logger.info(f"[{ticker}] DART 재무제표 수집...")
        dart_metrics = self.dart.get_financial_metrics(
            stock_code=ticker,
            company_name=company_name,
        )

        # ── 2. Yahoo Finance로 주가·PER·PBR 보완 ─────────────────────────

        logger.info(f"[{ticker}] Yahoo Finance 주가 수집...")
        price_series = self.yf.get_price_series(ticker, days_back=120, end_date=ref_date)

        if dart_metrics:
            fin_metrics = self.yf.get_financial_metrics(ticker, company_name, dart_metrics)
        else:
            logger.warning(f"[{ticker}] DART 실패 → Yahoo Finance 단독 재무 수집")
            fin_metrics = self.yf.get_financial_metrics(ticker, company_name)

        if fin_metrics is None:
            logger.error(f"[{ticker}] 재무 데이터 수집 완전 실패")
            fin_metrics = _empty_metrics(ticker, company_name)

        as_of_str = fin_metrics.fiscal_year_end or str(ref_date)

        # ── 3. DART 사업보고서 텍스트 수집 ────────────────────────────────

        logger.info(f"[{ticker}] DART 사업보고서 텍스트 수집...")
        sec_text = self.dart.get_securities_report_text(
            stock_code=ticker,
            company_name=company_name,
        )
        if sec_text is None:
            sec_text = _empty_sec_text(ticker, company_name)

        # ── 4. 뉴스 수집 (Yahoo Finance) ──────────────────────────────────

        logger.info(f"[{ticker}] 뉴스 수집...")
        news_items = self.yf.get_news(ticker, max_items=20)

        # ── 5. 거시지표 수집 ──────────────────────────────────────────────

        if shared_macro is None:
            logger.info("거시지표 수집 (FRED + Yahoo Finance)...")
            try:
                macro_inds = self.macro_collector.fetch()
            except Exception as e:
                logger.warning(f"거시지표 수집 실패: {e} → stub 사용")
                macro_inds = _stub_macro_kr()
        else:
            macro_inds = shared_macro

        # ── 6. Technical Agent ─────────────────────────────────────────────

        if _TECH_AVAILABLE and len(price_series) >= 30:
            ta = TechnicalAgent()
            tech_score = ta.score_series(price_series)[-1]
            logger.info(f"[{ticker}] Technical: {tech_score:.1f}/100 (실제 계산)")
        else:
            tech_score = _momentum_score(price_series)
            logger.info(f"[{ticker}] Technical: {tech_score:.1f}/100 (모멘텀 추정)")

        # ── 7. Quant Agent ─────────────────────────────────────────────────

        quant_out = self.quant_agent.run(fin_metrics)
        logger.info(f"[{ticker}] Quant: {quant_out.composite_score:.1f}/100  HardSell={quant_out.hard_sell_flag}")

        # ── 8. Qual Agent ──────────────────────────────────────────────────

        excerpt = SecuritiesReportExcerpt(
            ticker=ticker,
            company_name=company_name,
            info_updated=sec_text.info_updated,
            business_overview=sec_text.business_overview,
            business_risks=sec_text.business_risks,
            mda=sec_text.mda,
            governance=sec_text.governance,
        )

        if self.qual_agent and (sec_text.business_overview or sec_text.mda):
            qual_out = self.qual_agent.run(excerpt)
            logger.info(f"[{ticker}] Qual: {qual_out.composite_score:.1f}/100 (LLM)")
        else:
            qual_out = _rule_qual(sec_text)
            logger.info(f"[{ticker}] Qual: {qual_out.composite_score:.1f}/100 (규칙 기반)")

        # ── 9. News Agent ──────────────────────────────────────────────────

        if self.news_agent and news_items:
            news_out = self.news_agent.run(news_items, ticker)
            logger.info(f"[{ticker}] News: {news_out.net_score:.1f}/100 (LLM)")
        else:
            news_out = _rule_news(news_items, company_name)
            logger.info(f"[{ticker}] News: {news_out.net_score:.1f}/100 (규칙 기반)")

        # ── 10. Macro Agent ────────────────────────────────────────────────

        if self.macro_agent:
            macro_out = self.macro_agent.run(macro_inds)
            logger.info(f"[{ticker}] Macro: {macro_out.composite_score:.1f}/100 (LLM)")
        else:
            macro_out = _rule_macro(macro_inds)
            logger.info(f"[{ticker}] Macro: {macro_out.composite_score:.1f}/100 (규칙 기반)")

        macro_regime = MacroAgent.classify_regime(macro_out)
        logger.info(f"[{ticker}] Regime: {macro_regime}")

        # ── 11. Sector Agent ───────────────────────────────────────────────

        sector_avgs = _get_sector_avg_stub(sector)
        roc_data = _build_roc(fin_metrics, sector_avgs)

        analyst_reports = [
            AnalystReport("Technical Analyst",    tech_score,
                          f"기술적 분석 스코어 {tech_score:.1f}/100"),
            AnalystReport("Quant Fundamental",    quant_out.composite_score,
                          quant_out.to_analyst_report_comment()),
            AnalystReport("Qualitative Strategic", qual_out.composite_score,
                          f"BM={qual_out.business_momentum} RS={qual_out.immediate_risk_severity} MT={qual_out.management_trust}"),
            AnalystReport("News Analyst",          news_out.net_score,
                          f"Ret={news_out.return_outlook} Risk={news_out.risk_outlook}. {news_out.reason[:50]}"),
        ]

        if self.sector_agent:
            sector_out = self.sector_agent.run(
                ticker=ticker,
                analyst_reports=analyst_reports,
                sector_name=sector,
                roc_data=roc_data,
            )
            logger.info(f"[{ticker}] Sector: {sector_out.conviction_score}/100 (LLM)")
        else:
            sector_out = _rule_sector(analyst_reports)
            logger.info(f"[{ticker}] Sector: {sector_out.conviction_score}/100 (규칙 기반)")

        # ── 12. PM Agent ───────────────────────────────────────────────────

        if self.pm_agent:
            pm_out = self.pm_agent.run(ticker, macro_out, sector_out, macro_regime)
            logger.info(f"[{ticker}] PM: {pm_out.final_score}/100 (LLM)")
        else:
            pm_out = _rule_pm(tech_score, quant_out, qual_out, news_out,
                               macro_out, sector_out, macro_regime)
            logger.info(f"[{ticker}] PM: {pm_out.final_score}/100 (규칙 기반)")

        regime_weights = REGIME_WEIGHTS.get(macro_regime, REGIME_WEIGHTS["Transitional"])
        logger.info(f"[{ticker}] ★ FINAL: {pm_out.final_score}/100 → {_label(pm_out.final_score)}")

        return KospiHMASResult(
            ticker=ticker,
            company_name=company_name,
            sector=sector,
            as_of_date=as_of_str,
            technical_score=tech_score,
            quant_output=quant_out,
            qual_output=qual_out,
            news_output=news_out,
            sector_output=sector_out,
            macro_output=macro_out,
            macro_regime=macro_regime,
            pm_output=pm_out,
            regime_weights=regime_weights,
        )

    def run_batch(
        self,
        tickers: list[str],
        as_of_date: Optional[date] = None,
    ) -> list[KospiHMASResult]:
        """복수 종목 배치 실행. 거시지표는 1회만 수집하여 공유."""
        logger.info(f"배치 실행: {len(tickers)}개 종목")

        logger.info("거시지표 공통 수집...")
        try:
            shared_macro = self.macro_collector.fetch()
        except Exception as e:
            logger.warning(f"거시지표 수집 실패: {e}")
            shared_macro = _stub_macro_kr()

        results = []
        for ticker in tickers:
            try:
                r = self.run(ticker, as_of_date=as_of_date, shared_macro=shared_macro)
                results.append(r)
            except Exception as e:
                logger.error(f"[{ticker}] 실패: {e}")

        logger.info(f"배치 완료: {len(results)}/{len(tickers)} 종목 성공")
        return results


# ──────────────────────────────────────────────────────────────────────────────
# LLM 없이 동작하는 규칙 기반 fallback들
# ──────────────────────────────────────────────────────────────────────────────

def _momentum_score(prices: list[float]) -> float:
    if len(prices) < 20:
        return 50.0
    r = sum(prices[-5:])  / 5
    m = sum(prices[-20:]) / 20
    l = sum(prices)       / len(prices)
    s = 50.0
    if r > m:  s += 10
    if m > l:  s += 10
    if r > l:  s += 5
    return min(max(s, 20.0), 80.0)


def _rule_qual(sec_text: SecuritiesReportText) -> QualAgentOutput:
    """
    텍스트 키워드 기반 정성 평가.
    텍스트가 없으면 중립(3점)으로 처리 — 데이터 부재를 패널티로 주지 않음.
    """
    full_text = " ".join([
        sec_text.business_overview or "",
        sec_text.business_risks    or "",
        sec_text.mda               or "",
        sec_text.governance        or "",
    ]).lower()

    # 업황·실적 모멘텀 키워드
    pos_bm = ["성장", "증가", "확대", "호조", "상승", "개선", "신규", "수주", "출시", "혁신", "ai", "반도체"]
    neg_bm = ["감소", "하락", "부진", "축소", "어려움", "둔화", "경쟁 심화"]
    bm_score = 3 + sum(1 for k in pos_bm if k in full_text) // 2                  - sum(1 for k in neg_bm if k in full_text) // 2
    bm = max(1, min(5, bm_score))

    # 리스크 키워드 (많을수록 점수 낮음)
    risk_kw = ["소송", "조사", "제재", "환율", "원자재", "규제", "공급망", "경쟁", "불확실"]
    rs_score = 4 - sum(1 for k in risk_kw if k in full_text) // 2
    rs = max(1, min(5, rs_score))

    # 거버넌스 키워드
    gov_pos = ["사외이사", "감사위원회", "투명", "주주", "배당", "esg", "공시"]
    mt_score = 3 + sum(1 for k in gov_pos if k in full_text) // 2
    mt = max(1, min(5, mt_score))

    has_text = bool(full_text.strip())
    if not has_text:
        # 텍스트 수집 실패시 중립값 유지 (패널티 없음)
        bm, rs, mt = 3, 3, 3
        insight = "DART 사업보고서 텍스트 수집 미완료. 중립값 적용. LLM 활성화 시 정확한 분석 제공."
    else:
        insight = f"DART 텍스트 기반 분석. 긍정 신호 {sum(1 for k in pos_bm if k in full_text)}건, 리스크 키워드 {sum(1 for k in risk_kw if k in full_text)}건 감지."

    return QualAgentOutput(
        business_momentum=bm,
        immediate_risk_severity=rs,
        management_trust=mt,
        insight=insight,
        raw={},
    )


def _rule_news(items: list[NewsItem], company_name: str) -> NewsAgentOutput:
    POS = ["상향", "호실적", "증가", "성장", "최고", "beat", "strong", "record",
           "어닝서프라이즈", "신제품", "수주", "계약"]
    NEG = ["하향", "부진", "감소", "적자", "리스크", "우려", "miss", "weak",
           "소송", "조사", "감사", "손실"]
    pos = neg = 0
    for n in items:
        t = (n.headline + n.summary).lower()
        pos += sum(1 for k in POS if k in t)
        neg += sum(1 for k in NEG if k in t)
    ret  = min(5, max(1, 3 + (pos - neg)))
    risk = min(5, max(1, 3 - (pos - neg) // 2))
    reason = f"포지티브 신호 {pos}건, 네거티브 신호 {neg}건 감지. 총 뉴스 {len(items)}건 분석."
    return NewsAgentOutput(return_outlook=ret, risk_outlook=risk, reason=reason, raw={})


def _rule_macro(inds: MacroIndicators) -> MacroAgentOutput:
    vix   = inds.us_vix.value      if inds.us_vix      else 20.0
    kospi = inds.nikkei_225.roc_pct if inds.nikkei_225  else 0.0
    sp    = inds.sp500.roc_pct      if inds.sp500       else 0.0
    krw   = inds.usd_jpy.value      if inds.usd_jpy     else 1300.0

    mkt = 50 + (10 if sp > 3 else 5 if sp > 0 else -10 if sp < -3 else 0) \
             + (10 if kospi > 3 else 5 if kospi > 0 else -10 if kospi < -3 else 0) \
             + (10 if vix < 15 else -15 if vix > 25 else 0)
    mkt = max(10, min(90, mkt))
    risk = max(10, min(90, 85 - vix * 2))

    # USD/KRW: 원화 강세(낮을수록) = 한국 증시에 긍정
    krw_score = 60 if krw < 1300 else 50 if krw < 1350 else 40

    return MacroAgentOutput(
        market_trend={"label": "Market Direction", "score": int(mkt)},
        risk={"label": "Risk Level",     "score": int(risk)},
        economy={"label": "Economy",     "score": 55},
        rates={"label": "Rates",         "score": krw_score},
        inflation={"label": "Inflation", "score": 55},
        summary=f"KOSPI RoC={kospi:+.1f}%, VIX={vix:.1f}, USD/KRW={krw:.0f} 기반 추정.",
        raw={},
    )


def _rule_sector(reports: list[AnalystReport]) -> SectorAgentOutput:
    avg = sum(r.score for r in reports) / len(reports) if reports else 50
    return SectorAgentOutput(
        conviction_score=int(avg),
        investment_thesis=f"L1 에이전트 평균 점수 {avg:.1f}. LLM 활성화 시 섹터 비교 분석 제공.",
        raw={},
    )


def _rule_pm(tech, quant, qual, news, macro, sector_or_regime=None, regime=None) -> PMAgentOutput:
    # 구버전 호환: sector 없이 regime이 6번째로 오는 경우
    if isinstance(sector_or_regime, str):
        regime = sector_or_regime
        sector_or_regime = None
    if regime is None:
        regime = "Transitional"
    if sector_or_regime is not None:
        sector = sector_or_regime
    else:
        sector = None
    w = REGIME_WEIGHTS.get(regime, REGIME_WEIGHTS["Transitional"])
    # 40% macro + 60% (sector × weights)
    sector_blend = (
        tech * w["technical"]
        + quant.composite_score * w["fundamental"]
        + qual.composite_score  * w["qualitative"]
    )
    # sector가 없으면 sector_blend를 conviction_score로 사용
    if sector is not None:
        conviction = float(sector.conviction_score)
    else:
        conviction = float(sector_blend)
    score = int(0.4 * macro.composite_score + 0.6 * conviction)

    # Hard Sell 제한
    if quant.hard_sell_flag:
        score = min(score, 40)

    sector_conv = sector.conviction_score if sector is not None else int(conviction)
    reason = (
        f"{regime} 환경 (Tech={w['technical']:.0%}/Fund={w['fundamental']:.0%}/Qual={w['qualitative']:.0%}). "
        f"섹터 신뢰도={sector_conv}/100. 최종={score}."
    )
    return PMAgentOutput(final_score=score, reason=reason, raw={})


def _get_sector_avg_stub(sector: str) -> dict:
    """섹터별 업종 평균 (실제 집계 전 임시값)."""
    DEFAULTS = {
        "반도체·IT":     {"roe": 18.0, "per": 20.0, "net_sales_roc": 8.0},
        "자동차":        {"roe": 10.0, "per": 8.0,  "net_sales_roc": 5.0},
        "배터리":        {"roe": 5.0,  "per": 35.0, "net_sales_roc": 15.0},
        "은행·금융":     {"roe": 8.0,  "per": 6.0,  "net_sales_roc": 3.0},
        "화학":          {"roe": 6.0,  "per": 12.0, "net_sales_roc": 2.0},
        "바이오·제약":   {"roe": 4.0,  "per": 50.0, "net_sales_roc": 20.0},
        "인터넷·플랫폼": {"roe": 12.0, "per": 25.0, "net_sales_roc": 10.0},
    }
    return DEFAULTS.get(sector, {"roe": 10.0, "per": 15.0, "net_sales_roc": 5.0})


def _build_roc(m: FinancialMetrics, avg: dict) -> SectorRoCData:
    def ri(name, troc, akey):
        return RoCItem(name=name, target_roc=troc or 0.0,
                       sector_avg_roc=avg.get(akey) or 0.0)
    return SectorRoCData(
        sales=ri("Sales", m.net_sales_roc, "net_sales_roc"),
        op_profit=ri("Operating Profit", m.operating_income_roc, "operating_income_roc"),
        net_income=ri("Net Income", m.net_income_roc, "net_income_roc"),
        total_assets=ri("Total Assets", m.total_assets_roc, "total_assets_roc"),
        equity=ri("Equity", None, None),
        cash=ri("Cash", None, None),
        receivables=ri("Receivables", None, None),
        inventory=ri("Inventory", None, None),
        financial_assets=ri("Financial Assets", None, None),
        interest_bearing_debt=ri("Debt", m.interest_bearing_debt_roc, None),
        current_liabilities=ri("Current Liab", None, None),
        issued_shares=ri("Shares", None, None),
        op_cf=ri("Operating CF", m.operating_cf_roc, "operating_cf_roc"),
        investing_cf=ri("Investing CF", None, None),
        dividends=ri("Dividends", m.dps_roc, None),
        monthly_close=ri("Monthly Close", None, None),
    )


def _empty_metrics(ticker: str, name: str) -> FinancialMetrics:
    return FinancialMetrics(ticker=ticker, company_name=name,
                            period="N/A", fiscal_year_end=str(date.today()))


def _empty_sec_text(ticker: str, name: str) -> SecuritiesReportText:
    return SecuritiesReportText(ticker=ticker, company_name=name,
                                corp_code="", fiscal_year_end=str(date.today()))


def _stub_macro_kr() -> MacroIndicators:
    from hmas_macro_agent import MacroIndicator
    return MacroIndicators(
        us_fed_rate=MacroIndicator("US Fed Rate", 5.25, 0.0, "%"),
        us_10y_yield=MacroIndicator("US 10Y Yield", 4.2, -0.5, "%"),
        jp_policy_rate=MacroIndicator("KR Policy Rate", 3.5, 0.0, "%"),
        jp_10y_yield=MacroIndicator("KR 10Y Yield", 3.2, -0.2, "%"),
        us_cpi=MacroIndicator("US CPI YoY", 3.1, -0.3, ""),
        jp_cpi=MacroIndicator("KR CPI YoY", 2.3, -0.2, ""),
        gold=MacroIndicator("Gold", 2050.0, +1.2, " USD/oz"),
        crude_oil=MacroIndicator("Crude Oil", 72.0, -3.0, " USD/bbl"),
        us_payrolls=MacroIndicator("US Payrolls", 0.216, +5.0, "M"),
        industrial_production=MacroIndicator("Industrial Prod", 103.0, +0.3, ""),
        housing_starts=MacroIndicator("Housing Starts", 1.46, +4.0, "M"),
        unemployment_rate=MacroIndicator("US Unemployment", 3.7, +0.1, "%"),
        jp_business_index=MacroIndicator("KOSPI", 2500.0, +2.0, ""),
        usd_jpy=MacroIndicator("USD/KRW", 1330.0, +0.5, ""),
        nikkei_225=MacroIndicator("KOSPI", 2500.0, +2.0, ""),
        sp500=MacroIndicator("S&P 500", 4800.0, +2.1, ""),
        us_vix=MacroIndicator("US VIX", 15.0, -8.0, ""),
        nikkei_vi=MacroIndicator("VKOSPI", 18.0, -5.0, ""),
    )


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────────────────────
# 결과물 저장
# ──────────────────────────────────────────────────────────────────────────────

def _save_result(result: KospiHMASResult) -> None:
    """단일 종목 결과를 JSON + HTML 리포트로 저장."""
    import os
    from pathlib import Path

    out_dir = Path("hmas_results")
    out_dir.mkdir(exist_ok=True)

    stamp = date.today().strftime("%Y%m%d")
    base  = out_dir / f"{result.ticker}_{stamp}"

    # ── JSON 저장 ──────────────────────────────────────────────────────
    json_path = base.with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
    logger.info(f"JSON 저장: {json_path}")

    # ── HTML 리포트 저장 ───────────────────────────────────────────────
    html_path = base.with_suffix(".html")
    html = _build_html_report(result)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    logger.info(f"HTML 리포트 저장: {html_path}")
    print(f"\n📁 결과 저장 완료:")
    print(f"   JSON  → {json_path}")
    print(f"   HTML  → {html_path}  (브라우저로 열면 됩니다)")


def _save_batch_results(results: list) -> None:
    """배치 결과를 JSON + HTML 리포트로 저장."""
    import os
    from pathlib import Path

    out_dir = Path("hmas_results")
    out_dir.mkdir(exist_ok=True)

    stamp = date.today().strftime("%Y%m%d")
    base  = out_dir / f"batch_{stamp}"

    # JSON
    json_path = base.with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([r.to_dict() for r in results], f, ensure_ascii=False, indent=2)

    # HTML
    html_path = base.with_suffix(".html")
    html = _build_batch_html(results)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\n📁 배치 결과 저장 완료:")
    print(f"   JSON  → {json_path}")
    print(f"   HTML  → {html_path}  (브라우저로 열면 됩니다)")


def _score_color(score: int) -> str:
    if score >= 75: return "#16a34a"   # 진한 초록
    if score >= 60: return "#2563eb"   # 파랑
    if score >= 45: return "#d97706"   # 주황
    return "#dc2626"                   # 빨강


def _score_bg(score: int) -> str:
    if score >= 75: return "#dcfce7"
    if score >= 60: return "#dbeafe"
    if score >= 45: return "#fef9c3"
    return "#fee2e2"


def _build_html_report(r: KospiHMASResult) -> str:
    w = r.regime_weights
    color = _score_color(r.pm_output.final_score)
    bg    = _score_bg(r.pm_output.final_score)

    def bar(score, max_score=100):
        pct = int(score / max_score * 100)
        c = _score_color(int(score))
        return f'''<div style="background:#e5e7eb;border-radius:4px;height:12px;width:100%">
          <div style="background:{c};width:{pct}%;height:12px;border-radius:4px"></div></div>'''

    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>H-MAS | {r.company_name} ({r.ticker})</title>
<style>
  body{{font-family:'Malgun Gothic',sans-serif;background:#f8fafc;margin:0;padding:20px;color:#1e293b}}
  .card{{background:#fff;border-radius:12px;padding:24px;margin-bottom:16px;box-shadow:0 1px 3px rgba(0,0,0,.1)}}
  .header{{background:linear-gradient(135deg,#1e293b,#334155);color:#fff;border-radius:12px;padding:28px;margin-bottom:16px}}
  .final-score{{font-size:64px;font-weight:900;color:{color}}}
  .score-row{{display:flex;align-items:center;gap:12px;margin:8px 0}}
  .score-label{{width:140px;font-size:14px;color:#64748b}}
  .score-val{{width:50px;font-weight:700;color:{color};text-align:right}}
  .score-bar{{flex:1}}
  .tag{{display:inline-block;padding:4px 10px;border-radius:20px;font-size:12px;font-weight:600}}
  .section-title{{font-size:13px;font-weight:700;color:#94a3b8;text-transform:uppercase;letter-spacing:.05em;margin-bottom:12px}}
  .insight-box{{background:#f1f5f9;border-left:4px solid #3b82f6;padding:12px 16px;border-radius:0 8px 8px 0;font-size:14px;line-height:1.6}}
  table{{width:100%;border-collapse:collapse;font-size:14px}}
  th{{background:#f1f5f9;padding:10px 12px;text-align:left;font-weight:600;color:#475569}}
  td{{padding:10px 12px;border-bottom:1px solid #f1f5f9}}
  .grid2{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
  @media(max-width:600px){{.grid2{{grid-template-columns:1fr}}}}
</style>
</head>
<body>
<div style="max-width:860px;margin:0 auto">

<!-- 헤더 -->
<div class="header">
  <div style="font-size:13px;color:#94a3b8;margin-bottom:4px">{r.sector} · 기준일 {r.as_of_date} · Regime: {r.macro_regime}</div>
  <div style="font-size:28px;font-weight:800;margin-bottom:8px">{r.company_name} <span style="color:#94a3b8;font-size:18px">({r.ticker})</span></div>
  <div style="display:flex;align-items:center;gap:20px;flex-wrap:wrap">
    <div>
      <div style="font-size:12px;color:#94a3b8">최종 점수</div>
      <div class="final-score" style="color:{color}">{r.pm_output.final_score}</div>
    </div>
    <div>
      <span class="tag" style="background:{bg};color:{color};font-size:16px">{_label(r.pm_output.final_score)}</span>
      <div style="margin-top:12px;font-size:13px;color:#cbd5e1">
        가중치: 기술적 {w['technical']:.0%} · 펀더멘털 {w['fundamental']:.0%} · 정성 {w['qualitative']:.0%}
      </div>
    </div>
  </div>
</div>

<!-- L1 에이전트 -->
<div class="card">
  <div class="section-title">L1 — Specialist Agents</div>
  <div class="score-row">
    <div class="score-label">📈 Technical</div>
    <div class="score-val">{r.technical_score:.0f}</div>
    <div class="score-bar">{bar(r.technical_score)}</div>
  </div>
  <div class="score-row">
    <div class="score-label">🔢 Quant {'⚠️' if r.quant_output.hard_sell_flag else ''}</div>
    <div class="score-val">{r.quant_output.composite_score:.0f}</div>
    <div class="score-bar">{bar(r.quant_output.composite_score)}</div>
  </div>
  <div style="font-size:12px;color:#94a3b8;margin:-4px 0 8px 152px">
    수익성={r.quant_output.profitability_score} 밸류={r.quant_output.valuation_score} 현금={r.quant_output.cashflow_score} 건전성={r.quant_output.financial_health_score} 성장={r.quant_output.growth_score}
  </div>
  <div class="score-row">
    <div class="score-label">📄 Qual</div>
    <div class="score-val">{r.qual_output.composite_score:.0f}</div>
    <div class="score-bar">{bar(r.qual_output.composite_score)}</div>
  </div>
  <div class="score-row">
    <div class="score-label">📰 News</div>
    <div class="score-val">{r.news_output.net_score:.0f}</div>
    <div class="score-bar">{bar(r.news_output.net_score)}</div>
  </div>
</div>

<!-- L2 에이전트 -->
<div class="grid2">
  <div class="card">
    <div class="section-title">L2 — Sector Agent</div>
    <div style="font-size:40px;font-weight:800;color:{_score_color(r.sector_output.conviction_score)}">{r.sector_output.conviction_score}<span style="font-size:18px;color:#94a3b8">/100</span></div>
    <div class="insight-box" style="margin-top:12px">{r.sector_output.investment_thesis[:200]}...</div>
  </div>
  <div class="card">
    <div class="section-title">L2 — Macro Agent</div>
    <div style="font-size:40px;font-weight:800;color:{_score_color(int(r.macro_output.composite_score))}">{r.macro_output.composite_score:.0f}<span style="font-size:18px;color:#94a3b8">/100</span></div>
    <div style="margin-top:8px"><span class="tag" style="background:#f1f5f9;color:#334155">{r.macro_regime}</span></div>
    <div class="insight-box" style="margin-top:12px">{r.macro_output.summary[:200]}</div>
  </div>
</div>

<!-- 인사이트 -->
<div class="card">
  <div class="section-title">분석 근거</div>
  <table>
    <tr><th>에이전트</th><th>주요 인사이트</th></tr>
    <tr><td>Quant</td><td>{r.quant_output.highlight}</td></tr>
    <tr><td>Qual</td><td>{r.qual_output.insight}</td></tr>
    <tr><td>News</td><td>{r.news_output.reason}</td></tr>
    <tr><td>Macro</td><td>{r.macro_output.summary}</td></tr>
    <tr><td><strong>PM 최종 판단</strong></td><td><strong>{r.pm_output.reason}</strong></td></tr>
  </table>
</div>

<div style="text-align:center;font-size:12px;color:#94a3b8;margin-top:8px">
  H-MAS KOSPI · 생성일 {date.today()} · 투자 참고용 (투자 권유 아님)
</div>
</div>
</body>
</html>"""


def _build_batch_html(results: list) -> str:
    rows = ""
    for i, r in enumerate(results, 1):
        color = _score_color(r.pm_output.final_score)
        bg    = _score_bg(r.pm_output.final_score)
        hs    = "⚠️" if r.quant_output.hard_sell_flag else ""
        rows += f"""<tr>
          <td style="font-weight:700;color:#64748b">{i}</td>
          <td><strong>{r.company_name}</strong><br><span style="color:#94a3b8;font-size:12px">{r.ticker}</span></td>
          <td>{r.sector}</td>
          <td style="font-size:22px;font-weight:900;color:{color}">{r.pm_output.final_score}</td>
          <td>{r.quant_output.composite_score:.0f}</td>
          <td>{r.technical_score:.0f}</td>
          <td><span style="background:#f1f5f9;padding:2px 8px;border-radius:10px;font-size:12px">{r.macro_regime}</span></td>
          <td><span style="background:{bg};color:{color};padding:4px 10px;border-radius:12px;font-size:12px;font-weight:600">{_label(r.pm_output.final_score)}</span>{hs}</td>
        </tr>"""

    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>H-MAS KOSPI 배치 결과</title>
<style>
  body{{font-family:'Malgun Gothic',sans-serif;background:#f8fafc;margin:0;padding:20px;color:#1e293b}}
  .header{{background:linear-gradient(135deg,#1e293b,#334155);color:#fff;border-radius:12px;padding:24px;margin-bottom:20px}}
  table{{width:100%;border-collapse:collapse;background:#fff;border-radius:12px;overflow:hidden;box-shadow:0 1px 3px rgba(0,0,0,.1)}}
  th{{background:#f1f5f9;padding:12px 14px;text-align:left;font-size:13px;font-weight:700;color:#475569}}
  td{{padding:12px 14px;border-bottom:1px solid #f1f5f9;vertical-align:middle}}
  tr:hover td{{background:#f8fafc}}
</style>
</head>
<body>
<div style="max-width:1000px;margin:0 auto">
<div class="header">
  <div style="font-size:22px;font-weight:800">H-MAS KOSPI 배치 분석 결과</div>
  <div style="color:#94a3b8;margin-top:4px">{len(results)}개 종목 · {date.today()} · 최종 점수 내림차순</div>
</div>
<table>
  <thead><tr>
    <th>#</th><th>종목</th><th>섹터</th>
    <th>최종점수</th><th>Quant</th><th>Technical</th>
    <th>Regime</th><th>투자의견</th>
  </tr></thead>
  <tbody>{rows}</tbody>
</table>
<div style="text-align:center;font-size:12px;color:#94a3b8;margin-top:16px">
  H-MAS KOSPI · 투자 참고용 (투자 권유 아님)
</div>
</div>
</body>
</html>"""

def main():
    parser = argparse.ArgumentParser(description="H-MAS KOSPI Pipeline")
    parser.add_argument("--ticker",  default="005930", help="6자리 종목코드")
    parser.add_argument("--tickers", nargs="+",        help="배치 실행용 복수 종목")
    parser.add_argument("--mode",    choices=["single", "batch"], default="single")
    parser.add_argument("--json",    action="store_true")
    args = parser.parse_args()

    # API 키 확인
    openai_key = os.getenv("OPENAI_API_KEY")
    dart_key   = os.getenv("DART_API_KEY")
    fred_key   = os.getenv("FRED_API_KEY")

    if not openai_key:
        logger.error("OPENAI_API_KEY 환경변수가 없습니다.")
        sys.exit(1)
    if not dart_key:
        logger.warning("DART_API_KEY 없음 → 사업보고서 텍스트 수집 불가 (Qual Agent 규칙 기반)")
    if not fred_key:
        logger.warning("FRED_API_KEY 없음 → 한국 기준금리·CPI Yahoo로 대체")

    llm = LLMClient(
        model="gpt-4o",
        api_key=openai_key,
    )

    pipeline = KospiHMASPipeline(
        llm=llm,
        dart_key=dart_key,
        fred_key=fred_key,
        granularity=GranularitySetting.FINE,
    )

    if args.mode == "batch" or args.tickers:
        tickers = args.tickers or list(KOSPI100_MASTER.keys())[:10]
        results = pipeline.run_batch(tickers)

        if args.json:
            print(json.dumps([r.to_dict() for r in results],
                             ensure_ascii=False, indent=2))
        else:
            results.sort(key=lambda r: r.pm_output.final_score, reverse=True)
            print(f"\n{'='*72}")
            print(f"  KOSPI H-MAS BATCH RESULTS  ({len(results)} tickers)  {date.today()}")
            print(f"{'='*72}")
            print(f"{'Rank':<5}{'Ticker':<10}{'Company':<16}{'Final':>6}{'Quant':>7}{'Tech':>7}{'Regime':<14}{'추천'}")
            print(f"{'-'*75}")
            for i, r in enumerate(results, 1):
                hs = "⚠️" if r.quant_output.hard_sell_flag else ""
                print(f"{i:<5}{r.ticker:<10}{r.company_name[:14]:<16}"
                      f"{r.pm_output.final_score:>6}"
                      f"{r.quant_output.composite_score:>7.1f}"
                      f"{r.technical_score:>7.1f}"
                      f"  {r.macro_regime:<12}"
                      f"  {_label(r.pm_output.final_score)} {hs}")
            _save_batch_results(results)
    else:
        result = pipeline.run(ticker=args.ticker)
        if args.json:
            print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        else:
            print(result.summary())
            _save_result(result)


if __name__ == "__main__":
    main()
