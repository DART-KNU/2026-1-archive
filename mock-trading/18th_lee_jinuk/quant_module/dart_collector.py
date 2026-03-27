"""
DART Collector — 금융감독원 전자공시 API
=========================================
KOSPI 종목의 사업보고서 텍스트 + 재무제표 수치 수집.

환경변수:
  DART_API_KEY : OpenDART API 키 (https://opendart.fss.or.kr)

수집 데이터:
  1. 재무제표 (FinancialMetrics) — Quant Agent용
     DART 재무정보 API → 매출·영업이익·순이익·총자산·자기자본·부채 등
     전년 대비 RoC 자동 계산

  2. 사업보고서 텍스트 (SecuritiesReportText) — Qual Agent용
     사업의 내용·위험요소·MD&A·이사회 구성 등 텍스트 섹션
"""

from __future__ import annotations

import os
import time
import logging
import re
from dataclasses import dataclass
from typing import Optional
from datetime import date, timedelta

import requests

logger = logging.getLogger(__name__)

DART_BASE = "https://opendart.fss.or.kr/api"
DEFAULT_TIMEOUT = 20


# ──────────────────────────────────────────────────────────────────────────────
# 재무 지표 데이터 구조 (KOSPI 버전)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class FinancialMetrics:
    """Quant Agent 입력용 재무 지표 (전년 대비 RoC 포함)."""
    ticker: str
    company_name: str
    period: str               # e.g. "FY2023"
    fiscal_year_end: str      # e.g. "2023-12-31"

    # 수익성
    net_sales: Optional[float] = None          # 백만원
    net_sales_roc: Optional[float] = None      # %
    operating_income: Optional[float] = None
    operating_income_roc: Optional[float] = None
    net_income: Optional[float] = None
    net_income_roc: Optional[float] = None
    roe: Optional[float] = None
    roa: Optional[float] = None

    # 밸류에이션
    eps: Optional[float] = None
    eps_roc: Optional[float] = None
    bps: Optional[float] = None
    per: Optional[float] = None
    pbr: Optional[float] = None

    # 재무 건전성
    total_assets: Optional[float] = None
    total_assets_roc: Optional[float] = None
    equity: Optional[float] = None
    equity_ratio: Optional[float] = None
    de_ratio: Optional[float] = None
    interest_bearing_debt: Optional[float] = None
    interest_bearing_debt_roc: Optional[float] = None

    # 현금흐름
    operating_cf: Optional[float] = None
    operating_cf_roc: Optional[float] = None
    investing_cf: Optional[float] = None
    free_cf: Optional[float] = None

    # 성장
    dividends_per_share: Optional[float] = None
    dps_roc: Optional[float] = None
    market_cap: Optional[float] = None

    def to_prompt_block(self) -> str:
        def fmt(val, unit="", roc=None):
            if val is None:
                return "N/A"
            s = f"{val:,.1f}{unit}"
            if roc is not None:
                s += f" (RoC: {roc:+.1f}%)"
            return s

        return f"""
[Financial Metrics: {self.company_name} ({self.ticker}) | {self.period}]

Profitability:
  Net Sales (백만원)       : {fmt(self.net_sales, "", self.net_sales_roc)}
  Operating Income (백만원) : {fmt(self.operating_income, "", self.operating_income_roc)}
  Net Income (백만원)      : {fmt(self.net_income, "", self.net_income_roc)}
  ROE                      : {fmt(self.roe, "%")}
  ROA                      : {fmt(self.roa, "%")}

Valuation:
  EPS (원)                 : {fmt(self.eps, "", self.eps_roc)}
  BPS (원)                 : {fmt(self.bps, "원")}
  PER                      : {fmt(self.per, "x")}
  PBR                      : {fmt(self.pbr, "x")}

Financial Health:
  Total Assets (백만원)    : {fmt(self.total_assets, "", self.total_assets_roc)}
  Equity (백만원)          : {fmt(self.equity)}
  Equity Ratio             : {fmt(self.equity_ratio, "%")}
  D/E Ratio                : {fmt(self.de_ratio, "x")}
  Int. Bearing Debt (백만원): {fmt(self.interest_bearing_debt, "", self.interest_bearing_debt_roc)}

Cash Flow:
  Operating CF (백만원)    : {fmt(self.operating_cf, "", self.operating_cf_roc)}
  Investing CF (백만원)    : {fmt(self.investing_cf)}
  Free CF (백만원)         : {fmt(self.free_cf)}

Growth:
  DPS (원)                 : {fmt(self.dividends_per_share, "", self.dps_roc)}
  Market Cap (백만원)      : {fmt(self.market_cap)}
""".strip()


@dataclass
class SecuritiesReportText:
    """Qual Agent 입력용 사업보고서 텍스트."""
    ticker: str
    company_name: str
    corp_code: str
    fiscal_year_end: str
    info_updated: bool = False

    business_overview: str = ""   # 사업의 내용
    business_risks: str = ""      # 위험 요소
    mda: str = ""                 # MD&A (경영진의 논의 및 분석)
    governance: str = ""          # 이사회 및 지배구조


# ──────────────────────────────────────────────────────────────────────────────
# DART API 클라이언트
# ──────────────────────────────────────────────────────────────────────────────

class DartClient:
    """
    DART(전자공시) API v1 클라이언트.

    주요 메서드:
      get_corp_code()          → 종목코드 → DART 고유번호
      get_financial_metrics()  → 재무제표 수치 수집
      get_securities_report_text() → 사업보고서 텍스트 수집
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key or os.getenv("DART_API_KEY", "")
        if not self.api_key:
            raise ValueError("DART_API_KEY 환경변수가 설정되지 않았습니다.")
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "HMAS-Research/1.0"})

    def _get(self, endpoint: str, params: dict) -> dict:
        params["crtfc_key"] = self.api_key
        url = f"{DART_BASE}/{endpoint}"
        resp = self.session.get(url, params=params, timeout=DEFAULT_TIMEOUT)
        resp.raise_for_status()
        time.sleep(0.3)
        return resp.json()

    # ── 고유번호 조회 ─────────────────────────────────────────────────────────
    # DART corp_code 캐시 (ZIP 다운로드 후 메모리 저장)
    _corp_code_map: dict[str, str] = {}   # {stock_code: corp_code}

    def _load_corp_code_map(self) -> None:
        """
        DART 전체 기업 고유번호 목록 ZIP 다운로드 → stock_code 매핑 테이블 구축.
        DART 공식 방법: /api/corpCode.xml (ZIP)
        """
        if self._corp_code_map:
            return  # 이미 로드됨

        import zipfile
        import io
        import xml.etree.ElementTree as ET

        url = f"{DART_BASE}/corpCode.xml"
        try:
            resp = self.session.get(url, params={"crtfc_key": self.api_key}, timeout=30)
            resp.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                xml_filename = [n for n in zf.namelist() if n.endswith(".xml")][0]
                with zf.open(xml_filename) as f:
                    tree = ET.parse(f)
                    root = tree.getroot()
                    for item in root.findall("list"):
                        sc = item.findtext("stock_code", "").strip()
                        cc = item.findtext("corp_code", "").strip()
                        if sc and cc:
                            DartClient._corp_code_map[sc] = cc
            logger.info(f"DART corp_code 목록 로드 완료: {len(self._corp_code_map)}개")
        except Exception as e:
            logger.warning(f"DART corp_code ZIP 로드 실패: {e}")

    def get_corp_code(self, stock_code: str) -> Optional[str]:
        """종목코드(6자리) → DART 고유번호."""
        self._load_corp_code_map()
        return self._corp_code_map.get(stock_code)

    def get_company_info(self, stock_code: str) -> dict:
        """기업 기본 정보 조회."""
        corp_code = self.get_corp_code(stock_code)
        if not corp_code:
            return {}
        try:
            data = self._get("company.json", {"corp_code": corp_code})
            if data.get("status") == "000":
                return data
        except Exception as e:
            logger.warning(f"DART company info failed for {stock_code}: {e}")
        return {}

    # ── 재무제표 수집 ─────────────────────────────────────────────────────────

    def _get_financial_statement(
        self,
        corp_code: str,
        year: str,
        report_code: str = "11011",   # 11011=사업보고서, 11012=반기, 11013=분기
        fs_div: str = "CFS",          # CFS=연결, OFS=별도
    ) -> list[dict]:
        """DART 단일회사 주요계정 조회."""
        try:
            data = self._get("fnlttSinglAcntAll.json", {
                "corp_code": corp_code,
                "bsns_year": year,
                "reprt_code": report_code,
                "fs_div": fs_div,
            })
            if data.get("status") == "000":
                return data.get("list", [])
        except Exception as e:
            logger.warning(f"DART financial statement failed: {e}")
        return []

    @staticmethod
    def _find_account(items: list[dict], *keywords) -> Optional[float]:
        """계정명 키워드로 항목 탐색 후 금액 반환 (당기 기준)."""
        for kw in keywords:
            for item in items:
                nm = item.get("account_nm", "")
                if kw in nm:
                    val_str = item.get("thstrm_amount", "").replace(",", "").strip()
                    try:
                        return float(val_str) / 1_000_000  # 원 → 백만원
                    except (ValueError, TypeError):
                        pass
        return None

    @staticmethod
    def _find_account_prior(items: list[dict], *keywords) -> Optional[float]:
        """전기(전년) 금액 반환."""
        for kw in keywords:
            for item in items:
                nm = item.get("account_nm", "")
                if kw in nm:
                    val_str = item.get("frmtrm_amount", "").replace(",", "").strip()
                    try:
                        return float(val_str) / 1_000_000
                    except (ValueError, TypeError):
                        pass
        return None

    @staticmethod
    def _roc(cur: Optional[float], prior: Optional[float]) -> Optional[float]:
        if cur is None or prior is None or prior == 0:
            return None
        return round((cur - prior) / abs(prior) * 100, 1)

    def get_financial_metrics(
        self,
        stock_code: str,
        company_name: str,
        market_price: Optional[float] = None,
        shares_outstanding: Optional[float] = None,
        bsns_year: Optional[int] = None,
    ) -> Optional[FinancialMetrics]:
        """
        DART에서 연간 재무지표 수집.

        Parameters
        ----------
        stock_code        : 6자리 종목코드 (e.g. "005930")
        company_name      : 기업명
        market_price      : 현재 주가 (PER/PBR 계산용)
        shares_outstanding: 발행주식수 (백만주)
        bsns_year         : 수집할 회계연도 (None이면 직전연도 자동 결정)
        """
        corp_code = self.get_corp_code(stock_code)
        if not corp_code:
            logger.warning(f"corp_code not found for {stock_code}")
            return None

        if bsns_year:
            year = str(bsns_year)
            items = self._get_financial_statement(corp_code, year)
            if not items:
                # 1년 전도 시도
                year = str(bsns_year - 1)
                items = self._get_financial_statement(corp_code, year)
        else:
            year = str(date.today().year - 1)  # 직전 연도 사업보고서
            items = self._get_financial_statement(corp_code, year)
            if not items:
                # 전전년도 시도
                year = str(date.today().year - 2)
                items = self._get_financial_statement(corp_code, year)

        if not items:
            logger.warning(f"No financial data from DART for {stock_code}")
            return None

        # 주요 계정 추출
        net_sales     = self._find_account(items, "매출액", "수익(매출액)", "영업수익")
        net_sales_p   = self._find_account_prior(items, "매출액", "수익(매출액)", "영업수익")
        op_income     = self._find_account(items, "영업이익")
        op_income_p   = self._find_account_prior(items, "영업이익")
        net_income    = self._find_account(items, "당기순이익", "분기순이익")
        net_income_p  = self._find_account_prior(items, "당기순이익", "분기순이익")
        total_assets  = self._find_account(items, "자산총계")
        total_assets_p= self._find_account_prior(items, "자산총계")
        equity        = self._find_account(items, "자본총계", "지배기업소유주지분")
        equity_p      = self._find_account_prior(items, "자본총계", "지배기업소유주지분")
        total_liab    = self._find_account(items, "부채총계")
        op_cf         = self._find_account(items, "영업활동현금흐름", "영업활동으로인한현금흐름")
        op_cf_p       = self._find_account_prior(items, "영업활동현금흐름", "영업활동으로인한현금흐름")
        inv_cf        = self._find_account(items, "투자활동현금흐름", "투자활동으로인한현금흐름")

        # 파생 지표
        roe = round(net_income / equity * 100, 2) if net_income and equity else None
        roa = round(net_income / total_assets * 100, 2) if net_income and total_assets else None
        equity_ratio = round(equity / total_assets * 100, 1) if equity and total_assets else None

        # 부채비율: 총부채/자기자본
        de_ratio = None
        if total_liab and equity and equity > 0:
            de_ratio = round(total_liab / equity, 2)

        free_cf = None
        if op_cf is not None and inv_cf is not None:
            free_cf = round(op_cf + inv_cf, 0)

        # PER / PBR (주가 필요)
        per = pbr = None
        eps_val = None
        bps_val = None
        if shares_outstanding and shares_outstanding > 0:
            if net_income:
                eps_val = round(net_income * 1_000_000 / shares_outstanding, 0)
            if equity:
                bps_val = round(equity * 1_000_000 / shares_outstanding, 0)
        if market_price and eps_val and eps_val > 0:
            per = round(market_price / eps_val, 1)
        if market_price and bps_val and bps_val > 0:
            pbr = round(market_price / bps_val, 2)

        market_cap = None
        if market_price and shares_outstanding:
            market_cap = round(market_price * shares_outstanding / 1_000_000, 0)

        return FinancialMetrics(
            ticker=stock_code,
            company_name=company_name,
            period=f"FY{year}",
            fiscal_year_end=f"{year}-12-31",
            net_sales=net_sales,
            net_sales_roc=self._roc(net_sales, net_sales_p),
            operating_income=op_income,
            operating_income_roc=self._roc(op_income, op_income_p),
            net_income=net_income,
            net_income_roc=self._roc(net_income, net_income_p),
            roe=roe,
            roa=roa,
            eps=eps_val,
            bps=bps_val,
            total_assets=total_assets,
            total_assets_roc=self._roc(total_assets, total_assets_p),
            equity=equity,
            equity_ratio=equity_ratio,
            de_ratio=de_ratio,
            interest_bearing_debt=total_liab,
            interest_bearing_debt_roc=None,
            operating_cf=op_cf,
            operating_cf_roc=self._roc(op_cf, op_cf_p),
            investing_cf=inv_cf,
            free_cf=free_cf,
            per=per,
            pbr=pbr,
            market_cap=market_cap,
        )

    # ── 사업보고서 텍스트 수집 ────────────────────────────────────────────────

    def _get_report_list(self, corp_code: str, bsns_year: Optional[int] = None) -> list[dict]:
        """사업보고서 목록 조회. bsns_year 지정 시 해당 연도 전후 2년 범위 조회."""
        try:
            if bsns_year:
                # 해당 회계연도 사업보고서는 이듬해 3월 제출 → 전후 범위로 검색
                bgn_de = f"{bsns_year}0101"
                end_de = f"{bsns_year + 1}1231"
            else:
                bgn_de = (date.today() - timedelta(days=400)).strftime("%Y%m%d")
                end_de = date.today().strftime("%Y%m%d")
            data = self._get("list.json", {
                "corp_code": corp_code,
                "bgn_de": bgn_de,
                "end_de": end_de,
                "pblntf_ty": "A",    # A=정기공시 (사업보고서)
                "page_count": "10",
            })
            if data.get("status") == "000":
                return data.get("list", [])
        except Exception as e:
            logger.warning(f"DART report list failed: {e}")
        return []

    def _get_document_text(self, rcept_no: str, section_keyword: str,
                            max_chars: int = 2000) -> str:
        """
        DART 공시 문서에서 특정 섹션 텍스트 추출.
        document.json → dcm_no 목록 조회 후 각 문서 HTML에서 키워드 검색.
        """
        try:
            # 1단계: 문서 번호 목록 조회
            data = self._get("document.json", {"rcept_no": rcept_no})
            docs = data.get("list", [])

            for doc in docs:
                dcm_no = doc.get("dcm_no", "")
                if not dcm_no:
                    continue
                # 2단계: dcm_no로 실제 문서 HTML 조회
                try:
                    url = "https://dart.fss.or.kr/report/viewer.do"
                    params = {"rcpNo": rcept_no, "dcmNo": dcm_no, "eleId": "0",
                              "offset": "0", "length": "0", "dtd": "dart3.xsd"}
                    resp = self.session.get(url, params=params, timeout=20)
                    resp.encoding = "utf-8"
                    # HTML → 텍스트 변환
                    text = re.sub(r"<script[^>]*>.*?</script>", " ", resp.text, flags=re.DOTALL)
                    text = re.sub(r"<style[^>]*>.*?</style>",  " ", text, flags=re.DOTALL)
                    text = re.sub(r"<[^>]+>", " ", text)
                    text = re.sub(r"\s+", " ", text).strip()

                    idx = text.find(section_keyword)
                    if idx != -1:
                        return text[idx: idx + max_chars].strip()
                except Exception:
                    continue
        except Exception as e:
            logger.debug(f"Document text extraction failed: {e}")
        return ""

    def get_securities_report_text(
        self,
        stock_code: str,
        company_name: str,
        previous_rcept_no: Optional[str] = None,
        bsns_year: Optional[int] = None,
    ) -> Optional[SecuritiesReportText]:
        """
        사업보고서 텍스트 섹션 추출.
        bsns_year 지정 시 해당 연도 보고서, None이면 최신 보고서.

        수집 섹션:
          - 사업의 내용 (business_overview)
          - 위험 요소 (business_risks)
          - 경영진의 논의 및 분석 (mda)
          - 이사회 및 지배구조 (governance)
        """
        corp_code = self.get_corp_code(stock_code)
        if not corp_code:
            return None

        reports = self._get_report_list(corp_code, bsns_year=bsns_year)

        # 사업보고서만 필터
        annual_reports = [r for r in reports
                          if "사업보고서" in r.get("report_nm", "")
                          and "분기" not in r.get("report_nm", "")
                          and "반기" not in r.get("report_nm", "")]

        if not annual_reports:
            return None

        latest = annual_reports[0]
        rcept_no = latest.get("rcept_no", "")
        rcept_dt = latest.get("rcept_dt", "")
        info_updated = (previous_rcept_no is not None and rcept_no != previous_rcept_no)

        # 각 섹션 텍스트 추출
        SECTIONS = {
            "business_overview": ["사업의 내용", "사업의 개요", "주요 사업"],
            "business_risks":    ["위험 요소", "위험요인", "사업등의 위험요소"],
            "mda":               ["경영진의 논의", "재무상태", "영업 및 재무현황"],
            "governance":        ["이사회 구성", "지배구조", "임원 현황"],
        }

        extracted = {}
        for field_name, keywords in SECTIONS.items():
            text = ""
            for kw in keywords:
                text = self._get_document_text(rcept_no, kw, max_chars=2000)
                if text:
                    break
            extracted[field_name] = text

        return SecuritiesReportText(
            ticker=stock_code,
            company_name=company_name,
            corp_code=corp_code,
            fiscal_year_end=f"{rcept_dt[:4]}-12-31" if rcept_dt else "",
            info_updated=info_updated,
            **extracted,
        )


# ──────────────────────────────────────────────────────────────────────────────
# KOSPI 100 종목 마스터
# ──────────────────────────────────────────────────────────────────────────────

KOSPI100_MASTER: dict[str, dict] = {
    # IT·반도체
    "005930": {"name": "삼성전자",       "yf": "005930.KS", "sector": "반도체·IT"},
    "000660": {"name": "SK하이닉스",     "yf": "000660.KS", "sector": "반도체·IT"},
    "035420": {"name": "NAVER",         "yf": "035420.KS", "sector": "인터넷·플랫폼"},
    "035720": {"name": "카카오",         "yf": "035720.KS", "sector": "인터넷·플랫폼"},
    "066570": {"name": "LG전자",         "yf": "066570.KS", "sector": "전기전자"},
    "009150": {"name": "삼성전기",       "yf": "009150.KS", "sector": "전기전자"},
    # 자동차
    "005380": {"name": "현대자동차",     "yf": "005380.KS", "sector": "자동차"},
    "000270": {"name": "기아",           "yf": "000270.KS", "sector": "자동차"},
    "012330": {"name": "현대모비스",     "yf": "012330.KS", "sector": "자동차부품"},
    # 배터리·에너지
    "373220": {"name": "LG에너지솔루션", "yf": "373220.KS", "sector": "배터리"},
    "006400": {"name": "삼성SDI",        "yf": "006400.KS", "sector": "배터리"},
    "247540": {"name": "에코프로비엠",   "yf": "247540.KQ", "sector": "배터리소재"},
    # 금융
    "105560": {"name": "KB금융",         "yf": "105560.KS", "sector": "은행·금융"},
    "055550": {"name": "신한지주",       "yf": "055550.KS", "sector": "은행·금융"},
    "086790": {"name": "하나금융지주",   "yf": "086790.KS", "sector": "은행·금융"},
    "316140": {"name": "우리금융지주",   "yf": "316140.KS", "sector": "은행·금융"},
    "032830": {"name": "삼성생명",       "yf": "032830.KS", "sector": "보험"},
    # 화학·소재
    "051910": {"name": "LG화학",         "yf": "051910.KS", "sector": "화학"},
    "096770": {"name": "SK이노베이션",   "yf": "096770.KS", "sector": "화학·에너지"},
    "010950": {"name": "S-Oil",          "yf": "010950.KS", "sector": "정유"},
    "011170": {"name": "롯데케미칼",     "yf": "011170.KS", "sector": "화학"},
    # 바이오·제약
    "207940": {"name": "삼성바이오로직스","yf": "207940.KS", "sector": "바이오·제약"},
    "068270": {"name": "셀트리온",       "yf": "068270.KS", "sector": "바이오·제약"},
    "128940": {"name": "한미약품",       "yf": "128940.KS", "sector": "바이오·제약"},
    # 소비·유통
    "018260": {"name": "삼성물산",       "yf": "018260.KS", "sector": "건설·유통"},
    "028260": {"name": "삼성물산(우)",   "yf": "028260.KS", "sector": "건설·유통"},
    "139480": {"name": "이마트",         "yf": "139480.KS", "sector": "유통"},
    # 통신
    "017670": {"name": "SK텔레콤",       "yf": "017670.KS", "sector": "통신"},
    "030200": {"name": "KT",             "yf": "030200.KS", "sector": "통신"},
    # 철강·중공업
    "005490": {"name": "POSCO홀딩스",    "yf": "005490.KS", "sector": "철강"},
    "042660": {"name": "한화오션",       "yf": "042660.KS", "sector": "조선"},
    "009540": {"name": "HD한국조선해양", "yf": "009540.KS", "sector": "조선"},
}

# 섹터별 peer 종목
KOSPI_SECTOR_PEERS: dict[str, list[str]] = {
    "반도체·IT":     ["005930", "000660"],
    "인터넷·플랫폼": ["035420", "035720"],
    "자동차":        ["005380", "000270", "012330"],
    "배터리":        ["373220", "006400"],
    "은행·금융":     ["105560", "055550", "086790", "316140"],
    "화학":          ["051910", "096770", "011170"],
    "바이오·제약":   ["207940", "068270", "128940"],
    "조선":          ["042660", "009540"],
}
