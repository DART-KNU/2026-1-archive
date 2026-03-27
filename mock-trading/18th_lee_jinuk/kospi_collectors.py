"""
KOSPI Yahoo Finance + FRED Collector
======================================
KOSPI 종목의 주가·재무·뉴스·거시지표 수집.

환경변수:
  FRED_API_KEY : FRED API 키 (한국 기준금리·CPI 등)
"""

from __future__ import annotations

import os
import logging
from datetime import date, datetime, timedelta
from typing import Optional

import requests

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Yahoo Finance KOSPI 수집기
# ──────────────────────────────────────────────────────────────────────────────

class KospiYFinanceCollector:
    """
    yfinance로 KOSPI 종목 주가·재무·뉴스 수집.
    종목코드 형식: "005930.KS"
    """

    def __init__(self) -> None:
        import yfinance as yf
        self._yf = yf

    def get_financial_metrics(self, stock_code: str, company_name: str,
                               dart_metrics=None):
        """
        Yahoo Finance에서 재무 지표 수집.
        DART 데이터가 있으면 주가 관련 지표(PER·PBR)만 보완.
        """
        from dart_collector import FinancialMetrics

        # DART 데이터가 충분하면 주가 지표만 Yahoo로 보완
        if dart_metrics is not None:
            try:
                from dart_collector import KOSPI100_MASTER
                yf_sym = KOSPI100_MASTER.get(stock_code, {}).get("yf", f"{stock_code}.KS")
                t = self._yf.Ticker(yf_sym)
                d = t.info
                price = d.get("currentPrice") or d.get("regularMarketPrice")
                dart_metrics.per = d.get("trailingPE")
                dart_metrics.pbr = d.get("priceToBook")
                dart_metrics.market_cap = round(d.get("marketCap", 0) / 1_000_000, 0) or dart_metrics.market_cap
            except Exception as e:
                logger.debug(f"Yahoo Finance price supplement failed: {e}")
            return dart_metrics

        # DART 없이 Yahoo만으로 수집
        from dart_collector import KOSPI100_MASTER
        yf_sym = KOSPI100_MASTER.get(stock_code, {}).get("yf", f"{stock_code}.KS")

        try:
            t = self._yf.Ticker(yf_sym)
            d = t.info
            fin = t.financials
            bal = t.balance_sheet
            cf  = t.cashflow

            def _get(df, key, col=0):
                if df is None or df.empty:
                    return None
                matches = [i for i in df.index if key.lower() in str(i).lower()]
                if not matches:
                    return None
                v = df.loc[matches[0]].iloc[col]
                return float(v) if v is not None and str(v) != "nan" else None

            def _roc(c, p):
                if c is None or p is None or p == 0:
                    return None
                return round((c - p) / abs(p) * 100, 1)

            def to_m(v):
                return round(v / 1_000_000, 0) if v else None

            ns = _get(fin, "Total Revenue", 0)
            ns_p = _get(fin, "Total Revenue", 1)
            oi = _get(fin, "Operating Income", 0)
            oi_p = _get(fin, "Operating Income", 1)
            ni = _get(fin, "Net Income", 0)
            ni_p = _get(fin, "Net Income", 1)
            ta = _get(bal, "Total Assets", 0)
            ta_p = _get(bal, "Total Assets", 1)
            eq = _get(bal, "Stockholders Equity", 0) or _get(bal, "Common Stock Equity", 0)
            ld = _get(bal, "Long Term Debt", 0)
            ocf = _get(cf, "Operating Cash Flow", 0)
            ocf_p = _get(cf, "Operating Cash Flow", 1)
            icf = _get(cf, "Investing Cash Flow", 0)

            roe = round(ni / eq * 100, 2) if ni and eq else None
            roa = round(ni / ta * 100, 2) if ni and ta else None
            eq_ratio = round(eq / ta * 100, 1) if eq and ta else None
            de = round(ld / eq, 2) if ld and eq else None
            fcf = round(to_m(ocf or 0) + to_m(icf or 0), 0) if ocf or icf else None

            return FinancialMetrics(
                ticker=stock_code, company_name=company_name,
                period=f"FY{date.today().year - 1}",
                fiscal_year_end=f"{date.today().year - 1}-12-31",
                net_sales=to_m(ns), net_sales_roc=_roc(ns, ns_p),
                operating_income=to_m(oi), operating_income_roc=_roc(oi, oi_p),
                net_income=to_m(ni), net_income_roc=_roc(ni, ni_p),
                roe=roe, roa=roa,
                eps=d.get("trailingEps"), bps=None,
                per=d.get("trailingPE"), pbr=d.get("priceToBook"),
                total_assets=to_m(ta), total_assets_roc=_roc(ta, ta_p),
                equity=to_m(eq), equity_ratio=eq_ratio, de_ratio=de,
                interest_bearing_debt=to_m(ld),
                operating_cf=to_m(ocf), operating_cf_roc=_roc(ocf, ocf_p),
                investing_cf=to_m(icf), free_cf=fcf,
                market_cap=round(d.get("marketCap", 0) / 1_000_000, 0) or None,
            )
        except Exception as e:
            logger.warning(f"[{stock_code}] Yahoo Finance 재무 수집 실패: {e}")
            return None

    def get_price_series(self, stock_code: str, days_back: int = 120,
                          end_date: Optional[date] = None) -> list[float]:
        from dart_collector import KOSPI100_MASTER
        yf_sym = KOSPI100_MASTER.get(stock_code, {}).get("yf", f"{stock_code}.KS")
        end = end_date or date.today()
        start = end - timedelta(days=days_back + 10)
        try:
            df = self._yf.download(yf_sym,
                                    start=start.strftime("%Y-%m-%d"),
                                    end=end.strftime("%Y-%m-%d"),
                                    progress=False, auto_adjust=True)
            if df.empty:
                return []
            close = df["Close"]; close = close.iloc[:, 0] if hasattr(close, "columns") else close; prices = close.dropna().tolist()
            logger.info(f"[{stock_code}] 주가 {len(prices)}일 수집")
            return prices
        except Exception as e:
            logger.warning(f"[{stock_code}] 주가 수집 실패: {e}")
            return []

    def get_news(self, stock_code: str, max_items: int = 20) -> list:
        """Yahoo Finance 뉴스 수집 → NewsItem 리스트."""
        from hmas_news_agent import NewsItem
        from dart_collector import KOSPI100_MASTER
        yf_sym = KOSPI100_MASTER.get(stock_code, {}).get("yf", f"{stock_code}.KS")
        results = []
        try:
            t = self._yf.Ticker(yf_sym)
            for item in (t.news or [])[:max_items]:
                ts = item.get("providerPublishTime", 0)
                try:
                    pub = datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
                except Exception:
                    pub = str(date.today())
                results.append(NewsItem(
                    date=pub,
                    headline=item.get("title", ""),
                    summary=item.get("summary", ""),
                ))
            logger.info(f"[{stock_code}] 뉴스 {len(results)}건 수집")
        except Exception as e:
            logger.warning(f"[{stock_code}] 뉴스 수집 실패: {e}")
        return results


# ──────────────────────────────────────────────────────────────────────────────
# 한국 거시경제 지표 수집기
# ──────────────────────────────────────────────────────────────────────────────

class KoreaMarcoCollector:
    """
    FRED + Yahoo Finance로 한국 관련 거시경제 지표 수집.

    FRED 수집 지표:
      한국 기준금리, 한국 CPI, 미국 지표 (Fed Rate, 10Y, CPI, 고용 등)

    Yahoo Finance 수집 지표:
      KOSPI, S&P500, VIX, USD/KRW, 금, 원유
    """

    def __init__(self, fred_api_key: Optional[str] = None) -> None:
        self.fred_key = fred_api_key or os.getenv("FRED_API_KEY", "")

    def _fred(self, series_id: str, periods: int = 13) -> tuple[float, float]:
        """FRED에서 시계열 수집 → (최근값, RoC%)."""
        try:
            from fredapi import Fred
            fred = Fred(api_key=self.fred_key)
            s = fred.get_series(series_id).dropna()
            if len(s) < 2:
                return 0.0, 0.0
            latest = float(s.iloc[-1])
            prior  = float(s.iloc[-min(periods, len(s))])
            roc = (latest - prior) / abs(prior) * 100 if prior != 0 else 0.0
            return round(latest, 4), round(roc, 2)
        except Exception as e:
            logger.debug(f"FRED {series_id}: {e}")
            return 0.0, 0.0

    def _yahoo(self, symbol: str, days: int = 30) -> tuple[float, float]:
        """Yahoo Finance에서 가격 + RoC 수집."""
        try:
            import yfinance as yf
            end = date.today()
            start = end - timedelta(days=days + 10)
            df = yf.download(symbol,
                             start=start.strftime("%Y-%m-%d"),
                             end=end.strftime("%Y-%m-%d"),
                             progress=False, auto_adjust=True)
            if df.empty or len(df) < 2:
                return 0.0, 0.0
            latest = float(df["Close"].iloc[-1])
            prior  = float(df["Close"].iloc[0])
            roc = (latest - prior) / abs(prior) * 100 if prior != 0 else 0.0
            return round(latest, 4), round(roc, 2)
        except Exception as e:
            logger.debug(f"Yahoo {symbol}: {e}")
            return 0.0, 0.0

    def fetch(self):
        """전체 한국 관련 거시지표 수집 → MacroIndicators 반환."""
        from hmas_macro_agent import MacroIndicators, MacroIndicator

        logger.info("한국 거시지표 수집 중 (FRED + Yahoo Finance)...")

        # ── FRED ─────────────────────────────────────────────
        fed_rate, fed_roc     = self._fred("FEDFUNDS", 3)
        us_10y, us_10y_roc    = self._fred("DGS10", 3)
        us_cpi, us_cpi_roc    = self._fred("CPIAUCSL", 12)
        payrolls, pay_roc     = self._fred("PAYEMS", 12)
        unemp, unemp_roc      = self._fred("UNRATE", 12)
        ind_prod, ip_roc      = self._fred("INDPRO", 12)
        housing, house_roc    = self._fred("HOUST", 12)

        # 한국 기준금리 (FRED: IRSTCI01KRM156N)
        kr_rate, kr_rate_roc  = self._fred("IRSTCI01KRM156N", 3)
        # 한국 10년물 금리 (FRED: IRLTLT01KRM156N)
        kr_10y, kr_10y_roc    = self._fred("IRLTLT01KRM156N", 3)
        # 한국 CPI (FRED: KORCPIALLMINMEI)
        kr_cpi, kr_cpi_roc    = self._fred("KORCPIALLMINMEI", 12)

        # ── Yahoo Finance ─────────────────────────────────────
        kospi, kospi_roc      = self._yahoo("^KS11", 30)    # KOSPI
        sp500, sp500_roc      = self._yahoo("^GSPC", 30)    # S&P500
        vix, vix_roc          = self._yahoo("^VIX",  30)    # VIX
        usdkrw, krw_roc       = self._yahoo("KRW=X", 30)    # USD/KRW
        gold, gold_roc        = self._yahoo("GC=F",  30)    # 금
        crude, crude_roc      = self._yahoo("CL=F",  30)    # WTI

        # 한국 공포지수 (VKOSPI) — Yahoo에 없는 경우 KRX API 필요, 임시 0
        vkospi = 0.0
        vkospi_roc = 0.0

        return MacroIndicators(
            # 금리·정책 (한국 기준으로 재매핑)
            us_fed_rate=MacroIndicator("US Fed Rate",         fed_rate,  fed_roc,   "%"),
            us_10y_yield=MacroIndicator("US 10Y Yield",       us_10y,    us_10y_roc, "%"),
            jp_policy_rate=MacroIndicator("KR Policy Rate",   kr_rate,   kr_rate_roc, "%"),
            jp_10y_yield=MacroIndicator("KR 10Y Yield",       kr_10y,    kr_10y_roc, "%"),
            # 인플레이션·원자재
            us_cpi=MacroIndicator("US CPI YoY",               us_cpi,    us_cpi_roc, ""),
            jp_cpi=MacroIndicator("KR CPI YoY",               kr_cpi,    kr_cpi_roc, ""),
            gold=MacroIndicator("Gold",                        gold,      gold_roc,  " USD/oz"),
            crude_oil=MacroIndicator("Crude Oil (WTI)",        crude,     crude_roc, " USD/bbl"),
            # 성장·경제 (미국 지표 활용)
            us_payrolls=MacroIndicator("US Non-Farm Payrolls", payrolls / 1000, pay_roc, "M"),
            industrial_production=MacroIndicator("US Industrial Prod", ind_prod, ip_roc, ""),
            housing_starts=MacroIndicator("US Housing Starts", housing / 1000, house_roc, "M"),
            unemployment_rate=MacroIndicator("US Unemployment", unemp, unemp_roc, "%"),
            jp_business_index=MacroIndicator("KOSPI Index",    kospi, kospi_roc, ""),
            # 시장·리스크
            usd_jpy=MacroIndicator("USD/KRW",                  usdkrw, krw_roc, ""),
            nikkei_225=MacroIndicator("KOSPI",                 kospi,  kospi_roc, ""),
            sp500=MacroIndicator("S&P 500",                    sp500,  sp500_roc, ""),
            us_vix=MacroIndicator("US VIX",                    vix,    vix_roc,   ""),
            nikkei_vi=MacroIndicator("VKOSPI",                 vkospi, vkospi_roc, ""),
        )