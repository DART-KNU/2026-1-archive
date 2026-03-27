"""
H-MAS KOSPI 에이전트 백테스트  (hmas_agent_backtest.py)
=========================================================

H-MAS 에이전트(Macro/News/Qual/Quant/Technical)가 날짜마다
실제 그 시점 데이터를 받아 점수를 내고, 그 점수로 포트폴리오를 운용합니다.

전략 2가지 (--strategy 옵션):
  threshold (기본, 방식B)
    - 점수 >= buy_threshold  (기본 70) : 매수
    - 점수 <= sell_threshold (기본 50) : 매도
    - 그 사이                          : 보유 유지
    - 보유 종목 수 제한 없음 (점수 통과 종목 전부)
    - 균등 비중 배분

  topk (방식A)
    - 매 리밸런싱마다 점수 상위 K종목 유지
    - 나머지 전량 매도

실행:
  # 방식 B (임계값) — API 없이 구조 검증
  python hmas_agent_backtest.py --strategy threshold --mode stub \\
      --tickers 005930 000660 005380 035420 373220 105560 051910 207940

  # 방식 B — 실제 에이전트 신호
  python hmas_agent_backtest.py --strategy threshold --mode live \\
      --buy 70 --sell 50 \\
      --tickers 005930 000660 005380 035420 \\
      --start 2022-01-01 --end 2024-12-31

  # 방식 A (TopK)
  python hmas_agent_backtest.py --strategy topk --topk 3 --mode stub

환경변수:
  OPENAI_API_KEY, DART_API_KEY, FRED_API_KEY
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import warnings
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

# Windows에서 폴더명(korea2 등)이 모듈명과 충돌하는 문제 방지
# 파일이 있는 폴더를 절대경로로 sys.path 맨 앞에 추가
_THIS_DIR = str(Path(__file__).resolve().parent)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class AnalystReport:
    """hmas_sector_agent 없이 사용할 수 있는 인라인 대체 클래스"""
    name:    str
    score:   float
    comment: str = ""


@dataclass
class BacktestConfig:
    tickers:         list[str]
    start_date:      str   = "2022-01-01"
    end_date:        str   = "2024-12-31"
    freq:            str   = "monthly"        # monthly | weekly
    strategy:        str   = "threshold"      # threshold | topk
    # 방식 B (threshold) 파라미터
    buy_threshold:   float = 70.0             # 이 점수 이상이면 매수
    sell_threshold:  float = 50.0             # 이 점수 이하면 매도
    max_positions:   int   = 10               # 최대 보유 종목 수 (0=무제한)
    # 방식 A (topk) 파라미터
    topk:            int   = 3
    # 공통
    init_cash:       float = 100_000_000      # 초기 자본 1억
    cost_bps:        float = 15.0             # 편도 거래비용 15bp
    mode:            str   = "stub"           # stub | live
    out_dir:         str   = "backtest_results"
    cache_dir:       str   = "backtest_cache"


# ──────────────────────────────────────────────────────────────────────────────
# 캐시
# ──────────────────────────────────────────────────────────────────────────────

class AgentCache:
    def __init__(self, cache_dir: str):
        self.path = Path(cache_dir)
        self.path.mkdir(parents=True, exist_ok=True)

    def _key(self, date_str: str, ticker: str, agent: str) -> Path:
        return self.path / f"{agent}_{ticker}_{date_str}.json"

    def get(self, date_str: str, ticker: str, agent: str) -> Optional[dict]:
        p = self._key(date_str, ticker, agent)
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                pass
        return None

    def set(self, date_str: str, ticker: str, agent: str, data: dict):
        self._key(date_str, ticker, agent).write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )


# ──────────────────────────────────────────────────────────────────────────────
# 날짜 헬퍼
# ──────────────────────────────────────────────────────────────────────────────

def rebalance_dates(start: str, end: str, freq: str) -> list[date]:
    s, e = date.fromisoformat(start), date.fromisoformat(end)
    out, cur = [], s
    while cur <= e:
        out.append(cur)
        if freq == "monthly":
            cur = date(cur.year + (cur.month == 12), cur.month % 12 + 1, 1)
        else:
            cur += timedelta(weeks=1)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# 데이터 수집
# ──────────────────────────────────────────────────────────────────────────────

def fetch_price_data(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    import yfinance as yf
    logger.info(f"주가 수집: {len(tickers)}종목 ({start}~{end})")
    dfs = {}
    for t in tickers:
        try:
            df = yf.download(f"{t}.KS", start=start, end=end,
                             progress=False, auto_adjust=True)
            if df.empty:
                continue
            c = df["Close"]
            if hasattr(c, "columns"):
                c = c.iloc[:, 0]
            c.index = pd.to_datetime(c.index)
            dfs[t] = c
            logger.info(f"  [{t}] {len(c)}일")
        except Exception as e:
            logger.warning(f"  [{t}] 실패: {e}")
    if not dfs:
        raise RuntimeError("주가 데이터 없음")
    return pd.DataFrame(dfs).sort_index()


def fetch_macro_timeseries(start: str, end: str) -> pd.DataFrame:
    import yfinance as yf
    logger.info("거시지표 시계열 수집...")
    frames = {}
    for name, sym in {"kospi": "^KS11", "sp500": "^GSPC", "vix": "^VIX",
                       "usdkrw": "KRW=X", "gold": "GC=F", "crude": "CL=F"}.items():
        try:
            df = yf.download(sym, start=start, end=end, progress=False, auto_adjust=True)
            if not df.empty:
                c = df["Close"]
                if hasattr(c, "columns"):
                    c = c.iloc[:, 0]
                frames[name] = c
        except Exception:
            pass

    fred_key = os.getenv("FRED_API_KEY", "")
    if fred_key:
        try:
            from fredapi import Fred
            fred = Fred(api_key=fred_key)
            for name, sid in {"fed_rate": "FEDFUNDS", "us_10y": "DGS10",
                               "us_cpi": "CPIAUCSL", "kr_rate": "IRSTCI01KRM156N",
                               "kr_cpi": "KORCPIALLMINMEI", "payrolls": "PAYEMS",
                               "unemp": "UNRATE"}.items():
                try:
                    frames[name] = fred.get_series(sid, start, end)
                except Exception:
                    pass
        except ImportError:
            pass

    if not frames:
        idx = pd.date_range(start, end, freq="ME")
        return pd.DataFrame({"vix": 20.0, "kospi": 2500.0, "sp500": 4000.0}, index=idx)

    df = pd.DataFrame(frames)
    df.index = pd.to_datetime(df.index)
    return df.resample("ME").last().ffill().bfill()


# ──────────────────────────────────────────────────────────────────────────────
# 날짜별 MacroIndicators 생성
# ──────────────────────────────────────────────────────────────────────────────

def row_to_macro_indicators(row: pd.Series, prev: Optional[pd.Series]):
    _here = str(Path(__file__).resolve().parent)
    if _here not in sys.path:
        sys.path.insert(0, _here)
    from hmas_macro_agent import MacroIndicators, MacroIndicator

    def mi(label, col, unit=""):
        v = float(row.get(col) or 0)
        roc = 0.0
        if prev is not None:
            p = float(prev.get(col) or 0)
            if p != 0:
                roc = round((v - p) / abs(p) * 100, 2)
        return MacroIndicator(label, round(v, 4), roc, unit)

    return MacroIndicators(
        us_fed_rate=mi("US Fed Rate",   "fed_rate", "%"),
        us_10y_yield=mi("US 10Y",       "us_10y",   "%"),
        jp_policy_rate=mi("KR Rate",    "kr_rate",  "%"),
        jp_10y_yield=mi("KR 10Y",       "kr_10y",   "%") if "kr_10y" in row.index
                     else mi("KR 10Y",  "us_10y",   "%"),
        us_cpi=mi("US CPI",             "us_cpi",   ""),
        jp_cpi=mi("KR CPI",             "kr_cpi",   ""),
        gold=mi("Gold",                 "gold",     " USD/oz"),
        crude_oil=mi("Crude",           "crude",    " USD/bbl"),
        us_payrolls=mi("Payrolls",      "payrolls", "K"),
        industrial_production=mi("IP",  "indpro",   ""),
        housing_starts=mi("Housing",    "housing",  "M"),
        unemployment_rate=mi("Unemp",   "unemp",    "%"),
        jp_business_index=mi("KOSPI",   "kospi",    ""),
        usd_jpy=mi("USD/KRW",           "usdkrw",   ""),
        nikkei_225=mi("KOSPI",          "kospi",    ""),
        sp500=mi("S&P500",              "sp500",    ""),
        us_vix=mi("VIX",                "vix",      ""),
        nikkei_vi=mi("VKOSPI",          "vix",      ""),
    )


# ──────────────────────────────────────────────────────────────────────────────
# 연도별 재무 / 사업보고서 캐시
# ──────────────────────────────────────────────────────────────────────────────

class YearlyCache:
    def __init__(self):
        self._d: dict = {}

    def get(self, ticker: str, year: int):
        return self._d.get((ticker, year))

    def set(self, ticker: str, year: int, val):
        self._d[(ticker, year)] = val


# ──────────────────────────────────────────────────────────────────────────────
# Stub 점수 (API 없이)
# ──────────────────────────────────────────────────────────────────────────────

def stub_score(ticker: str, as_of: date,
               price_df: pd.DataFrame, macro_row: pd.Series) -> float:
    if ticker not in price_df.columns:
        return 50.0
    p = price_df[ticker].dropna()
    p = p[p.index <= pd.Timestamp(as_of)]
    if len(p) < 5:
        return 50.0

    r1m = (p.iloc[-1] / p.iloc[-min(22, len(p))] - 1)
    r3m = (p.iloc[-1] / p.iloc[-min(66, len(p))] - 1)
    r6m = (p.iloc[-1] / p.iloc[-min(126, len(p))] - 1)
    vol = p.pct_change().iloc[-20:].std() * np.sqrt(252) + 1e-6
    vix = float(macro_row.get("vix", 20) or 20)

    score = 50.0
    score += float(np.clip(r1m * 100, -15, 15))
    score += float(np.clip(r3m *  50, -10, 10))
    score += float(np.clip(r6m *  30,  -8,  8))
    score += float(np.clip(r1m / vol *  5,  -5,  5))
    score += 5 if vix < 18 else (-5 if vix > 28 else 0)
    return float(np.clip(score, 5, 95))


# ──────────────────────────────────────────────────────────────────────────────
# Live 점수 (H-MAS 에이전트 실행)
# ──────────────────────────────────────────────────────────────────────────────

def live_score(
    ticker:      str,
    as_of:       date,
    price_df:    pd.DataFrame,
    macro_inds,
    macro_row:   pd.Series,
    fin_cache:   YearlyCache,
    qual_cache:  YearlyCache,
    acache:      AgentCache,
    llm,
    dart,
) -> float:
    # 현재 파일이 있는 폴더를 sys.path 맨 앞에 추가
    # (Windows에서 폴더명이 korea2처럼 패키지명과 충돌하는 문제 방지)
    _here = str(Path(__file__).resolve().parent)
    if _here not in sys.path:
        sys.path.insert(0, _here)

    from dart_collector import KOSPI100_MASTER
    from quant_agent import QuantAgent
    from hmas_macro_agent import MacroAgent
    from hmas_news_agent import NewsAgent, NewsItem
    from hmas_qual_agent import QualAgent, SecuritiesReportExcerpt
    from hmas_pm_agent import PMAgent, REGIME_WEIGHTS
    from kospi_pipeline import (
        _rule_qual, _rule_news, _rule_macro, _rule_pm,
        _empty_metrics, _empty_sec_text,
    )

    ds   = as_of.isoformat()
    info = KOSPI100_MASTER.get(ticker, {"name": ticker, "sector": "기타"})
    name, sector = info["name"], info["sector"]

    # ── Technical ────────────────────────────────────────────────────────
    try:
        from technical_agent import TechnicalAgent
        px = price_df[ticker].dropna()
        px = px[px.index <= pd.Timestamp(as_of)].tolist()[-120:]
        tech = TechnicalAgent().score_series(px)[-1] if len(px) >= 30 else 50.0
    except Exception:
        tech = stub_score(ticker, as_of, price_df, macro_row)

    # ── Quant (연 1회, as_of 기준 회계연도) ─────────────────────────────
    # 12월 결산법인 사업보고서는 3월 말 제출
    # 1~3월: 전년도 보고서 아직 미제출이므로 전전년도가 원칙이나
    #        실제로는 전년도 잠정실적이 이미 공개되어 있어 전년도(FY-1) 우선 시도
    fy_primary   = as_of.year - 1          # 전년도 우선 시도
    fy_fallback  = as_of.year - 2          # 없으면 전전년도
    fy = fy_primary
    fin = fin_cache.get(ticker, fy)
    if fin is None:
        try:
            fin = dart.get_financial_metrics(ticker, name, bsns_year=fy_primary)
            if fin is None:                # DART에 없으면 전전년도
                fin = dart.get_financial_metrics(ticker, name, bsns_year=fy_fallback)
                fy = fy_fallback
            if fin:
                fin_cache.set(ticker, fy, fin)
                logger.info(f"  [{ticker}] DART 재무 수집 완료: FY{fy}")
        except Exception as e:
            logger.warning(f"  [{ticker}] DART 재무 수집 실패: {e}")
            fin = None
    fin = fin or _empty_metrics(ticker, name)
    quant = QuantAgent(llm).run(fin)

    # ── Qual (연 1회, as_of 기준 회계연도) ──────────────────────────────
    sec = qual_cache.get(ticker, fy)
    if sec is None:
        try:
            sec = dart.get_securities_report_text(ticker, name, bsns_year=fy)
            qual_cache.set(ticker, fy, sec)
        except Exception:
            sec = None
    sec = sec or _empty_sec_text(ticker, name)

    excerpt = SecuritiesReportExcerpt(
        ticker=ticker, company_name=name, info_updated=False,
        business_overview=sec.business_overview,
        business_risks=sec.business_risks,
        mda=sec.mda, governance=sec.governance,
    )
    cached_q = acache.get(ds[:4], ticker, "qual")
    if cached_q:
        from hmas_base import QualAgentOutput
        qual = QualAgentOutput(**cached_q)
    elif llm and (sec.business_overview or sec.mda):
        qual = QualAgent(llm).run(excerpt)
        acache.set(ds[:4], ticker, "qual", {
            "business_momentum": qual.business_momentum,
            "immediate_risk_severity": qual.immediate_risk_severity,
            "management_trust": qual.management_trust,
            "insight": qual.insight, "raw": {},
        })
    else:
        qual = _rule_qual(sec)

    # ── News (매월) ──────────────────────────────────────────────────────
    cached_n = acache.get(ds, ticker, "news")
    if cached_n:
        from hmas_base import NewsAgentOutput
        news = NewsAgentOutput(**cached_n)
    else:
        # 주가 모멘텀으로 뉴스 센티멘트 시뮬레이션
        r1m = 0.0
        px_all = price_df[ticker].dropna() if ticker in price_df else pd.Series(dtype=float)
        px_now = px_all[px_all.index <= pd.Timestamp(as_of)]
        if len(px_now) >= 22:
            r1m = float(px_now.iloc[-1] / px_now.iloc[-22] - 1) * 100
        vix = float(macro_row.get("vix", 20) or 20)

        # 실제 뉴스 없이 모멘텀 기반 → LLM 호출 불필요, 규칙으로 처리
        ret_o = 5 if r1m > 10 else 4 if r1m > 3 else 2 if r1m < -5 else 3
        risk_o = 1 if vix < 15 else 5 if vix > 28 else 3
        from hmas_base import NewsAgentOutput
        news = NewsAgentOutput(return_outlook=ret_o, risk_outlook=risk_o,
                               reason=f"1M={r1m:+.1f}% VIX={vix:.1f}", raw={})
        acache.set(ds, ticker, "news", {
            "return_outlook": news.return_outlook,
            "risk_outlook":   news.risk_outlook,
            "reason":         news.reason, "raw": {},
        })

    # ── Macro (월별 공유 캐시 — 종목 무관, 같은 달은 1회만 호출) ──────────
    cached_m = acache.get(ds[:7], "SHARED", "macro")
    if cached_m:
        from hmas_base import MacroAgentOutput
        macro_out = MacroAgentOutput(**cached_m)
    else:
        macro_out = MacroAgent(llm).run(macro_inds) if llm else _rule_macro(macro_inds)
        acache.set(ds[:7], "SHARED", "macro", {
            "market_trend": macro_out.market_trend,
            "risk":         macro_out.risk,
            "economy":      macro_out.economy,
            "rates":        macro_out.rates,
            "inflation":    macro_out.inflation,
            "summary":      macro_out.summary, "raw": {},
        })

    regime = MacroAgent.classify_regime(macro_out)

    # ── PM (5개 에이전트 직접 통합: Macro + Technical + Quant + Qual + News) ──
    pm = PMAgent(llm).run(
             ticker, macro_out, regime,
             tech_score=tech,
             quant_output=quant,
             qual_output=qual,
             news_output=news,
         ) if llm else _rule_pm(tech, quant, qual, news, macro_out, regime)

    # ── 에이전트 결과 출력 ───────────────────────────────────────────────
    logger.info(
        f"[{ticker}] {ds} | Regime={regime} | "
        f"Tech={tech:.0f}  Quant={quant.composite_score:.0f}  "
        f"Qual={qual.composite_score:.0f}  News={news.net_score:.0f}  "
        f"Macro={macro_out.composite_score:.0f}  "
        f"→ PM={pm.final_score}"
    )
    logger.info(
        f"  Quant: {quant.highlight[:60]}"
    )
    logger.info(
        f"  Qual : {qual.insight[:60]}"
    )
    logger.info(
        f"  News : {news.reason[:60]}"
    )
    logger.info(
        f"  Macro: {macro_out.summary[:60]}"
    )
    logger.info(
        f"  PM   : {pm.reason[:80]}"
    )

    return float(pm.final_score)


# ──────────────────────────────────────────────────────────────────────────────
# 시그널 행렬 생성
# ──────────────────────────────────────────────────────────────────────────────

def build_signals(
    cfg: BacktestConfig,
    price_df: pd.DataFrame,
    macro_df: pd.DataFrame,
    llm=None,
    dart=None,
) -> pd.DataFrame:
    """Returns DataFrame: index=rebalance_date, columns=ticker, values=score(0~100)"""
    dates   = rebalance_dates(cfg.start_date, cfg.end_date, cfg.freq)
    tickers = [t for t in cfg.tickers if t in price_df.columns]
    acache  = AgentCache(cfg.cache_dir)
    fin_c   = YearlyCache()
    qual_c  = YearlyCache()

    rows  = []
    total = len(dates) * len(tickers)
    done  = 0

    for d in dates:
        ts   = pd.Timestamp(d)
        midx = macro_df.index.asof(ts)
        mrow = macro_df.loc[midx] if midx in macro_df.index else macro_df.iloc[-1]
        pidx = macro_df.index.get_loc(midx) - 1 if midx in macro_df.index else 0
        prev = macro_df.iloc[pidx] if pidx >= 0 else None

        row = {"date": d.isoformat()}
        for t in tickers:
            done += 1
            if done % 10 == 0:
                logger.info(f"  신호 계산 {done}/{total} ({done/total*100:.0f}%)")
            try:
                if cfg.mode == "stub":
                    row[t] = stub_score(t, d, price_df, mrow)
                else:
                    macro_inds = row_to_macro_indicators(mrow, prev)
                    row[t] = live_score(t, d, price_df, macro_inds, mrow,
                                        fin_c, qual_c, acache, llm, dart)
            except Exception as e:
                import traceback
                logger.warning(f"  [{t}] {d} 점수 실패: {e}")
                logger.warning(traceback.format_exc())
                row[t] = 50.0
        rows.append(row)

    sig = pd.DataFrame(rows).set_index("date")
    sig.index = pd.to_datetime(sig.index)

    # ── 횡단면 정규화 (Cross-sectional z-score → 0~100 rescale) ──────────
    # LLM이 40~65 구간에 점수를 몰아주는 경향을 보정.
    # 종목이 1개일 때는 시계열 정규화(min-max)로 대체.
    if sig.shape[1] >= 2:
        # 종목 여러 개: 날짜별 횡단면 z-score 후 시그모이드 스케일
        mean_ = sig.mean(axis=1)
        std_  = sig.std(axis=1).replace(0, 1)
        z = sig.sub(mean_, axis=0).div(std_, axis=0)          # z-score
        sig_norm = 50 + z * 15                                 # ±1σ = ±15점
        sig = sig_norm.clip(5, 95)
    else:
        # 종목 1개: 시계열 min-max → 20~80 범위로 스케일
        mn, mx = sig.min().min(), sig.max().max()
        if mx > mn:
            sig = 20 + (sig - mn) / (mx - mn) * 60
        # 그래도 전부 같은 값이면 50으로 채움
        sig = sig.fillna(50)

    logger.info(f"시그널 완성 (정규화 후): {sig.shape[0]}×{sig.shape[1]}")
    logger.info(f"  점수 범위: min={sig.min().min():.1f}  max={sig.max().max():.1f}  mean={sig.mean().mean():.1f}")
    return sig


# ──────────────────────────────────────────────────────────────────────────────
# 포트폴리오 시뮬레이션 — 방식 B (threshold) 핵심 로직
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Trade:
    date:   str
    ticker: str
    action: str    # buy | sell
    price:  float
    shares: float
    amount: float
    cost:   float
    score:  float  # 매매 시점 H-MAS 점수


def run_portfolio_sim(
    sig:      pd.DataFrame,
    price_df: pd.DataFrame,
    cfg:      BacktestConfig,
) -> tuple[pd.Series, list[Trade], pd.DataFrame]:
    """
    Returns
    -------
    nav    : 일별 NAV Series
    trades : 거래 내역
    pos_log: 보유 종목 로그 DataFrame
    """
    cost_rate = cfg.cost_bps / 1e4
    reb_dates = list(sig.index)  # Timestamp 리스트 명시적 보장
    cash      = cfg.init_cash
    holdings: dict[str, float] = {}     # ticker → shares
    trades:   list[Trade]      = []
    nav_dict: dict             = {}
    pos_log:  list             = []     # (date, ticker, score, action)

    all_days = pd.date_range(
        start=reb_dates[0],
        end=reb_dates[-1] + pd.Timedelta(days=40),
        freq="B",
    )

    def price_at(ticker: str, ts) -> Optional[float]:
        if ticker not in price_df.columns:
            return None
        s = price_df[ticker].dropna()
        s = s[s.index <= ts]
        return float(s.iloc[-1]) if not s.empty else None

    for i, ts in enumerate(reb_dates):
        row = sig.loc[ts]
        # 단일 종목일 때 sig.loc[ts]가 scalar가 되는 경우 방어
        if not isinstance(row, pd.Series):
            row = pd.Series(row, index=sig.columns)
        scores = row.dropna()

        # ── 전략 분기 ────────────────────────────────────────────────────
        if cfg.strategy == "threshold":
            # 매수 대상: buy_threshold 이상
            buy_targets  = {t: s for t, s in scores.items()
                            if s >= cfg.buy_threshold and price_at(t, ts)}
            # 보유 중이면서 sell_threshold 이하: 매도
            sell_targets = {t for t in holdings
                            if scores.get(t, 0) <= cfg.sell_threshold}
            # 최대 보유 제한 (0 = 무제한)
            if cfg.max_positions > 0:
                current_hold = set(holdings.keys()) - sell_targets
                slots = cfg.max_positions - len(current_hold)
                # 아직 보유 안 한 종목 중 점수 내림차순으로 slots만큼만 허용
                new_buys = sorted(
                    [(t, s) for t, s in buy_targets.items() if t not in holdings],
                    key=lambda x: x[1], reverse=True
                )[:max(0, slots)]
                buy_targets = {t: s for t, s in new_buys}
                buy_targets.update({t: scores[t] for t in holdings
                                    if t not in sell_targets and t in scores})
            new_tickers = set(buy_targets) - set(holdings)

        else:  # topk
            top_tickers = {t for t, _ in
                           sorted(scores.items(), key=lambda x: x[1], reverse=True)[:cfg.topk]
                           if price_at(t, ts)}
            sell_targets = set(holdings) - top_tickers
            new_tickers  = top_tickers - set(holdings)

        # ── 매도 ─────────────────────────────────────────────────────────
        for t in sell_targets:
            px = price_at(t, ts)
            if not px:
                continue
            sh       = holdings.pop(t)
            proceeds = sh * px
            cost     = proceeds * cost_rate
            cash    += proceeds - cost
            sc       = float(scores.get(t, 0))
            trades.append(Trade(str(ts.date()), t, "sell", px, sh, proceeds, cost, sc))
            pos_log.append({"date": ts.date(), "ticker": t, "score": sc,
                            "action": "sell", "price": px})

        # ── 매수 ─────────────────────────────────────────────────────────
        if new_tickers:
            # 남은 현금을 신규 진입 종목 수로 균등 분배
            alloc = cash / len(new_tickers)
            for t in sorted(new_tickers):
                px = price_at(t, ts)
                if not px or px <= 0:
                    continue
                cost   = alloc * cost_rate
                shares = (alloc - cost) / px
                holdings[t] = holdings.get(t, 0) + shares
                cash -= alloc
                sc    = float(scores.get(t, 0))
                trades.append(Trade(str(ts.date()), t, "buy", px, shares, alloc, cost, sc))
                pos_log.append({"date": ts.date(), "ticker": t, "score": sc,
                                "action": "buy", "price": px})

        # 보유 종목 현황 로그
        for t, sh in holdings.items():
            pos_log.append({"date": ts.date(), "ticker": t,
                            "score": float(scores.get(t, 0)), "action": "hold",
                            "price": price_at(t, ts) or 0})

        # ── 일별 NAV 기록 ─────────────────────────────────────────────────
        next_ts = reb_dates[i + 1] if i + 1 < len(reb_dates) \
                  else ts + pd.Timedelta(days=40)
        for d in [dd for dd in all_days if ts <= dd < next_ts]:
            pv = cash + sum((price_at(t, d) or 0) * sh
                            for t, sh in holdings.items())
            nav_dict[d] = pv

    nav = pd.Series(nav_dict).sort_index().dropna()
    pos_df = pd.DataFrame(pos_log) if pos_log else pd.DataFrame()
    return nav, trades, pos_df


# ──────────────────────────────────────────────────────────────────────────────
# 성과 지표
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class BacktestMetrics:
    # 수익
    total_return:      float   # 전체 수익률 %
    annual_return:     float   # ARR: 연환산 수익률 %
    # 리스크
    annual_vol:        float   # 연환산 변동성 %
    max_drawdown:      float   # MDD %
    sharpe:            float   # Sharpe Ratio (Rf=0)
    calmar:            float   # Calmar = ARR / |MDD|
    win_rate:          float   # 월별 승률 %
    # 신호 품질
    ic:                float   # IC  = mean(Pearson(signal, fwd_ret))
    icir:              float   # ICIR = mean(IC) / std(IC)
    rank_ic:           float   # Rank IC = mean(Spearman(signal, fwd_ret))
    rank_icir:         float   # Rank ICIR = mean(RankIC) / std(RankIC)
    ic_monthly:        list    # 월별 IC 시계열
    rank_ic_monthly:   list    # 월별 Rank IC 시계열
    # 벤치마크 대비
    benchmark_return:  float   # KOSPI 전체 수익률 %
    excess_return:     float   # ARR - 벤치마크 연환산
    information_ratio: float   # IR = mean(월별초과수익) / std(월별초과수익) × √12
    # 전략별 추가
    avg_positions:     float   # 평균 보유 종목 수
    turnover:          float   # 월평균 회전율 %


def compute_metrics(
    nav:      pd.Series,
    sig:      pd.DataFrame,
    price_df: pd.DataFrame,
    pos_df:   pd.DataFrame,
    trades:   list[Trade],
    benchmark_ticker: str = "^KS11",
) -> BacktestMetrics:
    nav = nav.dropna()
    if len(nav) < 5:
        raise ValueError("NAV 너무 짧음")

    ret     = nav.pct_change().dropna()
    total   = (nav.iloc[-1] / nav.iloc[0] - 1) * 100
    n_years = (nav.index[-1] - nav.index[0]).days / 365.25
    arr     = ((1 + total / 100) ** (1 / n_years) - 1) * 100 if n_years > 0 else 0.0
    vol     = ret.std() * np.sqrt(252) * 100
    sharpe  = arr / vol if vol > 0 else 0.0

    # MDD
    cum     = (1 + ret).cumprod()
    mdd     = float(((cum - cum.cummax()) / cum.cummax()).min()) * 100
    calmar  = arr / abs(mdd) if mdd != 0 else 0.0

    monthly_ret = nav.resample("ME").last().pct_change().dropna()
    win_rate    = (monthly_ret > 0).mean() * 100

    # IC / ICIR / Rank IC / Rank ICIR
    ic_list, ric_list = [], []
    sig_dates = sig.index.tolist()
    for i in range(len(sig_dates) - 1):
        d0, d1 = sig_dates[i], sig_dates[i + 1]
        s_row  = sig.loc[d0].dropna()
        fwd    = {}
        for t in s_row.index:
            if t not in price_df.columns:
                continue
            p = price_df[t].dropna()
            p0s = p[p.index <= d0]
            p1s = p[p.index <= d1]
            if p0s.empty or p1s.empty:
                continue
            p0, p1 = float(p0s.iloc[-1]), float(p1s.iloc[-1])
            if p0 > 0:
                fwd[t] = (p1 - p0) / p0
        common = [t for t in s_row.index if t in fwd]
        if len(common) < 3:
            continue
        sv = np.array([s_row[t] for t in common])
        rv = np.array([fwd[t]   for t in common])
        if np.std(sv) > 0 and np.std(rv) > 0:
            ic_list.append(float(np.corrcoef(sv, rv)[0, 1]))
            try:
                from scipy.stats import spearmanr
                rc, _ = spearmanr(sv, rv)
                ric_list.append(float(rc))
            except Exception:
                pass

    def safe_icir(lst):
        if len(lst) < 2:
            return 0.0
        m, s = np.mean(lst), np.std(lst, ddof=1)
        return round(m / s, 3) if s > 0 else 0.0

    ic      = round(float(np.mean(ic_list)),  4) if ic_list  else 0.0
    icir    = safe_icir(ic_list)
    rank_ic = round(float(np.mean(ric_list)), 4) if ric_list else 0.0
    rank_icir = safe_icir(ric_list)

    # 벤치마크
    bench_ret = 0.0
    try:
        import yfinance as yf
        bdf = yf.download(benchmark_ticker,
                          start=nav.index[0].strftime("%Y-%m-%d"),
                          end=nav.index[-1].strftime("%Y-%m-%d"),
                          progress=False, auto_adjust=True)
        if not bdf.empty:
            bc = bdf["Close"]
            if hasattr(bc, "columns"):
                bc = bc.iloc[:, 0]
            bench_ret = float((bc.iloc[-1] / bc.iloc[0] - 1) * 100)
    except Exception:
        pass

    bench_arr   = ((1 + bench_ret / 100) ** (1 / n_years) - 1) * 100 if n_years > 0 else 0
    excess      = arr - bench_arr

    # IR (월별 초과수익 기준)
    ir = 0.0
    try:
        import yfinance as yf
        bdf2 = yf.download(benchmark_ticker,
                           start=nav.index[0].strftime("%Y-%m-%d"),
                           end=nav.index[-1].strftime("%Y-%m-%d"),
                           progress=False, auto_adjust=True)
        if not bdf2.empty:
            bc2 = bdf2["Close"]
            if hasattr(bc2, "columns"):
                bc2 = bc2.iloc[:, 0]
            bm = bc2.resample("ME").last().pct_change().dropna()
            cidx = monthly_ret.index.intersection(bm.index)
            if len(cidx) > 2:
                ex_m = monthly_ret.loc[cidx] - bm.loc[cidx]
                ir = float((ex_m.mean() / ex_m.std(ddof=1)) * np.sqrt(12)) if ex_m.std() > 0 else 0.0
    except Exception:
        pass

    # 평균 보유 종목 수
    avg_pos = 0.0
    if not pos_df.empty and "action" in pos_df.columns:
        hold_counts = pos_df[pos_df["action"] == "hold"].groupby("date")["ticker"].count()
        avg_pos = float(hold_counts.mean()) if not hold_counts.empty else 0.0

    # 월평균 회전율
    buys_per_month = len([t for t in trades if t.action == "buy"]) / max(n_years * 12, 1)
    turnover = buys_per_month / max(avg_pos, 1) * 100 if avg_pos > 0 else 0.0

    return BacktestMetrics(
        total_return=round(total, 2),
        annual_return=round(arr, 2),
        annual_vol=round(vol, 2),
        max_drawdown=round(mdd, 2),
        sharpe=round(sharpe, 3),
        calmar=round(calmar, 3),
        win_rate=round(win_rate, 1),
        ic=ic, icir=icir,
        rank_ic=rank_ic, rank_icir=rank_icir,
        ic_monthly=[round(v, 4) for v in ic_list],
        rank_ic_monthly=[round(v, 4) for v in ric_list],
        benchmark_return=round(bench_ret, 2),
        excess_return=round(excess, 2),
        information_ratio=round(ir, 3),
        avg_positions=round(avg_pos, 1),
        turnover=round(turnover, 1),
    )


# ──────────────────────────────────────────────────────────────────────────────
# 출력
# ──────────────────────────────────────────────────────────────────────────────

def print_summary(m: BacktestMetrics, cfg: BacktestConfig):
    strat = (f"Threshold (buy≥{cfg.buy_threshold} / sell≤{cfg.sell_threshold})"
             if cfg.strategy == "threshold" else f"TopK (k={cfg.topk})")
    print(f"\n{'='*64}")
    print(f"  H-MAS KOSPI 백테스트 결과")
    print(f"  전략: {strat}")
    print(f"  기간: {cfg.start_date} ~ {cfg.end_date} | {cfg.freq} | {cfg.mode}")
    print(f"  종목: {', '.join(cfg.tickers)}")
    print(f"{'='*64}")
    print(f"  ── 수익률 ─────────────────────────────────────────────")
    print(f"  전체 수익률 (Total Return)   : {m.total_return:+.2f}%")
    print(f"  연환산 수익률 (ARR)          : {m.annual_return:+.2f}%")
    print(f"  벤치마크 KOSPI               : {m.benchmark_return:+.2f}%")
    print(f"  초과 수익 (vs KOSPI)         : {m.excess_return:+.2f}%")
    print(f"  ── 리스크 ─────────────────────────────────────────────")
    print(f"  연환산 변동성                : {m.annual_vol:.2f}%")
    print(f"  최대낙폭 (MDD)               : {m.max_drawdown:.2f}%")
    print(f"  샤프 비율 (Sharpe)           : {m.sharpe:.3f}")
    print(f"  칼마 비율 (Calmar)           : {m.calmar:.3f}")
    print(f"  월별 승률                    : {m.win_rate:.1f}%")
    print(f"  정보 비율 (IR)               : {m.information_ratio:.3f}")
    print(f"  ── 신호 품질 ──────────────────────────────────────────")
    print(f"  IC                           : {m.ic:.4f}   (n={len(m.ic_monthly)})")
    print(f"  ICIR                         : {m.icir:.3f}")
    print(f"  Rank IC                      : {m.rank_ic:.4f}")
    print(f"  Rank ICIR                    : {m.rank_icir:.3f}")
    print(f"  ── 포트폴리오 ─────────────────────────────────────────")
    print(f"  평균 보유 종목 수            : {m.avg_positions:.1f}개")
    print(f"  월평균 회전율                : {m.turnover:.1f}%")
    print(f"{'='*64}")


def save_results(nav, sig, trades, pos_df, metrics, cfg):
    from datetime import date as _date
    out   = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = _date.today().strftime("%Y%m%d")
    strat = cfg.strategy

    nav.to_csv(out / f"nav_{strat}_{stamp}.csv", header=["NAV"])
    sig.to_csv(out / f"signals_{strat}_{stamp}.csv")

    if trades:
        pd.DataFrame([t.__dict__ for t in trades]).to_csv(
            out / f"trades_{strat}_{stamp}.csv", index=False)
    if not pos_df.empty:
        pos_df.to_csv(out / f"positions_{strat}_{stamp}.csv", index=False)

    md = metrics.__dict__.copy()
    md["config"] = {
        "strategy": cfg.strategy, "buy_threshold": cfg.buy_threshold,
        "sell_threshold": cfg.sell_threshold, "topk": cfg.topk,
        "tickers": cfg.tickers, "start": cfg.start_date,
        "end": cfg.end_date, "freq": cfg.freq, "mode": cfg.mode,
    }
    (out / f"metrics_{strat}_{stamp}.json").write_text(
        json.dumps(md, ensure_ascii=False, indent=2), encoding="utf-8")

    html = _build_html(nav, sig, trades, pos_df, metrics, cfg)
    hp   = out / f"backtest_{strat}_{stamp}.html"
    hp.write_text(html, encoding="utf-8")

    print(f"\n[저장] {out}/")
    print(f"   NAV       → nav_{strat}_{stamp}.csv")
    print(f"   신호      → signals_{strat}_{stamp}.csv")
    print(f"   거래내역  → trades_{strat}_{stamp}.csv")
    print(f"   보유현황  → positions_{strat}_{stamp}.csv")
    print(f"   지표      → metrics_{strat}_{stamp}.json")
    print(f"   HTML 리포트 → backtest_{strat}_{stamp}.html  ← 브라우저로 열면 차트")


def _sc(v): return "#16a34a" if v >= 60 else ("#d97706" if v >= 40 else "#dc2626")
def _bg(v): return "#dcfce7" if v >= 60 else ("#fef9c3" if v >= 40 else "#fee2e2")


def _build_html(nav, sig, trades, pos_df, m, cfg):
    nav_n  = (nav / nav.iloc[0] * 100).dropna()
    n_skip = max(1, len(nav_n) // 300)
    nav_labels = json.dumps([str(d.date()) for d in nav_n.index[::n_skip]])
    nav_vals   = json.dumps([round(v, 2)   for v in nav_n.values[::n_skip]])

    # 월별 수익률 표
    mret  = nav.resample("ME").last().pct_change().dropna() * 100
    mrows = ""
    for dt, r in mret.items():
        c = "#16a34a" if r > 0 else "#dc2626"
        b = "#dcfce7" if r > 0 else "#fee2e2"
        mrows += (f"<tr><td>{dt.strftime('%Y-%m')}</td>"
                  f"<td style='color:{c};font-weight:700'>{r:+.2f}%</td>"
                  f"<td style='background:{b}'>"
                  f"<div style='width:{min(abs(r)*6,100):.0f}px;height:12px;"
                  f"background:{c};border-radius:2px'></div></td></tr>")

    # IC 차트 데이터
    ic_labels   = json.dumps(list(range(1, len(m.ic_monthly) + 1)))
    ic_data     = json.dumps(m.ic_monthly)
    rank_ic_data = json.dumps(m.rank_ic_monthly)

    # 보유 현황 스냅샷 (최근 리밸런싱)
    pos_rows = ""
    if not pos_df.empty and "action" in pos_df.columns:
        last_date = pos_df["date"].max()
        snap = pos_df[(pos_df["date"] == last_date) & (pos_df["action"].isin(["hold", "buy"]))]
        for _, r in snap.iterrows():
            c = _sc(r["score"])
            pos_rows += (f"<tr><td>{r['ticker']}</td>"
                         f"<td style='color:{c};font-weight:700'>{r['score']:.0f}</td>"
                         f"<td>{r['price']:,.0f}원</td></tr>")

    # 거래 내역 (최근 20건)
    trade_rows = ""
    for t in (trades or [])[-20:]:
        c = "#16a34a" if t.action == "buy" else "#dc2626"
        trade_rows += (f"<tr><td>{t.date}</td><td>{t.ticker}</td>"
                       f"<td style='color:{c};font-weight:700'>"
                       f"{'매수' if t.action == 'buy' else '매도'}</td>"
                       f"<td>{t.score:.0f}</td>"
                       f"<td>{t.price:,.0f}</td>"
                       f"<td>{t.amount:,.0f}</td></tr>")

    strat_label = (f"Threshold (매수≥{cfg.buy_threshold} / 매도≤{cfg.sell_threshold})"
                   if cfg.strategy == "threshold" else f"TopK (k={cfg.topk})")
    ret_c = _sc(50 + m.annual_return)
    ic_c  = "#16a34a" if m.ic > 0.03 else ("#d97706" if m.ic > 0 else "#dc2626")

    def card(label, val, sub="", color="#1e293b"):
        return (f"<div class='mc'><div class='ml'>{label}</div>"
                f"<div class='mv' style='color:{color}'>{val}</div>"
                f"<div class='ms'>{sub}</div></div>")

    return f"""<!DOCTYPE html>
<html lang="ko"><head><meta charset="UTF-8">
<title>H-MAS 백테스트 | {cfg.strategy}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
body{{font-family:'Malgun Gothic',sans-serif;background:#f8fafc;margin:0;padding:20px;color:#1e293b}}
.hdr{{background:linear-gradient(135deg,#0f172a,#1e3a5f);color:#fff;border-radius:14px;padding:26px;margin-bottom:18px}}
.card{{background:#fff;border-radius:12px;padding:20px;margin-bottom:14px;box-shadow:0 1px 4px rgba(0,0,0,.08)}}
.mg{{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px;margin-bottom:14px}}
.mc{{background:#fff;border-radius:10px;padding:14px;box-shadow:0 1px 3px rgba(0,0,0,.07);text-align:center}}
.ml{{font-size:10px;color:#94a3b8;font-weight:700;text-transform:uppercase;letter-spacing:.05em}}
.mv{{font-size:24px;font-weight:900;margin:3px 0}}
.ms{{font-size:11px;color:#94a3b8}}
.st{{font-size:12px;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:.05em;margin-bottom:12px}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
th{{background:#f1f5f9;padding:8px 10px;text-align:left;font-weight:600;color:#475569;font-size:12px}}
td{{padding:8px 10px;border-bottom:1px solid #f8fafc}}
.g2{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}
@media(max-width:680px){{.g2{{grid-template-columns:1fr}}}}
</style></head><body>
<div style="max-width:1100px;margin:0 auto">
<div class="hdr">
  <div style="font-size:12px;color:#94a3b8">{cfg.start_date} ~ {cfg.end_date} · {cfg.freq} · {cfg.mode} 모드</div>
  <div style="font-size:22px;font-weight:800;margin:6px 0">H-MAS KOSPI — {strat_label}</div>
  <div style="font-size:13px;color:#cbd5e1">{', '.join(cfg.tickers)}</div>
</div>

<div class="mg">
  {card("ARR", f"{m.annual_return:+.1f}%", f"전체 {m.total_return:+.1f}%", ret_c)}
  {card("MDD", f"{m.max_drawdown:.1f}%", f"Calmar {m.calmar:.2f}", "#dc2626")}
  {card("Sharpe", f"{m.sharpe:.3f}", "Rf=0%", "#2563eb" if m.sharpe > 1 else "#64748b")}
  {card("IR", f"{m.information_ratio:.3f}", "vs KOSPI", "#16a34a" if m.information_ratio > 0.5 else "#64748b")}
  {card("초과수익", f"{m.excess_return:+.1f}%", f"KOSPI {m.benchmark_return:+.1f}%", ret_c)}
  {card("월별승률", f"{m.win_rate:.0f}%", "", "#16a34a" if m.win_rate > 55 else "#64748b")}
  {card("IC", f"{m.ic:.4f}", f"ICIR {m.icir:.3f}", ic_c)}
  {card("Rank IC", f"{m.rank_ic:.4f}", f"Rank ICIR {m.rank_icir:.3f}", ic_c)}
  {card("평균보유", f"{m.avg_positions:.1f}종목", f"회전율 {m.turnover:.0f}%", "#64748b")}
</div>

<div class="card">
  <div class="st">📈 누적 수익률 (기준 100)</div>
  <canvas id="navC" height="70"></canvas>
</div>

<div class="card">
  <div class="st">🎯 월별 IC / Rank IC</div>
  <canvas id="icC" height="55"></canvas>
  <div style="text-align:center;font-size:11px;color:#94a3b8;margin-top:6px">
    IC &gt; 0.03 = 약한 신호 &nbsp;|&nbsp; IC &gt; 0.05 = 유의미 &nbsp;|&nbsp;
    ICIR &gt; 0.5 = 안정적 신호
  </div>
</div>

<div class="g2">
  <div class="card">
    <div class="st">📅 월별 수익률</div>
    <div style="max-height:320px;overflow-y:auto">
    <table><thead><tr><th>월</th><th>수익률</th><th>바</th></tr></thead>
    <tbody>{mrows}</tbody></table></div>
  </div>
  <div class="card">
    <div class="st">💼 현재 보유 종목 (최근 리밸런싱)</div>
    <table><thead><tr><th>종목</th><th>H-MAS 점수</th><th>가격</th></tr></thead>
    <tbody>{pos_rows or '<tr><td colspan=3 style="color:#94a3b8;text-align:center">없음</td></tr>'}</tbody></table>
    <div style="margin-top:14px"><div class="st">최근 거래 내역</div>
    <div style="max-height:200px;overflow-y:auto">
    <table><thead><tr><th>날짜</th><th>종목</th><th>구분</th><th>점수</th><th>가격</th><th>금액</th></tr></thead>
    <tbody>{trade_rows or '<tr><td colspan=6 style="color:#94a3b8;text-align:center">없음</td></tr>'}</tbody></table>
    </div></div>
  </div>
</div>

<div style="text-align:center;font-size:11px;color:#94a3b8;margin-top:6px">
  H-MAS KOSPI Backtest · {date.today()} · 투자 참고용
</div>
</div>

<script>
new Chart(document.getElementById('navC').getContext('2d'), {{
  type:'line', data:{{
    labels:{nav_labels},
    datasets:[{{label:'H-MAS 포트폴리오',data:{nav_vals},
      borderColor:'#2563eb',backgroundColor:'rgba(37,99,235,0.05)',
      borderWidth:2,pointRadius:0,fill:true,tension:0.1}}]
  }},
  options:{{responsive:true,plugins:{{legend:{{position:'top'}}}},
    scales:{{y:{{grid:{{color:'#f1f5f9'}}}},
             x:{{grid:{{display:false}},ticks:{{maxTicksLimit:12}}}}}}}}
}});
new Chart(document.getElementById('icC').getContext('2d'), {{
  type:'bar', data:{{
    labels:{ic_labels},
    datasets:[
      {{label:'IC',data:{ic_data},
        backgroundColor:{ic_data}.map(v=>v>=0?'rgba(37,99,235,0.6)':'rgba(220,38,38,0.5)'),
        borderWidth:0,order:2}},
      {{label:'Rank IC',data:{rank_ic_data},type:'line',
        borderColor:'#f59e0b',backgroundColor:'transparent',
        borderWidth:2,pointRadius:3,order:1}}
    ]
  }},
  options:{{responsive:true,plugins:{{legend:{{position:'top'}}}},
    scales:{{y:{{ticks:{{callback:v=>v.toFixed(2)}},grid:{{color:'#f1f5f9'}}}},
             x:{{grid:{{display:false}},
               title:{{display:true,text:'리밸런싱 회차'}}}}}}}}
}});
</script>
</body></html>"""


# ──────────────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="H-MAS KOSPI 에이전트 백테스트")
    p.add_argument("--strategy", choices=["threshold", "topk"], default="threshold",
                   help="threshold=방식B(임계값) | topk=방식A(상위K종목)")
    p.add_argument("--mode",  choices=["stub", "live"], default="stub")
    p.add_argument("--tickers", nargs="+",
                   default=["005930", "000660", "005380", "035420",
                            "373220", "105560", "051910", "207940"])
    p.add_argument("--start", default="2022-01-01")
    p.add_argument("--end",   default="2024-12-31")
    p.add_argument("--freq",  choices=["monthly", "weekly"], default="monthly")
    # threshold 파라미터
    p.add_argument("--buy",   type=float, default=70.0, help="매수 임계값 (기본 70)")
    p.add_argument("--sell",  type=float, default=50.0, help="매도 임계값 (기본 50)")
    p.add_argument("--maxpos",type=int,   default=10,   help="최대 보유 종목 수 (0=무제한)")
    # topk 파라미터
    p.add_argument("--topk",  type=int, default=3)
    # 공통
    p.add_argument("--cash",  type=float, default=100_000_000)
    p.add_argument("--cost",  type=float, default=15.0)
    p.add_argument("--out",   default="backtest_results")
    p.add_argument("--cache", default="backtest_cache")
    args = p.parse_args()

    cfg = BacktestConfig(
        tickers=args.tickers,
        start_date=args.start, end_date=args.end, freq=args.freq,
        strategy=args.strategy,
        buy_threshold=args.buy, sell_threshold=args.sell,
        max_positions=args.maxpos,
        topk=args.topk,
        init_cash=args.cash, cost_bps=args.cost,
        mode=args.mode, out_dir=args.out, cache_dir=args.cache,
    )

    llm = dart = None
    if cfg.mode == "live":
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            logger.error("OPENAI_API_KEY 없음")
            sys.exit(1)
        _here = str(Path(__file__).resolve().parent)
        if _here not in sys.path:
            sys.path.insert(0, _here)
        from hmas_base import LLMClient
        from dart_collector import DartClient
        llm  = LLMClient(model="gpt-4o", api_key=key)
        dart = DartClient(api_key=os.getenv("DART_API_KEY", ""))

    logger.info(f"전략: {cfg.strategy} | 모드: {cfg.mode}")
    if cfg.strategy == "threshold":
        logger.info(f"  매수 임계값: {cfg.buy_threshold} | 매도 임계값: {cfg.sell_threshold} | 최대보유: {cfg.max_positions}")
    else:
        logger.info(f"  TopK: {cfg.topk}")

    price_df = fetch_price_data(
        cfg.tickers, cfg.start_date,
        (date.fromisoformat(cfg.end_date) + timedelta(days=40)).isoformat()
    )
    macro_df = fetch_macro_timeseries(cfg.start_date, cfg.end_date)

    logger.info("H-MAS 시그널 생성...")
    sig = build_signals(cfg, price_df, macro_df, llm, dart)

    # ── 시그널 점수 요약 출력 ────────────────────────────────────
    print("\n" + "="*56)
    print("  [H-MAS] 시그널 점수 요약")
    print("="*56)
    for t in sig.columns:
        s = sig[t]
        print(f"  [{t}] min={s.min():.1f}  max={s.max():.1f}  "
              f"mean={s.mean():.1f}  매수>={cfg.buy_threshold} 횟수={int((s >= cfg.buy_threshold).sum())}")
    print(f"\n  최근 6개월 점수:")
    print(sig.tail(6).to_string())
    print("="*56 + "\n")

    logger.info(f"포트폴리오 시뮬레이션 ({cfg.strategy})...")
    nav, trades, pos_df = run_portfolio_sim(sig, price_df, cfg)

    logger.info("성과 분석...")
    metrics = compute_metrics(nav, sig, price_df, pos_df, trades)

    print_summary(metrics, cfg)
    save_results(nav, sig, trades, pos_df, metrics, cfg)


if __name__ == "__main__":
    main()