"""
H-MAS KOSPI 백테스트  (hmas_backtest.py)
==========================================

Qlib 데이터 인프라(build_qlib_kr_data.py로 구축)를 기반으로
Qlib 모델 클래스를 직접 사용하는 백테스트입니다.

  import lightgbm  ← 안씀
  import xgboost   ← 안씀
  import torch     ← lstm만 내부적으로 필요

  대신:
  from qlib.contrib.model.gbdt import LGBModel      ← LightGBM
  from qlib.contrib.model.xgboost import XGBModel   ← XGBoost
  from qlib.contrib.model.pytorch_lstm import LSTM   ← LSTM

파이프라인:
  1. qlib.init(provider_uri)
       → build_qlib_kr_data.py로 구축한 ~/.qlib/qlib_data/kr_data 로드
  2. DatasetH + Alpha158
       → Qlib이 kr_data에서 직접 OHLCV 읽어 158개 피처 생성
  3. LGBModel / XGBModel / LSTM
       → DatasetH를 그대로 .fit(dataset) 으로 전달
  4. model.predict(dataset, segment="test")
       → 종목별 수익률 예측 점수
  5. TopK 포트폴리오 시뮬레이션
  6. Sharpe / IC / ICIR / Rank IC / Rank ICIR / IR / ARR / MDD

전제조건:
  python build_qlib_kr_data.py  ← 먼저 실행

실행:
  python hmas_backtest.py --model lgbm
  python hmas_backtest.py --model xgb  --topk 3
  python hmas_backtest.py --model lstm
  python hmas_backtest.py --stub        # 데이터 없이 구조 검증

설치:
  pip install pyqlib lightgbm xgboost
  pip install torch  # lstm만 필요
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

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

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
class BacktestConfig:
    model:        str   = "lgbm"
    tickers:      list  = field(default_factory=lambda: [
        "005930", "000660", "005380", "035420",
        "373220", "105560", "051910", "207940",
    ])
    qlib_data:    str   = "~/.qlib/qlib_data/kr_data"
    train_start:  str   = "2019-01-01"
    train_end:    str   = "2021-12-31"
    valid_start:  str   = "2022-01-01"
    valid_end:    str   = "2022-12-31"
    test_start:   str   = "2023-01-01"
    test_end:     str   = "2024-12-31"
    topk:         int   = 3
    freq:         str   = "monthly"
    init_cash:    float = 100_000_000
    cost_bps:     float = 15.0
    out_dir:      str   = "backtest_results"
    stub:         bool  = False


# ──────────────────────────────────────────────────────────────────────────────
# Qlib 초기화 + DatasetH 구성
# ──────────────────────────────────────────────────────────────────────────────

def init_qlib_and_dataset(cfg: BacktestConfig):
    """
    qlib.init → DatasetH(Alpha158) 생성 및 반환.

    Alpha158: OHLCV 기반 158개 팩터
      - kbar 피처 (캔들 패턴)
      - price 피처 (OPEN/HIGH/LOW 정규화)
      - rolling 피처 (ROC/MA/STD/MAX/MIN/QTLU/QTLD, window=5/10/20/30/60)
    """
    import qlib
    from qlib.data.dataset import DatasetH
    from qlib.contrib.data.handler import Alpha158

    qlib_path = str(Path(cfg.qlib_data).expanduser())
    logger.info(f"qlib.init: {qlib_path}")
    qlib.init(provider_uri=qlib_path)

    instruments = cfg.tickers
    logger.info(f"종목: {instruments}")

    handler = Alpha158(
        instruments=instruments,
        start_time=cfg.train_start,
        end_time=cfg.test_end,
        fit_start_time=cfg.train_start,
        fit_end_time=cfg.train_end,
        infer_processors=[
            {"class": "RobustZScoreNorm",
             "kwargs": {"fields_group": "feature", "clip_outlier": True}},
            {"class": "Fillna",
             "kwargs": {"fields_group": "feature"}},
        ],
        learn_processors=[
            {"class": "DropnaLabel"},
            {"class": "CSZScoreNorm",
             "kwargs": {"fields_group": "label"}},
        ],
    )

    dataset = DatasetH(
        handler=handler,
        segments={
            "train": (cfg.train_start, cfg.train_end),
            "valid": (cfg.valid_start, cfg.valid_end),
            "test":  (cfg.test_start,  cfg.test_end),
        },
    )

    logger.info("DatasetH(Alpha158) 구성 완료")
    return dataset


# ──────────────────────────────────────────────────────────────────────────────
# 모델 1: LightGBM — Qlib LGBModel
# ──────────────────────────────────────────────────────────────────────────────

def run_lgbm(dataset, cfg) -> pd.DataFrame:
    """
    Qlib LGBModel.fit(dataset) → predict(dataset, segment='test')
    import lightgbm 없이 Qlib 모델 클래스만 사용합니다.
    """
    from qlib.contrib.model.gbdt import LGBModel

    logger.info("[Qlib LGBModel] 학습 시작")

    model = LGBModel(
        loss="mse",
        num_boost_round=1000,
        early_stopping_rounds=50,
        num_leaves=63,
        learning_rate=0.05,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        min_child_samples=20,
        verbose=-1,
        n_jobs=-1,
    )

    model.fit(dataset, verbose_eval=100)
    logger.info(f"[Qlib LGBModel] 최적 반복: {model.model.best_iteration}")

    # 피처 중요도
    x_test = dataset.prepare("test", col_set="feature")
    feat_names = x_test.columns.tolist()
    imp = pd.Series(
        model.model.feature_importance("gain"), index=feat_names
    ).sort_values(ascending=False)
    logger.info(f"[Qlib LGBModel] Top5 피처: {imp.head(5).to_dict()}")

    # 예측: pd.Series(index=(datetime, instrument))
    pred = model.predict(dataset, segment="test")
    return _pred_series_to_df(pred)


# ──────────────────────────────────────────────────────────────────────────────
# 모델 2: XGBoost — Qlib XGBModel
# ──────────────────────────────────────────────────────────────────────────────

def run_xgb(dataset, cfg) -> pd.DataFrame:
    """
    Qlib XGBModel.fit(dataset) → predict(dataset, segment='test')
    import xgboost 없이 Qlib 모델 클래스만 사용합니다.
    """
    from qlib.contrib.model.xgboost import XGBModel

    logger.info("[Qlib XGBModel] 학습 시작")

    model = XGBModel(
        objective="reg:squarederror",
        eval_metric="rmse",
        max_depth=6,
        eta=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=20,
        tree_method="hist",
        verbosity=0,
    )

    model.fit(dataset, num_boost_round=1000, early_stopping_rounds=50, verbose_eval=100)
    logger.info(f"[Qlib XGBModel] 최적 반복: {model.model.best_iteration}")

    imp = pd.Series(
        model.model.get_score(importance_type="gain")
    ).sort_values(ascending=False)
    logger.info(f"[Qlib XGBModel] Top5 피처: {imp.head(5).to_dict()}")

    pred = model.predict(dataset, segment="test")
    return _pred_series_to_df(pred)


# ──────────────────────────────────────────────────────────────────────────────
# 모델 3: LSTM — Qlib pytorch_lstm.LSTM
# ──────────────────────────────────────────────────────────────────────────────

def run_lstm(dataset, cfg) -> pd.DataFrame:
    """
    Qlib LSTM.fit(dataset) → predict(dataset, segment='test')
    import torch / nn 없이 Qlib 모델 클래스만 사용합니다.
    (torch는 내부적으로 필요: pip install torch)
    """
    from qlib.contrib.model.pytorch_lstm import LSTM as QlibLSTM

    # d_feat: Alpha158 피처 수 확인
    x_test = dataset.prepare("test", col_set="feature")
    n_feat  = x_test.shape[1]
    logger.info(f"[Qlib LSTM] 학습 시작 (d_feat={n_feat})")

    model = QlibLSTM(
        d_feat=n_feat,
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
        n_epochs=100,
        lr=1e-3,
        batch_size=256,
        early_stop=15,
        loss="mse",
        optimizer="adam",
        GPU=0,
    )

    model.fit(dataset)
    pred = model.predict(dataset, segment="test")
    return _pred_series_to_df(pred)


# ──────────────────────────────────────────────────────────────────────────────
# Stub (Qlib 데이터 없이 구조 검증)
# ──────────────────────────────────────────────────────────────────────────────

def run_stub(cfg) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Qlib 데이터 없이 더미 예측 + 더미 가격으로 파이프라인 검증.
    """
    logger.info("[Stub] 더미 데이터 생성")
    np.random.seed(42)

    dates   = pd.date_range(cfg.test_start, cfg.test_end, freq="B")
    tickers = cfg.tickers
    records = []
    for t in tickers:
        for d in dates:
            records.append({"date": d, "ticker": t,
                            "score": np.random.normal(0, 1)})
    pred_df = pd.DataFrame(records)
    pred_df["date"] = pd.to_datetime(pred_df["date"])

    all_dates = pd.date_range(cfg.train_start, cfg.test_end, freq="B")
    price_dict = {}
    for t in tickers:
        price_dict[t] = pd.Series(
            50000 * np.cumprod(1 + np.random.normal(0.0003, 0.015, len(all_dates))),
            index=all_dates,
        )
    price_df = pd.DataFrame(price_dict)
    return pred_df, price_df


def _pred_series_to_df(pred: pd.Series) -> pd.DataFrame:
    """
    Qlib predict() 반환값 (MultiIndex Series) → (date, ticker, score) DataFrame.
    index: (datetime, instrument) MultiIndex
    """
    df = pred.reset_index()
    df.columns = ["date", "ticker", "score"]
    df["date"] = pd.to_datetime(df["date"])
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 주가 데이터 (포트폴리오 시뮬레이션용) — Qlib D.features에서 직접 수집
# ──────────────────────────────────────────────────────────────────────────────

def load_price_from_qlib(cfg: BacktestConfig) -> pd.DataFrame:
    """
    qlib.init이 완료된 상태에서 D.features로 종가 로드.
    반환: DataFrame(index=datetime, columns=ticker)
    """
    from qlib.data import D

    logger.info("Qlib에서 종가 로드...")
    df = D.features(
        cfg.tickers,
        ["$close"],
        start_time=cfg.train_start,
        end_time=(date.fromisoformat(cfg.test_end) + timedelta(days=40)).isoformat(),
        freq="day",
    )
    # MultiIndex (instrument, datetime) → pivot
    price_df = df["$close"].unstack(level="instrument")
    price_df.index = pd.to_datetime(price_df.index)
    logger.info(f"종가 로드 완료: {price_df.shape}")
    return price_df


# ──────────────────────────────────────────────────────────────────────────────
# 포트폴리오 시뮬레이션 (TopK)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Trade:
    date: str; ticker: str; action: str
    price: float; shares: float; amount: float; cost: float


def run_portfolio_sim(
    pred_df: pd.DataFrame,
    price_df: pd.DataFrame,
    cfg: BacktestConfig,
) -> tuple[pd.Series, list[Trade]]:
    cost_rate = cfg.cost_bps / 10_000
    reb_dates = sorted(pred_df["date"].unique())
    cash      = cfg.init_cash
    holdings: dict[str, float] = {}
    trades:   list[Trade]      = []
    nav_dict: dict             = {}

    all_days = pd.date_range(
        start=min(reb_dates),
        end=max(reb_dates) + pd.Timedelta(days=40),
        freq="B",
    )

    def get_price(ticker, as_of):
        if ticker not in price_df.columns:
            return None
        s = price_df[ticker].dropna()
        s = s[s.index <= as_of]
        return float(s.iloc[-1]) if not s.empty else None

    for i, reb_ts in enumerate(reb_dates):
        scores = pred_df[pred_df["date"] == reb_ts].set_index("ticker")["score"]
        valid  = {t: s for t, s in scores.items() if get_price(t, reb_ts) is not None}
        if not valid:
            continue

        top_tickers = set(sorted(valid, key=lambda t: valid[t], reverse=True)[:cfg.topk])
        cur_tickers = set(holdings.keys())

        for ticker in (cur_tickers - top_tickers):
            price = get_price(ticker, reb_ts)
            if not price: continue
            shares   = holdings.pop(ticker)
            proceeds = shares * price
            cost     = proceeds * cost_rate
            cash    += proceeds - cost
            trades.append(Trade(str(reb_ts.date()), ticker, "sell",
                                price, shares, proceeds, cost))

        new_tickers = top_tickers - cur_tickers
        if new_tickers:
            alloc = cash / len(new_tickers)
            for ticker in new_tickers:
                price = get_price(ticker, reb_ts)
                if not price or price <= 0: continue
                cost   = alloc * cost_rate
                shares = (alloc - cost) / price
                holdings[ticker] = holdings.get(ticker, 0) + shares
                cash -= alloc
                trades.append(Trade(str(reb_ts.date()), ticker, "buy",
                                    price, shares, alloc, cost))

        next_ts = reb_dates[i + 1] if i + 1 < len(reb_dates) \
                  else reb_ts + pd.Timedelta(days=40)
        for d in [dd for dd in all_days if reb_ts <= dd < next_ts]:
            pv = cash + sum((get_price(t, d) or 0) * sh
                            for t, sh in holdings.items())
            nav_dict[d] = pv

    return pd.Series(nav_dict).sort_index().dropna(), trades


# ──────────────────────────────────────────────────────────────────────────────
# 성과 분석
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class BacktestMetrics:
    total_return:      float
    annual_return:     float   # ARR
    annual_vol:        float
    sharpe:            float
    max_drawdown:      float   # MDD
    calmar:            float
    win_rate:          float
    ic:                float
    icir:              float
    rank_ic:           float
    rank_icir:         float
    ic_monthly:        list
    rank_ic_monthly:   list
    benchmark_return:  float
    excess_return:     float
    information_ratio: float   # IR


def compute_metrics(
    nav: pd.Series,
    pred_df: pd.DataFrame,
    price_df: pd.DataFrame,
    benchmark_ticker: str = "^KS11",
) -> BacktestMetrics:
    nav = nav.dropna()
    if len(nav) < 5:
        raise ValueError("NAV 데이터 부족")

    ret     = nav.pct_change().dropna()
    total   = float((nav.iloc[-1] / nav.iloc[0] - 1) * 100)
    n_years = (nav.index[-1] - nav.index[0]).days / 365.25
    annual  = float(((1 + total / 100) ** (1 / n_years) - 1) * 100) if n_years > 0 else 0.0
    vol     = float(ret.std() * np.sqrt(252) * 100)
    sharpe  = annual / vol if vol > 0 else 0.0

    cum    = (1 + ret).cumprod()
    mdd    = float((cum / cum.cummax() - 1).min() * 100)
    calmar = annual / abs(mdd) if mdd != 0 else 0.0

    monthly_ret = nav.resample("ME").last().pct_change().dropna()
    win_rate    = float((monthly_ret > 0).mean() * 100)

    # IC / ICIR / Rank IC / Rank ICIR
    ic_list, rank_ic_list = [], []
    for dt, grp in pred_df.groupby("date"):
        grp   = grp.set_index("ticker")
        dt_ts = pd.Timestamp(dt)
        fwd   = {}
        for ticker in grp.index:
            if ticker not in price_df.columns:
                continue
            p  = price_df[ticker].dropna()
            p0 = p[p.index <= dt_ts]
            p1 = p[p.index > dt_ts]
            if p0.empty or len(p1) < 1:
                continue
            v0 = float(p0.iloc[-1])
            v1 = float(p1.iloc[min(19, len(p1) - 1)])
            if v0 > 0:
                fwd[ticker] = (v1 - v0) / v0
        common = [t for t in grp.index if t in fwd]
        if len(common) < 3:
            continue
        s = np.array([grp.loc[t, "score"] for t in common])
        r = np.array([fwd[t] for t in common])
        if np.std(s) > 0 and np.std(r) > 0:
            ic_list.append(float(np.corrcoef(s, r)[0, 1]))
            rc, _ = spearmanr(s, r)
            rank_ic_list.append(float(rc))

    def safe_icir(vals):
        if not vals: return 0.0
        m, s = np.mean(vals), np.std(vals, ddof=1) if len(vals) > 1 else 1.0
        return round(m / s, 3) if s > 0 else 0.0

    ic        = round(float(np.mean(ic_list))      if ic_list      else 0.0, 4)
    icir      = safe_icir(ic_list)
    rank_ic   = round(float(np.mean(rank_ic_list)) if rank_ic_list else 0.0, 4)
    rank_icir = safe_icir(rank_ic_list)

    # 벤치마크 / IR
    bench_ret, information_ratio = 0.0, 0.0
    try:
        import yfinance as yf
        bdf = yf.download(benchmark_ticker,
                          start=nav.index[0].strftime("%Y-%m-%d"),
                          end=nav.index[-1].strftime("%Y-%m-%d"),
                          progress=False, auto_adjust=True)
        if not bdf.empty:
            bc = bdf["Close"]
            if hasattr(bc, "columns"): bc = bc.iloc[:, 0]
            bench_ret = float((bc.iloc[-1] / bc.iloc[0] - 1) * 100)
            bench_m   = bc.resample("ME").last().pct_change().dropna()
            cidx      = monthly_ret.index.intersection(bench_m.index)
            if len(cidx) > 2:
                exc = monthly_ret.loc[cidx] - bench_m.loc[cidx]
                information_ratio = float(exc.mean() / exc.std(ddof=1) * np.sqrt(12))
    except Exception:
        pass

    return BacktestMetrics(
        total_return=round(total, 2),
        annual_return=round(annual, 2),
        annual_vol=round(vol, 2),
        sharpe=round(sharpe, 2),
        max_drawdown=round(mdd, 2),
        calmar=round(calmar, 2),
        win_rate=round(win_rate, 1),
        ic=ic, icir=icir,
        rank_ic=rank_ic, rank_icir=rank_icir,
        ic_monthly=[round(v, 4) for v in ic_list],
        rank_ic_monthly=[round(v, 4) for v in rank_ic_list],
        benchmark_return=round(bench_ret, 2),
        excess_return=round(annual - bench_ret, 2),
        information_ratio=round(information_ratio, 3),
    )


# ──────────────────────────────────────────────────────────────────────────────
# 출력 / 저장
# ──────────────────────────────────────────────────────────────────────────────

def print_summary(m: BacktestMetrics, cfg: BacktestConfig):
    label = "Stub" if cfg.stub else cfg.model.upper()
    print(f"\n{'='*62}")
    print(f"  H-MAS KOSPI 백테스트  [{label}]")
    print(f"  Qlib 데이터: {cfg.qlib_data}")
    print(f"  학습: {cfg.train_start}~{cfg.train_end}  테스트: {cfg.test_start}~{cfg.test_end}")
    print(f"  종목: {', '.join(cfg.tickers[:6])}{'...' if len(cfg.tickers)>6 else ''}")
    print(f"{'='*62}")
    print(f"  전체 수익률 (Total)    : {m.total_return:+.2f}%")
    print(f"  연환산 수익률 (ARR)    : {m.annual_return:+.2f}%")
    print(f"  벤치마크 (KOSPI)       : {m.benchmark_return:+.2f}%")
    print(f"  초과 수익률            : {m.excess_return:+.2f}%")
    print(f"  연환산 변동성          : {m.annual_vol:.2f}%")
    print(f"  최대낙폭 (MDD)         : {m.max_drawdown:.2f}%")
    print(f"  샤프 비율 (Sharpe)     : {m.sharpe:.3f}")
    print(f"  칼마 비율 (Calmar)     : {m.calmar:.3f}")
    print(f"  월별 승률              : {m.win_rate:.1f}%")
    print(f"  정보 비율 (IR)         : {m.information_ratio:.3f}")
    print(f"  IC                     : {m.ic:.4f}  (샘플:{len(m.ic_monthly)})")
    print(f"  ICIR                   : {m.icir:.3f}")
    print(f"  Rank IC                : {m.rank_ic:.4f}")
    print(f"  Rank ICIR              : {m.rank_icir:.3f}")
    print(f"{'='*62}")


def save_results(nav, pred_df, trades, metrics, cfg):
    out   = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    label = "stub" if cfg.stub else cfg.model
    stamp = date.today().strftime("%Y%m%d")
    pfx   = f"{label}_{stamp}"

    nav.to_csv(out / f"nav_{pfx}.csv", header=["NAV"])
    pred_df.to_csv(out / f"pred_{pfx}.csv", index=False)

    trades_df = pd.DataFrame([t.__dict__ for t in trades])
    if not trades_df.empty:
        trades_df.to_csv(out / f"trades_{pfx}.csv", index=False)

    metrics_d = {k: v for k, v in metrics.__dict__.items()}
    metrics_d["config"] = {
        "model": label, "qlib_data": cfg.qlib_data,
        "tickers": cfg.tickers,
        "train": f"{cfg.train_start}~{cfg.train_end}",
        "test":  f"{cfg.test_start}~{cfg.test_end}",
        "topk": cfg.topk,
    }
    (out / f"metrics_{pfx}.json").write_text(
        json.dumps(metrics_d, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    html = _build_html(nav, trades_df if not trades_df.empty else pd.DataFrame(), metrics, cfg)
    html_path = out / f"report_{pfx}.html"
    html_path.write_text(html, encoding="utf-8")

    print(f"\n📁 결과 저장: {out}/")
    print(f"   report_{pfx}.html  ← 브라우저로 열기")
    print(f"   metrics_{pfx}.json")
    print(f"   nav_{pfx}.csv")


def _build_html(nav, trades_df, m: BacktestMetrics, cfg: BacktestConfig) -> str:
    label    = "Stub" if cfg.stub else cfg.model.upper()
    nav_norm = (nav / nav.iloc[0] * 100).dropna()
    step     = max(1, len(nav_norm) // 250)
    nav_labels = [str(d.date()) for d in nav_norm.index[::step]]
    nav_values = [round(v, 2) for v in nav_norm.values[::step]]

    monthly_rows = ""
    for dt, r in nav.resample("ME").last().pct_change().dropna().items():
        r = float(r) * 100
        c = "#dcfce7" if r > 0 else "#fee2e2"
        t = "#16a34a" if r > 0 else "#dc2626"
        monthly_rows += (
            f"<tr><td>{dt.strftime('%Y-%m')}</td>"
            f"<td style='color:{t};font-weight:700'>{r:+.2f}%</td>"
            f"<td><div style='width:{min(abs(r)*8,120):.0f}px;height:12px;"
            f"background:{t};border-radius:2px'></div></td></tr>"
        )

    trades_html = ""
    if not trades_df.empty:
        for _, row in trades_df.tail(30).iterrows():
            ac = "#16a34a" if row["action"] == "buy" else "#dc2626"
            trades_html += (
                f"<tr><td>{row['date']}</td><td>{row['ticker']}</td>"
                f"<td style='color:{ac};font-weight:700'>"
                f"{'매수' if row['action']=='buy' else '매도'}</td>"
                f"<td>{row['price']:,.0f}원</td><td>{row['amount']:,.0f}원</td></tr>"
            )

    def mc(lbl, val, sub="", color="#1e293b"):
        return (f"<div class='mc'><div class='ml'>{lbl}</div>"
                f"<div class='mv' style='color:{color}'>{val}</div>"
                f"<div class='ms'>{sub}</div></div>")

    rc   = "#16a34a" if m.annual_return > 0 else "#dc2626"
    ec   = "#16a34a" if m.excess_return > 0 else "#dc2626"
    ic_c = "#16a34a" if m.ic > 0.03 else ("#d97706" if m.ic > 0 else "#dc2626")

    return f"""<!DOCTYPE html>
<html lang="ko"><head><meta charset="UTF-8">
<title>H-MAS [{label}] 백테스트</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  body{{font-family:'Malgun Gothic',sans-serif;background:#f8fafc;margin:0;padding:20px;color:#1e293b}}
  .hdr{{background:linear-gradient(135deg,#0f172a,#1e3a5f);color:#fff;border-radius:14px;padding:24px;margin-bottom:16px}}
  .card{{background:#fff;border-radius:12px;padding:20px;margin-bottom:14px;box-shadow:0 1px 4px rgba(0,0,0,.08)}}
  .grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px;margin-bottom:14px}}
  .mc{{background:#fff;border-radius:10px;padding:14px;box-shadow:0 1px 3px rgba(0,0,0,.07);text-align:center}}
  .ml{{font-size:10px;color:#94a3b8;font-weight:700;text-transform:uppercase}}
  .mv{{font-size:23px;font-weight:900;margin:3px 0}}
  .ms{{font-size:10px;color:#94a3b8}}
  .st{{font-size:12px;font-weight:700;color:#64748b;text-transform:uppercase;margin-bottom:12px}}
  table{{width:100%;border-collapse:collapse;font-size:12px}}
  th{{background:#f1f5f9;padding:8px 10px;text-align:left;font-weight:700;color:#475569}}
  td{{padding:8px 10px;border-bottom:1px solid #f8fafc}}
  .g2{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}
  @media(max-width:700px){{.g2{{grid-template-columns:1fr}}}}
</style></head><body>
<div style="max-width:1100px;margin:0 auto">
<div class="hdr">
  <div style="font-size:11px;color:#94a3b8">
    [{label}] · Qlib Alpha158 · 학습 {cfg.train_start}~{cfg.train_end}
    · 테스트 {cfg.test_start}~{cfg.test_end} · Top{cfg.topk}
  </div>
  <div style="font-size:22px;font-weight:800;margin-top:6px">H-MAS KOSPI [{label}] 백테스트</div>
  <div style="font-size:12px;color:#94a3b8;margin-top:4px">{', '.join(cfg.tickers)}</div>
</div>

<div class="grid">
  {mc("ARR", f"{m.annual_return:+.1f}%", f"전체 {m.total_return:+.1f}%", rc)}
  {mc("변동성", f"{m.annual_vol:.1f}%", "연환산")}
  {mc("Sharpe", f"{m.sharpe:.2f}", "Rf=0%", "#2563eb" if m.sharpe > 1 else "#64748b")}
  {mc("MDD", f"{m.max_drawdown:.1f}%", f"Calmar {m.calmar:.2f}", "#dc2626")}
  {mc("승률", f"{m.win_rate:.0f}%", "월별", "#16a34a" if m.win_rate > 55 else "#64748b")}
  {mc("초과수익", f"{m.excess_return:+.1f}%", f"KOSPI {m.benchmark_return:+.1f}%", ec)}
  {mc("IR", f"{m.information_ratio:.3f}", "월별 초과수익 안정성")}
  {mc("IC", f"{m.ic:.4f}", f"ICIR {m.icir:.3f}", ic_c)}
  {mc("Rank IC", f"{m.rank_ic:.4f}", f"Rank ICIR {m.rank_icir:.3f}", ic_c)}
</div>

<div class="card">
  <div class="st">📈 누적 수익률 (기준 100)</div>
  <canvas id="nav" height="75"></canvas>
</div>
<div class="card">
  <div class="st">🎯 월별 IC / Rank IC</div>
  <canvas id="ic" height="55"></canvas>
  <div style="text-align:center;font-size:11px;color:#94a3b8;margin-top:6px">
    IC&gt;0.03 약한 신호 · IC&gt;0.05 유의미 · ICIR&gt;0.5 안정적
  </div>
</div>

<div class="g2">
  <div class="card">
    <div class="st">📅 월별 수익률</div>
    <div style="max-height:300px;overflow-y:auto">
    <table><thead><tr><th>월</th><th>수익률</th><th>바</th></tr></thead>
    <tbody>{monthly_rows}</tbody></table></div>
  </div>
  <div class="card">
    <div class="st">💼 최근 거래 (30건)</div>
    <div style="max-height:300px;overflow-y:auto">
    <table><thead><tr><th>날짜</th><th>종목</th><th>구분</th><th>가격</th><th>금액</th></tr></thead>
    <tbody>{trades_html or "<tr><td colspan=5 style='text-align:center;color:#94a3b8'>없음</td></tr>"}</tbody>
    </table></div>
  </div>
</div>

<div style="text-align:center;font-size:11px;color:#94a3b8;margin-top:6px">
  H-MAS KOSPI · Qlib {cfg.model.upper()} · {date.today()} · 투자 참고용
</div>

<script>
new Chart(document.getElementById('nav'),{{
  type:'line',
  data:{{labels:{json.dumps(nav_labels)},datasets:[{{
    label:'포트폴리오',data:{json.dumps(nav_values)},
    borderColor:'#2563eb',backgroundColor:'rgba(37,99,235,0.05)',
    borderWidth:2,pointRadius:0,fill:true,tension:0.1
  }}]}},
  options:{{responsive:true,plugins:{{legend:{{position:'top'}}}},
    scales:{{y:{{grid:{{color:'#f1f5f9'}}}},x:{{grid:{{display:false}},ticks:{{maxTicksLimit:12}}}}}}}}
}});
const icData={json.dumps(m.ic_monthly)};
new Chart(document.getElementById('ic'),{{
  type:'bar',
  data:{{
    labels:{json.dumps(list(range(1,len(m.ic_monthly)+1)))},
    datasets:[
      {{label:'IC',data:icData,
        backgroundColor:icData.map(v=>v>=0?'rgba(37,99,235,0.6)':'rgba(220,38,38,0.5)'),
        borderWidth:0,order:2}},
      {{label:'Rank IC',data:{json.dumps(m.rank_ic_monthly)},
        type:'line',borderColor:'#f59e0b',backgroundColor:'transparent',
        borderWidth:2,pointRadius:3,order:1}}
    ]
  }},
  options:{{responsive:true,plugins:{{legend:{{position:'top'}}}},
    scales:{{y:{{grid:{{color:'#f1f5f9'}},ticks:{{callback:v=>v.toFixed(2)}}}},
      x:{{grid:{{display:false}}}}}}}}
}});
</script>
</div></body></html>"""


# ──────────────────────────────────────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="H-MAS KOSPI 백테스트 (Qlib 기반)")
    p.add_argument("--model",       choices=["lgbm", "xgb", "lstm"], default="lgbm")
    p.add_argument("--qlib_data",   default="~/.qlib/qlib_data/kr_data",
                   help="build_qlib_kr_data.py로 구축한 Qlib 데이터 경로")
    p.add_argument("--tickers",     nargs="+",
                   default=["005930","000660","005380","035420",
                             "373220","105560","051910","207940"])
    p.add_argument("--train_start", default="2019-01-01")
    p.add_argument("--train_end",   default="2021-12-31")
    p.add_argument("--valid_start", default="2022-01-01")
    p.add_argument("--valid_end",   default="2022-12-31")
    p.add_argument("--test_start",  default="2023-01-01")
    p.add_argument("--test_end",    default="2024-12-31")
    p.add_argument("--topk",        type=int,   default=3)
    p.add_argument("--freq",        choices=["monthly","weekly"], default="monthly")
    p.add_argument("--cash",        type=float, default=100_000_000)
    p.add_argument("--cost",        type=float, default=15.0)
    p.add_argument("--out",         default="backtest_results")
    p.add_argument("--stub",        action="store_true",
                   help="Qlib 데이터 없이 더미 신호로 파이프라인 구조 검증")
    args = p.parse_args()

    cfg = BacktestConfig(
        model=args.model, tickers=args.tickers,
        qlib_data=args.qlib_data,
        train_start=args.train_start, train_end=args.train_end,
        valid_start=args.valid_start, valid_end=args.valid_end,
        test_start=args.test_start,   test_end=args.test_end,
        topk=args.topk, freq=args.freq,
        init_cash=args.cash, cost_bps=args.cost,
        out_dir=args.out, stub=args.stub,
    )

    label = "Stub" if cfg.stub else cfg.model.upper()
    logger.info(f"모델: {label} | Qlib 데이터: {cfg.qlib_data}")
    logger.info(f"테스트: {cfg.test_start}~{cfg.test_end} | Top{cfg.topk}")

    if cfg.stub:
        # Qlib 데이터 없이 더미로 파이프라인 검증
        pred_df, price_df = run_stub(cfg)
    else:
        # 1. Qlib 초기화 + DatasetH(Alpha158) 구성
        dataset = init_qlib_and_dataset(cfg)

        # 2. 모델 학습 + 예측
        if cfg.model == "lgbm":
            pred_df = run_lgbm(dataset, cfg)
        elif cfg.model == "xgb":
            pred_df = run_xgb(dataset, cfg)
        elif cfg.model == "lstm":
            pred_df = run_lstm(dataset, cfg)

        # 3. 포트폴리오 시뮬용 종가 — Qlib D.features에서 로드
        price_df = load_price_from_qlib(cfg)

    # 4. 포트폴리오 시뮬레이션
    logger.info("포트폴리오 시뮬레이션...")
    nav, trades = run_portfolio_sim(pred_df, price_df, cfg)
    if len(nav) < 5:
        logger.error("NAV 생성 실패")
        sys.exit(1)

    # 5. 성과 분석
    logger.info("성과 분석...")
    metrics = compute_metrics(nav, pred_df, price_df)

    # 6. 출력 + 저장
    print_summary(metrics, cfg)
    save_results(nav, pred_df, trades, metrics, cfg)


if __name__ == "__main__":
    main()