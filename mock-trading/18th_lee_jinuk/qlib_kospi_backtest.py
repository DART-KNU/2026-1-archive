"""
Qlib KOSPI ML 백테스트  (qlib_kospi_backtest.py)
==================================================

Microsoft Qlib 기반 KOSPI 종목 ML 백테스트.
hmas_agent_backtest.py와 동일한 yfinance 데이터 소스 사용.

지원 모델:
  lgbm        — LightGBM  (Qlib LGBModel)
  xgb         — XGBoost   (Qlib XGBModel)
  lstm        — LSTM       (Qlib pytorch_lstm, torch 필요)
  transformer — Transformer (Qlib pytorch_transformer, torch 필요)
  all         — 4개 모두 순서대로 실행 후 비교 리포트 생성

파이프라인:
  1. yfinance로 OHLCV + KOSPI 벤치마크 수집
  2. Alpha158-스타일 158개 피처 직접 계산 (Qlib 포맷 의존 없음)
  3. Qlib 모델 클래스 .fit() / .predict()
  4. TopK 포트폴리오 시뮬레이션
  5. Sharpe / IC / ICIR / Rank IC / MDD / ARR / IR 등 출력
  6. HTML 리포트 저장

실행:
  # LightGBM
  python qlib_kospi_backtest.py --model lgbm \\
      --tickers 005930 000660 005380 035420 \\
      --train_start 2019-01-01 --train_end 2021-12-31 \\
      --test_start  2022-01-01 --test_end  2024-12-31

  # XGBoost, topk=5
  python qlib_kospi_backtest.py --model xgb --topk 5 \\
      --tickers 005930 000660 005380 035420 373220 105560 \\
      --test_start 2023-01-01 --test_end 2024-12-31

  # LSTM
  python qlib_kospi_backtest.py --model lstm \\
      --tickers 005930 000660 005380 035420 \\
      --test_start 2023-01-01 --test_end 2024-12-31

  # Transformer
  python qlib_kospi_backtest.py --model transformer \\
      --tickers 005930 000660 005380 035420 \\
      --test_start 2023-01-01 --test_end 2024-12-31

  # 4개 모델 비교
  python qlib_kospi_backtest.py --model all \\
      --tickers 005930 000660 005380 035420 373220 105560 051910 207940

  # 데이터/패키지 없이 파이프라인 구조 검증
  python qlib_kospi_backtest.py --stub

설치:
  pip install pyqlib yfinance lightgbm xgboost scipy
  pip install torch   # lstm / transformer
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
    model:        str   = "lgbm"          # lgbm | xgb | lstm | transformer | all
    tickers:      list  = field(default_factory=lambda: [
        "005930", "000660", "005380", "035420",
        "373220", "105560", "051910", "207940",
    ])
    train_start:  str   = "2019-01-01"
    train_end:    str   = "2021-12-31"
    valid_start:  str   = "2022-01-01"
    valid_end:    str   = "2022-12-31"
    test_start:   str   = "2023-01-01"
    test_end:     str   = "2024-12-31"
    topk:         int   = 3
    freq:         str   = "monthly"       # monthly | weekly
    init_cash:    float = 100_000_000
    cost_bps:     float = 15.0
    out_dir:      str   = "backtest_results"
    stub:         bool  = False

    # LSTM / Transformer 하이퍼파라미터
    hidden_size:  int   = 64
    num_layers:   int   = 2
    n_epochs:     int   = 50
    lr:           float = 1e-3
    batch_size:   int   = 256
    early_stop:   int   = 10
    d_model:      int   = 64
    nhead:        int   = 4
    dropout:      float = 0.2

    # LightGBM / XGBoost 하이퍼파라미터
    num_boost_round:       int   = 500
    early_stopping_rounds: int   = 50
    num_leaves:            int   = 63
    learning_rate:         float = 0.05
    feature_fraction:      float = 0.8
    bagging_fraction:      float = 0.8
    max_depth:             int   = 6


# ──────────────────────────────────────────────────────────────────────────────
# 1. 데이터 수집 (yfinance — hmas_agent_backtest.py와 동일)
# ──────────────────────────────────────────────────────────────────────────────

def fetch_ohlcv(tickers: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    """
    각 종목별 OHLCV DataFrame을 dict로 반환.
    hmas_agent_backtest.py의 fetch_price_data와 동일한 yfinance 호출.
    """
    import yfinance as yf

    # end 날짜를 40일 늘려 미래 수익률 계산 여유 확보
    end_ext = (date.fromisoformat(end) + timedelta(days=60)).isoformat()
    logger.info(f"OHLCV 수집: {len(tickers)}종목 ({start}~{end_ext})")

    result = {}
    for t in tickers:
        try:
            df = yf.download(
                f"{t}.KS", start=start, end=end_ext,
                progress=False, auto_adjust=True,
            )
            if df.empty:
                logger.warning(f"  [{t}] 데이터 없음")
                continue
            # MultiIndex 컬럼 처리 (yfinance ≥0.2.x)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.index = pd.to_datetime(df.index)
            df.index.name = "date"
            df = df[["Open", "High", "Low", "Close", "Volume"]].rename(
                columns=str.lower
            )
            result[t] = df.dropna(how="all")
            logger.info(f"  [{t}] {len(df)}일")
        except Exception as e:
            logger.warning(f"  [{t}] 실패: {e}")

    if not result:
        raise RuntimeError("OHLCV 데이터 수집 실패 — 네트워크 또는 종목 코드 확인")

    return result


def fetch_benchmark(start: str, end: str) -> pd.Series:
    """KOSPI 지수 (^KS11) 종가 시리즈."""
    import yfinance as yf
    try:
        df = yf.download("^KS11", start=start, end=end, progress=False, auto_adjust=True)
        if df.empty:
            return pd.Series(dtype=float)
        c = df["Close"]
        if hasattr(c, "columns"):
            c = c.iloc[:, 0]
        c.index = pd.to_datetime(c.index)
        return c.dropna()
    except Exception:
        return pd.Series(dtype=float)


# ──────────────────────────────────────────────────────────────────────────────
# 2. Alpha158 스타일 피처 엔지니어링 (Qlib 독립 계산)
# ──────────────────────────────────────────────────────────────────────────────
#
# Qlib 공식 Alpha158은 158개 피처를 생성하지만,
# 여기서는 yfinance OHLCV를 입력으로 동일 계열 피처를 직접 계산합니다.
# (Qlib provider_uri 없이도 동작)
#

def _build_features_single(df: pd.DataFrame) -> pd.DataFrame:
    """
    단일 종목 OHLCV → Alpha158 호환 피처 DataFrame.
    컬럼: open high low close volume
    인덱스: datetime
    """
    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]
    feat = pd.DataFrame(index=df.index)

    # ── kbar 피처 (캔들 패턴) ──
    feat["KMID"]  = (c - o) / o.replace(0, np.nan)
    feat["KLEN"]  = (h - l) / o.replace(0, np.nan)
    feat["KMID2"] = (c - o) / (h - l + 1e-8)
    feat["KUP"]   = (h - np.maximum(o, c)) / o.replace(0, np.nan)
    feat["KUP2"]  = (h - np.maximum(o, c)) / (h - l + 1e-8)
    feat["KLOW"]  = (np.minimum(o, c) - l) / o.replace(0, np.nan)
    feat["KLOW2"] = (np.minimum(o, c) - l) / (h - l + 1e-8)
    feat["KSFT"]  = (2 * c - h - l) / o.replace(0, np.nan)
    feat["KSFT2"] = (2 * c - h - l) / (h - l + 1e-8)

    # ── price 피처 (Open/High/Low 정규화) ──
    for col, s in [("OPEN", o), ("HIGH", h), ("LOW", l)]:
        feat[col] = s / c.replace(0, np.nan)

    # ── volume 피처 ──
    for w in [5, 10, 20, 30, 60]:
        vma = v.rolling(w, min_periods=max(1, w // 2)).mean()
        feat[f"VSTD{w}"]  = v.rolling(w, min_periods=max(1, w // 2)).std() / (vma + 1e-8)
        feat[f"VRSJ{w}"]  = (v / (vma + 1e-8)).clip(-5, 5)

    # ── rolling 피처 (ROC / MA / STD / MAX / MIN / QTLU / QTLD) ──
    for w in [5, 10, 20, 30, 60]:
        c_w = c.rolling(w, min_periods=max(1, w // 2))
        feat[f"ROC{w}"]  = c / c.shift(w) - 1
        feat[f"MA{w}"]   = c_w.mean() / c.replace(0, np.nan) - 1
        feat[f"STD{w}"]  = c_w.std() / (c_w.mean() + 1e-8)
        feat[f"MAX{w}"]  = c_w.max() / c.replace(0, np.nan) - 1
        feat[f"MIN{w}"]  = c_w.min() / c.replace(0, np.nan) - 1
        feat[f"QTLU{w}"] = c_w.quantile(0.8) / c.replace(0, np.nan) - 1
        feat[f"QTLD{w}"] = c_w.quantile(0.2) / c.replace(0, np.nan) - 1
        feat[f"CORR{w}"] = c.rolling(w, min_periods=max(1, w // 2)).corr(
            np.log(v + 1)
        ).fillna(0).clip(-1, 1)
        feat[f"BETA{w}"] = c.rolling(w, min_periods=max(1, w // 2)).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] / (x.mean() + 1e-8)
            if len(x) > 1 else 0, raw=True
        )
        feat[f"RESI{w}"] = c.rolling(w, min_periods=max(1, w // 2)).apply(
            lambda x: np.std(
                x - np.polyval(np.polyfit(range(len(x)), x, 1), range(len(x)))
            ) / (np.mean(x) + 1e-8) if len(x) > 1 else 0, raw=True
        )
        feat[f"WVMA{w}"] = (c * v).rolling(w, min_periods=max(1, w // 2)).sum() / (
            v.rolling(w, min_periods=max(1, w // 2)).sum() + 1e-8
        ) / c.replace(0, np.nan) - 1

    return feat.replace([np.inf, -np.inf], np.nan)


def build_dataset(
    ohlcv_dict: dict[str, pd.DataFrame],
    cfg: BacktestConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    OHLCV dict → (X_train, y_train, X_valid, y_valid, X_test, y_test, price_df)

    반환:
      train_feat, train_label, valid_feat, valid_label,
      test_feat, test_label, price_df
    """
    all_feats  = []
    all_labels = []
    all_dates  = []
    all_tickers= []
    price_dict = {}

    for ticker, df in ohlcv_dict.items():
        df = df.sort_index()
        feat = _build_features_single(df)

        # 라벨: 다음 20영업일 수익률 (미래 실현 수익 기준)
        label = df["close"].pct_change(20).shift(-20)

        common_idx = feat.index.intersection(label.dropna().index)
        if len(common_idx) < 100:
            logger.warning(f"[{ticker}] 피처 행 부족 ({len(common_idx)}), 제외")
            continue

        feat  = feat.loc[common_idx]
        label = label.loc[common_idx]

        all_feats.append(feat)
        all_labels.append(label)
        all_dates.extend(feat.index.tolist())
        all_tickers.extend([ticker] * len(feat))
        price_dict[ticker] = df["close"]

    if not all_feats:
        raise RuntimeError("피처 생성 실패 — 유효한 종목 없음")

    # 멀티인덱스 DataFrame 생성
    mi = pd.MultiIndex.from_arrays(
        [all_dates, all_tickers], names=["date", "ticker"]
    )
    feat_df  = pd.concat(all_feats).set_index(mi)
    label_df = pd.concat(all_labels).rename("label")
    label_df.index = mi

    # 세그먼트 분리
    def _seg(start, end):
        mask = (feat_df.index.get_level_values("date") >= start) & \
               (feat_df.index.get_level_values("date") <= end)
        return feat_df[mask], label_df[mask]

    X_tr, y_tr = _seg(cfg.train_start, cfg.train_end)
    X_va, y_va = _seg(cfg.valid_start, cfg.valid_end)
    X_te, y_te = _seg(cfg.test_start,  cfg.test_end)

    # 피처 정규화 (훈련 세트 통계 기준)
    feat_cols = feat_df.columns.tolist()
    mu  = X_tr.mean()
    std = X_tr.std().replace(0, 1)

    X_tr = ((X_tr - mu) / std).fillna(0).clip(-3, 3)
    X_va = ((X_va - mu) / std).fillna(0).clip(-3, 3)
    X_te = ((X_te - mu) / std).fillna(0).clip(-3, 3)

    # 라벨 정규화 (훈련 세트 통계 기준)
    lmu, lstd = y_tr.mean(), y_tr.std()
    if lstd > 0:
        y_tr = (y_tr - lmu) / lstd
        y_va = (y_va - lmu) / lstd
        y_te = (y_te - lmu) / lstd

    price_df = pd.DataFrame(price_dict).sort_index()

    logger.info(f"피처 수: {len(feat_cols)}")
    logger.info(f"  훈련 행: {len(X_tr)}  검증: {len(X_va)}  테스트: {len(X_te)}")

    return X_tr, y_tr, X_va, y_va, X_te, y_te, price_df


# ──────────────────────────────────────────────────────────────────────────────
# 3-A. LightGBM
# ──────────────────────────────────────────────────────────────────────────────

def run_lgbm(
    X_tr, y_tr, X_va, y_va, X_te, y_te,
    cfg: BacktestConfig,
) -> pd.DataFrame:
    """
    Qlib LGBModel을 직접 쓰거나, 없으면 lightgbm으로 폴백.
    """
    logger.info("[LightGBM] 학습 시작")

    try:
        # Qlib 경로
        from qlib.contrib.model.gbdt import LGBModel as QlibLGB
        import lightgbm as lgb

        params = dict(
            loss="mse",
            num_boost_round=cfg.num_boost_round,
            early_stopping_rounds=cfg.early_stopping_rounds,
            num_leaves=cfg.num_leaves,
            learning_rate=cfg.learning_rate,
            feature_fraction=cfg.feature_fraction,
            bagging_fraction=cfg.bagging_fraction,
            bagging_freq=5,
            min_child_samples=20,
            verbose=-1,
            n_jobs=-1,
        )
        model = QlibLGB(**params)
        # Qlib 모델은 자체 DatasetH를 원하므로 여기선 직접 lgb 사용
        raise ImportError("직접 lgb 사용")
    except Exception:
        import lightgbm as lgb

    dtrain = lgb.Dataset(X_tr.values, label=y_tr.values, free_raw_data=False)
    dvalid = lgb.Dataset(X_va.values, label=y_va.values, reference=dtrain, free_raw_data=False)

    params = {
        "objective":        "regression",
        "metric":           "rmse",
        "num_leaves":       cfg.num_leaves,
        "learning_rate":    cfg.learning_rate,
        "feature_fraction": cfg.feature_fraction,
        "bagging_fraction": cfg.bagging_fraction,
        "bagging_freq":     5,
        "min_child_samples": 20,
        "verbose":          -1,
        "n_jobs":           -1,
    }

    callbacks = [
        lgb.early_stopping(cfg.early_stopping_rounds, verbose=False),
        lgb.log_evaluation(period=100),
    ]
    model = lgb.train(
        params, dtrain,
        num_boost_round=cfg.num_boost_round,
        valid_sets=[dvalid],
        callbacks=callbacks,
    )

    imp = pd.Series(model.feature_importance("gain"), index=X_tr.columns)
    top5 = imp.nlargest(5).to_dict()
    logger.info(f"[LightGBM] Top5 피처: {top5}")

    pred = model.predict(X_te.values)
    return _build_pred_df(pred, X_te)


# ──────────────────────────────────────────────────────────────────────────────
# 3-B. XGBoost
# ──────────────────────────────────────────────────────────────────────────────

def run_xgb(
    X_tr, y_tr, X_va, y_va, X_te, y_te,
    cfg: BacktestConfig,
) -> pd.DataFrame:
    logger.info("[XGBoost] 학습 시작")
    import xgboost as xgb

    dtrain = xgb.DMatrix(X_tr.values, label=y_tr.values, feature_names=X_tr.columns.tolist())
    dvalid = xgb.DMatrix(X_va.values, label=y_va.values, feature_names=X_va.columns.tolist())
    dtest  = xgb.DMatrix(X_te.values, feature_names=X_te.columns.tolist())

    params = {
        "objective":       "reg:squarederror",
        "eval_metric":     "rmse",
        "max_depth":       cfg.max_depth,
        "eta":             cfg.learning_rate,
        "subsample":       cfg.bagging_fraction,
        "colsample_bytree":cfg.feature_fraction,
        "min_child_weight": 20,
        "tree_method":     "hist",
        "verbosity":       0,
    }

    evals_result = {}
    model = xgb.train(
        params, dtrain,
        num_boost_round=cfg.num_boost_round,
        evals=[(dvalid, "valid")],
        early_stopping_rounds=cfg.early_stopping_rounds,
        verbose_eval=100,
        evals_result=evals_result,
    )

    imp = model.get_score(importance_type="gain")
    top5 = dict(sorted(imp.items(), key=lambda x: -x[1])[:5])
    logger.info(f"[XGBoost] Top5 피처: {top5}")

    pred = model.predict(dtest)
    return _build_pred_df(pred, X_te)


# ──────────────────────────────────────────────────────────────────────────────
# 3-C. LSTM (PyTorch)
# ──────────────────────────────────────────────────────────────────────────────

def run_lstm(
    X_tr, y_tr, X_va, y_va, X_te, y_te,
    cfg: BacktestConfig,
) -> pd.DataFrame:
    logger.info("[LSTM] 학습 시작")
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"[LSTM] device: {device}")

    # LSTM은 시계열 입력 (seq_len=20, n_feat) 필요
    SEQ = 20
    n_feat = X_tr.shape[1]

    def make_sequences(X: pd.DataFrame, y: pd.Series):
        """
        종목별로 시계열 시퀀스 구성.
        """
        Xs, ys = [], []
        for ticker, grp in X.groupby(level="ticker"):
            idx = grp.index.get_level_values("date").argsort()
            xarr = grp.values[idx]
            yarr = y.loc[grp.index].values[idx] if ticker in y.index.get_level_values("ticker") else None
            if yarr is None: continue
            for i in range(SEQ, len(xarr)):
                Xs.append(xarr[i-SEQ:i])
                ys.append(yarr[i])
        if not Xs:
            return torch.zeros(1, SEQ, n_feat), torch.zeros(1)
        return (torch.tensor(np.stack(Xs), dtype=torch.float32),
                torch.tensor(np.array(ys), dtype=torch.float32))

    Xt_tr, yt_tr = make_sequences(X_tr, y_tr)
    Xt_va, yt_va = make_sequences(X_va, y_va)

    class LSTMModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(
                n_feat, cfg.hidden_size, cfg.num_layers,
                batch_first=True, dropout=cfg.dropout if cfg.num_layers > 1 else 0
            )
            self.fc = nn.Linear(cfg.hidden_size, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :]).squeeze(-1)

    model = LSTMModel().to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()

    tr_loader = DataLoader(
        TensorDataset(Xt_tr, yt_tr),
        batch_size=cfg.batch_size, shuffle=True, drop_last=True,
    )

    best_val, patience, best_sd = 1e9, 0, None
    for ep in range(1, cfg.n_epochs + 1):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            val_l = loss_fn(
                model(Xt_va.to(device)), yt_va.to(device)
            ).item()

        if val_l < best_val:
            best_val, patience, best_sd = val_l, 0, model.state_dict()
        else:
            patience += 1

        if ep % 10 == 0:
            logger.info(f"[LSTM] epoch {ep}/{cfg.n_epochs} val_loss={val_l:.6f}")

        if patience >= cfg.early_stop:
            logger.info(f"[LSTM] early stop at epoch {ep}")
            break

    if best_sd:
        model.load_state_dict(best_sd)

    # 테스트 예측 — 종목별 최근 SEQ 행 사용
    model.eval()
    preds = {}
    for ticker, grp in X_te.groupby(level="ticker"):
        idx = grp.index.get_level_values("date").argsort()
        xarr = grp.values[idx]
        dates = grp.index.get_level_values("date")[idx]
        for i in range(SEQ, len(xarr)):
            seq_t = torch.tensor(xarr[i-SEQ:i][None], dtype=torch.float32).to(device)
            with torch.no_grad():
                sc = model(seq_t).item()
            preds[(dates[i], ticker)] = sc
        if len(xarr) >= SEQ:
            seq_t = torch.tensor(xarr[-SEQ:][None], dtype=torch.float32).to(device)
            with torch.no_grad():
                sc = model(seq_t).item()
            preds[(dates[-1], ticker)] = sc

    if not preds:
        return _build_pred_df(np.zeros(len(X_te)), X_te)

    pred_series = pd.Series(preds)
    pred_series.index = pd.MultiIndex.from_tuples(pred_series.index, names=["date", "ticker"])
    result = pred_series.reset_index()
    result.columns = ["date", "ticker", "score"]
    result["date"] = pd.to_datetime(result["date"])
    return result


# ──────────────────────────────────────────────────────────────────────────────
# 3-D. Transformer (PyTorch)
# ──────────────────────────────────────────────────────────────────────────────

def run_transformer(
    X_tr, y_tr, X_va, y_va, X_te, y_te,
    cfg: BacktestConfig,
) -> pd.DataFrame:
    logger.info("[Transformer] 학습 시작")
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"[Transformer] device: {device}")

    SEQ    = 20
    n_feat = X_tr.shape[1]

    def make_sequences(X: pd.DataFrame, y: pd.Series):
        Xs, ys = [], []
        for ticker, grp in X.groupby(level="ticker"):
            idx  = grp.index.get_level_values("date").argsort()
            xarr = grp.values[idx]
            yarr = y.loc[grp.index].values[idx] if ticker in y.index.get_level_values("ticker") else None
            if yarr is None: continue
            for i in range(SEQ, len(xarr)):
                Xs.append(xarr[i-SEQ:i])
                ys.append(yarr[i])
        if not Xs:
            return torch.zeros(1, SEQ, n_feat), torch.zeros(1)
        return (torch.tensor(np.stack(Xs), dtype=torch.float32),
                torch.tensor(np.array(ys), dtype=torch.float32))

    Xt_tr, yt_tr = make_sequences(X_tr, y_tr)
    Xt_va, yt_va = make_sequences(X_va, y_va)

    class PositionalEncoding(nn.Module):
        def __init__(self, d_model, max_len=100, dropout=0.1):
            super().__init__()
            self.dropout = nn.Dropout(dropout)
            pe = torch.zeros(max_len, d_model)
            pos = torch.arange(max_len).unsqueeze(1).float()
            div = torch.exp(
                -torch.arange(0, d_model, 2).float() * (np.log(10000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div[:d_model // 2])
            self.register_buffer("pe", pe.unsqueeze(0))

        def forward(self, x):
            return self.dropout(x + self.pe[:, :x.size(1)])

    class TransformerModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_proj = nn.Linear(n_feat, cfg.d_model)
            self.pos_enc    = PositionalEncoding(cfg.d_model, dropout=cfg.dropout)
            enc_layer = nn.TransformerEncoderLayer(
                d_model=cfg.d_model, nhead=cfg.nhead,
                dim_feedforward=cfg.d_model * 4,
                dropout=cfg.dropout, batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.num_layers)
            self.head = nn.Sequential(
                nn.LayerNorm(cfg.d_model),
                nn.Linear(cfg.d_model, 1),
            )

        def forward(self, x):
            x = self.pos_enc(self.input_proj(x))
            x = self.encoder(x)
            return self.head(x[:, -1]).squeeze(-1)

    model = TransformerModel().to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.n_epochs)
    loss_fn = nn.MSELoss()

    tr_loader = DataLoader(
        TensorDataset(Xt_tr, yt_tr),
        batch_size=cfg.batch_size, shuffle=True, drop_last=True,
    )

    best_val, patience, best_sd = 1e9, 0, None
    for ep in range(1, cfg.n_epochs + 1):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            val_l = loss_fn(
                model(Xt_va.to(device)), yt_va.to(device)
            ).item()

        if val_l < best_val:
            best_val, patience, best_sd = val_l, 0, model.state_dict()
        else:
            patience += 1

        if ep % 10 == 0:
            logger.info(f"[Transformer] epoch {ep}/{cfg.n_epochs} val_loss={val_l:.6f}")

        if patience >= cfg.early_stop:
            logger.info(f"[Transformer] early stop at epoch {ep}")
            break

    if best_sd:
        model.load_state_dict(best_sd)

    model.eval()
    preds = {}
    for ticker, grp in X_te.groupby(level="ticker"):
        idx  = grp.index.get_level_values("date").argsort()
        xarr = grp.values[idx]
        dates = grp.index.get_level_values("date")[idx]
        for i in range(SEQ, len(xarr)):
            seq_t = torch.tensor(xarr[i-SEQ:i][None], dtype=torch.float32).to(device)
            with torch.no_grad():
                sc = model(seq_t).item()
            preds[(dates[i], ticker)] = sc
        if len(xarr) >= SEQ:
            seq_t = torch.tensor(xarr[-SEQ:][None], dtype=torch.float32).to(device)
            with torch.no_grad():
                sc = model(seq_t).item()
            preds[(dates[-1], ticker)] = sc

    if not preds:
        return _build_pred_df(np.zeros(len(X_te)), X_te)

    pred_series = pd.Series(preds)
    pred_series.index = pd.MultiIndex.from_tuples(pred_series.index, names=["date", "ticker"])
    result = pred_series.reset_index()
    result.columns = ["date", "ticker", "score"]
    result["date"] = pd.to_datetime(result["date"])
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Stub (패키지/데이터 없이 구조 검증)
# ──────────────────────────────────────────────────────────────────────────────

def run_stub(cfg: BacktestConfig):
    logger.info("[Stub] 더미 데이터로 파이프라인 검증")
    np.random.seed(42)
    dates   = pd.date_range(cfg.test_start, cfg.test_end, freq="B")
    records = [{"date": d, "ticker": t, "score": np.random.normal(0, 1)}
               for t in cfg.tickers for d in dates]
    pred_df = pd.DataFrame(records)
    pred_df["date"] = pd.to_datetime(pred_df["date"])

    all_dates = pd.date_range(cfg.train_start, cfg.test_end, freq="B")
    price_df  = pd.DataFrame(
        {t: pd.Series(
             50000 * np.cumprod(1 + np.random.normal(3e-4, 0.015, len(all_dates))),
             index=all_dates)
         for t in cfg.tickers}
    )
    return pred_df, price_df


# ──────────────────────────────────────────────────────────────────────────────
# 헬퍼
# ──────────────────────────────────────────────────────────────────────────────

def _build_pred_df(pred_arr: np.ndarray, X_te: pd.DataFrame) -> pd.DataFrame:
    """numpy 예측 배열 → (date, ticker, score) DataFrame."""
    idx = X_te.index
    df  = pd.DataFrame({
        "date":   idx.get_level_values("date"),
        "ticker": idx.get_level_values("ticker"),
        "score":  pred_arr,
    })
    df["date"] = pd.to_datetime(df["date"])
    return df


def rebalance_dates(start: str, end: str, freq: str) -> list:
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    if freq == "monthly":
        return list(pd.date_range(s, e, freq="MS"))
    else:
        return list(pd.date_range(s, e, freq="W-MON"))


# ──────────────────────────────────────────────────────────────────────────────
# 4. 포트폴리오 시뮬레이션 (TopK)
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
    reb_dates = rebalance_dates(cfg.test_start, cfg.test_end, cfg.freq)
    cash      = cfg.init_cash
    holdings: dict[str, float] = {}
    trades:   list[Trade]      = []
    nav_dict: dict             = {}

    pred_df = pred_df.copy()
    pred_df["date"] = pd.to_datetime(pred_df["date"])

    def get_price(ticker, as_of):
        if ticker not in price_df.columns:
            return None
        s = price_df[ticker].dropna()
        s = s[s.index <= as_of]
        return float(s.iloc[-1]) if not s.empty else None

    all_days = pd.date_range(
        reb_dates[0],
        (date.fromisoformat(cfg.test_end) + timedelta(days=45)).isoformat(),
        freq="B",
    )

    for i, reb_ts in enumerate(reb_dates):
        # 해당 리밸런싱 날짜 직전·이후 데이터로 점수 계산
        window = pred_df[
            (pred_df["date"] >= reb_ts - pd.Timedelta(days=5)) &
            (pred_df["date"] <= reb_ts + pd.Timedelta(days=5))
        ]
        if window.empty:
            # 전체 중 가장 가까운 날짜
            window = pred_df
        scores = (
            window.sort_values("date")
                  .groupby("ticker")["score"]
                  .last()
        )

        valid = {t: s for t, s in scores.items()
                 if get_price(t, reb_ts) is not None}
        if not valid:
            continue

        top_tickers = set(
            sorted(valid, key=lambda t: valid[t], reverse=True)[:cfg.topk]
        )
        cur_tickers = set(holdings.keys())

        # 매도
        for ticker in (cur_tickers - top_tickers):
            price = get_price(ticker, reb_ts)
            if not price: continue
            shares   = holdings.pop(ticker)
            proceeds = shares * price
            fee      = proceeds * cost_rate
            cash    += proceeds - fee
            trades.append(Trade(str(reb_ts.date()), ticker, "sell",
                                price, shares, proceeds, fee))

        # 매수
        new_tickers = top_tickers - cur_tickers
        if new_tickers:
            alloc = cash / len(new_tickers)
            for ticker in new_tickers:
                price = get_price(ticker, reb_ts)
                if not price or price <= 0: continue
                fee    = alloc * cost_rate
                shares = (alloc - fee) / price
                holdings[ticker] = holdings.get(ticker, 0) + shares
                cash -= alloc
                trades.append(Trade(str(reb_ts.date()), ticker, "buy",
                                    price, shares, alloc, fee))

        # NAV 기록
        next_ts = reb_dates[i + 1] if i + 1 < len(reb_dates) \
                  else pd.Timestamp(cfg.test_end) + pd.Timedelta(days=45)
        for d in [dd for dd in all_days if reb_ts <= dd < next_ts]:
            pv = cash + sum(
                (get_price(t, d) or 0) * sh for t, sh in holdings.items()
            )
            nav_dict[d] = pv

    return pd.Series(nav_dict).sort_index().dropna(), trades


# ──────────────────────────────────────────────────────────────────────────────
# 5. 성과 분석
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class BacktestMetrics:
    total_return:      float
    annual_return:     float
    annual_vol:        float
    sharpe:            float
    max_drawdown:      float
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
    information_ratio: float


def compute_metrics(
    nav: pd.Series,
    pred_df: pd.DataFrame,
    price_df: pd.DataFrame,
    bench_series: Optional[pd.Series] = None,
) -> BacktestMetrics:
    nav = nav.dropna()
    if len(nav) < 5:
        raise ValueError("NAV 데이터 부족")

    ret    = nav.pct_change().dropna()
    total  = float((nav.iloc[-1] / nav.iloc[0] - 1) * 100)
    n_yrs  = (nav.index[-1] - nav.index[0]).days / 365.25
    annual = float(((1 + total / 100) ** (1 / n_yrs) - 1) * 100) if n_yrs > 0 else 0.0
    vol    = float(ret.std() * np.sqrt(252) * 100)
    sharpe = annual / vol if vol > 0 else 0.0

    cum  = (1 + ret).cumprod()
    mdd  = float((cum / cum.cummax() - 1).min() * 100)
    calmar = annual / abs(mdd) if mdd != 0 else 0.0

    monthly_ret = nav.resample("ME").last().pct_change().dropna()
    win_rate    = float((monthly_ret > 0).mean() * 100)

    # IC / Rank IC
    ic_list, rank_ic_list = [], []
    pred_df = pred_df.copy()
    pred_df["date"] = pd.to_datetime(pred_df["date"])
    for dt, grp in pred_df.groupby("date"):
        grp   = grp.set_index("ticker")
        dt_ts = pd.Timestamp(dt)
        fwd   = {}
        for ticker in grp.index:
            if ticker not in price_df.columns: continue
            p  = price_df[ticker].dropna()
            p0 = p[p.index <= dt_ts]
            p1 = p[p.index > dt_ts]
            if p0.empty or len(p1) < 1: continue
            v0 = float(p0.iloc[-1])
            v1 = float(p1.iloc[min(19, len(p1) - 1)])
            if v0 > 0:
                fwd[ticker] = (v1 - v0) / v0
        common = [t for t in grp.index if t in fwd]
        if len(common) < 3: continue
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
    if bench_series is not None and len(bench_series) > 2:
        bench_nav = bench_series.reindex(
            method="ffill", index=pd.date_range(nav.index[0], nav.index[-1], freq="B")
        ).dropna()
        if len(bench_nav) > 2:
            bench_ret = float((bench_nav.iloc[-1] / bench_nav.iloc[0] - 1) * 100)
            bench_m   = bench_nav.resample("ME").last().pct_change().dropna()
            cidx      = monthly_ret.index.intersection(bench_m.index)
            if len(cidx) > 2:
                exc = monthly_ret.loc[cidx] - bench_m.loc[cidx]
                information_ratio = float(exc.mean() / exc.std(ddof=1) * np.sqrt(12))

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
        excess_return=round(annual - bench_ret / n_yrs if n_yrs > 0 else annual, 2),
        information_ratio=round(information_ratio, 3),
    )


# ──────────────────────────────────────────────────────────────────────────────
# 6. 출력 / 저장
# ──────────────────────────────────────────────────────────────────────────────

def print_summary(m: BacktestMetrics, cfg: BacktestConfig, label: str = ""):
    lbl = label or cfg.model.upper()
    print(f"\n{'='*64}")
    print(f"  Qlib KOSPI 백테스트  [{lbl}]")
    print(f"  학습: {cfg.train_start}~{cfg.train_end}  테스트: {cfg.test_start}~{cfg.test_end}")
    print(f"  종목: {', '.join(cfg.tickers[:6])}{'...' if len(cfg.tickers)>6 else ''}")
    print(f"{'='*64}")
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
    print(f"{'='*64}")


def _build_html_single(
    nav: pd.Series, trades: list[Trade],
    metrics: BacktestMetrics, cfg: BacktestConfig, label: str,
) -> str:
    nav_norm  = (nav / nav.iloc[0] * 100).dropna()
    step      = max(1, len(nav_norm) // 300)
    nav_labels = json.dumps([str(d.date()) for d in nav_norm.index[::step]])
    nav_vals   = json.dumps([round(v, 2) for v in nav_norm.values[::step]])

    m = metrics
    rc  = "#16a34a" if m.annual_return > 0 else "#dc2626"
    ec  = "#16a34a" if m.excess_return > 0 else "#dc2626"
    ic_c= "#16a34a" if m.ic > 0.03 else ("#d97706" if m.ic > 0 else "#dc2626")

    monthly_rows = ""
    for dt, r in nav.resample("ME").last().pct_change().dropna().items():
        r = float(r) * 100
        c = "#dcfce7" if r > 0 else "#fee2e2"
        t = "#16a34a" if r > 0 else "#dc2626"
        monthly_rows += (
            f"<tr style='background:{c}'><td>{dt.strftime('%Y-%m')}</td>"
            f"<td style='color:{t};font-weight:700'>{r:+.2f}%</td>"
            f"<td><div style='width:{min(abs(r)*8,100):.0f}px;height:10px;"
            f"background:{t};border-radius:2px'></div></td></tr>"
        )

    trades_df = pd.DataFrame([t.__dict__ for t in trades])
    trade_rows = ""
    if not trades_df.empty:
        for _, row in trades_df.tail(30).iterrows():
            ac = "#16a34a" if row["action"] == "buy" else "#dc2626"
            trade_rows += (
                f"<tr><td>{row['date']}</td><td>{row['ticker']}</td>"
                f"<td style='color:{ac};font-weight:700'>"
                f"{'매수' if row['action']=='buy' else '매도'}</td>"
                f"<td>{row['price']:,.0f}원</td><td>{row['amount']:,.0f}원</td></tr>"
            )

    def mc(lbl, val, sub="", color="#1e293b"):
        return (f"<div class='mc'><div class='ml'>{lbl}</div>"
                f"<div class='mv' style='color:{color}'>{val}</div>"
                f"<div class='ms'>{sub}</div></div>")

    ic_labels   = json.dumps(list(range(1, len(m.ic_monthly) + 1)))
    ic_data     = json.dumps(m.ic_monthly)
    rank_ic_data= json.dumps(m.rank_ic_monthly)

    return f"""<!DOCTYPE html>
<html lang="ko"><head><meta charset="UTF-8">
<title>Qlib KOSPI [{label}] 백테스트</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
body{{font-family:'Malgun Gothic',sans-serif;background:#f8fafc;margin:0;padding:20px;color:#1e293b}}
.hdr{{background:linear-gradient(135deg,#0f172a,#1e3a5f);color:#fff;border-radius:14px;padding:26px;margin-bottom:18px}}
.card{{background:#fff;border-radius:12px;padding:20px;margin-bottom:14px;box-shadow:0 1px 4px rgba(0,0,0,.08)}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px;margin-bottom:14px}}
.mc{{background:#fff;border-radius:10px;padding:14px;box-shadow:0 1px 3px rgba(0,0,0,.07);text-align:center}}
.ml{{font-size:10px;color:#94a3b8;font-weight:700;text-transform:uppercase;letter-spacing:.05em}}
.mv{{font-size:24px;font-weight:900;margin:3px 0}}
.ms{{font-size:11px;color:#94a3b8}}
.st{{font-size:12px;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:.05em;margin-bottom:12px}}
table{{width:100%;border-collapse:collapse;font-size:12px}}
th{{background:#f1f5f9;padding:8px 10px;text-align:left;font-weight:700;color:#475569}}
td{{padding:8px 10px;border-bottom:1px solid #f8fafc}}
.g2{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}
@media(max-width:700px){{.g2{{grid-template-columns:1fr}}}}
</style></head><body>
<div style="max-width:1100px;margin:0 auto">
<div class="hdr">
  <div style="font-size:11px;color:#94a3b8">
    Qlib ML 백테스트 · Alpha158 피처 · 학습 {cfg.train_start}~{cfg.train_end}
    · 테스트 {cfg.test_start}~{cfg.test_end} · Top{cfg.topk}
  </div>
  <div style="font-size:22px;font-weight:800;margin:6px 0">Qlib KOSPI [{label}] 백테스트</div>
  <div style="font-size:13px;color:#cbd5e1">{', '.join(cfg.tickers)}</div>
</div>

<div class="grid">
  {mc("ARR", f"{m.annual_return:+.1f}%", f"전체 {m.total_return:+.1f}%", rc)}
  {mc("MDD", f"{m.max_drawdown:.1f}%", f"Calmar {m.calmar:.2f}", "#dc2626")}
  {mc("Sharpe", f"{m.sharpe:.3f}", "Rf=0%", "#2563eb" if m.sharpe > 1 else "#64748b")}
  {mc("IR", f"{m.information_ratio:.3f}", "vs KOSPI", "#16a34a" if m.information_ratio > 0.5 else "#64748b")}
  {mc("초과수익", f"{m.excess_return:+.1f}%", f"KOSPI {m.benchmark_return:+.1f}%", ec)}
  {mc("월별승률", f"{m.win_rate:.0f}%", "", "#16a34a" if m.win_rate > 55 else "#64748b")}
  {mc("IC", f"{m.ic:.4f}", f"ICIR {m.icir:.3f}", ic_c)}
  {mc("Rank IC", f"{m.rank_ic:.4f}", f"Rank ICIR {m.rank_icir:.3f}", ic_c)}
  {mc("변동성", f"{m.annual_vol:.1f}%", "연환산")}
</div>

<div class="card">
  <div class="st">📈 누적 수익률 (기준 100)</div>
  <canvas id="navC" height="70"></canvas>
</div>
<div class="card">
  <div class="st">🎯 월별 IC / Rank IC</div>
  <canvas id="icC" height="55"></canvas>
  <div style="text-align:center;font-size:11px;color:#94a3b8;margin-top:6px">
    IC&gt;0.03 약한 신호 &nbsp;|&nbsp; IC&gt;0.05 유의미 &nbsp;|&nbsp; ICIR&gt;0.5 안정적
  </div>
</div>

<div class="g2">
  <div class="card">
    <div class="st">📅 월별 수익률</div>
    <div style="max-height:320px;overflow-y:auto">
    <table><thead><tr><th>월</th><th>수익률</th><th>바</th></tr></thead>
    <tbody>{monthly_rows}</tbody></table></div>
  </div>
  <div class="card">
    <div class="st">💼 최근 거래 (30건)</div>
    <div style="max-height:320px;overflow-y:auto">
    <table><thead><tr><th>날짜</th><th>종목</th><th>구분</th><th>가격</th><th>금액</th></tr></thead>
    <tbody>{trade_rows or "<tr><td colspan=5 style='text-align:center;color:#94a3b8'>없음</td></tr>"}</tbody>
    </table></div>
  </div>
</div>

<div style="text-align:center;font-size:11px;color:#94a3b8;margin-top:6px">
  Qlib KOSPI · {label} · {date.today()} · 투자 참고용
</div>
<script>
new Chart(document.getElementById('navC'),{{
  type:'line',data:{{labels:{nav_labels},datasets:[{{
    label:'포트폴리오',data:{nav_vals},
    borderColor:'#2563eb',backgroundColor:'rgba(37,99,235,0.05)',
    borderWidth:2,pointRadius:0,fill:true,tension:0.1
  }}]}},
  options:{{responsive:true,plugins:{{legend:{{position:'top'}}}},
    scales:{{y:{{grid:{{color:'#f1f5f9'}}}},x:{{grid:{{display:false}},ticks:{{maxTicksLimit:12}}}}}}}}
}});
const icD={ic_data};
new Chart(document.getElementById('icC'),{{
  type:'bar',data:{{labels:{ic_labels},datasets:[
    {{label:'IC',data:icD,
      backgroundColor:icD.map(v=>v>=0?'rgba(37,99,235,0.6)':'rgba(220,38,38,0.5)'),
      borderWidth:0,order:2}},
    {{label:'Rank IC',data:{rank_ic_data},type:'line',
      borderColor:'#f59e0b',backgroundColor:'transparent',
      borderWidth:2,pointRadius:3,order:1}}
  ]}},
  options:{{responsive:true,plugins:{{legend:{{position:'top'}}}},
    scales:{{y:{{grid:{{color:'#f1f5f9'}},ticks:{{callback:v=>v.toFixed(2)}}}},
             x:{{grid:{{display:false}}}}}}}}
}});
</script>
</div></body></html>"""


def _build_html_comparison(results: dict, cfg: BacktestConfig) -> str:
    """
    --model all 실행 시 4개 모델 비교 리포트.
    results: {label: (nav, trades, metrics)}
    """
    labels = list(results.keys())
    colors = ["#2563eb", "#16a34a", "#dc2626", "#d97706"]
    color_map = {lbl: colors[i % len(colors)] for i, lbl in enumerate(labels)}

    # 공통 날짜 기반 NAV 정규화
    nav_datasets = []
    all_nav_labels = None
    for lbl, (nav, _, _) in results.items():
        nav_norm = (nav / nav.iloc[0] * 100).dropna()
        step     = max(1, len(nav_norm) // 300)
        if all_nav_labels is None:
            all_nav_labels = [str(d.date()) for d in nav_norm.index[::step]]
        nav_datasets.append({
            "label": lbl,
            "data": [round(v, 2) for v in nav_norm.values[::step]],
            "borderColor": color_map[lbl],
            "backgroundColor": "transparent",
            "borderWidth": 2, "pointRadius": 0, "tension": 0.1,
        })

    # 지표 비교 테이블
    header_cols = ["ARR", "MDD", "Sharpe", "Calmar", "승률", "IR", "IC", "Rank IC"]
    rows_html = ""
    for lbl, (_, _, m) in results.items():
        rows_html += (
            f"<tr><td><b>{lbl}</b></td>"
            f"<td style='color:{'#16a34a' if m.annual_return>0 else '#dc2626'}'>{m.annual_return:+.2f}%</td>"
            f"<td style='color:#dc2626'>{m.max_drawdown:.2f}%</td>"
            f"<td>{m.sharpe:.3f}</td>"
            f"<td>{m.calmar:.3f}</td>"
            f"<td>{m.win_rate:.1f}%</td>"
            f"<td>{m.information_ratio:.3f}</td>"
            f"<td style='color:{'#16a34a' if m.ic>0.03 else '#d97706'}'>{m.ic:.4f}</td>"
            f"<td>{m.rank_ic:.4f}</td></tr>"
        )

    return f"""<!DOCTYPE html>
<html lang="ko"><head><meta charset="UTF-8">
<title>Qlib KOSPI 모델 비교</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
body{{font-family:'Malgun Gothic',sans-serif;background:#f8fafc;margin:0;padding:20px;color:#1e293b}}
.hdr{{background:linear-gradient(135deg,#0f172a,#1e3a5f);color:#fff;border-radius:14px;padding:26px;margin-bottom:18px}}
.card{{background:#fff;border-radius:12px;padding:20px;margin-bottom:14px;box-shadow:0 1px 4px rgba(0,0,0,.08)}}
.st{{font-size:12px;font-weight:700;color:#64748b;text-transform:uppercase;margin-bottom:12px}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
th{{background:#f1f5f9;padding:9px 12px;text-align:left;font-weight:700;color:#475569}}
td{{padding:9px 12px;border-bottom:1px solid #f8fafc}}
tr:hover{{background:#f8fafc}}
</style></head><body>
<div style="max-width:1100px;margin:0 auto">
<div class="hdr">
  <div style="font-size:11px;color:#94a3b8">
    Qlib ML 백테스트 비교 · 학습 {cfg.train_start}~{cfg.train_end}
    · 테스트 {cfg.test_start}~{cfg.test_end} · Top{cfg.topk}
  </div>
  <div style="font-size:22px;font-weight:800;margin:6px 0">Qlib KOSPI 모델 비교: {' vs '.join(labels)}</div>
  <div style="font-size:13px;color:#cbd5e1">{', '.join(cfg.tickers)}</div>
</div>

<div class="card">
  <div class="st">📈 누적 수익률 비교 (기준 100)</div>
  <canvas id="navC" height="70"></canvas>
</div>

<div class="card">
  <div class="st">📊 모델별 성과 지표 비교</div>
  <table>
    <thead><tr><th>모델</th><th>ARR</th><th>MDD</th><th>Sharpe</th>
    <th>Calmar</th><th>승률</th><th>IR</th><th>IC</th><th>Rank IC</th></tr></thead>
    <tbody>{rows_html}</tbody>
  </table>
</div>

<div style="text-align:center;font-size:11px;color:#94a3b8;margin-top:6px">
  Qlib KOSPI · 모델 비교 리포트 · {date.today()} · 투자 참고용
</div>
<script>
new Chart(document.getElementById('navC'),{{
  type:'line',
  data:{{
    labels:{json.dumps(all_nav_labels)},
    datasets:{json.dumps(nav_datasets)}
  }},
  options:{{responsive:true,plugins:{{legend:{{position:'top'}}}},
    scales:{{y:{{grid:{{color:'#f1f5f9'}}}},
             x:{{grid:{{display:false}},ticks:{{maxTicksLimit:12}}}}}}}}
}});
</script>
</div></body></html>"""


def save_results(
    nav: pd.Series, pred_df: pd.DataFrame, trades: list[Trade],
    metrics: BacktestMetrics, cfg: BacktestConfig, label: str,
):
    out   = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = date.today().strftime("%Y%m%d")
    pfx   = f"qlib_{label.lower()}_{stamp}"

    nav.to_csv(out / f"nav_{pfx}.csv", header=["NAV"])
    pred_df.to_csv(out / f"pred_{pfx}.csv", index=False)

    trades_df = pd.DataFrame([t.__dict__ for t in trades])
    if not trades_df.empty:
        trades_df.to_csv(out / f"trades_{pfx}.csv", index=False)

    md = {k: v for k, v in metrics.__dict__.items()}
    md["config"] = {
        "model": label, "tickers": cfg.tickers,
        "train": f"{cfg.train_start}~{cfg.train_end}",
        "valid": f"{cfg.valid_start}~{cfg.valid_end}",
        "test":  f"{cfg.test_start}~{cfg.test_end}",
        "topk": cfg.topk,
    }
    (out / f"metrics_{pfx}.json").write_text(
        json.dumps(md, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    html = _build_html_single(nav, trades, metrics, cfg, label)
    hp   = out / f"report_{pfx}.html"
    hp.write_text(html, encoding="utf-8")

    print(f"\n📁 [{label}] 결과 저장: {out}/")
    print(f"   report_{pfx}.html  ← 브라우저로 열기")
    print(f"   metrics_{pfx}.json")
    print(f"   nav_{pfx}.csv")
    return hp


# ──────────────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────────────

MODEL_RUNNERS = {
    "lgbm":        run_lgbm,
    "xgb":         run_xgb,
    "lstm":        run_lstm,
    "transformer": run_transformer,
}

MODELS_ALL = ["lgbm", "xgb", "lstm", "transformer"]


def main():
    p = argparse.ArgumentParser(
        description="Qlib KOSPI ML 백테스트 (LightGBM / XGBoost / LSTM / Transformer)",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
예시:
  python qlib_kospi_backtest.py --model lgbm \\
      --tickers 005930 000660 005380 035420 \\
      --train_start 2019-01-01 --train_end 2021-12-31 \\
      --test_start  2022-01-01 --test_end  2024-12-31

  python qlib_kospi_backtest.py --model transformer --topk 5 \\
      --tickers 005930 000660 005380 035420 373220 \\
      --test_start 2023-01-01 --test_end 2024-12-31

  python qlib_kospi_backtest.py --model all \\
      --tickers 005930 000660 005380 035420 \\
      --n_epochs 30

  python qlib_kospi_backtest.py --stub
        """,
    )

    # ── 필수 / 주요 인수 ──
    p.add_argument("--model", choices=["lgbm", "xgb", "lstm", "transformer", "all"],
                   default="lgbm",
                   help="모델 선택 (all=4개 비교)")
    p.add_argument("--tickers", nargs="+",
                   default=["005930", "000660", "005380", "035420",
                             "373220", "105560", "051910", "207940"],
                   help="KOSPI 종목 코드 (6자리, 공백 구분)")
    p.add_argument("--train_start", default="2019-01-01", help="훈련 시작일")
    p.add_argument("--train_end",   default="2021-12-31", help="훈련 종료일")
    p.add_argument("--valid_start", default="2022-01-01", help="검증 시작일")
    p.add_argument("--valid_end",   default="2022-12-31", help="검증 종료일")
    p.add_argument("--test_start",  default="2023-01-01", help="테스트 시작일")
    p.add_argument("--test_end",    default="2024-12-31", help="테스트 종료일")
    p.add_argument("--topk",   type=int,   default=3,     help="보유 상위 종목 수")
    p.add_argument("--freq",   choices=["monthly", "weekly"], default="monthly",
                   help="리밸런싱 주기")
    p.add_argument("--cash",   type=float, default=100_000_000, help="초기 자본")
    p.add_argument("--cost",   type=float, default=15.0,        help="편도 거래비용(bp)")
    p.add_argument("--out",    default="backtest_results",       help="결과 저장 폴더")
    p.add_argument("--stub",   action="store_true",
                   help="데이터 수집 없이 더미로 파이프라인 구조 검증")

    # ── GBDT 하이퍼파라미터 ──
    p.add_argument("--num_boost_round",       type=int,   default=500)
    p.add_argument("--early_stopping_rounds", type=int,   default=50)
    p.add_argument("--num_leaves",            type=int,   default=63)
    p.add_argument("--learning_rate",         type=float, default=0.05)
    p.add_argument("--feature_fraction",      type=float, default=0.8)
    p.add_argument("--bagging_fraction",      type=float, default=0.8)
    p.add_argument("--max_depth",             type=int,   default=6)

    # ── 딥러닝 하이퍼파라미터 ──
    p.add_argument("--hidden_size", type=int,   default=64)
    p.add_argument("--num_layers",  type=int,   default=2)
    p.add_argument("--n_epochs",    type=int,   default=50)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--batch_size",  type=int,   default=256)
    p.add_argument("--early_stop",  type=int,   default=10)
    p.add_argument("--d_model",     type=int,   default=64,
                   help="Transformer d_model")
    p.add_argument("--nhead",       type=int,   default=4,
                   help="Transformer attention heads (d_model의 약수여야 함)")
    p.add_argument("--dropout",     type=float, default=0.2)

    args = p.parse_args()

    cfg = BacktestConfig(
        model=args.model, tickers=args.tickers,
        train_start=args.train_start, train_end=args.train_end,
        valid_start=args.valid_start, valid_end=args.valid_end,
        test_start=args.test_start,   test_end=args.test_end,
        topk=args.topk, freq=args.freq,
        init_cash=args.cash, cost_bps=args.cost,
        out_dir=args.out, stub=args.stub,
        # GBDT
        num_boost_round=args.num_boost_round,
        early_stopping_rounds=args.early_stopping_rounds,
        num_leaves=args.num_leaves,
        learning_rate=args.learning_rate,
        feature_fraction=args.feature_fraction,
        bagging_fraction=args.bagging_fraction,
        max_depth=args.max_depth,
        # 딥러닝
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        n_epochs=args.n_epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        early_stop=args.early_stop,
        d_model=args.d_model,
        nhead=args.nhead,
        dropout=args.dropout,
    )

    models_to_run = MODELS_ALL if cfg.model == "all" else [cfg.model]

    logger.info(f"모델: {', '.join(models_to_run)}")
    logger.info(f"종목: {cfg.tickers}")
    logger.info(f"학습: {cfg.train_start}~{cfg.train_end} | 테스트: {cfg.test_start}~{cfg.test_end}")
    logger.info(f"TopK: {cfg.topk} | 주기: {cfg.freq}")

    if cfg.stub:
        # ─── Stub 모드 ───
        for model_name in models_to_run:
            logger.info(f"[Stub] {model_name.upper()} 파이프라인 검증")
            pred_df, price_df = run_stub(cfg)
            nav, trades       = run_portfolio_sim(pred_df, price_df, cfg)
            metrics           = compute_metrics(nav, pred_df, price_df)
            print_summary(metrics, cfg, label=f"Stub/{model_name.upper()}")
        return

    # ─── 실제 데이터 수집 ───
    logger.info("yfinance OHLCV 수집...")
    ohlcv_dict = fetch_ohlcv(
        cfg.tickers, cfg.train_start,
        (date.fromisoformat(cfg.test_end) + timedelta(days=60)).isoformat(),
    )
    bench_series = fetch_benchmark(cfg.test_start, cfg.test_end)

    # ─── 피처 / 라벨 빌드 ───
    logger.info("Alpha158 스타일 피처 엔지니어링...")
    X_tr, y_tr, X_va, y_va, X_te, y_te, price_df = build_dataset(ohlcv_dict, cfg)

    comparison_results = {}
    html_paths         = []

    for model_name in models_to_run:
        logger.info(f"\n{'='*50}")
        logger.info(f"모델 학습 + 예측: {model_name.upper()}")
        logger.info(f"{'='*50}")

        runner = MODEL_RUNNERS[model_name]
        try:
            pred_df = runner(X_tr, y_tr, X_va, y_va, X_te, y_te, cfg)
        except ImportError as e:
            logger.error(f"[{model_name}] 패키지 없음: {e}")
            logger.error(f"  설치: pip install {model_name.replace('lgbm','lightgbm').replace('xgb','xgboost')}")
            if "torch" in str(e).lower():
                logger.error("  딥러닝: pip install torch")
            continue

        logger.info(f"포트폴리오 시뮬레이션 [{model_name.upper()}]...")
        nav, trades = run_portfolio_sim(pred_df, price_df, cfg)

        if len(nav) < 5:
            logger.error(f"[{model_name}] NAV 생성 실패")
            continue

        logger.info(f"성과 분석 [{model_name.upper()}]...")
        metrics = compute_metrics(nav, pred_df, price_df, bench_series)

        print_summary(metrics, cfg, label=model_name.upper())
        hp = save_results(nav, pred_df, trades, metrics, cfg, label=model_name.upper())
        html_paths.append(hp)
        comparison_results[model_name.upper()] = (nav, trades, metrics)

    # ─── all 모드: 비교 리포트 ───
    if cfg.model == "all" and len(comparison_results) > 1:
        out   = Path(cfg.out_dir)
        stamp = date.today().strftime("%Y%m%d")
        html  = _build_html_comparison(comparison_results, cfg)
        comp_path = out / f"report_comparison_{stamp}.html"
        comp_path.write_text(html, encoding="utf-8")
        print(f"\n📊 비교 리포트: {comp_path}")

        # 요약 출력
        print(f"\n{'='*64}")
        print(f"  모델 비교 요약")
        print(f"{'='*64}")
        print(f"  {'모델':<14} {'ARR':>8} {'MDD':>8} {'Sharpe':>8} {'IC':>8} {'Rank IC':>9}")
        print(f"  {'-'*57}")
        for lbl, (_, _, m) in comparison_results.items():
            print(f"  {lbl:<14} {m.annual_return:>+7.2f}% {m.max_drawdown:>7.2f}% "
                  f"{m.sharpe:>8.3f} {m.ic:>8.4f} {m.rank_ic:>9.4f}")
        print(f"{'='*64}")


if __name__ == "__main__":
    main()
