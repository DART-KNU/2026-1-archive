# -*- coding: utf-8 -*-
"""
Minimal Samsung Electronics + SK hynix standalone strategy.

This file is self-contained except for optional KRX credentials in a nearby .env file.
The strategy uses only:
- Samsung core weight driven by KOSPI above/below 60-day moving average
- Hynix tactical overlay driven by ret10 / rs20 / close above ma20
- Buy & Hold benchmarks: Samsung 25% + Hynix 15%, and KOSPI * 0.4

Required packages:
    pip install pykrx pandas numpy openpyxl matplotlib
"""

from __future__ import annotations

import os
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from pykrx import stock
from pykrx.website.comm.auth import build_krx_session
from pykrx.website.comm.webio import set_session

warnings.filterwarnings("ignore")


# =========================================================
# 0. Settings
# =========================================================
START_DATE = "2023-01-02"
END_DATE = pd.Timestamp.now().strftime("%Y-%m-%d")

FIXED_UNIVERSE = [
    {"ticker": "005930", "name": "Samsung Electronics", "market": "KOSPI"},
    {"ticker": "000660", "name": "SK hynix", "market": "KOSPI"},
]

INITIAL_CAPITAL = 100_000_000
FEE_RATE = 0.0015
SLEEP_BETWEEN_CALLS = 0.15
REBALANCE_LOG_MIN_DELTA = 0.05
BENCHMARK_MA_WINDOW = 60

STRATEGY_LABEL = "minimal_core_hynix_overlay"
STRATEGY_DISPLAY_NAME = "Minimal Samsung Core + Hynix Overlay"
STRATEGY_SLEEVE_WEIGHT = 0.40
PORTFOLIO_TICKER_CAPS: Dict[str, float] = {
    "005930": 0.25,
    "000660": 0.15,
}
CAP_EXCESS_HANDLING = "redirect_hynix_to_samsung"
CAP_EXCESS_REDIRECT_MAP: Dict[str, str] = {
    "000660": "005930",
}
MAX_SAMSUNG_STRATEGY_TARGET = PORTFOLIO_TICKER_CAPS["005930"] / STRATEGY_SLEEVE_WEIGHT
MAX_HYNIX_STRATEGY_TARGET = PORTFOLIO_TICKER_CAPS["000660"] / STRATEGY_SLEEVE_WEIGHT

WINNER_PARAMS: Dict[str, float | str] = {
    "variant_name": "minimal_aggressive",
    "model": "minimal_core_hynix_overlay",
    "samsung_core_on": MAX_SAMSUNG_STRATEGY_TARGET,
    "samsung_core_off": 0.30,
    "hynix_base": 0.16,
    "hynix_tactical_scale": 0.36,
    "ret10_threshold": 0.06,
    "rs20_threshold": 0.00,
}

BENCHMARK_FIXED_WEIGHTS: Dict[str, float] = {
    "005930": 0.25,
    "000660": 0.15,
}
BENCHMARK_LABEL = "buy_hold_25_15_benchmark"
BENCHMARK_DISPLAY_NAME = "Benchmark BH 25/15"

KOSPI_BENCHMARK_WEIGHT = 0.40
KOSPI_BENCHMARK_LABEL = "kospi_40_benchmark"
KOSPI_BENCHMARK_DISPLAY_NAME = "KOSPI * 0.4"

OUTPUT_XLSX = "DART/semicon_minimal_core_hynix_overlay_backtest.xlsx"
OUTPUT_PLOT = "DART/semicon_minimal_core_hynix_overlay_backtest.png"
PROJECT_DIR = Path(__file__).resolve().parent


# =========================================================
# 1. Environment / helpers
# =========================================================
def load_local_env(env_path: str | Path | None = None) -> None:
    candidates: List[Path] = []
    if env_path is not None:
        candidates.append(Path(env_path))

    candidates.extend(
        [
            PROJECT_DIR / ".env",
            PROJECT_DIR / "DART" / ".env",
            Path.cwd() / ".env",
            Path.cwd() / "DART" / ".env",
        ]
    )

    seen: set[str] = set()
    for candidate in candidates:
        path = candidate.resolve(strict=False)
        path_key = str(path)
        if path_key in seen or not path.exists():
            continue
        seen.add(path_key)

        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
        return


def initialize_krx_session() -> None:
    load_local_env()

    login_id = os.getenv("KRX_ID")
    login_pw = os.getenv("KRX_PW")
    if not (login_id and login_pw):
        print("[WARN] KRX_ID / KRX_PW environment variables are missing. Trying anonymous session.")
        return

    try:
        session = build_krx_session(login_id=login_id, login_pw=login_pw)
        set_session(session)
        if session is None:
            print("[WARN] Failed to create KRX login session. Falling back to anonymous session.")
        else:
            print("[INFO] KRX login session initialized.")
    except Exception as exc:
        print(f"[WARN] Failed to initialize KRX login session: {exc}")


def normalize_yyyymmdd(value: str) -> str:
    return pd.Timestamp(value).strftime("%Y%m%d")


def safe_to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def reset_date_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.reset_index().copy()
    first_col = out.columns[0]
    out = out.rename(columns={first_col: "date"})
    out["date"] = pd.to_datetime(out["date"])
    return out.sort_values("date").reset_index(drop=True)


def get_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column in df.columns:
        return pd.to_numeric(df[column], errors="coerce")
    return pd.Series(0.0, index=df.index, dtype=float)


def standardize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    known_map = {
        "date": "date",
        "Date": "date",
        "날짜": "date",
        "open": "open",
        "시가": "open",
        "high": "high",
        "고가": "high",
        "low": "low",
        "저가": "low",
        "close": "close",
        "종가": "close",
        "volume": "volume",
        "거래량": "volume",
        "value": "value",
        "거래대금": "value",
        "ret_pct": "ret_pct",
        "등락률": "ret_pct",
    }
    out = out.rename(columns={key: value for key, value in known_map.items() if key in out.columns})

    if "date" not in out.columns:
        out = out.reset_index()
        out = out.rename(columns={out.columns[0]: "date"})

    if "open" not in out.columns or "close" not in out.columns:
        cols = list(out.columns)
        aliases = ["date", "open", "high", "low", "close", "volume", "value", "ret_pct"]
        rename_map: Dict[str, str] = {}
        for idx, alias in enumerate(aliases):
            if idx < len(cols) and alias not in out.columns:
                rename_map[cols[idx]] = alias
        out = out.rename(columns=rename_map)

    required = ["date", "open", "high", "low", "close", "volume"]
    missing = [column for column in required if column not in out.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out["date"] = pd.to_datetime(out["date"])
    for column in ["open", "high", "low", "close", "volume"]:
        out[column] = safe_to_numeric(out[column])

    if "value" not in out.columns:
        out["value"] = out["close"] * out["volume"]
    out["value"] = safe_to_numeric(out["value"])

    keep = ["date", "open", "high", "low", "close", "volume", "value"]
    return out[keep].sort_values("date").drop_duplicates("date").reset_index(drop=True)


def calc_mdd(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min()) if not dd.empty else 0.0


# =========================================================
# 2. Data download
# =========================================================
def build_universe() -> pd.DataFrame:
    universe = pd.DataFrame(FIXED_UNIVERSE)
    universe["ticker"] = universe["ticker"].astype(str).str.zfill(6)
    return universe


def build_strategy_config_frame() -> pd.DataFrame:
    rows = [
        {"item": "start_date", "value": START_DATE},
        {"item": "end_date", "value": END_DATE},
        {"item": "strategy_label", "value": STRATEGY_LABEL},
        {"item": "strategy_display_name", "value": STRATEGY_DISPLAY_NAME},
        {"item": "strategy_sleeve_weight", "value": STRATEGY_SLEEVE_WEIGHT},
        {"item": "cap_excess_handling", "value": CAP_EXCESS_HANDLING},
        {"item": "cap_excess_redirect_map", "value": str(CAP_EXCESS_REDIRECT_MAP)},
        {"item": "benchmark_label", "value": BENCHMARK_LABEL},
        {"item": "benchmark_display_name", "value": BENCHMARK_DISPLAY_NAME},
        {"item": "kospi_benchmark_label", "value": KOSPI_BENCHMARK_LABEL},
        {"item": "kospi_benchmark_display_name", "value": KOSPI_BENCHMARK_DISPLAY_NAME},
        {"item": "kospi_benchmark_weight", "value": KOSPI_BENCHMARK_WEIGHT},
    ]
    for ticker, cap in PORTFOLIO_TICKER_CAPS.items():
        rows.append({"item": f"portfolio_cap_{ticker}", "value": cap})
    for ticker, weight in BENCHMARK_FIXED_WEIGHTS.items():
        rows.append({"item": f"benchmark_weight_{ticker}", "value": weight})
    for key, value in WINNER_PARAMS.items():
        rows.append({"item": f"winner_{key}", "value": value})
    return pd.DataFrame(rows)


def download_one_ticker_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    raw = stock.get_market_ohlcv_by_date(normalize_yyyymmdd(start), normalize_yyyymmdd(end), ticker)
    df = reset_date_frame(raw)
    if df.empty:
        return pd.DataFrame()
    return standardize_ohlcv_columns(df)


def build_price_panel(universe: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for row in universe.itertuples(index=False):
        print(f"[INFO] Downloading OHLCV: {row.ticker} {row.name}")
        df = download_one_ticker_ohlcv(row.ticker, start, end)
        if df.empty:
            continue
        df["ticker"] = row.ticker
        df["name"] = row.name
        df["market"] = row.market
        frames.append(df)
        time.sleep(SLEEP_BETWEEN_CALLS)

    if not frames:
        raise RuntimeError("No price data downloaded.")

    panel = pd.concat(frames, ignore_index=True)
    panel["ticker"] = panel["ticker"].astype(str).str.zfill(6)
    return panel.sort_values(["date", "ticker"]).reset_index(drop=True)


def download_benchmark_index(start: str, end: str) -> pd.DataFrame:
    # Avoid pykrx's extra ticker-name lookup, which still fails on some KRX index responses.
    raw = stock.get_index_ohlcv_by_date(
        normalize_yyyymmdd(start),
        normalize_yyyymmdd(end),
        "1001",
        name_display=False,
    )
    df = reset_date_frame(raw)
    if df.empty:
        raise RuntimeError("Failed to download KOSPI benchmark data.")

    bm = standardize_ohlcv_columns(df)
    bm["ret1"] = bm["close"].pct_change().fillna(0.0)
    bm["bm_ret20"] = bm["close"].pct_change(20)
    bm["bm_ma60"] = bm["close"].rolling(BENCHMARK_MA_WINDOW).mean()
    return bm


# =========================================================
# 3. Features and strategy
# =========================================================
def add_features_one_ticker(df: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
    one = df.copy().sort_values("date").reset_index(drop=True)
    bm = benchmark[["date", "close", "bm_ret20", "bm_ma60"]].rename(columns={"close": "bm_close"})
    one = one.merge(bm, on="date", how="left")

    one["ret1"] = one["close"].pct_change().fillna(0.0)
    one["ret10"] = one["close"].pct_change(10)
    one["ret20"] = one["close"].pct_change(20)
    one["ma20"] = one["close"].rolling(20).mean()
    one["rs20"] = one["ret20"] - one["bm_ret20"]
    one["market_on"] = (one["bm_close"] > one["bm_ma60"]).astype(float)
    return one


def add_features_panel(panel: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for _, group in panel.groupby("ticker", sort=False):
        frames.append(add_features_one_ticker(group, benchmark))
    return pd.concat(frames, ignore_index=True).sort_values(["date", "ticker"]).reset_index(drop=True)


def apply_portfolio_constraints(panel: pd.DataFrame, target_col: str = "target_weight") -> pd.DataFrame:
    df = panel.copy().sort_values(["date", "ticker"]).reset_index(drop=True)
    df["ticker"] = df["ticker"].astype(str).str.zfill(6)

    df["strategy_target_weight"] = get_series(df, target_col).fillna(0.0).clip(lower=0.0)
    strategy_sum = df.groupby("date")["strategy_target_weight"].transform("sum")
    df["strategy_stock_weight"] = strategy_sum
    df["strategy_cash_target_weight"] = (1.0 - strategy_sum).clip(lower=0.0, upper=1.0)
    df["strategy_sleeve_weight"] = STRATEGY_SLEEVE_WEIGHT

    df["portfolio_target_weight_uncapped"] = STRATEGY_SLEEVE_WEIGHT * df["strategy_target_weight"]
    df["ticker_portfolio_cap"] = df["ticker"].map(PORTFOLIO_TICKER_CAPS).fillna(STRATEGY_SLEEVE_WEIGHT)
    df["target_weight"] = np.minimum(df["portfolio_target_weight_uncapped"], df["ticker_portfolio_cap"])
    df["portfolio_cap_excess_before_redirect"] = (
        df["portfolio_target_weight_uncapped"] - df["target_weight"]
    ).clip(lower=0.0)
    df["cap_excess_redirected_weight"] = 0.0
    df["portfolio_cap_clipped_weight"] = df["portfolio_cap_excess_before_redirect"].copy()

    if CAP_EXCESS_HANDLING == "redirect_hynix_to_samsung":
        for _, date_index in df.groupby("date").groups.items():
            for donor_ticker, receiver_ticker in CAP_EXCESS_REDIRECT_MAP.items():
                donor_rows = [idx for idx in date_index if df.at[idx, "ticker"] == donor_ticker]
                receiver_rows = [idx for idx in date_index if df.at[idx, "ticker"] == receiver_ticker]
                if not donor_rows or not receiver_rows:
                    continue

                donor_idx = donor_rows[0]
                receiver_idx = receiver_rows[0]
                donor_excess = float(df.at[donor_idx, "portfolio_cap_clipped_weight"])
                if donor_excess <= 0:
                    continue

                receiver_room = max(
                    0.0,
                    float(df.at[receiver_idx, "ticker_portfolio_cap"]) - float(df.at[receiver_idx, "target_weight"]),
                )
                redirected_weight = min(donor_excess, receiver_room)
                if redirected_weight <= 0:
                    continue

                df.at[receiver_idx, "target_weight"] = float(df.at[receiver_idx, "target_weight"]) + redirected_weight
                df.at[receiver_idx, "cap_excess_redirected_weight"] = (
                    float(df.at[receiver_idx, "cap_excess_redirected_weight"]) + redirected_weight
                )
                df.at[donor_idx, "portfolio_cap_clipped_weight"] = donor_excess - redirected_weight

    actual_stock_sum = df.groupby("date")["target_weight"].transform("sum")
    clipped_sum = df.groupby("date")["portfolio_cap_clipped_weight"].transform("sum")
    redirected_sum = df.groupby("date")["cap_excess_redirected_weight"].transform("sum")
    pre_redirect_excess_sum = df.groupby("date")["portfolio_cap_excess_before_redirect"].transform("sum")

    df["cash_target_weight"] = (1.0 - actual_stock_sum).clip(lower=0.0, upper=1.0)
    df["strategy_stock_weight_after_caps"] = actual_stock_sum
    df["strategy_cash_weight_after_caps"] = (
        STRATEGY_SLEEVE_WEIGHT - actual_stock_sum
    ).clip(lower=0.0, upper=STRATEGY_SLEEVE_WEIGHT)
    df["outside_sleeve_cash_weight"] = max(0.0, 1.0 - STRATEGY_SLEEVE_WEIGHT)
    df["portfolio_cap_clipped_total"] = clipped_sum
    df["portfolio_cap_excess_before_redirect_total"] = pre_redirect_excess_sum
    df["cap_excess_redirected_total"] = redirected_sum
    return df


def build_minimal_strategy_panel(panel_feat: pd.DataFrame) -> pd.DataFrame:
    df = panel_feat.copy().sort_values(["date", "ticker"]).reset_index(drop=True)
    df["ticker"] = df["ticker"].astype(str).str.zfill(6)

    samsung_mask = df["ticker"].eq("005930")
    hynix_mask = df["ticker"].eq("000660")

    market_on = get_series(df, "market_on").fillna(0.0).clip(lower=0.0, upper=1.0)
    ret10_raw = get_series(df, "ret10").fillna(0.0)
    rs20_raw = get_series(df, "rs20").fillna(0.0)
    above_ma20 = (get_series(df, "close") > get_series(df, "ma20")).astype(float)

    ret10_flag = (ret10_raw > float(WINNER_PARAMS["ret10_threshold"])).astype(float)
    rs20_flag = (rs20_raw > float(WINNER_PARAMS["rs20_threshold"])).astype(float)
    hynix_simple_signal = (0.45 * ret10_flag + 0.35 * rs20_flag + 0.20 * above_ma20) * (0.5 + 0.5 * market_on)
    hynix_signal_strength = hynix_simple_signal.clip(0.0, 1.0)

    samsung_core_target = np.where(
        market_on > 0,
        float(WINNER_PARAMS["samsung_core_on"]),
        float(WINNER_PARAMS["samsung_core_off"]),
    )
    samsung_core_target = np.clip(samsung_core_target, 0.10, MAX_SAMSUNG_STRATEGY_TARGET)

    hynix_target = (
        float(WINNER_PARAMS["hynix_base"])
        + float(WINNER_PARAMS["hynix_tactical_scale"]) * hynix_signal_strength
    )
    hynix_target = np.clip(hynix_target, 0.0, MAX_HYNIX_STRATEGY_TARGET)

    df["alpha_score"] = np.where(hynix_mask, hynix_simple_signal, samsung_core_target)
    df["ret10_flag"] = ret10_flag
    df["rs20_flag"] = rs20_flag
    df["above_ma20_flag"] = above_ma20
    df["hynix_simple_signal"] = np.where(hynix_mask, hynix_simple_signal, np.nan)
    df["hynix_signal_strength"] = np.where(hynix_mask, hynix_signal_strength, np.nan)
    df["samsung_core_target"] = np.where(samsung_mask, samsung_core_target, np.nan)
    df["alloc_score"] = np.where(hynix_mask, hynix_simple_signal, samsung_core_target)
    df["target_weight"] = 0.0
    df.loc[samsung_mask, "target_weight"] = samsung_core_target[samsung_mask]
    df.loc[hynix_mask, "target_weight"] = hynix_target[hynix_mask]
    df["strategy"] = STRATEGY_LABEL
    df["overlay_profile"] = str(WINNER_PARAMS["variant_name"])
    return apply_portfolio_constraints(df, target_col="target_weight")


# =========================================================
# 4. Benchmarks
# =========================================================
def build_benchmark_buy_hold(panel_feat: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if panel_feat.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    panel = panel_feat.copy().sort_values(["date", "ticker"]).reset_index(drop=True)
    panel["ticker"] = panel["ticker"].astype(str).str.zfill(6)
    benchmark_sleeve = float(sum(BENCHMARK_FIXED_WEIGHTS.values()))

    panel["strategy_target_weight"] = panel["ticker"].map(BENCHMARK_FIXED_WEIGHTS).fillna(0.0)
    panel["strategy_cash_target_weight"] = max(0.0, 1.0 - benchmark_sleeve)
    panel["strategy_sleeve_weight"] = benchmark_sleeve
    panel["portfolio_target_weight_uncapped"] = panel["strategy_target_weight"]
    panel["ticker_portfolio_cap"] = panel["ticker"].map(BENCHMARK_FIXED_WEIGHTS).fillna(0.0)
    panel["target_weight"] = panel["portfolio_target_weight_uncapped"]
    panel["portfolio_cap_clipped_weight"] = 0.0
    actual_stock_sum = panel.groupby("date")["target_weight"].transform("sum")
    panel["cash_target_weight"] = (1.0 - actual_stock_sum).clip(lower=0.0, upper=1.0)
    panel["strategy_stock_weight_after_caps"] = actual_stock_sum
    panel["strategy_cash_weight_after_caps"] = (benchmark_sleeve - actual_stock_sum).clip(lower=0.0, upper=benchmark_sleeve)
    panel["outside_sleeve_cash_weight"] = max(0.0, 1.0 - benchmark_sleeve)
    panel["portfolio_cap_clipped_total"] = 0.0
    panel["portfolio_cap_excess_before_redirect_total"] = 0.0
    panel["cap_excess_redirected_total"] = 0.0
    if "alpha_score" not in panel.columns:
        panel["alpha_score"] = 0.0
    eq, trades, rebalances = backtest_weight_strategy(panel, weight_col="target_weight")
    return panel, eq, trades, rebalances


def build_kospi_weighted_benchmark(
    benchmark: pd.DataFrame,
    weight: float = KOSPI_BENCHMARK_WEIGHT,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if benchmark.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    bm = benchmark[["date", "close", "ret1"]].copy().sort_values("date").reset_index(drop=True)
    panel = pd.DataFrame({
        "date": bm["date"],
        "ticker": "KOSPI_1001",
        "name": "KOSPI",
        "close": bm["close"],
        "ret1": bm["ret1"],
    })
    panel["target_weight"] = weight
    panel["cash_target_weight"] = 1.0 - weight
    panel["alpha_score"] = 0.0
    panel["strategy_stock_weight_after_caps"] = weight
    panel["strategy_cash_weight_after_caps"] = 0.0
    panel["outside_sleeve_cash_weight"] = 1.0 - weight
    panel["portfolio_cap_clipped_total"] = 0.0
    panel["portfolio_cap_excess_before_redirect_total"] = 0.0
    panel["cap_excess_redirected_total"] = 0.0
    eq, trades, rebalances = backtest_weight_strategy(panel, weight_col="target_weight")
    return panel, eq, trades, rebalances


# =========================================================
# 5. Diagnostics
# =========================================================
def build_recent_watchlist(dynamic_panel: pd.DataFrame, lookback_days: int = 20) -> pd.DataFrame:
    if dynamic_panel.empty:
        return pd.DataFrame()

    latest_date = pd.to_datetime(dynamic_panel["date"]).max()
    cutoff = latest_date - pd.Timedelta(days=lookback_days)
    recent = dynamic_panel.loc[dynamic_panel["date"] >= cutoff].copy()
    cols = [
        "date",
        "ticker",
        "name",
        "close",
        "ret10",
        "rs20",
        "market_on",
        "ret10_flag",
        "rs20_flag",
        "above_ma20_flag",
        "samsung_core_target",
        "hynix_simple_signal",
        "alpha_score",
        "target_weight",
        "cash_target_weight",
    ]
    cols = [column for column in cols if column in recent.columns]
    return recent.sort_values(["date", "target_weight", "alpha_score"], ascending=[False, False, False])[cols].reset_index(drop=True)


def build_current_allocation_snapshot(dynamic_panel: pd.DataFrame) -> pd.DataFrame:
    if dynamic_panel.empty:
        return pd.DataFrame()

    latest_date = pd.to_datetime(dynamic_panel["date"]).max()
    latest = dynamic_panel.loc[dynamic_panel["date"] == latest_date].copy()
    cols = [
        "date",
        "ticker",
        "name",
        "close",
        "ret10",
        "rs20",
        "market_on",
        "samsung_core_target",
        "hynix_simple_signal",
        "alpha_score",
        "target_weight",
        "cash_target_weight",
        "strategy_stock_weight_after_caps",
    ]
    cols = [column for column in cols if column in latest.columns]
    return latest.sort_values(["target_weight", "alpha_score"], ascending=[False, False])[cols].reset_index(drop=True)


# =========================================================
# 6. Backtest
# =========================================================
def build_position_trade_log(
    effective_weights: pd.DataFrame,
    close_px: pd.DataFrame,
    score_px: pd.DataFrame,
    name_map: Dict[str, str],
) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    for ticker in effective_weights.columns:
        w = effective_weights[ticker].fillna(0.0)
        in_pos = False
        entry_date = None
        entry_price = None
        entry_score = None

        for current_date in w.index:
            current_weight = float(w.loc[current_date])
            current_price = float(close_px.at[current_date, ticker]) if pd.notna(close_px.at[current_date, ticker]) else np.nan
            current_score = float(score_px.at[current_date, ticker]) if pd.notna(score_px.at[current_date, ticker]) else np.nan

            if not in_pos and current_weight > 0:
                in_pos = True
                entry_date = current_date
                entry_price = current_price
                entry_score = current_score
                continue

            if in_pos and current_weight <= 0:
                ret = np.nan
                if pd.notna(entry_price) and pd.notna(current_price) and entry_price != 0:
                    ret = current_price / entry_price - 1.0
                records.append({
                    "ticker": ticker,
                    "name": name_map.get(ticker, ticker),
                    "entry_date": entry_date,
                    "exit_date": current_date,
                    "entry_price": entry_price,
                    "exit_price": current_price,
                    "entry_score": entry_score,
                    "exit_score": current_score,
                    "trade_return": ret,
                })
                in_pos = False
                entry_date = None
                entry_price = None
                entry_score = None

        if in_pos and entry_date is not None:
            last_date = w.index[-1]
            exit_price = float(close_px.at[last_date, ticker]) if pd.notna(close_px.at[last_date, ticker]) else np.nan
            exit_score = float(score_px.at[last_date, ticker]) if pd.notna(score_px.at[last_date, ticker]) else np.nan
            ret = np.nan
            if pd.notna(entry_price) and pd.notna(exit_price) and entry_price != 0:
                ret = exit_price / entry_price - 1.0
            records.append({
                "ticker": ticker,
                "name": name_map.get(ticker, ticker),
                "entry_date": entry_date,
                "exit_date": last_date,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "entry_score": entry_score,
                "exit_score": exit_score,
                "trade_return": ret,
            })

    return pd.DataFrame(records)


def build_rebalance_event_log(
    effective_weights: pd.DataFrame,
    close_px: pd.DataFrame,
    score_px: pd.DataFrame,
    name_map: Dict[str, str],
) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    prev_weights = effective_weights.shift(1).fillna(0.0)
    delta = effective_weights.fillna(0.0) - prev_weights

    for current_date in effective_weights.index:
        for ticker in effective_weights.columns:
            change = float(delta.at[current_date, ticker])
            if abs(change) < REBALANCE_LOG_MIN_DELTA:
                continue
            price = float(close_px.at[current_date, ticker]) if pd.notna(close_px.at[current_date, ticker]) else np.nan
            score = float(score_px.at[current_date, ticker]) if pd.notna(score_px.at[current_date, ticker]) else np.nan
            records.append({
                "date": current_date,
                "ticker": ticker,
                "name": name_map.get(ticker, ticker),
                "prev_weight": float(prev_weights.at[current_date, ticker]),
                "new_weight": float(effective_weights.at[current_date, ticker]),
                "delta_weight": change,
                "price": price,
                "alpha_score": score,
            })

    return pd.DataFrame(records)


def backtest_weight_strategy(signal_panel: pd.DataFrame, weight_col: str = "target_weight") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if signal_panel.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    panel = signal_panel.copy().sort_values(["date", "ticker"]).reset_index(drop=True)
    panel["date"] = pd.to_datetime(panel["date"])

    close_px = panel.pivot_table(index="date", columns="ticker", values="close", aggfunc="last").sort_index()
    ret1_px = panel.pivot_table(index="date", columns="ticker", values="ret1", aggfunc="last").sort_index()
    target_weights = panel.pivot_table(index="date", columns="ticker", values=weight_col, aggfunc="last").sort_index().fillna(0.0)
    score_px = panel.pivot_table(index="date", columns="ticker", values="alpha_score", aggfunc="last").sort_index().fillna(0.0)

    effective_weights = target_weights.shift(1).fillna(0.0)
    turnover = effective_weights.diff().abs().sum(axis=1).fillna(effective_weights.abs().sum(axis=1))
    gross_return = (effective_weights * ret1_px.fillna(0.0)).sum(axis=1)
    cost = turnover * FEE_RATE
    net_return = gross_return - cost
    equity = (1.0 + net_return).cumprod() * INITIAL_CAPITAL

    eq = pd.DataFrame({
        "date": net_return.index,
        "daily_return": net_return.values,
        "gross_return": gross_return.values,
        "turnover": turnover.values,
        "fee_cost": cost.values,
        "stock_weight": effective_weights.sum(axis=1).values,
        "cash_weight": 1.0 - effective_weights.sum(axis=1).values,
        "equity": equity.values,
    })

    name_map = (
        panel[["ticker", "name"]]
        .drop_duplicates("ticker")
        .set_index("ticker")["name"]
        .to_dict()
    )
    position_trades = build_position_trade_log(effective_weights, close_px, score_px, name_map)
    rebalance_log = build_rebalance_event_log(effective_weights, close_px, score_px, name_map)
    return eq, position_trades, rebalance_log


def calc_performance_metrics(eq: pd.DataFrame, label: str, trades: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    if eq.empty:
        return pd.DataFrame({"label": [label], "metric": [], "value": []})

    daily = pd.to_numeric(eq["daily_return"], errors="coerce").fillna(0.0)
    total_return = float(eq["equity"].iloc[-1] / INITIAL_CAPITAL - 1.0)
    days = max(len(eq), 1)
    cagr = float((eq["equity"].iloc[-1] / INITIAL_CAPITAL) ** (252 / days) - 1.0) if days > 1 else total_return
    volatility = float(daily.std(ddof=0) * np.sqrt(252))
    sharpe = float(daily.mean() / daily.std(ddof=0) * np.sqrt(252)) if daily.std(ddof=0) > 0 else 0.0
    mdd = calc_mdd(eq["equity"])
    avg_stock_weight = float(pd.to_numeric(eq["stock_weight"], errors="coerce").mean())
    current_stock_weight = float(pd.to_numeric(eq["stock_weight"], errors="coerce").iloc[-1])
    avg_cash_weight = float(pd.to_numeric(eq["cash_weight"], errors="coerce").mean())
    current_cash_weight = float(pd.to_numeric(eq["cash_weight"], errors="coerce").iloc[-1])
    rebalance_days = int((pd.to_numeric(eq["turnover"], errors="coerce") > 0).sum())
    n_trades = int(len(trades)) if trades is not None and not trades.empty else 0

    rows = [
        ("total_return", total_return),
        ("CAGR", cagr),
        ("sharpe", sharpe),
        ("MDD", mdd),
        ("volatility", volatility),
        ("avg_stock_weight", avg_stock_weight),
        ("current_stock_weight", current_stock_weight),
        ("avg_cash_weight", avg_cash_weight),
        ("current_cash_weight", current_cash_weight),
        ("rebalance_days", rebalance_days),
        ("n_trades", n_trades),
    ]
    return pd.DataFrame({"label": label, "metric": [metric for metric, _ in rows], "value": [value for _, value in rows]})


# =========================================================
# 7. Output helpers
# =========================================================
def build_summary_pivot(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    return summary.pivot(index="metric", columns="label", values="value").reset_index()


def save_equity_plot(curves: List[Tuple[str, pd.DataFrame]], path: str = OUTPUT_PLOT) -> None:
    valid_curves = [(label, eq.copy()) for label, eq in curves if eq is not None and not eq.empty]
    if not valid_curves:
        return

    plt.style.use("default")
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True, gridspec_kw={"height_ratios": [3, 2]})

    for label, eq in valid_curves:
        data = eq.copy().sort_values("date").reset_index(drop=True)
        data["cum_return"] = data["equity"] / INITIAL_CAPITAL - 1.0
        data["drawdown"] = data["equity"] / data["equity"].cummax() - 1.0
        axes[0].plot(data["date"], data["cum_return"], label=label, linewidth=2.2)
        axes[1].plot(data["date"], data["drawdown"], label=label, linewidth=2.0)

    axes[0].set_title("Minimal Samsung Core + Hynix Overlay vs Benchmarks", fontsize=18, pad=14)
    axes[0].set_ylabel("Cumulative Return", fontsize=12)
    axes[0].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="upper left", fontsize=10)

    axes[1].set_ylabel("Drawdown", fontsize=12)
    axes[1].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="lower left", fontsize=10)

    plt.xticks(rotation=30)
    plt.tight_layout()
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_results(
    dynamic_panel: pd.DataFrame,
    dynamic_eq: pd.DataFrame,
    dynamic_trades: pd.DataFrame,
    dynamic_rebalances: pd.DataFrame,
    benchmark_bh_panel: pd.DataFrame,
    benchmark_bh_eq: pd.DataFrame,
    benchmark_kospi_panel: pd.DataFrame,
    benchmark_kospi_eq: pd.DataFrame,
    summary: pd.DataFrame,
    recent_watchlist: pd.DataFrame,
    current_alloc: pd.DataFrame,
    strategy_config: pd.DataFrame,
    path: str = OUTPUT_XLSX,
) -> Path:
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    def write_excel(excel_path: Path) -> None:
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            strategy_config.to_excel(writer, sheet_name="strategy_config", index=False)
            summary.to_excel(writer, sheet_name="summary", index=False)
            build_summary_pivot(summary).to_excel(writer, sheet_name="summary_pivot", index=False)
            dynamic_panel.to_excel(writer, sheet_name="strategy_panel", index=False)
            dynamic_eq.to_excel(writer, sheet_name="strategy_equity", index=False)
            dynamic_trades.to_excel(writer, sheet_name="strategy_trades", index=False)
            dynamic_rebalances.to_excel(writer, sheet_name="strategy_rebalances", index=False)
            benchmark_bh_panel.to_excel(writer, sheet_name="benchmark_bh_panel", index=False)
            benchmark_bh_eq.to_excel(writer, sheet_name="benchmark_bh_equity", index=False)
            benchmark_kospi_panel.to_excel(writer, sheet_name="benchmark_kospi_panel", index=False)
            benchmark_kospi_eq.to_excel(writer, sheet_name="benchmark_kospi_equity", index=False)
            recent_watchlist.to_excel(writer, sheet_name="recent_watchlist", index=False)
            current_alloc.to_excel(writer, sheet_name="current_allocation", index=False)

    def build_fallback_path(base_path: Path, attempt: int) -> Path:
        if attempt == 1:
            suffix = "_autosave"
        else:
            suffix = f"_autosave{attempt}"
        return base_path.with_name(f"{base_path.stem}{suffix}{base_path.suffix}")

    attempt = 0
    while True:
        attempt += 1
        try_path = target_path if attempt == 1 else build_fallback_path(target_path, attempt - 1)
        try:
            write_excel(try_path)
            return try_path
        except PermissionError:
            if attempt >= 10:
                raise
            continue


# =========================================================
# 8. Main
# =========================================================
def main() -> None:
    initialize_krx_session()

    print("=" * 90)
    print(f"{STRATEGY_DISPLAY_NAME} | {START_DATE} ~ {END_DATE}")
    print("=" * 90)

    print("[1] Build universe")
    universe = build_universe()
    print(universe[["ticker", "name", "market"]].to_string(index=False))

    print("[2] Download price panel")
    panel = build_price_panel(universe, START_DATE, END_DATE)
    print(f"[INFO] price rows = {len(panel)}")
    if not panel.empty:
        actual_price_end = pd.to_datetime(panel["date"]).max().strftime("%Y-%m-%d")
        print(f"[INFO] latest price date = {actual_price_end}")

    print("[3] Download KOSPI benchmark")
    benchmark = download_benchmark_index(START_DATE, END_DATE)
    print(f"[INFO] benchmark rows = {len(benchmark)}")
    if not benchmark.empty:
        actual_benchmark_end = pd.to_datetime(benchmark["date"]).max().strftime("%Y-%m-%d")
        print(f"[INFO] latest benchmark date = {actual_benchmark_end}")

    print("[4] Build minimal features")
    panel_feat = add_features_panel(panel, benchmark)
    feature_cols = [col for col in ["date", "ticker", "name", "close", "ret10", "rs20", "market_on"] if col in panel_feat.columns]
    print(panel_feat[feature_cols].tail(4).to_string(index=False))

    print("[5] Build strategy panel")
    strategy_panel = build_minimal_strategy_panel(panel_feat)
    show_cols = [
        "date",
        "ticker",
        "name",
        "ret10",
        "rs20",
        "market_on",
        "samsung_core_target",
        "hynix_simple_signal",
        "target_weight",
        "cash_target_weight",
    ]
    show_cols = [column for column in show_cols if column in strategy_panel.columns]
    print(strategy_panel[show_cols].tail(6).to_string(index=False))

    print("[6] Backtest strategy and benchmarks")
    strategy_eq, strategy_trades, strategy_rebalances = backtest_weight_strategy(strategy_panel, weight_col="target_weight")
    benchmark_bh_panel, benchmark_bh_eq, benchmark_bh_trades, benchmark_bh_rebalances = build_benchmark_buy_hold(panel_feat)
    benchmark_kospi_panel, benchmark_kospi_eq, benchmark_kospi_trades, benchmark_kospi_rebalances = build_kospi_weighted_benchmark(benchmark)
    print(f"[INFO] strategy rows = {len(strategy_eq)}, trades = {len(strategy_trades)}, rebalances = {len(strategy_rebalances)}")
    print(f"[INFO] benchmark bh rows = {len(benchmark_bh_eq)}, rebalances = {len(benchmark_bh_rebalances)}")
    print(f"[INFO] benchmark kospi rows = {len(benchmark_kospi_eq)}, rebalances = {len(benchmark_kospi_rebalances)}")

    print("[7] Build diagnostics")
    recent_watchlist = build_recent_watchlist(strategy_panel, lookback_days=20)
    current_alloc = build_current_allocation_snapshot(strategy_panel)
    if not current_alloc.empty:
        print(current_alloc.to_string(index=False))

    print("[8] Summary")
    summary = pd.concat(
        [
            calc_performance_metrics(strategy_eq, STRATEGY_LABEL, strategy_trades),
            calc_performance_metrics(benchmark_bh_eq, BENCHMARK_LABEL, benchmark_bh_trades),
            calc_performance_metrics(benchmark_kospi_eq, KOSPI_BENCHMARK_LABEL, benchmark_kospi_trades),
        ],
        ignore_index=True,
    )
    print(build_summary_pivot(summary).to_string(index=False))

    print("[9] Save outputs")
    strategy_config = build_strategy_config_frame()
    saved_excel = save_results(
        dynamic_panel=strategy_panel,
        dynamic_eq=strategy_eq,
        dynamic_trades=strategy_trades,
        dynamic_rebalances=strategy_rebalances,
        benchmark_bh_panel=benchmark_bh_panel,
        benchmark_bh_eq=benchmark_bh_eq,
        benchmark_kospi_panel=benchmark_kospi_panel,
        benchmark_kospi_eq=benchmark_kospi_eq,
        summary=summary,
        recent_watchlist=recent_watchlist,
        current_alloc=current_alloc,
        strategy_config=strategy_config,
        path=OUTPUT_XLSX,
    )
    save_equity_plot(
        [
            (STRATEGY_DISPLAY_NAME, strategy_eq),
            (BENCHMARK_DISPLAY_NAME, benchmark_bh_eq),
            (KOSPI_BENCHMARK_DISPLAY_NAME, benchmark_kospi_eq),
        ],
        path=OUTPUT_PLOT,
    )
    print(f"[INFO] Excel saved to: {saved_excel}")
    print(f"[INFO] Plot saved to : {OUTPUT_PLOT}")


if __name__ == "__main__":
    main()
