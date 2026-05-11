from __future__ import annotations

import argparse
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm
from scipy import stats
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller


START_DATE = "2023-05-10"
END_DATE = "2026-05-10"
DEFAULT_MARKET_DATA = Path("data/kospi_sentiment_tradingday_merged_20230510_20260510.csv")
DEFAULT_OUTPUT_DIR = Path("data/statistical_tests")
YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
FRED_KOREA_10Y_SERIES = "IRLTLT01KRM156N"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run lag tests between news sentiment and KOSPI returns."
    )
    parser.add_argument("--market-data", type=Path, default=DEFAULT_MARKET_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start-date", default=START_DATE)
    parser.add_argument("--end-date", default=END_DATE)
    parser.add_argument("--max-lag", type=int, default=10)
    return parser.parse_args()


def date_to_timestamp(date: str) -> int:
    return int(datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())


def fetch_yahoo_chart(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    end_plus_one = (
        datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)
    ).strftime("%Y-%m-%d")
    url = YAHOO_CHART_URL.format(symbol=quote(symbol, safe=""))
    params = {
        "period1": date_to_timestamp(start_date),
        "period2": date_to_timestamp(end_plus_one),
        "interval": "1d",
        "events": "history",
    }
    response = requests.get(url, params=params, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    response.raise_for_status()
    payload = response.json()
    chart = payload.get("chart", {})
    if chart.get("error"):
        raise RuntimeError(chart["error"])

    result = chart["result"][0]
    timestamps = result["timestamp"]
    quote_data = result["indicators"]["quote"][0]
    close = quote_data["close"]
    df = pd.DataFrame(
        {
            "date": [
                datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
                for ts in timestamps
            ],
            "usdkrw_close": close,
        }
    )
    df = df.dropna(subset=["usdkrw_close"]).sort_values("date").reset_index(drop=True)
    df["usdkrw_log_return"] = np.log(df["usdkrw_close"] / df["usdkrw_close"].shift(1))
    return df


def fetch_fred_series(series_id: str) -> pd.DataFrame:
    url = FRED_CSV_URL.format(series_id=series_id)
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    path = DEFAULT_OUTPUT_DIR / f"{series_id}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(response.text, encoding="utf-8")
    df = pd.read_csv(path)
    value_col = series_id
    df = df.rename(columns={"observation_date": "date", value_col: "korea_10y_yield"})
    df["date"] = pd.to_datetime(df["date"])
    df["korea_10y_yield"] = pd.to_numeric(df["korea_10y_yield"], errors="coerce")
    df = df.dropna(subset=["korea_10y_yield"]).sort_values("date").reset_index(drop=True)
    return df


def prepare_dataset(args: argparse.Namespace) -> pd.DataFrame:
    base = pd.read_csv(args.market_data, dtype={"date": str})
    base["date"] = pd.to_datetime(base["date"])

    fx = fetch_yahoo_chart("KRW=X", args.start_date, args.end_date)
    fx["date"] = pd.to_datetime(fx["date"])

    rates = fetch_fred_series(FRED_KOREA_10Y_SERIES)

    df = base.merge(fx, on="date", how="left")
    df = pd.merge_asof(
        df.sort_values("date"),
        rates.sort_values("date"),
        on="date",
        direction="backward",
    )
    df["usdkrw_close"] = df["usdkrw_close"].ffill()
    df["usdkrw_log_return"] = df["usdkrw_log_return"].ffill().fillna(0)
    df["korea_10y_yield"] = df["korea_10y_yield"].ffill()
    df["korea_10y_yield_change"] = df["korea_10y_yield"].diff().fillna(0)
    df["kospi_log_return"] = pd.to_numeric(df["kospi_log_return"], errors="coerce")
    df["sentiment_index"] = pd.to_numeric(df["sentiment_index"], errors="coerce")
    return df.reset_index(drop=True)


def correlation_with_pvalue(x: pd.Series, y: pd.Series) -> tuple[float, float, int]:
    pair = pd.concat([x, y], axis=1).dropna()
    if len(pair) < 3:
        return np.nan, np.nan, len(pair)
    corr, pvalue = stats.pearsonr(pair.iloc[:, 0], pair.iloc[:, 1])
    return float(corr), float(pvalue), len(pair)


def run_cross_correlations(df: pd.DataFrame, max_lag: int) -> pd.DataFrame:
    rows = []
    for lag in range(1, max_lag + 1):
        future_return = df["kospi_log_return"].shift(-lag)
        corr, pvalue, n = correlation_with_pvalue(df["sentiment_index"], future_return)
        rows.append(
            {
                "direction": "sentiment_to_future_market",
                "lag_days": lag,
                "correlation": corr,
                "p_value": pvalue,
                "n": n,
            }
        )

        past_return = df["kospi_log_return"].shift(lag)
        corr, pvalue, n = correlation_with_pvalue(past_return, df["sentiment_index"])
        rows.append(
            {
                "direction": "market_to_future_sentiment",
                "lag_days": lag,
                "correlation": corr,
                "p_value": pvalue,
                "n": n,
            }
        )
    return pd.DataFrame(rows)


def ols_with_hac(y: pd.Series, x: pd.DataFrame, hac_lags: int = 5):
    data = pd.concat([y, x], axis=1).dropna()
    y_clean = data.iloc[:, 0]
    x_clean = sm.add_constant(data.iloc[:, 1:], has_constant="add")
    model = sm.OLS(y_clean, x_clean).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
    return model, len(data)


def run_lagged_regressions(df: pd.DataFrame, max_lag: int) -> pd.DataFrame:
    rows = []
    for lag in range(1, max_lag + 1):
        x = pd.DataFrame(
            {
                f"sentiment_lag_{lag}": df["sentiment_index"].shift(lag),
                "kospi_return_lag_1": df["kospi_log_return"].shift(1),
                "usdkrw_log_return": df["usdkrw_log_return"],
                "korea_10y_yield_change": df["korea_10y_yield_change"],
            }
        )
        model, n = ols_with_hac(df["kospi_log_return"], x)
        param = f"sentiment_lag_{lag}"
        rows.append(
            {
                "model": "market_return_on_lagged_sentiment",
                "lag_days": lag,
                "coef": model.params[param],
                "t_value": model.tvalues[param],
                "p_value": model.pvalues[param],
                "r_squared": model.rsquared,
                "n": n,
            }
        )

        reverse_x = pd.DataFrame(
            {
                f"kospi_return_lag_{lag}": df["kospi_log_return"].shift(lag),
                "sentiment_lag_1": df["sentiment_index"].shift(1),
                "usdkrw_log_return": df["usdkrw_log_return"],
                "korea_10y_yield_change": df["korea_10y_yield_change"],
            }
        )
        reverse_model, reverse_n = ols_with_hac(df["sentiment_index"], reverse_x)
        reverse_param = f"kospi_return_lag_{lag}"
        rows.append(
            {
                "model": "sentiment_on_lagged_market_return",
                "lag_days": lag,
                "coef": reverse_model.params[reverse_param],
                "t_value": reverse_model.tvalues[reverse_param],
                "p_value": reverse_model.pvalues[reverse_param],
                "r_squared": reverse_model.rsquared,
                "n": reverse_n,
            }
        )
    return pd.DataFrame(rows)


def build_granger_frame(
    df: pd.DataFrame,
    y_col: str,
    cause_col: str,
    max_lag: int,
) -> pd.DataFrame:
    data = pd.DataFrame(
        {
            "y": df[y_col],
            "cause": df[cause_col],
            "control_fx": df["usdkrw_log_return"],
            "control_rate": df["korea_10y_yield_change"],
        }
    )
    for lag in range(1, max_lag + 1):
        data[f"y_lag_{lag}"] = data["y"].shift(lag)
        data[f"cause_lag_{lag}"] = data["cause"].shift(lag)
        data[f"control_fx_lag_{lag}"] = data["control_fx"].shift(lag)
        data[f"control_rate_lag_{lag}"] = data["control_rate"].shift(lag)
    return data.dropna().reset_index(drop=True)


def run_granger_ols(df: pd.DataFrame, max_lag: int) -> pd.DataFrame:
    rows = []
    tests = [
        ("sentiment_causes_market", "kospi_log_return", "sentiment_index"),
        ("market_causes_sentiment", "sentiment_index", "kospi_log_return"),
    ]
    for direction, y_col, cause_col in tests:
        for lag in range(1, max_lag + 1):
            data = build_granger_frame(df, y_col, cause_col, lag)
            y = data["y"]
            base_cols = [f"y_lag_{i}" for i in range(1, lag + 1)]
            control_cols = (
                ["control_fx", "control_rate"]
                + [f"control_fx_lag_{i}" for i in range(1, lag + 1)]
                + [f"control_rate_lag_{i}" for i in range(1, lag + 1)]
            )
            cause_cols = [f"cause_lag_{i}" for i in range(1, lag + 1)]
            unrestricted_cols = base_cols + control_cols + cause_cols
            x = sm.add_constant(data[unrestricted_cols], has_constant="add")
            model = sm.OLS(y, x).fit()
            hypothesis = " = 0, ".join(cause_cols) + " = 0"
            f_test = model.f_test(hypothesis)
            rows.append(
                {
                    "direction": direction,
                    "lag_days": lag,
                    "f_stat": float(f_test.fvalue),
                    "p_value": float(f_test.pvalue),
                    "n": int(model.nobs),
                    "r_squared": model.rsquared,
                }
            )
    return pd.DataFrame(rows)


def adf_summary(series: pd.Series, name: str) -> dict[str, float | str | int]:
    cleaned = series.dropna()
    stat, pvalue, used_lag, nobs, _, _ = adfuller(cleaned, autolag="AIC")
    return {
        "variable": name,
        "adf_stat": stat,
        "p_value": pvalue,
        "used_lag": used_lag,
        "nobs": nobs,
    }


def run_var(df: pd.DataFrame, max_lag: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    var_cols = [
        "kospi_log_return",
        "sentiment_index",
        "usdkrw_log_return",
        "korea_10y_yield_change",
    ]
    var_df = df[var_cols].dropna().reset_index(drop=True)
    adf = pd.DataFrame([adf_summary(var_df[col], col) for col in var_cols])

    model = VAR(var_df)
    selected = model.select_order(maxlags=max_lag)
    selected_lag = int(selected.selected_orders.get("aic") or 1)
    selected_lag = max(1, min(selected_lag, max_lag))
    fitted = model.fit(selected_lag)

    rows = []
    tests = [
        ("sentiment_causes_market", "kospi_log_return", ["sentiment_index"]),
        ("market_causes_sentiment", "sentiment_index", ["kospi_log_return"]),
    ]
    for direction, caused, causing in tests:
        test = fitted.test_causality(caused=caused, causing=causing, kind="f")
        rows.append(
            {
                "direction": direction,
                "selected_lag_aic": selected_lag,
                "test_statistic": float(test.test_statistic),
                "p_value": float(test.pvalue),
                "df": str(test.df),
            }
        )

    order_summary = pd.DataFrame(
        [
            {
                "criterion": key,
                "selected_lag": value,
            }
            for key, value in selected.selected_orders.items()
        ]
    )
    return adf, order_summary, pd.DataFrame(rows)


def write_markdown_summary(
    path: Path,
    cross: pd.DataFrame,
    regressions: pd.DataFrame,
    granger: pd.DataFrame,
    var_tests: pd.DataFrame,
) -> None:
    best_cross = cross.assign(abs_corr=cross["correlation"].abs()).sort_values(
        "abs_corr", ascending=False
    )
    sig_reg = regressions[regressions["p_value"] < 0.05]
    sig_granger = granger[granger["p_value"] < 0.05]
    sig_var = var_tests[var_tests["p_value"] < 0.05]

    text = f"""# 뉴스 감성-코스피 시차 검정 요약

분석 기간: 2023-05-10 ~ 2026-05-10

통제변수:
- USD/KRW 로그수익률
- 한국 10년 국채수익률 변화분

주의:
- 코스피 마지막 거래일은 2026-05-08이다.
- 금리 변수는 FRED의 한국 10년 국채수익률 월별 자료를 거래일 기준으로 forward-fill했다.
- 아래 결과는 통계적 검정 결과이며, 인과의 경제적 해석은 모형 가정과 통제변수 한계를 함께 고려해야 한다.

## 1. 교차상관 분석

절대값 기준 상위 5개:

{best_cross.head(5).drop(columns=["abs_corr"]).to_markdown(index=False)}

## 2. 통제변수 포함 시차 회귀

5% 유의수준에서 유의한 결과 수: {len(sig_reg)}

{sig_reg.to_markdown(index=False) if not sig_reg.empty else "유의한 결과 없음"}

## 3. 통제변수 포함 Granger causality 검정

5% 유의수준에서 유의한 결과 수: {len(sig_granger)}

{sig_granger.to_markdown(index=False) if not sig_granger.empty else "유의한 결과 없음"}

## 4. VAR 모형 causality 검정

{var_tests.to_markdown(index=False)}

5% 유의수준 유의 결과:

{sig_var.to_markdown(index=False) if not sig_var.empty else "유의한 결과 없음"}
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = prepare_dataset(args)
    df.to_csv(args.output_dir / "analysis_dataset_with_controls.csv", index=False, encoding="utf-8-sig")

    cross = run_cross_correlations(df, args.max_lag)
    regressions = run_lagged_regressions(df, args.max_lag)
    granger = run_granger_ols(df, args.max_lag)
    adf, var_order, var_tests = run_var(df, args.max_lag)

    cross.to_csv(args.output_dir / "01_cross_correlations.csv", index=False, encoding="utf-8-sig")
    regressions.to_csv(args.output_dir / "02_lagged_regressions_controls.csv", index=False, encoding="utf-8-sig")
    granger.to_csv(args.output_dir / "03_granger_controls.csv", index=False, encoding="utf-8-sig")
    adf.to_csv(args.output_dir / "04_var_adf_stationarity.csv", index=False, encoding="utf-8-sig")
    var_order.to_csv(args.output_dir / "04_var_lag_order.csv", index=False, encoding="utf-8-sig")
    var_tests.to_csv(args.output_dir / "04_var_causality.csv", index=False, encoding="utf-8-sig")
    write_markdown_summary(
        args.output_dir / "summary.md",
        cross,
        regressions,
        granger,
        var_tests,
    )

    print(f"analysis rows: {len(df)}")
    print(f"date range: {df['date'].min().date()} ~ {df['date'].max().date()}")
    print(f"saved outputs to: {args.output_dir}")
    print("top cross-correlations:")
    print(
        cross.assign(abs_corr=cross["correlation"].abs())
        .sort_values("abs_corr", ascending=False)
        .head(6)
        .drop(columns=["abs_corr"])
        .to_string(index=False)
    )
    print("VAR causality:")
    print(var_tests.to_string(index=False))


if __name__ == "__main__":
    main()
