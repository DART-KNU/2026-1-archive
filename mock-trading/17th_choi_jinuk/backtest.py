from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pykrx import stock


DEFAULT_SELECTION_CSV = "quant_constrained_top20.csv"
DEFAULT_START = "20220224"
DEFAULT_END = "20220630"
DEFAULT_INITIAL_CAPITAL = 1_000_000_000
DEFAULT_MAX_TURNOVER = 0.05
DEFAULT_REBALANCE_RULE = "W-FRI"
KOSPI_INDEX_CODE = "1001"


def normalize_code(value: object) -> str:
    text = str(value).strip().replace("'", "")
    if text.endswith(".0"):
        text = text[:-2]
    text = text.replace("A", "").strip()
    return text.zfill(6)


def read_selection(selection_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(selection_path, encoding="utf-8-sig")
    if "종목코드" not in df.columns:
        raise ValueError("selection 파일에 '종목코드' 컬럼이 없습니다.")

    out = df.copy()
    out["종목코드"] = out["종목코드"].apply(normalize_code)

    if "final_weight" not in out.columns:
        out["final_weight"] = 1 / len(out)

    out["final_weight"] = pd.to_numeric(out["final_weight"], errors="coerce")
    out = out.dropna(subset=["종목코드", "final_weight"]).copy()

    if out.empty:
        raise ValueError("selection 파일이 비어 있습니다.")

    weight_sum = out["final_weight"].sum()
    if weight_sum <= 0:
        raise ValueError("가중치 합이 0 이하입니다.")
    out["final_weight"] = out["final_weight"] / weight_sum
    return out


def fetch_stock_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    series_list: list[pd.Series] = []
    valid_tickers: list[str] = []

    for ticker in tickers:
        df = stock.get_market_ohlcv_by_date(start, end, ticker)
        if df is None or df.empty:
            print(f"[경고] 가격 데이터 없음: {ticker}")
            continue

        close = pd.to_numeric(df["종가"], errors="coerce").replace(0, np.nan).dropna()
        if close.empty:
            print(f"[경고] 종가가 비어 있음: {ticker}")
            continue

        close.name = ticker
        series_list.append(close)
        valid_tickers.append(ticker)

    if not series_list:
        raise ValueError("불러온 종목 가격 데이터가 없습니다.")

    prices = pd.concat(series_list, axis=1).sort_index().ffill()
    prices = prices.loc[:, prices.iloc[0].notna()].copy()
    if prices.empty:
        raise ValueError("시작일 종가가 있는 종목이 없습니다.")

    return prices


def fetch_kospi_prices(start: str, end: str) -> pd.Series:
    kospi = stock.get_index_ohlcv_by_date(start, end, KOSPI_INDEX_CODE)
    if kospi is None or kospi.empty:
        raise ValueError("KOSPI 지수 데이터를 불러오지 못했습니다.")

    close_col = "종가" if "종가" in kospi.columns else kospi.columns[0]
    close = pd.to_numeric(kospi[close_col], errors="coerce").dropna()
    close.name = "KOSPI"
    return close


def get_target_weights(selection: pd.DataFrame, tickers: list[str]) -> pd.Series:
    target = selection.set_index("종목코드")["final_weight"].reindex(tickers).fillna(0.0)
    total = target.sum()
    if total <= 0:
        raise ValueError("유효 종목의 목표 비중 합이 0입니다.")
    return target / total


def get_rebalance_dates(index: pd.DatetimeIndex, rule: str = DEFAULT_REBALANCE_RULE) -> set[pd.Timestamp]:
    schedule = pd.Series(index=index, data=index).resample(rule).last().dropna()
    dates = set(pd.to_datetime(schedule.values))
    dates.add(pd.to_datetime(index[0]))
    return dates


def apply_turnover_limit(
    current_weights: pd.Series,
    target_weights: pd.Series,
    max_turnover: float,
) -> tuple[pd.Series, float]:
    current = current_weights.reindex(target_weights.index).fillna(0.0)
    target = target_weights.reindex(current.index).fillna(0.0)

    delta = target - current
    required_turnover = 0.5 * np.abs(delta).sum()

    if required_turnover <= max_turnover + 1e-12:
        new_weights = target.copy()
        actual_turnover = float(required_turnover)
    else:
        scale = max_turnover / required_turnover
        new_weights = current + scale * delta
        actual_turnover = float(max_turnover)

    new_weights = new_weights.clip(lower=0.0)
    new_weights = new_weights / new_weights.sum()
    return new_weights, actual_turnover


def run_backtest(
    prices: pd.DataFrame,
    target_weights: pd.Series,
    initial_capital: float,
    max_turnover: float,
    rebalance_rule: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    returns = prices.pct_change().fillna(0.0)
    rebalance_dates = get_rebalance_dates(prices.index, rule=rebalance_rule)

    tickers = prices.columns.tolist()
    current_weights = target_weights.reindex(tickers).fillna(0.0)
    current_weights = current_weights / current_weights.sum()

    portfolio_value = initial_capital
    prev_cum = 1.0

    result_rows: list[dict[str, object]] = []
    weight_history_rows: list[dict[str, object]] = []
    turnover_rows: list[dict[str, object]] = []

    for i, date in enumerate(prices.index):
        daily_ret = returns.loc[date]
        port_ret = float((current_weights * daily_ret).sum())
        portfolio_value *= (1 + port_ret)
        cum_ret = portfolio_value / initial_capital

        running_max = max([prev_cum] + [row["cum_ret"] for row in result_rows]) if result_rows else cum_ret
        drawdown = cum_ret / running_max - 1

        result_rows.append(
            {
                "Date": pd.to_datetime(date),
                "portfolio_value": portfolio_value,
                "portfolio_daily_return": port_ret,
                "cum_ret": cum_ret,
                "drawdown": drawdown,
            }
        )

        weight_row = {"Date": pd.to_datetime(date)}
        weight_row.update({ticker: float(current_weights.get(ticker, 0.0)) for ticker in tickers})
        weight_history_rows.append(weight_row)

        # 다음 날로 넘어가기 전에 비중 drift 반영
        growth = 1 + daily_ret
        denom = float((current_weights * growth).sum())
        if denom <= 0:
            raise RuntimeError("포트폴리오 가치가 0 이하가 되었습니다.")
        drifted_weights = (current_weights * growth) / denom

        # 당일 종가 기준 리밸런싱 -> 다음 거래일 시작 비중으로 사용
        actual_turnover = 0.0
        if pd.to_datetime(date) in rebalance_dates:
            current_weights, actual_turnover = apply_turnover_limit(
                current_weights=drifted_weights,
                target_weights=target_weights,
                max_turnover=max_turnover,
            )
        else:
            current_weights = drifted_weights

        turnover_rows.append(
            {
                "Date": pd.to_datetime(date),
                "turnover": actual_turnover,
                "is_rebalance_day": int(pd.to_datetime(date) in rebalance_dates),
            }
        )
        prev_cum = cum_ret

    result = pd.DataFrame(result_rows).set_index("Date")
    weight_history = pd.DataFrame(weight_history_rows).set_index("Date")
    turnover_history = pd.DataFrame(turnover_rows).set_index("Date")
    result["drawdown"] = result["cum_ret"] / result["cum_ret"].cummax() - 1
    return result, weight_history, turnover_history


def make_performance_summary(result: pd.DataFrame) -> dict[str, float]:
    total_return = float(result["cum_ret"].iloc[-1] - 1)
    mdd = float(result["drawdown"].min())

    daily_ret = result["portfolio_daily_return"].dropna()
    if len(daily_ret) > 1:
        annualized_return = float((1 + total_return) ** (252 / len(daily_ret)) - 1)
        annualized_vol = float(daily_ret.std(ddof=0) * np.sqrt(252))
        sharpe = float(annualized_return / annualized_vol) if annualized_vol > 0 else np.nan
    else:
        annualized_return = np.nan
        annualized_vol = np.nan
        sharpe = np.nan

    return {
        "total_return": total_return,
        "mdd": mdd,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_vol,
        "sharpe_like": sharpe,
    }


def save_plots(result: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(result.index, result["portfolio_cum_return"] * 100, label="Portfolio")
    ax.plot(result.index, result["kospi_cum_return"] * 100, label="KOSPI")
    ax.set_title("Cumulative Return: Portfolio vs KOSPI")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return (%)")
    ax.grid(True)
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%y.%m.%d"))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(out_dir / "cumulative_return.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    port_mdd = float(result["portfolio_drawdown"].min())
    kospi_mdd = float(result["kospi_drawdown"].min())
    ax.plot(result.index, result["portfolio_drawdown"] * 100, label=f"Portfolio (MDD {port_mdd:.2%})")
    ax.plot(result.index, result["kospi_drawdown"] * 100, label=f"KOSPI (MDD {kospi_mdd:.2%})")
    ax.set_title("Drawdown: Portfolio vs KOSPI")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (%)")
    ax.grid(True)
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%y.%m.%d"))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(out_dir / "drawdown.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="최종 선정 20종목 기준 백테스트 + 시각화")
    parser.add_argument("--selection", default=DEFAULT_SELECTION_CSV, help="최종 선정 CSV 경로")
    parser.add_argument("--start", default=DEFAULT_START, help="백테스트 시작일 YYYYMMDD")
    parser.add_argument("--end", default=DEFAULT_END, help="백테스트 종료일 YYYYMMDD")
    parser.add_argument("--initial-capital", type=float, default=DEFAULT_INITIAL_CAPITAL, help="초기 자본")
    parser.add_argument("--max-turnover", type=float, default=DEFAULT_MAX_TURNOVER, help="주간 회전율 한도")
    parser.add_argument("--rebalance-rule", default=DEFAULT_REBALANCE_RULE, help="pandas resample 규칙. 기본: W-FRI")
    parser.add_argument("--out-dir", default=".", help="결과 저장 폴더")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    selection = read_selection(args.selection)
    tickers = selection["종목코드"].tolist()

    prices = fetch_stock_prices(tickers=tickers, start=args.start, end=args.end)
    valid_tickers = prices.columns.tolist()
    target_weights = get_target_weights(selection, valid_tickers)

    result, weight_history, turnover_history = run_backtest(
        prices=prices,
        target_weights=target_weights,
        initial_capital=args.initial_capital,
        max_turnover=args.max_turnover,
        rebalance_rule=args.rebalance_rule,
    )

    kospi_close = fetch_kospi_prices(args.start, args.end).reindex(result.index).ffill()
    kospi_ret = kospi_close.pct_change().fillna(0.0)
    kospi_cum = (1 + kospi_ret).cumprod()
    kospi_drawdown = kospi_cum / kospi_cum.cummax() - 1

    result["portfolio_cum_return"] = result["cum_ret"] - 1
    result["portfolio_drawdown"] = result["drawdown"]
    result["kospi_cum_return"] = kospi_cum - 1
    result["kospi_drawdown"] = kospi_drawdown

    summary = make_performance_summary(result)
    kospi_total_return = float(result["kospi_cum_return"].iloc[-1])
    kospi_mdd = float(result["kospi_drawdown"].min())

    result.to_csv(out_dir / "backtest_result.csv", encoding="utf-8-sig")
    weight_history.to_csv(out_dir / "weight_history.csv", encoding="utf-8-sig")
    turnover_history.to_csv(out_dir / "turnover_history.csv", encoding="utf-8-sig")
    save_plots(result, out_dir)

    print("\n[백테스트 요약]")
    print(f"사용 종목 수: {len(valid_tickers)}")
    print(f"종목 코드: {valid_tickers}")
    print(f"포트폴리오 총수익률: {summary['total_return']:.2%}")
    print(f"포트폴리오 MDD: {summary['mdd']:.2%}")
    if pd.notna(summary["annualized_return"]):
        print(f"포트폴리오 연환산 수익률: {summary['annualized_return']:.2%}")
        print(f"포트폴리오 연환산 변동성: {summary['annualized_volatility']:.2%}")
        print(f"포트폴리오 샤프 비슷한 값: {summary['sharpe_like']:.2f}")
    print(f"KOSPI 총수익률: {kospi_total_return:.2%}")
    print(f"KOSPI MDD: {kospi_mdd:.2%}")
    print(f"최종 포트 가치: {result['portfolio_value'].iloc[-1]:,.0f} 원")
    print(f"평균 일일 회전율: {turnover_history['turnover'].mean():.2%}")
    print(f"최대 일일 회전율: {turnover_history['turnover'].max():.2%}")

    print("\n저장 완료")
    print(f"- {out_dir / 'backtest_result.csv'}")
    print(f"- {out_dir / 'weight_history.csv'}")
    print(f"- {out_dir / 'turnover_history.csv'}")
    print(f"- {out_dir / 'cumulative_return.png'}")
    print(f"- {out_dir / 'drawdown.png'}")


if __name__ == "__main__":
    main()
