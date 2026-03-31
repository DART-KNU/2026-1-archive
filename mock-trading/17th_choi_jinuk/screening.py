from __future__ import annotations

import argparse
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from pykrx import stock
from scipy.optimize import Bounds, LinearConstraint, milp


DEFAULT_UNIVERSE_CSV = "유니버스.csv"
DEFAULT_OP_XLSX = "영업이익.xlsx"
DEFAULT_OUT_FULL = "quant_factor_rank_full_constrained.csv"
DEFAULT_OUT_TOP20 = "quant_constrained_top20.csv"
DEFAULT_OUT_SECTOR = "quant_sector_caps.csv"

TOP_N = 20
FINAL_EQUAL_WEIGHT = 1.0 / TOP_N
SMALLCAP_THRESHOLD_WON = 1_000_000_000_000  # 1조원
SMALLCAP_PORT_CAP = 0.30
DEFAULT_LOOKBACK_DAYS = 15
DEFAULT_PRICE_LOOKBACK_CALENDAR_DAYS = 90
DEFAULT_SLEEP_SEC = 0.15


def normalize_code(value: object) -> str:
    """종목코드를 6자리 문자열로 정리."""
    text = str(value).strip().replace("'", "")
    if text.endswith(".0"):
        text = text[:-2]
    text = text.replace("A", "").strip()
    return text.zfill(6)


def zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    mean = values.mean()
    std = values.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(np.nan, index=series.index)
    return (values - mean) / std


def read_csv_with_fallback(path: str | Path) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "cp949", "utf-8"):
        try:
            return pd.read_csv(path, dtype=str, encoding=encoding)
        except Exception as exc:  # pragma: no cover - 환경별 인코딩 예외 대응
            last_error = exc
    raise RuntimeError(f"CSV를 읽지 못했습니다: {path}\n원인: {last_error}")


def find_first_existing_column(columns: Iterable[str], candidates: list[str]) -> str:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise ValueError(f"필요한 컬럼을 찾지 못했습니다. 후보: {candidates}")


def load_universe(csv_path: str) -> pd.DataFrame:
    df = read_csv_with_fallback(csv_path)

    code_col = find_first_existing_column(df.columns, ["종목코드", "ticker", "Ticker", "code", "Code"])
    name_col = find_first_existing_column(df.columns, ["기업명", "종목명", "Name", "name"])
    sector_col = find_first_existing_column(
        df.columns,
        ["GICS_sector", "GICS 산업군", "GICS Sector", "sector", "Sector"],
    )
    market_cap_col = find_first_existing_column(
        df.columns,
        ["시가총액", "KRX_시가총액", "market_cap", "MarketCap", "Market Cap"],
    )

    universe = df[[name_col, code_col, sector_col, market_cap_col]].copy()
    universe.columns = ["기업명", "종목코드", "GICS_sector", "시가총액"]

    universe["종목코드"] = universe["종목코드"].apply(normalize_code)
    universe["기업명"] = universe["기업명"].astype(str).str.strip()
    universe["GICS_sector"] = universe["GICS_sector"].astype(str).str.strip()
    universe["시가총액"] = pd.to_numeric(universe["시가총액"].str.replace(",", "", regex=False), errors="coerce")

    universe = universe.dropna(subset=["종목코드", "기업명", "GICS_sector", "시가총액"]).copy()
    universe = universe.drop_duplicates(subset="종목코드", keep="first").reset_index(drop=True)

    if universe.empty:
        raise ValueError("유니버스가 비어 있습니다. 입력 파일을 확인하세요.")

    return universe


def extract_latest_op_yoy(excel_path: str, target_codes: set[str]) -> pd.DataFrame:
    try:
        raw = pd.read_excel(excel_path, sheet_name="영업이익", header=None)
    except ValueError:
        raise ValueError("영업이익.xlsx 안에서 '영업이익' 시트를 찾지 못했습니다.")

    # 기존 업로드 코드의 DataGuide 구조를 그대로 사용
    names = raw.iloc[7, 2:]
    codes = raw.iloc[8, 2:].astype(str).map(normalize_code)
    item_names = raw.iloc[11, 2:].astype(str).str.strip()

    valid_cols: list[int] = []
    for col_idx, (code, item_name) in enumerate(zip(codes, item_names), start=2):
        if code in target_codes and item_name == "영업이익(천원)":
            valid_cols.append(col_idx)

    years = pd.to_numeric(raw.iloc[13:, 0], errors="coerce")
    quarters = raw.iloc[13:, 1].astype(str).str.strip()

    records: list[dict[str, object]] = []
    for col_idx in valid_cols:
        code = normalize_code(raw.iat[8, col_idx])
        name = str(raw.iat[7, col_idx]).strip()
        values = pd.to_numeric(raw.iloc[13:, col_idx], errors="coerce")

        frame = pd.DataFrame(
            {
                "year": years.values,
                "quarter": quarters.values,
                "op": values.values,
            }
        )
        frame = frame.dropna(subset=["year"]).copy()
        if frame.empty:
            continue

        frame["year"] = frame["year"].astype(int)
        non_null = frame.dropna(subset=["op"]).copy()
        if non_null.empty:
            continue

        latest = non_null.iloc[-1]
        prev = frame[
            (frame["year"] == int(latest["year"]) - 1)
            & (frame["quarter"] == str(latest["quarter"]))
        ]
        if prev.empty:
            continue

        prev_op = pd.to_numeric(prev["op"].iloc[0], errors="coerce")
        curr_op = pd.to_numeric(latest["op"], errors="coerce")
        if pd.isna(prev_op) or pd.isna(curr_op) or prev_op == 0:
            continue

        op_yoy = (curr_op - prev_op) / abs(prev_op)
        records.append(
            {
                "종목코드": code,
                "영업이익_엑셀기업명": name,
                "최근발표분기": f"{int(latest['year'])}{latest['quarter']}",
                "전년동기": f"{int(latest['year']) - 1}{latest['quarter']}",
                "최근분기_영업이익_천원": curr_op,
                "전년동기_영업이익_천원": prev_op,
                "영업이익YoY": op_yoy,
            }
        )

    op_df = pd.DataFrame(records)
    if op_df.empty:
        raise ValueError(
            "영업이익 YoY를 계산하지 못했습니다.\n"
            "영업이익.xlsx의 시트명/행 구조가 기존 코드와 다른지 확인하세요."
        )

    op_df = op_df.drop_duplicates(subset="종목코드", keep="last").reset_index(drop=True)
    return op_df


def find_last_trading_day(lookback_days: int = DEFAULT_LOOKBACK_DAYS) -> str:
    today = datetime.today()
    probe_code = "005930"

    for delta in range(lookback_days):
        candidate = (today - timedelta(days=delta)).strftime("%Y%m%d")
        try:
            px = stock.get_market_ohlcv_by_date(candidate, candidate, probe_code)
            if px is not None and not px.empty:
                close = pd.to_numeric(px["종가"], errors="coerce").dropna()
                close = close[close > 0]
                if not close.empty:
                    return candidate
        except Exception:
            continue

    raise RuntimeError("최근 거래일을 찾지 못했습니다.")


def fetch_price_factors(
    codes: list[str],
    end_date: str,
    min_points: int = 21,
    lookback_calendar_days: int = DEFAULT_PRICE_LOOKBACK_CALENDAR_DAYS,
    sleep_sec: float = DEFAULT_SLEEP_SEC,
) -> pd.DataFrame:
    """
    가격 팩터 계산
    - momentum_20d_mean: 최근 20거래일 일별 수익률 평균
    - volatility_20d: 최근 20거래일 일별 수익률 표준편차
    """
    start_date = (
        datetime.strptime(end_date, "%Y%m%d") - timedelta(days=lookback_calendar_days)
    ).strftime("%Y%m%d")

    rows: list[dict[str, object]] = []
    for idx, code in enumerate(codes, start=1):
        try:
            px = stock.get_market_ohlcv_by_date(start_date, end_date, code)
            if px is None or px.empty:
                continue

            close = pd.to_numeric(px["종가"], errors="coerce").dropna()
            close = close[close > 0]
            if len(close) < min_points:
                continue

            daily_ret = close.pct_change().dropna().tail(20)
            if len(daily_ret) < 20:
                continue

            momentum_20d_mean = daily_ret.mean()
            volatility_20d = daily_ret.std(ddof=0)

            rows.append(
                {
                    "종목코드": code,
                    "momentum_20d_mean": momentum_20d_mean,
                    "volatility_20d": volatility_20d,
                    "가격기준일": pd.to_datetime(close.index[-1]).strftime("%Y-%m-%d"),
                }
            )
        except Exception as exc:
            print(f"[WARN] {code} 가격 데이터 조회 실패: {exc}")

        if idx % 30 == 0:
            print(f"가격 데이터 진행: {idx}/{len(codes)}")
            time.sleep(sleep_sec)

    price_df = pd.DataFrame(rows)
    if price_df.empty:
        raise ValueError("가격 팩터를 계산하지 못했습니다.")
    return price_df


def build_rank_table(universe: pd.DataFrame, op_df: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
    merged = universe.merge(op_df, on="종목코드", how="left")
    merged = merged.merge(price_df, on="종목코드", how="left")

    rank_df = merged.dropna(
        subset=["영업이익YoY", "momentum_20d_mean", "volatility_20d", "GICS_sector", "시가총액"]
    ).copy()
    if rank_df.empty:
        raise ValueError("팩터 3개가 모두 계산된 종목이 없습니다.")

    rank_df["is_smallcap_lt_1tn"] = rank_df["시가총액"] < SMALLCAP_THRESHOLD_WON

    rank_df["z_op_yoy"] = zscore(rank_df["영업이익YoY"])
    rank_df["z_momentum"] = zscore(rank_df["momentum_20d_mean"])
    rank_df["z_lowvol"] = -zscore(rank_df["volatility_20d"])

    rank_df["composite_score"] = (
        rank_df["z_op_yoy"] + rank_df["z_momentum"] + rank_df["z_lowvol"]
    ) / 3.0

    rank_df = rank_df.dropna(subset=["composite_score"]).copy()
    rank_df = rank_df.sort_values("composite_score", ascending=False).reset_index(drop=True)
    rank_df["unconstrained_rank"] = np.arange(1, len(rank_df) + 1)
    return rank_df


def compute_sector_caps(universe: pd.DataFrame) -> pd.DataFrame:
    sector_mc = universe.groupby("GICS_sector", dropna=False)["시가총액"].sum().sort_values(ascending=False)
    total_mc = sector_mc.sum()

    sector_df = pd.DataFrame(
        {
            "GICS_sector": sector_mc.index,
            "sector_market_cap": sector_mc.values,
            "sector_market_weight": sector_mc.values / total_mc,
        }
    )
    sector_df["sector_port_cap"] = np.where(
        sector_df["sector_market_weight"] <= 0.05,
        0.10,
        sector_df["sector_market_weight"] * 2.0,
    )
    sector_df["max_names_at_equal_weight"] = np.floor(
        sector_df["sector_port_cap"] / FINAL_EQUAL_WEIGHT + 1e-12
    ).astype(int)
    return sector_df


def solve_top_n_selection(rank_df: pd.DataFrame, sector_caps: pd.DataFrame) -> pd.DataFrame:
    df = rank_df.copy().reset_index(drop=True)
    n = len(df)
    if n < TOP_N:
        raise RuntimeError(f"랭킹 가능 종목 수({n})가 TOP_N({TOP_N})보다 적습니다.")

    sector_cap_map = sector_caps.set_index("GICS_sector")["max_names_at_equal_weight"].to_dict()
    smallcap_max_names = int(np.floor(SMALLCAP_PORT_CAP / FINAL_EQUAL_WEIGHT + 1e-12))

    objective = -df["composite_score"].to_numpy(dtype=float)
    integrality = np.ones(n, dtype=int)
    bounds = Bounds(lb=np.zeros(n), ub=np.ones(n))
    constraints: list[LinearConstraint] = []

    # 정확히 20종목 선택
    A_total = np.ones((1, n), dtype=float)
    constraints.append(
        LinearConstraint(A_total, lb=np.array([TOP_N], dtype=float), ub=np.array([TOP_N], dtype=float))
    )

    # 섹터 비중 한도 -> 최대 종목 수로 변환
    sector_array = df["GICS_sector"].to_numpy()
    for sector, max_names in sector_cap_map.items():
        idx = np.where(sector_array == sector)[0]
        if len(idx) == 0:
            continue
        A_sector = np.zeros((1, n), dtype=float)
        A_sector[0, idx] = 1.0
        constraints.append(
            LinearConstraint(A_sector, lb=np.array([-np.inf]), ub=np.array([max_names], dtype=float))
        )

    # 시가총액 1조 미만 종목 총합 30% 이하 -> 최대 6종목
    small_idx = np.where(df["is_smallcap_lt_1tn"].to_numpy())[0]
    if len(small_idx) > 0:
        A_small = np.zeros((1, n), dtype=float)
        A_small[0, small_idx] = 1.0
        constraints.append(
            LinearConstraint(A_small, lb=np.array([-np.inf]), ub=np.array([smallcap_max_names], dtype=float))
        )

    result = milp(c=objective, integrality=integrality, bounds=bounds, constraints=constraints)
    if not result.success:
        raise RuntimeError(f"최적화 실패: {result.message}")

    selected_flag = np.rint(result.x).astype(int)
    selected = df.loc[selected_flag == 1].copy()
    if len(selected) != TOP_N:
        raise RuntimeError(f"최종 선정 종목 수가 20개가 아닙니다: {len(selected)}개")

    selected = selected.sort_values(["composite_score", "unconstrained_rank"], ascending=[False, True]).reset_index(drop=True)
    selected["constrained_rank"] = np.arange(1, len(selected) + 1)
    selected["final_weight"] = FINAL_EQUAL_WEIGHT
    selected["stock_weight_cap"] = np.where(selected["종목코드"] == "005930", 0.40, 0.15)
    selected["stock_weight_cap_pass"] = selected["final_weight"] <= selected["stock_weight_cap"] + 1e-12
    return selected


def build_sector_summary(selected: pd.DataFrame, sector_caps: pd.DataFrame) -> pd.DataFrame:
    port_sector_weight = (
        selected.groupby("GICS_sector")["final_weight"].sum().rename("portfolio_weight").reset_index()
    )
    summary = sector_caps.merge(port_sector_weight, on="GICS_sector", how="left")
    summary["portfolio_weight"] = summary["portfolio_weight"].fillna(0.0)
    summary["constraint_pass"] = summary["portfolio_weight"] <= summary["sector_port_cap"] + 1e-12
    return summary.sort_values("portfolio_weight", ascending=False).reset_index(drop=True)


def save_outputs(
    rank_df: pd.DataFrame,
    selected: pd.DataFrame,
    sector_summary: pd.DataFrame,
    out_full: str,
    out_top20: str,
    out_sector: str,
) -> None:
    rank_cols = [
        "unconstrained_rank",
        "기업명",
        "종목코드",
        "GICS_sector",
        "시가총액",
        "is_smallcap_lt_1tn",
        "최근발표분기",
        "전년동기",
        "최근분기_영업이익_천원",
        "전년동기_영업이익_천원",
        "영업이익YoY",
        "momentum_20d_mean",
        "volatility_20d",
        "z_op_yoy",
        "z_momentum",
        "z_lowvol",
        "composite_score",
        "가격기준일",
    ]
    selected_cols = [
        "constrained_rank",
        "unconstrained_rank",
        "기업명",
        "종목코드",
        "GICS_sector",
        "시가총액",
        "is_smallcap_lt_1tn",
        "최근발표분기",
        "영업이익YoY",
        "momentum_20d_mean",
        "volatility_20d",
        "z_op_yoy",
        "z_momentum",
        "z_lowvol",
        "composite_score",
        "final_weight",
        "stock_weight_cap",
        "stock_weight_cap_pass",
    ]

    rank_df[rank_cols].to_csv(out_full, index=False, encoding="utf-8-sig")
    selected[selected_cols].to_csv(out_top20, index=False, encoding="utf-8-sig")
    sector_summary.to_csv(out_sector, index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser(description="대회용 3팩터 스크리닝 + 제약 반영 상위 20종목 선택")
    parser.add_argument("--universe", default=DEFAULT_UNIVERSE_CSV, help="유니버스 CSV 경로")
    parser.add_argument("--op", default=DEFAULT_OP_XLSX, help="영업이익.xlsx 경로")
    parser.add_argument("--out-full", default=DEFAULT_OUT_FULL, help="전체 랭킹 CSV 저장 경로")
    parser.add_argument("--out-top20", default=DEFAULT_OUT_TOP20, help="최종 20종목 CSV 저장 경로")
    parser.add_argument("--out-sector", default=DEFAULT_OUT_SECTOR, help="섹터 제약 요약 CSV 저장 경로")
    parser.add_argument("--end-date", default=None, help="가격 팩터 기준일(YYYYMMDD). 비우면 최근 거래일 자동 탐색")
    args = parser.parse_args()

    universe = load_universe(args.universe)
    target_codes = set(universe["종목코드"])
    print(f"유니버스 종목 수: {len(universe)}")

    op_df = extract_latest_op_yoy(args.op, target_codes)
    print(f"영업이익 YoY 계산 가능 종목 수: {len(op_df)}")

    end_date = args.end_date or find_last_trading_day()
    print(f"가격 팩터 기준일: {end_date}")

    price_df = fetch_price_factors(sorted(target_codes), end_date=end_date)
    print(f"가격 팩터 계산 가능 종목 수: {len(price_df)}")

    rank_df = build_rank_table(universe=universe, op_df=op_df, price_df=price_df)
    print(f"최종 팩터 랭킹 가능 종목 수: {len(rank_df)}")

    sector_caps = compute_sector_caps(universe)
    selected = solve_top_n_selection(rank_df, sector_caps)
    sector_summary = build_sector_summary(selected, sector_caps)

    smallcap_total = selected.loc[selected["is_smallcap_lt_1tn"], "final_weight"].sum()
    print(f"1조 미만 종목 비중: {smallcap_total:.1%}")

    save_outputs(rank_df, selected, sector_summary, args.out_full, args.out_top20, args.out_sector)

    print("\n저장 완료")
    print(f"- 전체 랭킹: {args.out_full}")
    print(f"- 최종 20개: {args.out_top20}")
    print(f"- 섹터 요약: {args.out_sector}")
    print("\n[최종 20개 종목]")
    print(
        selected[
            ["constrained_rank", "기업명", "종목코드", "GICS_sector", "composite_score", "final_weight"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
