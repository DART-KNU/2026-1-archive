"""
regime_analysis.py
Regime별 BTC 상관계수 비교 + t-test → BTC 정체성 케이스 분류

입력 : regime_data.csv

출력 :
  · regime_analysis_result.csv  — 날짜별 BTC 상관계수 + Regime 라벨
  · ttest_summary.txt           — t-test 결과 및 케이스 분류
  · fig_regime_boxplot.png      — Regime별 BTC 상관계수 박스플롯
  · fig_regime_timeseries.png   — BTC 상관계수 시계열 + Regime 띠

케이스 분류
  케이스 1 (디지털 금)    : A·B 모두 BTC-금↑ 유지
  케이스 2 (디지털 은)    : A에서 BTC-금↑, B에서 BTC-구리↑  (이중적)
  케이스 3 (순수 위험자산): A·B 모두 BTC-구리↑
  케이스 4 (혼합/불명확) : 패턴 없음
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
from scipy.stats import ttest_ind

# ── 경로 ─────────────────────────────────────────────────────────────────────
INPUT_PATH  = "regime_data.csv"
OUT_CSV     = "regime_analysis_result.csv"
OUT_SUMMARY = "ttest_summary.txt"
FIG_BOX     = "fig_regime_boxplot.png"
FIG_TS      = "fig_regime_timeseries.png"

# ── 파라미터 ──────────────────────────────────────────────────────────────────
WINDOW      = 60          # Rolling Correlation 창 (일)
ALPHA       = 0.05        # 유의수준

# ── 색상 ──────────────────────────────────────────────────────────────────────
COLOR_A   = "#1565C0"    # Regime A (귀금속)
COLOR_B   = "#BF360C"    # Regime B (산업재)
COLOR_AU  = "#FFC107"    # 금
COLOR_AG  = "#90A4AE"    # 은
COLOR_CU  = "#8D4E2A"    # 구리


# ════════════════════════════════════════════════════════════════════════════
# 1. 데이터 로드 + BTC 상관계수 계산
# ════════════════════════════════════════════════════════════════════════════
def load_and_compute(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig",
                     parse_dates=["Date"], index_col="Date")
    df.sort_index(inplace=True)

    print("[데이터 로드]")
    print(f"  전체: {len(df)}행  ({df.index[0].date()} ~ {df.index[-1].date()})")

    # BTC-금/은/구리 Rolling Correlation (60일) 계산
    df["Corr_BTC_금"] = (df["BTC_Return"]
                         .rolling(WINDOW)
                         .corr(df["Gold_Return"]))
    df["Corr_BTC_은"] = (df["BTC_Return"]
                         .rolling(WINDOW)
                         .corr(df["Silver_Return"]))
    df["Corr_BTC_구리"] = (df["BTC_Return"]
                           .rolling(WINDOW)
                           .corr(df["Copper_Return"]))

    # Regime 있고 BTC 상관계수도 유효한 행만
    analysis_cols = ["Corr_BTC_금", "Corr_BTC_은", "Corr_BTC_구리",
                     "Regime_Label", "Regime"]
    sub = df[analysis_cols].dropna()

    print(f"  분석 가능: {len(sub)}행  "
          f"({sub.index[0].date()} ~ {sub.index[-1].date()})")
    print(f"  Regime A: {(sub['Regime_Label']=='A').sum()}일  "
          f"/ Regime B: {(sub['Regime_Label']=='B').sum()}일")

    return df, sub


# ════════════════════════════════════════════════════════════════════════════
# 2. Regime별 기술통계
# ════════════════════════════════════════════════════════════════════════════
def descriptive_stats(sub: pd.DataFrame) -> dict:
    groups = {}
    for label in ["A", "B"]:
        g = sub[sub["Regime_Label"] == label]
        groups[label] = {
            "BTC_금"  : g["Corr_BTC_금"].values,
            "BTC_은"  : g["Corr_BTC_은"].values,
            "BTC_구리": g["Corr_BTC_구리"].values,
        }

    print("\n[Regime별 BTC 상관계수 평균]")
    print(f"{'':>12} {'Regime A (귀금속)':>18} {'Regime B (산업재)':>18}")
    print("-" * 50)
    for name in ["BTC_금", "BTC_은", "BTC_구리"]:
        mA = groups["A"][name].mean()
        mB = groups["B"][name].mean()
        print(f"  {name:>10}   {mA:>+.4f}              {mB:>+.4f}")

    return groups


# ════════════════════════════════════════════════════════════════════════════
# 3. t-test (+ Mann-Whitney U test 보조)
# ════════════════════════════════════════════════════════════════════════════
def run_ttests(groups: dict) -> list:
    """
    귀무가설: Regime A와 B에서 BTC-X 평균 상관계수가 같다
    대립가설: 다르다 (two-sided)
    """
    results = []
    targets = [
        ("BTC_금",   "BTC–Gold"),
        ("BTC_은",   "BTC–Silver"),
        ("BTC_구리", "BTC–Copper"),
    ]

    print("\n[t-test 결과]")
    print(f"{'':>12} {'mean_A':>8} {'mean_B':>8} {'t-stat':>8} "
          f"{'p-value':>10} {'유의(5%)':>9}")
    print("-" * 60)

    for key, label in targets:
        a = groups["A"][key]
        b = groups["B"][key]

        t_stat, p_val = ttest_ind(a, b, equal_var=False)   # Welch t-test

        sig    = "✓" if p_val < ALPHA else "✗"
        mean_A = a.mean()
        mean_B = b.mean()

        print(f"  {label:>12}  {mean_A:>+.4f}  {mean_B:>+.4f}  "
              f"{t_stat:>+8.3f}  {p_val:>10.4f}  {sig:>5}")

        results.append({
            "label"  : label,
            "key"    : key,
            "mean_A" : mean_A,
            "mean_B" : mean_B,
            "t_stat" : t_stat,
            "p_value": p_val,
            "sig"    : p_val < ALPHA,
        })

    return results


# ════════════════════════════════════════════════════════════════════════════
# 4. 케이스 분류
# ════════════════════════════════════════════════════════════════════════════
def classify_case(results: list) -> str:
    """
    BTC 정체성 케이스 분류 로직

    핵심 판단 기준:
      - Regime A에서 BTC-금 vs BTC-구리 중 어느 쪽이 더 높은가?
      - Regime B에서 BTC-금 vs BTC-구리 중 어느 쪽이 더 높은가?
      - 두 Regime 간 차이가 통계적으로 유의한가?
    """
    r = {x["key"]: x for x in results}

    mean_A_금   = r["BTC_금"]["mean_A"]
    mean_B_금   = r["BTC_금"]["mean_B"]
    mean_A_구리 = r["BTC_구리"]["mean_A"]
    mean_B_구리 = r["BTC_구리"]["mean_B"]

    sig_금   = r["BTC_금"]["sig"]
    sig_구리 = r["BTC_구리"]["sig"]

    # Regime A: 금 우세 여부
    A_금_우세   = mean_A_금 > mean_A_구리
    # Regime B: 구리 우세 여부
    B_구리_우세 = mean_B_구리 > mean_B_금

    print("\n[케이스 분류 판단 기준]")
    print(f"  Regime A: BTC-금({mean_A_금:+.4f}) vs BTC-구리({mean_A_구리:+.4f})"
          f"  → {'금 우세' if A_금_우세 else '구리 우세'}")
    print(f"  Regime B: BTC-금({mean_B_금:+.4f}) vs BTC-구리({mean_B_구리:+.4f})"
          f"  → {'금 우세' if not B_구리_우세 else '구리 우세'}")
    print(f"  BTC-금 Regime간 차이 유의: {sig_금}")
    print(f"  BTC-구리 Regime간 차이 유의: {sig_구리}")

    # ── 분류 로직 ────────────────────────────────────────────────────────
    if A_금_우세 and not B_구리_우세:
        # 두 Regime 모두 금 우세
        case = "케이스 1: 디지털 금 (Digital Gold)"
        desc = ("Regime 무관하게 항상 BTC-금 상관이 BTC-구리보다 높음.\n"
                "BTC는 Regime에 관계없이 금과 유사하게 행동.")

    elif A_금_우세 and B_구리_우세 and (sig_금 or sig_구리):
        # A에서 금 우세, B에서 구리 우세 + 유의함
        case = "케이스 2: 디지털 은 (Digital Silver)"
        desc = ("Regime A(귀금속 국면)에서는 BTC-금 상관 우세,\n"
                "Regime B(산업재 국면)에서는 BTC-구리 상관 우세.\n"
                "BTC는 은처럼 시장 국면에 따라 정체성이 바뀌는 이중적 자산.")

    elif not A_금_우세 and B_구리_우세:
        # 두 Regime 모두 구리 우세
        case = "케이스 3: 순수 위험자산 (Pure Risk Asset)"
        desc = ("Regime 무관하게 항상 BTC-구리 상관이 BTC-금보다 높음.\n"
                "BTC는 Regime에 관계없이 위험자산으로 행동.")

    else:
        case = "케이스 4: 혼합/불명확 (Mixed)"
        desc = "일관된 패턴이 없어 정체성을 단일하게 분류하기 어려움."

    print(f"\n{'='*55}")
    print(f"  판정: {case}")
    print(f"{'='*55}")
    print(f"  {desc}")

    return case, desc


# ════════════════════════════════════════════════════════════════════════════
# 5. 결과 저장
# ════════════════════════════════════════════════════════════════════════════
def save_csv(df: pd.DataFrame, sub: pd.DataFrame):
    out = df.copy()
    out.to_csv(OUT_CSV, encoding="utf-8-sig")
    print(f"\n[저장] {OUT_CSV}")


def save_summary(results: list, case: str, desc: str):
    lines = []
    lines.append("=" * 60)
    lines.append("  BTC 정체성 분석: Regime별 t-test 결과  (STEP 4)")
    lines.append("=" * 60)
    lines.append(f"\n분석 창    : {WINDOW}일 Rolling Correlation")
    lines.append(f"유의수준   : α = {ALPHA}")
    lines.append(f"검정 방법  : Welch's t-test (two-sided)")
    lines.append("")
    lines.append("── Regime별 BTC 평균 상관계수 및 t-test ──")
    lines.append(f"{'':>14} {'mean_A':>8} {'mean_B':>8} "
                 f"{'t-stat':>8} {'p-value':>10} {'유의':>5}")
    lines.append("-" * 58)
    for r in results:
        sig = "✓" if r["sig"] else "✗"
        lines.append(f"  {r['label']:>12}  {r['mean_A']:>+.4f}  {r['mean_B']:>+.4f}  "
                     f"{r['t_stat']:>+8.3f}  {r['p_value']:>10.4f}  {sig:>5}")
    lines.append("")
    lines.append("── 케이스 분류 결과 ──")
    lines.append(f"  {case}")
    lines.append(f"  {desc}")

    text = "\n".join(lines)
    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[저장] {OUT_SUMMARY}")
    print("\n" + text)


# ════════════════════════════════════════════════════════════════════════════
# 6. 시각화
# ════════════════════════════════════════════════════════════════════════════
def plot_boxplot(groups: dict, results: list, case: str):
    """
    3×1 박스플롯: BTC-금 / BTC-은 / BTC-구리
    Regime A(파랑) vs Regime B(주황)
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 6))
    fig.suptitle(
        f"BTC Correlation by Silver Regime\n판정: {case}",
        fontsize=12, fontweight="bold"
    )

    targets = [
        ("BTC_금",   "BTC–Gold",   COLOR_AU),
        ("BTC_은",   "BTC–Silver", COLOR_AG),
        ("BTC_구리", "BTC–Copper", COLOR_CU),
    ]
    r_dict = {x["key"]: x for x in results}

    for ax, (key, label, color) in zip(axes, targets):
        data_A = groups["A"][key]
        data_B = groups["B"][key]

        bp = ax.boxplot(
            [data_A, data_B],
            patch_artist=True,
            medianprops=dict(color="white", linewidth=2),
            whiskerprops=dict(linewidth=1.2),
            capprops=dict(linewidth=1.2),
            flierprops=dict(marker="o", markersize=2, alpha=0.3),
            widths=0.5,
        )
        bp["boxes"][0].set_facecolor(COLOR_A)
        bp["boxes"][0].set_alpha(0.7)
        bp["boxes"][1].set_facecolor(COLOR_B)
        bp["boxes"][1].set_alpha(0.7)

        # 평균 표시
        ax.scatter([1, 2],
                   [data_A.mean(), data_B.mean()],
                   color="white", s=50, zorder=5, label="mean")

        # p-value 표시
        r = r_dict[key]
        sig_txt = (f"p = {r['p_value']:.4f} ✓"
                   if r["sig"] else f"p = {r['p_value']:.4f}")
        ax.set_title(f"{label}\n{sig_txt}", fontsize=10)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Regime A\n(귀금속)", "Regime B\n(산업재)"],
                           fontsize=9)
        ax.set_ylabel("Rolling Correlation (60d)", fontsize=9)
        ax.axhline(0, color="gray", lw=0.7, ls="--", alpha=0.5)
        ax.grid(axis="y", alpha=0.3)

    patch_A = mpatches.Patch(color=COLOR_A, alpha=0.7, label="Regime A (귀금속)")
    patch_B = mpatches.Patch(color=COLOR_B, alpha=0.7, label="Regime B (산업재)")
    fig.legend(handles=[patch_A, patch_B],
               loc="lower center", ncol=2, fontsize=10,
               bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(FIG_BOX, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[그림] {FIG_BOX}")


def plot_timeseries(df: pd.DataFrame, sub: pd.DataFrame):
    """
    3-panel 시계열: BTC-금/은/구리 상관계수 + Regime 띠
    """
    dates  = sub.index
    regime = sub["Regime"].values

    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    fig.suptitle(
        "BTC Rolling Correlation (60d) by Silver Regime\n"
        "Regime A = 귀금속 국면 │ Regime B = 산업재 국면",
        fontsize=12, fontweight="bold"
    )

    series = [
        ("Corr_BTC_금",   "BTC–Gold Corr (60d)",   COLOR_AU),
        ("Corr_BTC_은",   "BTC–Silver Corr (60d)", COLOR_AG),
        ("Corr_BTC_구리", "BTC–Copper Corr (60d)", COLOR_CU),
    ]

    for ax, (col, label, color) in zip(axes, series):
        # Regime 배경 띠
        i = 0
        while i < len(regime):
            j = i
            while j < len(regime) and regime[j] == regime[i]:
                j += 1
            c = COLOR_A if regime[i] == 0 else COLOR_B
            ax.axvspan(dates[i], dates[min(j, len(dates)-1)],
                       alpha=0.13, color=c, lw=0)
            i = j

        ax.plot(dates, sub[col].values, color=color, lw=0.9, label=label)
        ax.axhline(0, color="gray", lw=0.6, ls="--", alpha=0.5)
        ax.set_ylabel("Correlation", fontsize=9)
        ax.legend(loc="upper right", fontsize=9)
        ax.set_ylim(-1.05, 1.05)

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1].set_xlabel("Date", fontsize=10)

    patch_A = mpatches.Patch(color=COLOR_A, alpha=0.4, label="Regime A (귀금속)")
    patch_B = mpatches.Patch(color=COLOR_B, alpha=0.4, label="Regime B (산업재)")
    fig.legend(handles=[patch_A, patch_B],
               loc="lower center", ncol=2, fontsize=10,
               bbox_to_anchor=(0.5, -0.01))

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(FIG_TS, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[그림] {FIG_TS}")


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("Regime별 BTC 상관계수 비교 + t-test")
    print("=" * 60)

    # 1. 로드 + BTC 상관계수 계산
    df, sub = load_and_compute(INPUT_PATH)

    # 2. 기술통계
    groups = descriptive_stats(sub)

    # 3. t-test
    results = run_ttests(groups)

    # 4. 케이스 분류
    case, desc = classify_case(results)

    # 5. 저장
    save_csv(df, sub)
    save_summary(results, case, desc)

    # 6. 시각화
    plot_boxplot(groups, results, case)
    plot_timeseries(df, sub)

    print("\n" + "=" * 60)