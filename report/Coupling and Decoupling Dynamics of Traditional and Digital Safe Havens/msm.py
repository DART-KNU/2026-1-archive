"""
Markov Switching Model (2-state)

입력 : merged_data.csv
       feature: Corr_은_금_60d, Corr_은_구리_60d

출력 :
  · regime_data.csv           — 날짜별 Regime 판정(A/B) + 사후확률
  · msm_summary.txt           — 모델 파라미터 요약
  · fig_msm_prob.png          — Smoothed Posterior + 상관계수 시계열
  · fig_msm_regime_band.png   — Regime 띠 + 은-금/은-구리 Correlation

Regime 정의
  Regime A (귀금속 국면) : 은-금 상관↑, 은-구리 상관↓  → 안전자산 모드
  Regime B (산업재 국면) : 은-금 상관↓, 은-구리 상관↑  → 위험선호 모드

방법론 : Multivariate Gaussian HMM (= Markov Switching Model)
         Baum-Welch EM 알고리즘, scipy/numpy 순수 구현
"""

# ── 라이브러리 ───────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
from matplotlib import font_manager as fm
from scipy.stats import multivariate_normal

# ── 경로 설정 ─────────────────────────────────────────────────────────────────
INPUT_PATH  = "merged_data.csv"
OUT_CSV     = "regime_data.csv"
OUT_SUMMARY = "msm_summary.txt"
FIG_PROB    = "fig_msm_prob.png"
FIG_BAND    = "fig_msm_regime_band.png"

# ── 파라미터 ──────────────────────────────────────────────────────────────────
FEAT_AU   = "Corr_은_금_60d"    # 은-금  60일 Rolling Corr
FEAT_CU   = "Corr_은_구리_60d"  # 은-구리 60일 Rolling Corr
K         = 2                   # Regime 수
MAX_ITER  = 200                 # EM 최대 반복
TOL       = 1e-6                # 수렴 기준 (log-likelihood 변화량)
RANDOM_SEED = 42

# ── 색상 ──────────────────────────────────────────────────────────────────────
COLOR_A    = "#1565C0"   # Regime A: 귀금속 국면 (진파랑)
COLOR_B    = "#BF360C"   # Regime B: 산업재 국면 (진주황)
ALPHA_BAND = 0.15
COLOR_AU   = "#FFC107"   # 금색
COLOR_CU   = "#8D4E2A"   # 구리색

def configure_korean_font():
    """Select a Korean-capable font so Hangul labels render correctly."""
    preferred_fonts = [
        "Malgun Gothic",
        "AppleGothic",
        "NanumGothic",
        "Noto Sans CJK KR",
    ]
    available = {font.name for font in fm.fontManager.ttflist}
    selected = next((name for name in preferred_fonts if name in available), None)


# ════════════════════════════════════════════════════════════════════════════
# 1. 데이터 로드
# ════════════════════════════════════════════════════════════════════════════
def load_data(path: str):
    df = pd.read_csv(path, encoding="cp949", parse_dates=["Date"],
                     index_col="Date")
    df.sort_index(inplace=True)

    for col in [FEAT_AU, FEAT_CU]:
        if col not in df.columns:
            raise ValueError(f"필요한 컬럼 없음: {col}\n"
                             f"현재 컬럼: {df.columns.tolist()}")

    sub = df[[FEAT_AU, FEAT_CU]].dropna()
    print(f"[데이터 로드]")
    print(f"  전체: {len(df)}행  ({df.index[0].date()} ~ {df.index[-1].date()})")
    print(f"  MSM 적합용 (60d 유효): {len(sub)}행  "
          f"({sub.index[0].date()} ~ {sub.index[-1].date()})")
    return df, sub


# ════════════════════════════════════════════════════════════════════════════
# 2. Baum-Welch EM 알고리즘 (Multivariate Gaussian HMM)
#    Y : (T, D) 관측값
#    A : (K, K) 전이행렬
#    pi: (K,)   초기 상태 분포
#    mu: (K, D) 각 Regime 평균
#    cov: (K, D, D) 각 Regime 공분산
# ════════════════════════════════════════════════════════════════════════════
def _log_emission(Y: np.ndarray, mu: np.ndarray,
                  cov: np.ndarray) -> np.ndarray:
    """각 시점·상태별 log 방출 확률  →  (T, K)"""
    T = len(Y)
    log_b = np.zeros((T, K))
    for k in range(K):
        log_b[:, k] = multivariate_normal.logpdf(Y, mean=mu[k], cov=cov[k])
    return log_b


def _forward(log_b: np.ndarray, A: np.ndarray,
             pi: np.ndarray):
    """Log-scale forward pass  →  log_alpha (T, K)"""
    T, K_ = log_b.shape
    log_alpha = np.full((T, K_), -np.inf)
    log_alpha[0] = np.log(pi + 1e-300) + log_b[0]
    log_A = np.log(A + 1e-300)
    for t in range(1, T):
        for j in range(K_):
            log_alpha[t, j] = log_b[t, j] + np.logaddexp.reduce(
                log_alpha[t - 1] + log_A[:, j])
    return log_alpha


def _backward(log_b: np.ndarray, A: np.ndarray):
    """Log-scale backward pass  →  log_beta (T, K)"""
    T, K_ = log_b.shape
    log_beta = np.zeros((T, K_))      # log(1) = 0
    log_A = np.log(A + 1e-300)
    for t in range(T - 2, -1, -1):
        for i in range(K_):
            log_beta[t, i] = np.logaddexp.reduce(
                log_A[i] + log_b[t + 1] + log_beta[t + 1])
    return log_beta


def _e_step(Y: np.ndarray, A: np.ndarray, pi: np.ndarray,
            mu: np.ndarray, cov: np.ndarray):
    """
    E-step: 사후확률(gamma) + 전이기대횟수(xi) 계산
    gamma : (T, K)   — P(z_t = k | Y)
    xi    : (T-1, K, K) — P(z_t=i, z_{t+1}=j | Y)
    ll    : log-likelihood
    """
    log_b     = _log_emission(Y, mu, cov)
    log_alpha = _forward(log_b, A, pi)
    log_beta  = _backward(log_b, A)

    # gamma
    log_gamma = log_alpha + log_beta
    log_norm  = np.logaddexp.reduce(log_gamma, axis=1, keepdims=True)
    gamma     = np.exp(log_gamma - log_norm)

    # xi  (T-1, K, K)
    T     = len(Y)
    log_A = np.log(A + 1e-300)
    xi    = np.zeros((T - 1, K, K))
    for t in range(T - 1):
        for i in range(K):
            for j in range(K):
                xi[t, i, j] = (log_alpha[t, i] + log_A[i, j]
                                + log_b[t + 1, j] + log_beta[t + 1, j])
        row_max = xi[t].max()
        xi[t]   = np.exp(xi[t] - row_max)
        xi[t]  /= xi[t].sum() + 1e-300

    ll = np.logaddexp.reduce(log_alpha[-1])
    return gamma, xi, ll


def _m_step(Y: np.ndarray, gamma: np.ndarray, xi: np.ndarray):
    """M-step: 파라미터 업데이트"""
    T, K_ = gamma.shape
    D     = Y.shape[1]

    # 전이행렬
    A_new = xi.sum(0)
    A_new = A_new / (A_new.sum(1, keepdims=True) + 1e-300)

    # 초기 분포
    pi_new = gamma[0] / (gamma[0].sum() + 1e-300)

    # 평균 & 공분산
    mu_new  = np.zeros((K_, D))
    cov_new = np.zeros((K_, D, D))
    for k in range(K_):
        w = gamma[:, k]                          # (T,)
        w_sum = w.sum() + 1e-300
        mu_new[k]  = (w[:, None] * Y).sum(0) / w_sum
        diff       = Y - mu_new[k]               # (T, D)
        cov_new[k] = ((w[:, None, None]
                       * diff[:, :, None]
                       * diff[:, None, :]).sum(0) / w_sum)
        cov_new[k] += np.eye(D) * 1e-6          # 수치 안정

    return A_new, pi_new, mu_new, cov_new


def fit_msm(sub: pd.DataFrame):
    """
    Baum-Welch EM으로 2-state MSM 적합
    반환: gamma(T,K), 최종 파라미터(A, pi, mu, cov), ll 기록
    """
    Y = sub.values.copy()
    T, D = Y.shape

    # ── 초기화: 은-금 상관 중앙값 기준 2분할 ────────────────────────
    np.random.seed(RANDOM_SEED)
    median_au = np.median(Y[:, 0])
    mask_A    = Y[:, 0] >= median_au

    mu  = np.array([Y[mask_A].mean(0),  Y[~mask_A].mean(0)])
    cov = np.array([
        np.cov(Y[mask_A].T)  + np.eye(D) * 1e-6,
        np.cov(Y[~mask_A].T) + np.eye(D) * 1e-6,
    ])
    A  = np.array([[0.97, 0.03], [0.03, 0.97]])   # sticky 초기값
    pi = np.array([0.5, 0.5])

    print(f"\n[MSM 적합]  EM 알고리즘  (max_iter={MAX_ITER}, tol={TOL})")
    print(f"  초기 mu[0]: {mu[0]}  ← 은-금 높은 Regime 후보")
    print(f"  초기 mu[1]: {mu[1]}  ← 은-금 낮은 Regime 후보")

    ll_history = []
    prev_ll    = -np.inf

    for it in range(MAX_ITER):
        gamma, xi, ll = _e_step(Y, A, pi, mu, cov)
        A, pi, mu, cov = _m_step(Y, gamma, xi)
        ll_history.append(ll)

        if it % 20 == 0 or it < 5:
            print(f"  iter {it:4d}  log-likelihood = {ll:.6f}")

        if abs(ll - prev_ll) < TOL:
            print(f"  ✓ 수렴 완료  (iter={it},  Δll={abs(ll-prev_ll):.2e})")
            break
        prev_ll = ll
    else:
        print(f"  ⚠ 최대 반복 도달 ({MAX_ITER})")

    print(f"\n  최종 log-likelihood : {ll:.6f}")
    return gamma, A, pi, mu, cov, ll_history


# ════════════════════════════════════════════════════════════════════════════
# 3. Regime 라벨 정렬
#    은-금 평균이 높은 state → Regime A (귀금속 국면)
# ════════════════════════════════════════════════════════════════════════════
def relabel(gamma: np.ndarray, mu: np.ndarray):
    """
    state_A: 은-금 mean이 높은 state 인덱스
    regime : (T,) 정수 배열  0=A, 1=B
    prob_A : (T,) Regime A 사후확률
    """
    state_A = 0 if mu[0, 0] > mu[1, 0] else 1
    state_B = 1 - state_A

    regime = np.where(np.argmax(gamma, axis=1) == state_A, 0, 1)
    prob_A = gamma[:, state_A]
    prob_B = gamma[:, state_B]

    print(f"\n[Regime 라벨 정렬]")
    print(f"  State {state_A} → Regime A (귀금속 국면)")
    print(f"    은-금 평균={mu[state_A,0]:.4f},  은-구리 평균={mu[state_A,1]:.4f}")
    print(f"  State {state_B} → Regime B (산업재 국면)")
    print(f"    은-금 평균={mu[state_B,0]:.4f},  은-구리 평균={mu[state_B,1]:.4f}")

    n_A = (regime == 0).sum()
    n_B = (regime == 1).sum()
    T   = len(regime)
    print(f"\n  Regime A: {n_A}일 ({n_A/T*100:.1f}%)")
    print(f"  Regime B: {n_B}일 ({n_B/T*100:.1f}%)")

    return regime, prob_A, prob_B, state_A


# ════════════════════════════════════════════════════════════════════════════
# 4. 전환 횟수 및 평균 지속기간 계산
# ════════════════════════════════════════════════════════════════════════════
def regime_stats(regime: np.ndarray, A: np.ndarray, state_A: int):
    """전이행렬에서 평균 지속기간 계산"""
    p_AA = A[state_A,   state_A]
    p_BB = A[1-state_A, 1-state_A]
    dur_A = 1 / (1 - p_AA) if p_AA < 1 else np.inf
    dur_B = 1 / (1 - p_BB) if p_BB < 1 else np.inf

    # 전환 횟수
    switches = (np.diff(regime) != 0).sum()

    print(f"\n[Regime 통계]")
    print(f"  전환 횟수         : {switches}회")
    print(f"  평균 지속기간 A   : {dur_A:.1f}일")
    print(f"  평균 지속기간 B   : {dur_B:.1f}일")
    print(f"  전이확률 A→A      : {p_AA:.4f}")
    print(f"  전이확률 B→B      : {p_BB:.4f}")

    return {"switches": switches, "dur_A": dur_A, "dur_B": dur_B,
            "p_AA": p_AA, "p_BB": p_BB}


# ════════════════════════════════════════════════════════════════════════════
# 5. 저장
# ════════════════════════════════════════════════════════════════════════════
def save_csv(df_full: pd.DataFrame, sub: pd.DataFrame,
             regime: np.ndarray, prob_A: np.ndarray,
             prob_B: np.ndarray) -> pd.DataFrame:
    regime_df = pd.DataFrame({
        "Regime"      : regime,
        "Regime_Label": np.where(regime == 0, "A", "B"),
        "Prob_A"      : np.round(prob_A, 6),
        "Prob_B"      : np.round(prob_B, 6),
    }, index=sub.index)

    out = df_full.join(regime_df, how="left")
    out.to_csv(OUT_CSV, encoding="utf-8-sig")
    print(f"\n[저장] {OUT_CSV}  ({len(out)}행)")
    return out


def save_summary(A, pi, mu, cov, ll_history, stats, state_A):
    lines = []
    lines.append("=" * 62)
    lines.append("  Markov Switching Model 요약  (STEP 3)")
    lines.append("=" * 62)
    lines.append(f"\n입력 feature : {FEAT_AU},  {FEAT_CU}")
    lines.append(f"Regime 수    : {K}")
    lines.append(f"최종 log-lik : {ll_history[-1]:.6f}")
    lines.append(f"수렴 반복수  : {len(ll_history)}")
    lines.append("")
    lines.append("── 전이행렬 A ──")
    lines.append(f"  A→A = {A[state_A,   state_A]:.4f}   "
                 f"A→B = {A[state_A,   1-state_A]:.4f}")
    lines.append(f"  B→A = {A[1-state_A, state_A]:.4f}   "
                 f"B→B = {A[1-state_A, 1-state_A]:.4f}")
    lines.append("")
    lines.append("── 각 Regime 평균 ──")
    lines.append(f"  Regime A:  은-금={mu[state_A,0]:.4f},  "
                 f"은-구리={mu[state_A,1]:.4f}")
    lines.append(f"  Regime B:  은-금={mu[1-state_A,0]:.4f},  "
                 f"은-구리={mu[1-state_A,1]:.4f}")
    lines.append("")
    lines.append("── Regime 지속성 ──")
    lines.append(f"  평균 지속기간 A : {stats['dur_A']:.1f}일")
    lines.append(f"  평균 지속기간 B : {stats['dur_B']:.1f}일")
    lines.append(f"  전환 횟수       : {stats['switches']}회")
    lines.append("")
    lines.append(f"[Regime 정의]")
    lines.append(f"  Regime A (State {state_A})   : 귀금속 국면 (은-금 동조↑, 안전자산 모드)")
    lines.append(f"  Regime B (State {1-state_A}) : 산업재 국면 (은-구리 동조↑, 위험선호 모드)")

    text = "\n".join(lines)
    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[저장] {OUT_SUMMARY}")
    print("\n" + text)


# ════════════════════════════════════════════════════════════════════════════
# 6. 시각화
# ════════════════════════════════════════════════════════════════════════════
def _regime_band(ax, dates, regime):
    """Regime A/B 배경 띠를 axes에 추가"""
    colors = {0: COLOR_A, 1: COLOR_B}
    i = 0
    while i < len(regime):
        j = i
        while j < len(regime) and regime[j] == regime[i]:
            j += 1
        ax.axvspan(dates[i], dates[min(j, len(dates)-1)],
                   alpha=ALPHA_BAND, color=colors[regime[i]], lw=0)
        i = j


def plot_prob(sub: pd.DataFrame, prob_A: np.ndarray,
              regime: np.ndarray, ll_history: list):
    """
    Fig 1: 3-panel
      (a) Smoothed P(Regime A) + 배경 띠
      (b) 은-금 / 은-구리 60d Corr + 배경 띠
      (c) EM 수렴 곡선
    """
    fig, axes = plt.subplots(3, 1, figsize=(15, 11),
                             gridspec_kw={"height_ratios": [2.5, 2.5, 1.2]})
    fig.suptitle(
        "Markov Switching Model: Silver Regime Identification\n"
        "Feature: 60-day Rolling Correlation  (Ag–Au, Ag–Cu)",
        fontsize=13, fontweight="bold", y=0.99
    )
    dates = sub.index

    # ── (a) Posterior P(Regime A) ────────────────────────────────────────
    ax = axes[0]
    _regime_band(ax, dates, regime)
    ax.fill_between(dates, prob_A, alpha=0.35, color=COLOR_A)
    ax.plot(dates, prob_A, color=COLOR_A, lw=0.9, label="P(Regime A | data)")
    ax.axhline(0.5, color="gray", lw=0.8, ls="--", alpha=0.6, label="0.5 threshold")
    ax.set_ylabel("P(Regime A)\n귀금속 국면", fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(dates[0], dates[-1])
    ax.legend(loc="upper right", fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())

    patch_A = mpatches.Patch(color=COLOR_A, alpha=0.4, label="Regime A (귀금속)")
    patch_B = mpatches.Patch(color=COLOR_B, alpha=0.4, label="Regime B (산업재)")
    ax.legend(handles=[
        mpatches.Patch(color=COLOR_A, alpha=0.5, label="Regime A (귀금속)"),
        mpatches.Patch(color=COLOR_B, alpha=0.5, label="Regime B (산업재)"),
        plt.Line2D([0],[0], color=COLOR_A, lw=1.5, label="P(Regime A)"),
        plt.Line2D([0],[0], color="gray", lw=0.9, ls="--", label="0.5"),
    ], loc="upper right", fontsize=8.5, ncol=2)

    # ── (b) 은-금 / 은-구리 Corr ────────────────────────────────────────
    ax2 = axes[1]
    _regime_band(ax2, dates, regime)
    ax2.plot(dates, sub[FEAT_AU].values, color=COLOR_AU, lw=1.1,
             label="Ag–Au Corr (60d)", zorder=3)
    ax2.plot(dates, sub[FEAT_CU].values, color=COLOR_CU, lw=1.1,
             label="Ag–Cu Corr (60d)", zorder=3)
    ax2.axhline(0, color="gray", lw=0.6, ls="--", alpha=0.5)
    ax2.set_ylabel("Rolling Correlation", fontsize=10)
    ax2.set_xlim(dates[0], dates[-1])
    ax2.legend(loc="upper right", fontsize=9)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.xaxis.set_major_locator(mdates.YearLocator())

    # ── (c) EM 수렴 곡선 ─────────────────────────────────────────────────
    ax3 = axes[2]
    ax3.plot(ll_history, color="steelblue", lw=1.3)
    ax3.set_xlabel("EM Iteration", fontsize=9)
    ax3.set_ylabel("Log-Likelihood", fontsize=9)
    ax3.set_title("EM Convergence", fontsize=9, pad=3)
    ax3.grid(alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(FIG_PROB, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[그림] {FIG_PROB}")


def plot_regime_band(df_out: pd.DataFrame, sub: pd.DataFrame,
                     regime: np.ndarray):
    """
    Fig 2: 2-panel
      (a) BTC 가격 + Regime 띠  (STEP 4 예고)
      (b) 은-금 / 은-구리 Corr + Regime 띠  (상세)
    """
    dates = sub.index

    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    fig.suptitle("Regime Band: Silver Regime vs. Asset Prices\n"
                 "(Regime A = 귀금속 국면 │ Regime B = 산업재 국면)",
                 fontsize=13, fontweight="bold")

    # ── (a) BTC 가격 + 띠 ────────────────────────────────────────────────
    ax = axes[0]
    btc_sub = df_out["BTC_Close"].loc[dates]
    _regime_band(ax, dates, regime)
    ax.semilogy(dates, btc_sub.values, color="darkorange",
                lw=1.0, label="BTC (log scale)")
    ax.set_ylabel("BTC Price (USD, log)", fontsize=10)
    ax.legend(loc="upper left", fontsize=9)

    # ── (b) 은-금 / 은-구리 Corr ─────────────────────────────────────────
    ax2 = axes[1]
    _regime_band(ax2, dates, regime)
    ax2.plot(dates, sub[FEAT_AU].values, color=COLOR_AU, lw=1.1,
             label="Ag–Au Corr (60d)")
    ax2.plot(dates, sub[FEAT_CU].values, color=COLOR_CU, lw=1.1,
             label="Ag–Cu Corr (60d)")
    ax2.axhline(0, color="gray", lw=0.6, ls="--", alpha=0.5)
    ax2.set_ylabel("Rolling Correlation", fontsize=10)
    ax2.set_xlabel("Date", fontsize=10)
    ax2.legend(loc="upper right", fontsize=9)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.xaxis.set_major_locator(mdates.YearLocator())

    # 공통 범례
    patch_A = mpatches.Patch(color=COLOR_A, alpha=0.5, label="Regime A (귀금속)")
    patch_B = mpatches.Patch(color=COLOR_B, alpha=0.5, label="Regime B (산업재)")
    fig.legend(handles=[patch_A, patch_B],
               loc="upper right", bbox_to_anchor=(0.99, 0.97),
               fontsize=9, framealpha=0.85)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIG_BAND, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[그림] {FIG_BAND}")


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    configure_korean_font()
    print("=" * 62)
    print("  STEP 3 : Markov Switching Model (Silver Regime)")
    print("=" * 62)

    # 1. 로드
    df_full, sub = load_data(INPUT_PATH)

    # 2. MSM 적합
    gamma, A, pi, mu, cov, ll_history = fit_msm(sub)

    # 3. Regime 정렬
    regime, prob_A, prob_B, state_A = relabel(gamma, mu)

    # 4. 통계
    stats = regime_stats(regime, A, state_A)

    # 5. 저장
    df_out = save_csv(df_full, sub, regime, prob_A, prob_B)
    save_summary(A, pi, mu, cov, ll_history, stats, state_A)

    # 6. 시각화
    plot_prob(sub, prob_A, regime, ll_history)
    plot_regime_band(df_out, sub, regime)

    print("\n" + "=" * 62)
    print("HMM으로 Regime 추정 완료")
    print("=" * 62)
