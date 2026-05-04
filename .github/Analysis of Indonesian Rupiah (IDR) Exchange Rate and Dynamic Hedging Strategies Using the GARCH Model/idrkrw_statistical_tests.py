# IDR/KRW 로그수익률 기초통계 및 정규성/변동성 검정

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox

# CSV 파일 불러오기
df = pd.read_csv('IDR:KRW 계산.csv')

# 숫자형 데이터 정리
for col in ['USD/KRW', 'IDR/USD', 'IDR/KRW']:
    if df[col].dtype == object:
        df[col] = df[col].str.replace(',', '').astype(float)

# 날짜 변환
df['날짜'] = pd.to_datetime(df['날짜'], format='%m/%d/%y', errors='coerce')

if df['날짜'].isna().any():
    df['날짜'] = pd.to_datetime(df['날짜'], errors='coerce')

# 정렬 및 결측치 제거
df = df.sort_values('날짜').dropna(subset=['날짜', 'IDR/KRW'])

# 로그수익률 계산
df['Log_Return'] = np.log(df['IDR/KRW'] / df['IDR/KRW'].shift(1)) * 100
df_clean = df.dropna(subset=['Log_Return'])

returns = df_clean['Log_Return']

# -------------------------------
# 1. 기초통계량
# -------------------------------
print("=== Descriptive Statistics ===")
print(f"Observations      : {len(returns)}")
print(f"Mean (%)          : {returns.mean():.6f}")
print(f"Std Dev (%)       : {returns.std():.6f}")
print(f"Skewness          : {returns.skew():.6f}")
print(f"Excess Kurtosis   : {returns.kurtosis():.6f}")

# -------------------------------
# 2. Jarque-Bera 정규성 검정
# -------------------------------
jb_stat, jb_p = stats.jarque_bera(returns)

print("\n=== Jarque-Bera Test ===")
print(f"JB Statistic      : {jb_stat:.6f}")
print(f"P-value           : {jb_p:.6f}")

# -------------------------------
# 3. Ljung-Box 검정 (ARCH 효과 확인)
# 제곱수익률 기준
# -------------------------------
lb_result = acorr_ljungbox(returns**2, lags=[10], return_df=True)

lb_stat = lb_result['lb_stat'].iloc[0]
lb_p = lb_result['lb_pvalue'].iloc[0]

print("\n=== Ljung-Box Test on Squared Returns ===")
print(f"Q(10) Statistic   : {lb_stat:.6f}")
print(f"P-value           : {lb_p:.6f}")