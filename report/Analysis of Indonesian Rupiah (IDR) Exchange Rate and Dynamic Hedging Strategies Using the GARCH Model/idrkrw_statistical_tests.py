import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox

# 데이터 불러오기
df = pd.read_csv('IDR:KRW 계산.csv')

# 숫자형 데이터 정리
for col in ['USD/KRW', 'IDR/USD', 'IDR/KRW']:
    if df[col].dtype == object:
        df[col] = df[col].str.replace(',', '').astype(float)

# 날짜 변환
df['날짜'] = pd.to_datetime(df['날짜'], format='%m/%d/%y', errors='coerce')

if df['날짜'].isna().any():
    df['날짜'] = pd.to_datetime(df['날짜'], errors='coerce')

df = df.sort_values('날짜').dropna(subset=['날짜', 'IDR/KRW'])

# 로그수익률 계산
df['Log_Return'] = np.log(df['IDR/KRW'] / df['IDR/KRW'].shift(1)) * 100
returns = df['Log_Return'].dropna()

# 기초통계량
print('--- Descriptive Statistics ---')
print(f'Count: {len(returns)}')
print(f'Mean (%): {returns.mean():.4f}')
print(f'Std Dev (%): {returns.std():.4f}')
print(f'Skewness: {returns.skew():.4f}')
print(f'Excess Kurtosis: {returns.kurtosis():.4f}')

# Jarque-Bera 검정
jb_stat, jb_p = stats.jarque_bera(returns)

# Ljung-Box 검정
lb_test = acorr_ljungbox(returns**2, lags=[10], return_df=True)
lb_stat = lb_test['lb_stat'].iloc[0]
lb_p = lb_test['lb_pvalue'].iloc[0]

print(f'Jarque-Bera Stat: {jb_stat:.4f} (p-value: {jb_p:.4e})')
print(f'Ljung-Box Q(10): {lb_stat:.4f} (p-value: {lb_p:.4e})')