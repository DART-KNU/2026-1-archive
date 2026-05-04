import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
df_clean = df.dropna(subset=['Log_Return'])

# 이벤트 구간
events = [
    ('2013-05-01', '2013-09-30', 'Taper Tantrum', 'salmon'),
    ('2020-02-01', '2020-05-31', 'COVID-19', 'gold'),
    ('2022-01-01', '2022-12-31', 'Strong USD', 'lightgreen'),
    ('2025-01-01', '2026-04-27', 'Middle East Risk', 'lightblue')
]

# 그래프 생성
plt.figure(figsize=(12, 5))
plt.plot(df_clean['날짜'], df_clean['Log_Return'],
         color='dimgray', linewidth=0.5)

for start, end, label, color in events:
    plt.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                color=color, alpha=0.3)

plt.title('IDR/KRW Log Returns (2013-2026)')
plt.xlabel('Date')
plt.ylabel('Daily Return (%)')
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig('figure2_idrkrw_log_returns.png', dpi=150)
plt.show()