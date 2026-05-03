"""
전처리 및 병합 코드
"""

import pandas as pd
import numpy as np

# ============================================================
# 1. 파일 로드
# ============================================================

gold   = pd.read_csv('금_(2017).csv',      parse_dates=['Date'], index_col='Date')[['Close']].rename(columns={'Close': 'Gold_Close'})
silver = pd.read_csv('은_(2017).csv',      parse_dates=['Date'], index_col='Date')[['Close']].rename(columns={'Close': 'Silver_Close'})
copper = pd.read_csv('구리_(2017).csv',    parse_dates=['Date'], index_col='Date')[['Close']].rename(columns={'Close': 'Copper_Close'})
btc    = pd.read_csv('비트코인_(2017).csv', parse_dates=['Date'], index_col='Date')[['Close']].rename(columns={'Close': 'BTC_Close'})

for name, d in [('금', gold), ('은', silver), ('구리', copper), ('BTC', btc)]:
    print(f"  ✓ {name}: {len(d)}행  ({d.index[0].date()} ~ {d.index[-1].date()})")

# ============================================================
# 2. 병합
#    - 금·은·구리 공통 거래일 기준 inner join
#    - BTC는 left join 후 ffill (주말·공휴일 → 직전 거래일 값)
# ============================================================

df = gold.join(silver, how='inner').join(copper, how='inner')  # 공통 거래일
df = df.join(btc, how='left')                                  # BTC left join
df['BTC_Close'] = df['BTC_Close'].ffill()                      # 주말 BTC 채우기
df = df.dropna()

print(f"  ✓ 병합 완료: {len(df)}행  ({df.index[0].date()} ~ {df.index[-1].date()})")

# ============================================================
# 3. 파생변수 계산
# ============================================================

# 금-은 비율 (GSR)
df['GS_Ratio'] = df['Gold_Close'] / df['Silver_Close']
print("  ✓ 금-은 비율 (GS_Ratio)")

# 로그 수익률
for col, ret_col in [('Gold_Close',   'Gold_Return'),
                     ('Silver_Close', 'Silver_Return'),
                     ('Copper_Close', 'Copper_Return'),
                     ('BTC_Close',    'BTC_Return')]:
    df[ret_col] = np.log(df[col] / df[col].shift(1))
print("  ✓ 로그 수익률 (Gold / Silver / Copper / BTC)")

# Rolling Correlation: 은-금, 은-구리 (30·60·90일)
# 메인 분석: 60일 / 30·90일: Robustness Check용
for w in [30, 60, 90]:
    df[f'Corr_Ag_Au_{w}d'] = df['Silver_Return'].rolling(w).corr(df['Gold_Return'])
    df[f'Corr_Ag_Cu_{w}d'] = df['Silver_Return'].rolling(w).corr(df['Copper_Return'])
    df[f'Corr_BTC_Cu_{w}d'] = df['BTC_Return'].rolling(w).corr(df['Copper_Return'])
    df[f'Corr_BTC_Ag_{w}d'] = df['BTC_Return'].rolling(w).corr(df['Silver_Return'])
    df[f'Corr_BTC_Au_{w}d'] = df['BTC_Return'].rolling(w).corr(df['Gold_Return'])
print("  ✓ 은-금 / 은-구리 Rolling Correlation (30·60·90일)")

# ※ 30일 워밍업까지만 제거 → 60·90일 초반 NaN은 빈칸으로 CSV에 유지
core_cols = ['Gold_Close', 'Silver_Close', 'Copper_Close', 'BTC_Close',
             'GS_Ratio', 'Gold_Return', 'Silver_Return', 'Copper_Return',
             'BTC_Return', 'Corr_Ag_Au_30d', 'Corr_Ag_Cu_30d',
             'Corr_BTC_Cu_30d', 'Corr_BTC_Ag_30d', 'Corr_BTC_Au_30d']
df = df.dropna(subset=core_cols)

print(f"\n  결측치 현황 (60·90일 초반 구간은 NaN 정상):")
print(df.isna().sum())

# ============================================================
# 4. 저장
# ============================================================
df.to_csv('merged_data.csv')
print("  ✓ merged_data.csv 저장 완료")

# ============================================================
# 5. 요약
# ============================================================
print("\n" + "="*60)
print("완료!")
print("="*60)
print(f"\n📊 총 관측치 : {len(df)}행")
print(f"📅 분석 기간 : {df.index[0].date()} ~ {df.index[-1].date()}")
print(f"📁 컬럼 목록 :")
for c in df.columns:
    print(f"     {c}")

print(f"\n기초 통계:")
print(df[['Gold_Close', 'Silver_Close', 'Copper_Close', 'BTC_Close']].describe().round(2))
