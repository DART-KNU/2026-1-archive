import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pykrx import stock
import os

# 1. 환경 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

core_stocks = {'005930': '삼성전자', '000660': 'SK하이닉스'}
sub_stocks = {
    '금융': ['105560', '055550'], '자동차': ['005380', '000270'], 
    '헬스케어': ['207940'], '산업재': ['373220', '010620'], 
    '에너지': ['096770', '010950'], '유틸리티': ['015760', '036460']
}

# 2. 데이터 수집
start_date, end_date = "YYYYMMDD", "YYYYMMDD"
print("🚀 전략의 수익률/낙폭/샤프지수 통합 분석을 시작합니다...")

market_price = stock.get_market_ohlcv(start_date, end_date, "069500")['종가']
market_ret = market_price.pct_change()

all_codes = list(core_stocks.keys()) + [c for s in sub_stocks.values() for c in s]
all_prices = {}
for c in all_codes:
    try:
        df = stock.get_market_ohlcv(start_date, end_date, c)
        if not df.empty: all_prices[c] = df['종가']
    except: continue

price_df = pd.DataFrame(all_prices).ffill().dropna(axis=1)
ret_df = price_df.pct_change()

# 3. 시뮬레이션 
strategy_daily_rets = []
for i in range(len(ret_df)):
    if i < 20:
        strategy_daily_rets.append(0); continue
    
    # --- 역발상 지갑 ---
    core_part = (ret_df.iloc[i].get('005930', 0) * 0.40) + (ret_df.iloc[i].get('000660', 0) * 0.15)
    others_codes = [c for c in price_df.columns if c not in core_stocks.keys()]
    others_ret = ret_df[others_codes].iloc[i].mean() * 0.30
    contrarian_ret = core_part + others_ret
    
    # --- 모멘텀 지갑 ---
    sector_rs = {}
    m_ret_5d = market_ret.iloc[i-5:i].sum()
    for s_name, codes in sub_stocks.items():
        avail = [c for c in codes if c in price_df.columns and not pd.isna(price_df.iloc[i][c])]
        if not avail: continue
        s_ret_5d = ret_df[avail].iloc[i-5:i].mean(axis=1).sum()
        sector_rs[s_name] = s_ret_5d / abs(m_ret_5d) if m_ret_5d != 0 else s_ret_5d
    
    if sector_rs:
        top_s = max(sector_rs, key=sector_rs.get)
        stk_avail = [c for c in sub_stocks[top_s] if c in price_df.columns and not pd.isna(price_df.iloc[i][c])]
        best_stk = ret_df[stk_avail].iloc[i-5:i].sum().idxmax()
        momentum_ret = ret_df.iloc[i][best_stk] * 0.15
    else: momentum_ret = 0
    
    strategy_daily_rets.append(contrarian_ret + momentum_ret)

# 4. 마켓 타이밍 및 지표 계산
vko_df = pd.read_csv(os.path.join(os.path.dirname(__file__), "vkospi_2018_2026.csv"), encoding='cp949')
vko_df['Date'] = pd.to_datetime(vko_df['date']); vko_df.set_index('Date', inplace=True)

final_df = pd.DataFrame({'Raw_Ret': strategy_daily_rets}, index=ret_df.index)
final_df = final_df.join(vko_df['VKOSPI']).dropna()
final_df['Index'] = (1 + final_df['Raw_Ret']).cumprod()
final_df['MA20'] = final_df['Index'].rolling(20).mean()

def timing(row):
    if pd.isna(row['MA20']): return 0
    if row['Index'] < row['MA20']:
        return 0.5 if row['VKOSPI'] < 30 else 0.8
    return 1.0

final_df['Weight'] = final_df.apply(timing, axis=1)
final_df['Final_Ret'] = final_df['Raw_Ret'] * final_df['Weight'].shift(1)
final_df['Cum_Strategy'] = (1 + final_df['Final_Ret'].fillna(0)).cumprod()
final_df['Cum_KOSPI'] = (1 + market_ret.loc[final_df.index].fillna(0)).cumprod()
final_df['Cum_KOSPI'] = final_df['Cum_KOSPI'] / final_df['Cum_KOSPI'].iloc[0]

# --- 추가 지표 계산 ---
# 1. 낙폭 (Drawdown)
final_df['Strategy_DD'] = final_df['Cum_Strategy'] / final_df['Cum_Strategy'].cummax() - 1
final_df['KOSPI_DD'] = final_df['Cum_KOSPI'] / final_df['Cum_KOSPI'].cummax() - 1

# 2. 롤링 샤프 지수 (252일 윈도우)
def rolling_sharpe(ret_series, window=252):
    excess_ret = ret_series - (0.035 / 252)
    return np.sqrt(252) * (excess_ret.rolling(window).mean() / excess_ret.rolling(window).std())

final_df['Rolling_Sharpe'] = rolling_sharpe(final_df['Final_Ret'])

# 5. 시각화 (3단 차트)
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), sharex=True, gridspec_kw={'height_ratios': [2, 1, 1]})

# [1단: 누적 수익률]
ax1.plot(final_df['Cum_Strategy'], label='내 전략 (85:15)', color='navy', lw=2)
ax1.plot(final_df['Cum_KOSPI'], label='KOSPI 200', color='red', ls='--', alpha=0.5)
ax1.set_title('누적 수익률', fontsize=15); ax1.legend(); ax1.grid(True, alpha=0.2)

# [2단: 낙폭(Drawdown)]
ax2.fill_between(final_df.index, final_df['Strategy_DD'], 0, color='navy', alpha=0.3, label='전략 낙폭')
ax2.fill_between(final_df.index, final_df['KOSPI_DD'], 0, color='red', alpha=0.1, label='시장 낙폭')
ax2.set_title('낙폭 비교 (위험 관리 능력)', fontsize=13); ax2.legend(); ax2.grid(True, alpha=0.2)

# [3단: 롤링 샤프 지수]
ax3.plot(final_df['Rolling_Sharpe'], color='darkgreen', lw=1.5, label='Rolling Sharpe (1yr)')
ax3.axhline(1.0, color='orange', ls=':', label='Excellent (1.0)')
ax3.axhline(0, color='black', lw=0.5)
ax3.set_title('샤프 지수 추이 (투자 효율성)', fontsize=13); ax3.legend(); ax3.grid(True, alpha=0.2)

plt.tight_layout(); plt.show()

print(f"🏁 최종 수익률: {(final_df['Cum_Strategy'].iloc[-1]-1):.2%}")
print(f"📉 최대 낙폭(MDD): {final_df['Strategy_DD'].min():.2%}")
print(f"📊 평균 샤프 지수: {final_df['Rolling_Sharpe'].mean():.2f}")