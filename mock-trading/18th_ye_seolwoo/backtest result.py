import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
import warnings
import os

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# =========================================================
# 1. 데이터 로드 (설우님 로직)
# =========================================================
print("[1/4] 설우님의 데이터를 로드하고 전처리를 수행합니다...")
desktop_path = r"C:\Users\yswbi\OneDrive\바탕 화면"
file_path = os.path.join(desktop_path, "kospi200 수정종가,시총,거래대금.xlsx")

df_raw = pd.read_excel(file_path, skiprows=14, header=None)
df_raw[0] = pd.to_datetime(df_raw[0])
df_raw = df_raw.dropna(subset=[0]).sort_values(0).reset_index(drop=True)

meta_df = pd.read_excel(file_path, header=None, nrows=13)
codes = meta_df.iloc[8, 1:].values.astype(str)
types = meta_df.iloc[12, 1:].values.astype(str)

p_idx = [i+1 for i, t in enumerate(types) if "수정주가" in t]
m_idx = [i+1 for i, t in enumerate(types) if "시가총액" in t]
v_idx = [i+1 for i, t in enumerate(types) if "거래대금" in t]

df_p = pd.DataFrame(df_raw[p_idx].values, index=df_raw[0], columns=codes[[i-1 for i in p_idx]]).ffill()
df_m = pd.DataFrame(df_raw[m_idx].values, index=df_raw[0], columns=codes[[i-1 for i in m_idx]]).ffill()
df_v = pd.DataFrame(df_raw[v_idx].values, index=df_raw[0], columns=codes[[i-1 for i in v_idx]]).fillna(0)

# 중복 제거
df_p = df_p.loc[:, ~df_p.columns.duplicated()]
df_m = df_m.loc[:, ~df_m.columns.duplicated()]
df_v = df_v.loc[:, ~df_v.columns.duplicated()]

# 설우님 스타일 벤치마크 산출 (동일 가중 평균 수익률)
bm_idx = (1 + df_p.pct_change().mean(axis=1).fillna(0)).cumprod() * 1000
df_bm = pd.DataFrame({
    'Price': bm_idx,
    'MA3': bm_idx.rolling(3).mean(), 'MA5': bm_idx.rolling(5).mean(), 'MA10': bm_idx.rolling(10).mean(),
    'MA50': bm_idx.rolling(50).mean(), 'MA100': bm_idx.rolling(100).mean()
}, index=df_p.index)

# =========================================================
# 2. 하이브리드 백테스트 가동 (설우님 핵심 루프)
# =========================================================
print("[2/4] 전구간 백테스트 가동 중... (설우님 로직 100%)")

df_p['week_grp'] = df_p.index.to_series().dt.isocalendar().week + (df_p.index.to_series().dt.isocalendar().year * 100)
valid_weekly_idx = df_p.groupby('week_grp').apply(lambda x: x.index[-1]).tolist()
df_p = df_p.drop(columns=['week_grp'])

results = { 'Date': [], 'Benchmark': [], 'Strategy_6to4': [] }

start_idx = 0
for i, date in enumerate(valid_weekly_idx):
    if df_p.index.get_loc(date) >= 100:
        start_idx = i
        break

for t in range(start_idx, len(valid_weekly_idx)):
    target_date = valid_weekly_idx[t-1]
    curr_date = valid_weekly_idx[t]
    t_loc = df_p.index.get_loc(target_date)
    
    # [Core] PCA + Market Cap
    top20 = df_m.loc[target_date].nlargest(20).index
    remains = df_p.columns.intersection(df_m.loc[target_date].dropna().index).difference(top20)
    ret_remains = df_p[remains].iloc[max(0, t_loc-60):t_loc].pct_change().dropna(axis=1, how='all').fillna(0)
    
    if not ret_remains.empty and ret_remains.shape[1] >= 10:
        pca = PCA(n_components=1).fit(ret_remains)
        pca10 = pd.Series(np.abs(pca.components_[0]), index=ret_remains.columns).nlargest(10).index
        core_picks = top20.union(pca10)
    else: core_picks = top20
        
    w_core = df_m.loc[target_date, core_picks] / df_m.loc[target_date, core_picks].sum()
    core_ret = ((df_p.loc[curr_date, core_picks] / df_p.loc[target_date, core_picks] - 1) * w_core).sum()
    
    # [Satellite] Scoring + Regime
    row = df_bm.loc[target_date]
    is_bull, is_bear, is_long_up = row['MA3']>row['MA5']>row['MA10'], row['MA3']<row['MA5']<row['MA10'], row['MA50']>row['MA100']
    
    roc_1w = (df_p.loc[target_date] / df_p.loc[df_p.index[max(0, t_loc-5)]]) - 1
    turnover = (df_v.iloc[max(0, t_loc-5):t_loc].mean()) / (df_m.loc[target_date] + 1e-9)
    score = roc_1w.rank(ascending=False) + turnover.rank(ascending=False)
    
    sat_ret = 0
    if is_bull or (not is_bear and is_long_up):
        picks = score.nsmallest(10).index
        w_s = df_m.loc[target_date, picks] / df_m.loc[target_date, picks].sum()
        sat_ret = ((df_p.loc[curr_date, picks] / df_p.loc[target_date, picks] - 1) * w_s).sum()
    elif not is_bear:
        picks = score.nsmallest(5).index
        w_s = df_m.loc[target_date, picks] / df_m.loc[target_date, picks].sum()
        sat_ret = ((df_p.loc[curr_date, picks] / df_p.loc[target_date, picks] - 1) * w_s).sum() * 0.5
        
    bm_ret = (df_bm.loc[curr_date, 'Price'] / df_bm.loc[target_date, 'Price']) - 1
    port_ret = (core_ret * 0.6) + (sat_ret * 0.4)
    
    results['Date'].append(curr_date)
    results['Benchmark'].append(bm_ret)
    results['Strategy_6to4'].append(port_ret)

# =========================================================
# 3. 결과 분석 및 대시보드 생성
# =========================================================
print("[3/4] 통계 지표를 산출하고 대시보드 파일을 생성합니다...")
df_res = pd.DataFrame(results).set_index('Date').fillna(0)
df_cum = (1 + df_res).cumprod()

# 성과 지표 계산
def get_metrics_table(ret_series):
    cum = (1 + ret_series).cumprod()
    tr = (cum.iloc[-1] - 1) * 100
    years = len(ret_series) / 52
    cagr = (cum.iloc[-1] ** (1/years) - 1) * 100
    mdd = ((cum.cummax() - cum) / cum.cummax()).max() * 100
    sr = (ret_series.mean() / (ret_series.std() + 1e-9)) * np.sqrt(52)
    return [f"{cum.iloc[-1]:.2f}x", f"{tr:.2f}%", f"{cagr:.2f}%", f"{mdd:.2f}%", f"{sr:.2f}"]

stats_summary = pd.DataFrame({
    'Metric': ['Final Wealth', 'Total Return', 'CAGR', 'MDD', 'Sharpe Ratio'],
    'Strategy': get_metrics_table(df_res['Strategy_6to4']),
    'Benchmark': get_metrics_table(df_res['Benchmark'])
})

# (1) 히트맵 파일 저장
df_month = df_res['Strategy_6to4'].resample('MS').apply(lambda x: (1+x).prod()-1).reset_index()
df_month['Year'], df_month['Month'] = df_month['Date'].dt.year, df_month['Date'].dt.month
pivot = df_month.pivot(index='Year', columns='Month', values='Strategy_6to4') * 100

plt.figure(figsize=(12, 7))
sns.heatmap(pivot, annot=True, fmt=".1f", cmap="RdYlGn", center=0)
plt.title('월별 수익률 히트맵 (%) - 설우님 로직 적용', fontsize=15)
plt.savefig(os.path.join(desktop_path, '설우전략_월별_히트맵.png'))
plt.close()

# (2) Plotly HTML 리포트 저장
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=("누적 수익률 추이", "Drawdown"))
fig.add_trace(go.Scatter(x=df_cum.index, y=df_cum['Strategy_6to4'], name='Strategy 6:4 (PCA)', line=dict(color='dodgerblue', width=3)), row=1, col=1)
fig.add_trace(go.Scatter(x=df_cum.index, y=df_cum['Benchmark'], name='Benchmark (Internal)', line=dict(color='gray', dash='dot')), row=1, col=1)

strategy_mdd = (df_cum['Strategy_6to4'] - df_cum['Strategy_6to4'].cummax()) / df_cum['Strategy_6to4'].cummax()
fig.add_trace(go.Scatter(x=df_cum.index, y=strategy_mdd, name='MDD', fill='tozeroy', line=dict(color='red'), opacity=0.3), row=2, col=1)

fig.update_layout(height=800, title_text=f"<b>[DART 18기 예설우] 최종 하이브리드 백테스트 리포트</b>", showlegend=True)
fig.write_html(os.path.join(desktop_path, '설우전략_성과_리포트.html'))

print("\n" + "="*70)
print("             [BACKTEST SUMMARY - USER LOGIC]")
print("="*70)
print(stats_summary.to_string(index=False))
print("\n🎉 바탕화면에 '설우전략_월별_히트맵.png'와 '설우전략_성과_리포트.html'이 생성되었습니다!")
print("="*70)