# -*- coding: utf-8 -*-
"""
스크리닝 전략 3년 백테스트 (2022–2024)
조건: 부채비율 ≤ 200%, EPS(TTM) > 0, ROE > 5%, ETF/펀드/인버스/레버리지 제외
리밸런싱: 분기별 (12회), 동일가중
벤치마크: KOSPI (KS11)
데이터: data/debt.xlsx, data/EPS.xlsx, data/ROE.xlsx (FnGuide DataGuide)
"""

import os
import warnings
import platform

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

warnings.filterwarnings('ignore')

# ── 한글 폰트 ──
plt.rcParams['font.family'] = 'AppleGothic' if platform.system() == 'Darwin' else 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ── 가격 데이터 소스 ──
try:
    import FinanceDataReader as fdr
    USE_FDR = True
    print("✅ FinanceDataReader 사용")
except ImportError:
    import yfinance as yf
    USE_FDR = False
    print("✅ yfinance 사용")


# ================================================================
# 설정
# ================================================================
DEBT_MAX        = 200
ROE_MIN         = 5
EPS_TTM_MIN     = 0
EXCLUDE_PATTERN = r'TIGER|KODEX|RISE|ACE|HANARO|PLUS|레버리지|인버스|선물|ETN|SOL|KBSTAR|ARIRANG|파워|BNK'
START_DATE      = '2022-01-01'
END_DATE        = '2025-03-28'

# (분기 레이블, 진입일, 청산일)
QUARTERS = [
    ('2022_1Q', '2022-03-31', '2022-06-30'),
    ('2022_2Q', '2022-06-30', '2022-09-30'),
    ('2022_3Q', '2022-09-30', '2022-12-30'),
    ('2022_4Q', '2022-12-30', '2023-03-31'),
    ('2023_1Q', '2023-03-31', '2023-06-30'),
    ('2023_2Q', '2023-06-30', '2023-09-29'),
    ('2023_3Q', '2023-09-29', '2023-12-29'),
    ('2023_4Q', '2023-12-29', '2024-03-29'),
    ('2024_1Q', '2024-03-29', '2024-06-28'),
    ('2024_2Q', '2024-06-28', '2024-09-30'),
    ('2024_3Q', '2024-09-30', '2024-12-27'),
    ('2024_4Q', '2024-12-27', '2025-03-28'),
]


# ================================================================
# STEP 1: FnGuide xlsx 파싱
# ================================================================

def parse_fnguide(filepath, data_start_row, code_col, name_col, data_start_col, year_row, qtr_row):
    df_raw = pd.read_excel(filepath, sheet_name='Sheet1', header=None)
    years = df_raw.iloc[year_row, data_start_col:].values
    qtrs  = df_raw.iloc[qtr_row,  data_start_col:].values

    col_names = []
    for y, q in zip(years, qtrs):
        col_names.append(f"{int(y)}_{q}" if pd.notna(y) and pd.notna(q) else None)

    data   = df_raw.iloc[data_start_row:]
    result = pd.DataFrame(
        data.iloc[:, data_start_col:].values[:, [i for i, c in enumerate(col_names) if c]],
        columns=[c for c in col_names if c]
    )
    result.insert(0, 'name', data.iloc[:, name_col].values)
    result.insert(0, 'code', data.iloc[:, code_col].values)
    result = result[result['code'].notna()].copy()
    result['code'] = result['code'].astype(str).str.strip()
    return result


print("📂 데이터 로드 중...")
debt = parse_fnguide('data/debt.xlsx', data_start_row=14, code_col=5, name_col=6,
                     data_start_col=12, year_row=12, qtr_row=13)
eps  = parse_fnguide('data/EPS.xlsx',  data_start_row=9,  code_col=0, name_col=1,
                     data_start_col=7,  year_row=7,  qtr_row=8)
roe  = parse_fnguide('data/ROE.xlsx',  data_start_row=9,  code_col=0, name_col=1,
                     data_start_col=7,  year_row=7,  qtr_row=8)
def clean_code(df):
    """FnGuide 종목코드 앞의 'A' prefix 제거 후 6자리 zero-fill"""
    df = df.copy()
    df['code'] = df['code'].astype(str).str.strip().str.lstrip('A').str.zfill(6)
    return df

debt = clean_code(debt)
eps  = clean_code(eps)
roe  = clean_code(roe)
print(f"  debt {len(debt)}개 / EPS {len(eps)}개 / ROE {len(roe)}개")


# ================================================================
# STEP 2: 분기별 스크리닝
# ================================================================

def screen(q_label):
    """해당 분기 조건 통과 종목 코드 리스트 반환"""
    if q_label not in debt.columns or q_label not in roe.columns:
        return []

    # EPS TTM: 해당 분기 포함 직전 4개 분기 합산
    yr, qstr = q_label.split('_')
    yr, qn = int(yr), {'1Q': 1, '2Q': 2, '3Q': 3, '4Q': 4}[qstr]
    ttm_qs = []
    for i in range(4):
        q_i, y_i = qn - i, yr
        if q_i <= 0:
            q_i += 4; y_i -= 1
        ttm_qs.append(f"{y_i}_{['','1Q','2Q','3Q','4Q'][q_i]}")
    ttm_qs = [q for q in ttm_qs if q in eps.columns]
    if len(ttm_qs) < 2:
        return []

    eps_ttm   = eps[ttm_qs].apply(pd.to_numeric, errors='coerce').sum(axis=1, min_count=2)
    debt_vals = pd.to_numeric(debt[q_label], errors='coerce')
    roe_vals  = pd.to_numeric(roe[q_label],  errors='coerce')

    df = pd.DataFrame({'code': debt['code'], 'name': debt['name'], 'debt': debt_vals})
    df = df.merge(pd.DataFrame({'code': eps['code'], 'eps_ttm': eps_ttm}), on='code', how='left')
    df = df.merge(pd.DataFrame({'code': roe['code'], 'roe': roe_vals}),   on='code', how='left')
    for c in ['debt', 'eps_ttm', 'roe']:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    mask = (
        (df['debt']    <= DEBT_MAX) &
        (df['eps_ttm'] >  EPS_TTM_MIN) &
        (df['roe']     >  ROE_MIN) &
        (~df['name'].str.contains(EXCLUDE_PATTERN, na=False))
    )
    return df.loc[mask, 'code'].tolist()


print("\n📊 분기별 스크리닝:")
quarter_stocks = {}
for q_label, *_ in QUARTERS:
    stocks = screen(q_label)
    quarter_stocks[q_label] = stocks
    print(f"  {q_label}: {len(stocks):>4}개")


# ================================================================
# STEP 3: 가격 데이터 다운로드
# ================================================================

all_codes = list({c for v in quarter_stocks.values() for c in v})
print(f"\n📥 {len(all_codes)}개 종목 가격 다운로드 중...")

def _strip_tz(s):
    if hasattr(s.index, 'tz') and s.index.tz is not None:
        s = s.copy(); s.index = s.index.tz_localize(None)
    return s

price_data = {}
if USE_FDR:
    for i, code in enumerate(all_codes):
        try:
            df = fdr.DataReader(code, START_DATE, END_DATE)
            col = 'Close' if 'Close' in df.columns else df.columns[0]
            s = df[col].dropna()
            if len(s) > 10:
                price_data[code] = _strip_tz(s)
        except Exception:
            pass
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(all_codes)}...")
else:
    for i, code in enumerate(all_codes):
        for sfx in ['.KS', '.KQ']:
            try:
                df = yf.download(code + sfx, start=START_DATE, end=END_DATE,
                                 progress=False, auto_adjust=True)
                if df is None or len(df) < 10:
                    continue
                s = (df['Close'].iloc[:, 0] if isinstance(df.columns, pd.MultiIndex)
                     else df['Close']).dropna()
                if len(s) > 10:
                    price_data[code] = _strip_tz(s)
                    break
            except Exception:
                pass
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(all_codes)}...")

price_df = pd.DataFrame(price_data)
print(f"  다운로드 성공: {len(price_df.columns)}개 종목")

# KOSPI 벤치마크
try:
    if USE_FDR:
        kospi = _strip_tz(fdr.DataReader('KS11', START_DATE, END_DATE)['Close'])
    else:
        df = yf.download('^KS11', start=START_DATE, end=END_DATE,
                         progress=False, auto_adjust=True)
        kospi = _strip_tz(
            df['Close'].iloc[:, 0] if isinstance(df.columns, pd.MultiIndex) else df['Close']
        ).dropna()
except Exception:
    kospi = None
    print("  ⚠️ KOSPI 데이터 다운로드 실패")


# ================================================================
# STEP 4: 백테스트 실행
# ================================================================

def qret(codes, start_dt, end_dt):
    valid = [c for c in codes if c in price_df.columns]
    if not valid:
        return 0.0, 0
    sub = price_df.loc[start_dt:end_dt, valid].ffill()
    rets = [sub[c].dropna().iloc[-1] / sub[c].dropna().iloc[0] - 1
            for c in valid if len(sub[c].dropna()) >= 2]
    return (float(np.mean(rets)) if rets else 0.0), len(rets)

def bret(start_dt, end_dt):
    if kospi is None:
        return 0.0
    s = kospi.loc[start_dt:end_dt].dropna()
    return float(s.iloc[-1] / s.iloc[0] - 1) if len(s) >= 2 else 0.0


print("\n🚀 백테스트 실행:")
rows = []
for q_label, start_dt, end_dt in QUARTERS:
    stocks = quarter_stocks.get(q_label, [])
    pr, n_traded = qret(stocks, start_dt, end_dt)
    br = bret(start_dt, end_dt)
    rows.append({'quarter': q_label, 'n_stocks': len(stocks), 'n_traded': n_traded,
                 'port': pr, 'bench': br})
    print(f"  {q_label}: 포트 {pr*100:+.1f}%  KOSPI {br*100:+.1f}%  ({n_traded}/{len(stocks)}종목)")

rdf = pd.DataFrame(rows)
port_nav  = (1 + rdf['port']).cumprod()
bench_nav = (1 + rdf['bench']).cumprod()
alpha     = rdf['port'] - rdf['bench']


# ================================================================
# STEP 5: 성과 지표
# ================================================================

n = len(rdf)
port_ann  = (1 + rdf['port']).prod() ** (4/n) - 1
bench_ann = (1 + rdf['bench']).prod() ** (4/n) - 1
port_vol  = rdf['port'].std()  * np.sqrt(4)
bench_vol = rdf['bench'].std() * np.sqrt(4)
port_sharpe  = port_ann  / port_vol  if port_vol  > 0 else 0
bench_sharpe = bench_ann / bench_vol if bench_vol > 0 else 0
port_mdd  = ((port_nav  / port_nav.cummax())  - 1).min()
bench_mdd = ((bench_nav / bench_nav.cummax()) - 1).min()

print(f"\n{'='*52}")
print(f"{'항목':<18} {'포트폴리오':>12} {'KOSPI':>12}")
print(f"{'-'*42}")
print(f"{'연환산 수익률':<18} {port_ann*100:>11.1f}%  {bench_ann*100:>10.1f}%")
print(f"{'연환산 변동성':<18} {port_vol*100:>11.1f}%  {bench_vol*100:>10.1f}%")
print(f"{'샤프비율':<18} {port_sharpe:>12.2f}  {bench_sharpe:>11.2f}")
print(f"{'MDD':<18} {port_mdd*100:>11.1f}%  {bench_mdd*100:>10.1f}%")
print(f"{'누적 수익률':<18} {(port_nav.iloc[-1]-1)*100:>11.1f}%  {(bench_nav.iloc[-1]-1)*100:>10.1f}%")
print(f"{'KOSPI 초과 분기':<18} {(alpha>0).sum():>11}회")


# ================================================================
# STEP 6: 시각화
# ================================================================

xt = [q.replace('_', '\n') for q in rdf['quarter']]
x  = list(range(n))

fig, axes = plt.subplots(3, 2, figsize=(16, 15))
fig.suptitle(
    '스크리닝 전략 3년 백테스트  (2022–2024)\n'
    '[부채비율 ≤ 200%  |  EPS(TTM) > 0  |  ROE > 5%  |  ETF 제외  |  분기별 동일가중 리밸런싱]',
    fontsize=13, fontweight='bold', y=0.99
)

# ── 1. 누적 수익률 ──
ax = axes[0, 0]
nav_x      = list(range(n + 1))
port_pct   = [0] + [(v - 1) * 100 for v in port_nav]
bench_pct  = [0] + [(v - 1) * 100 for v in bench_nav]
ax.plot(nav_x, port_pct,  'b-o',  label='스크리닝 포트', lw=2, ms=5)
ax.plot(nav_x, bench_pct, 'r--s', label='KOSPI',         lw=2, ms=4, alpha=0.75)
ax.fill_between(nav_x, port_pct, bench_pct,
                where=[p >= b for p, b in zip(port_pct, bench_pct)],
                color='steelblue', alpha=0.12, label='초과수익 구간')
ax.axhline(0, color='gray', ls=':', alpha=0.5)
ax.set_title('누적 수익률 (%)', fontweight='bold')
ax.set_xticks(nav_x)
ax.set_xticklabels(['시작'] + xt, fontsize=7)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# ── 2. 분기별 수익률 비교 ──
ax = axes[0, 1]
w = 0.35
ax.bar([i - w/2 for i in x], rdf['port']  * 100, w, label='포트',  color='steelblue', alpha=0.85)
ax.bar([i + w/2 for i in x], rdf['bench'] * 100, w, label='KOSPI', color='tomato',    alpha=0.75)
ax.axhline(0, color='black', lw=0.8)
ax.set_title('분기별 수익률 비교 (%)', fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(xt, fontsize=7)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')

# ── 3. 초과 수익률 ──
ax = axes[1, 0]
colors = ['steelblue' if a >= 0 else 'tomato' for a in alpha]
ax.bar(x, alpha * 100, color=colors, alpha=0.85)
ax.axhline(0, color='black', lw=0.8)
ax.set_title('초과 수익률  포트 − KOSPI (%)', fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(xt, fontsize=7)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
ax.grid(True, alpha=0.3, axis='y')
ax.text(0.02, 0.95,
        f'KOSPI 초과  {(alpha>0).sum()}/{n} 분기\n누적 초과: {alpha.sum()*100:+.1f}%',
        transform=ax.transAxes, fontsize=9, va='top',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

# ── 4. 낙폭 ──
ax = axes[1, 1]
port_dd  = ((port_nav  / port_nav.cummax())  - 1) * 100
bench_dd = ((bench_nav / bench_nav.cummax()) - 1) * 100
ax.fill_between(x, port_dd.values,  0, alpha=0.35, color='steelblue', label='포트')
ax.fill_between(x, bench_dd.values, 0, alpha=0.25, color='tomato',    label='KOSPI')
ax.plot(x, port_dd.values,  'b-',  lw=1.8)
ax.plot(x, bench_dd.values, 'r--', lw=1.5)
ax.set_title('낙폭  Drawdown (%)', fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(xt, fontsize=7)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# ── 5. 분기별 종목 수 ──
ax = axes[2, 0]
ax.bar(x, rdf['n_stocks'],  color='lightsteelblue', alpha=0.9, label='스크리닝 통과')
ax.bar(x, rdf['n_traded'],  color='steelblue',      alpha=0.8, label='가격 데이터 있음')
ax.set_title('분기별 포트폴리오 종목 수', fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(xt, fontsize=7)
for i, v in enumerate(rdf['n_traded']):
    ax.text(i, v + 1, str(v), ha='center', va='bottom', fontsize=7)
ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')

# ── 6. 성과 요약 표 ──
ax = axes[2, 1]
ax.axis('off')
tdata = [
    ['항목',           '포트폴리오',                           'KOSPI'],
    ['연환산 수익률',  f'{port_ann*100:.1f}%',                 f'{bench_ann*100:.1f}%'],
    ['연환산 변동성',  f'{port_vol*100:.1f}%',                 f'{bench_vol*100:.1f}%'],
    ['샤프비율',       f'{port_sharpe:.2f}',                   f'{bench_sharpe:.2f}'],
    ['MDD',            f'{port_mdd*100:.1f}%',                 f'{bench_mdd*100:.1f}%'],
    ['누적 수익률',    f'{(port_nav.iloc[-1]-1)*100:.1f}%',    f'{(bench_nav.iloc[-1]-1)*100:.1f}%'],
    ['KOSPI 초과 분기', f'{(alpha>0).sum()}/{n}회',            '—'],
    ['평균 종목 수',   f'{rdf["n_traded"].mean():.0f}개',      '—'],
]
tbl = ax.table(cellText=tdata[1:], colLabels=tdata[0], loc='center', cellLoc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1.2, 2.0)
for j in range(3):
    tbl[0, j].set_facecolor('#2c3e50')
    tbl[0, j].set_text_props(color='white', fontweight='bold')
for i in range(1, len(tdata)):
    tbl[i, 1].set_facecolor('#d6eaf8')
ax.set_title('성과 요약', fontweight='bold', pad=25)

plt.tight_layout()
plt.savefig('backtest_result.png', dpi=150, bbox_inches='tight')
print("\n✅ backtest_result.png 저장 완료")
plt.show()

rdf.to_csv('backtest_detail.csv', index=False, encoding='utf-8-sig')
print("✅ backtest_detail.csv 저장 완료")
