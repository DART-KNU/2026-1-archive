import pandas as pd
import matplotlib.pyplot as plt

# 데이터 불러오기
idr_cds = pd.read_csv('IDR CDS.csv')
krw_cds = pd.read_csv('KRW CDS.csv')

# 날짜 변환
idr_cds['날짜'] = pd.to_datetime(idr_cds['날짜'])
krw_cds['날짜'] = pd.to_datetime(krw_cds['날짜'])

# 종가 숫자형 변환
def clean_numeric(val):
    if isinstance(val, str):
        return float(val.replace(',', ''))
    return val

idr_cds['종가'] = idr_cds['종가'].apply(clean_numeric)
krw_cds['종가'] = krw_cds['종가'].apply(clean_numeric)

# 날짜 기준 정렬
idr_cds = idr_cds.sort_values('날짜')
krw_cds = krw_cds.sort_values('날짜')

# 데이터 병합
merged_cds = pd.merge(
    idr_cds[['날짜', '종가']],
    krw_cds[['날짜', '종가']],
    on='날짜',
    suffixes=('_IDR', '_KRW')
)

# 그래프 생성
plt.figure(figsize=(12, 6))
plt.plot(
    merged_cds['날짜'],
    merged_cds['종가_IDR'],
    label='Indonesia 5Y CDS',
    color='orange'
)

plt.plot(
    merged_cds['날짜'],
    merged_cds['종가_KRW'],
    label='Korea 5Y CDS',
    color='blue'
)

plt.title('Comparison of 5Y CDS Premiums (IDR vs KRW)')
plt.xlabel('Date')
plt.ylabel('CDS Premium (bps)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig('cds_comparison.png', dpi=150)
plt.show()