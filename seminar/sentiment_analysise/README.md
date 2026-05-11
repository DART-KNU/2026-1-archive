```
# 감성분석

이 폴더는 2023년 5월 10일부터 2026년 5월 10일까지의 뉴스 기사 제목을 대상으로 감성분석을 수행하고, 감성지수와 KOSPI 수익률 사이의 관계를 통계적으로 검정하기 위한 코드입니다.

전체 분석은 크게 두 단계로 구성됩니다.

1. `sentiment_analysise.py`  
   뉴스 기사 제목에 대해 감성분석을 수행하고, 날짜별 감성지수를 계산합니다.

2. `run_statistical_tests.py`  
   계산된 감성지수와 KOSPI 수익률, 환율, 금리 데이터를 결합하여 상관관계 분석, 시차 회귀분석, Granger 인과성 검정, VAR 분석을 수행합니다.

---

## 1. sentiment_analysise.py

### 목적

`sentiment_analysise.py`는 BigKinds에서 수집한 뉴스 기사 제목 데이터를 이용해 각 기사 제목의 감성을 분류하는 코드입니다.

기사 제목은 금융 감성분석 모델인 `snunlp/KR-FinBert-SC`를 사용하여 다음 세 가지 감성으로 분류됩니다.

- `positive`: 긍정
- `negative`: 부정
- `neutral`: 중립

이후 기사별 감성분석 결과를 저장하고, 날짜별로 긍정 기사 수, 부정 기사 수, 중립 기사 수를 집계하여 일별 감성지수를 계산합니다.

---

### 입력 파일

기본 입력 파일은 다음 경로를 사용합니다.

```text
data/bigkinds_news_titles_20230510_20260510.csv
```

입력 CSV 파일에는 최소한 다음 두 개의 열이 있어야 합니다.

```text
date,title
```

- `date`: 기사 날짜
- `title`: 뉴스 기사 제목

---

### 주요 처리 과정

#### 1. 뉴스 제목 불러오기

```python
load_titles()
```

입력 CSV 파일에서 `date`와 `title` 열을 불러옵니다.

이 과정에서 다음 작업을 수행합니다.

- 날짜와 제목 데이터 정리
- 제목 앞뒤 공백 제거
- 제목 내부의 불필요한 중복 공백 제거
- 날짜와 제목이 비어 있는 행 제거
- 날짜와 제목 기준 정렬
- 중복 제목 제거

기본적으로 같은 제목이 여러 번 등장하면 첫 번째 제목만 남기고 제거합니다.

---

#### 2. 이미 분석한 데이터 건너뛰기

```python
--resume
```

옵션을 사용하면 기존 출력 파일에 이미 저장된 제목은 다시 분석하지 않습니다.

이 기능은 분석 도중 코드가 중단되었을 때 유용합니다. 처음부터 다시 실행하지 않고, 아직 처리하지 않은 기사 제목만 이어서 분석할 수 있습니다.

---

#### 3. 감성분석 모델 불러오기

```python
make_analyzer()
```

Hugging Face의 `transformers` 라이브러리를 사용하여 감성분석 모델을 불러옵니다.

기본 모델은 다음과 같습니다.

```text
snunlp/KR-FinBert-SC
```

이 모델은 한국어 금융 문장 감성분석에 사용할 수 있는 FinBERT 계열 모델입니다.

GPU 사용이 가능하면 CUDA를 사용하고, 그렇지 않으면 CPU로 실행됩니다.

---

#### 4. 기사 제목 감성분석

```python
analyze_titles()
```

뉴스 제목을 일정한 batch 단위로 나누어 감성분석을 수행합니다.

각 제목에 대해 다음 값을 저장합니다.

- `date`: 기사 날짜
- `title`: 기사 제목
- `sentiment`: 정규화된 감성 라벨
- `score`: 모델이 예측한 확률 점수
- `signed_score`: 감성 방향을 반영한 점수
- `raw_label`: 모델이 원래 출력한 라벨

`signed_score`는 다음 방식으로 계산됩니다.

```text
positive -> +score
negative -> -score
neutral  -> 0
```

즉, 긍정 기사는 양수, 부정 기사는 음수, 중립 기사는 0으로 변환됩니다.

---

#### 5. 날짜별 감성지수 계산

```python
write_daily_index()
```

기사별 감성분석 결과를 날짜별로 집계합니다.

날짜별로 다음 값을 계산합니다.

- 긍정 기사 수
- 부정 기사 수
- 중립 기사 수
- 전체 기사 수
- 감성지수

감성지수는 다음 공식으로 계산됩니다.

```text
sentiment_index = (positive - negative) / total
```

감성지수가 1에 가까울수록 긍정 기사가 많고, -1에 가까울수록 부정 기사가 많다는 의미입니다.

---

### 출력 파일

기본 출력 파일은 다음과 같습니다.

#### 기사별 감성분석 결과

```text
data/bigkinds_news_titles_sentiment.csv
```

각 뉴스 제목별 감성분석 결과가 저장됩니다.

#### 날짜별 감성지수

```text
data/bigkinds_daily_sentiment_index.csv
```

날짜별 긍정, 부정, 중립 기사 수와 감성지수가 저장됩니다.

---

### 실행 예시

전체 데이터를 분석하려면 다음과 같이 실행합니다.

```bash
python sentiment_analysise.py
```

일부 데이터만 테스트하려면 다음과 같이 실행할 수 있습니다.

```bash
python sentiment_analysise.py --limit 100
```

기존 결과를 이어서 분석하려면 다음과 같이 실행합니다.

```bash
python sentiment_analysise.py --resume
```

---

## 2. run_statistical_tests.py

### 목적

`run_statistical_tests.py`는 앞에서 계산한 뉴스 감성지수와 KOSPI 수익률 사이의 관계를 통계적으로 분석하는 코드입니다.

단순 상관관계만 확인하는 것이 아니라, 시차를 고려하여 다음 질문을 검정합니다.

- 오늘의 뉴스 감성이 이후 KOSPI 수익률과 관련이 있는가?
- 과거 KOSPI 수익률이 이후 뉴스 감성에 영향을 주는가?
- 감성지수와 시장 수익률 사이에 Granger 인과성이 존재하는가?
- 환율과 금리 변수를 통제한 후에도 관계가 유지되는가?

---

### 입력 파일

기본 입력 파일은 다음과 같습니다.

```text
data/kospi_sentiment_tradingday_merged_20230510_20260510.csv
```

이 파일에는 KOSPI 수익률과 일별 감성지수가 결합되어 있어야 합니다.

주요 변수는 다음과 같습니다.

- `date`: 날짜
- `kospi_log_return`: KOSPI 로그수익률
- `sentiment_index`: 뉴스 감성지수

---

### 추가로 사용하는 외부 데이터

이 코드는 분석의 신뢰도를 높이기 위해 두 가지 통제변수를 추가로 사용합니다.

#### 1. USD/KRW 환율

Yahoo Finance에서 `KRW=X` 데이터를 가져옵니다.

환율 데이터로부터 다음 변수를 계산합니다.

```text
usdkrw_log_return
```

이는 원/달러 환율의 로그수익률입니다.

#### 2. 한국 10년 국채금리

FRED에서 한국 10년 국채금리 데이터를 가져옵니다.

사용하는 FRED series ID는 다음과 같습니다.

```text
IRLTLT01KRM156N
```

금리 데이터로부터 다음 변수를 계산합니다.

```text
korea_10y_yield_change
```

이는 한국 10년 국채금리의 변화량입니다.

---

### 주요 분석 과정

#### 1. 분석 데이터 준비

```python
prepare_dataset()
```

기본 시장 데이터에 환율 데이터와 금리 데이터를 결합합니다.

이 과정에서 다음 작업을 수행합니다.

- KOSPI 수익률과 감성지수 불러오기
- Yahoo Finance에서 환율 데이터 수집
- FRED에서 한국 10년 국채금리 데이터 수집
- 날짜 기준 병합
- 결측값 보정
- 환율 로그수익률 계산
- 금리 변화량 계산

---

#### 2. 교차 상관관계 분석

```python
run_cross_correlations()
```

감성지수와 KOSPI 수익률 사이의 시차 상관관계를 계산합니다.

분석 방향은 두 가지입니다.

```text
sentiment_to_future_market
```

현재 감성지수가 미래 KOSPI 수익률과 관련이 있는지 확인합니다.

```text
market_to_future_sentiment
```

과거 KOSPI 수익률이 이후 감성지수와 관련이 있는지 확인합니다.

각 lag에 대해 상관계수와 p-value를 계산합니다.

---

#### 3. 시차 회귀분석

```python
run_lagged_regressions()
```

감성지수와 시장 수익률의 관계를 회귀분석으로 검정합니다.

첫 번째 회귀분석은 다음 관계를 확인합니다.

```text
과거 감성지수 -> 현재 KOSPI 수익률
```

두 번째 회귀분석은 반대 방향을 확인합니다.

```text
과거 KOSPI 수익률 -> 현재 감성지수
```

이때 다음 변수를 함께 통제합니다.

- 전일 KOSPI 수익률
- USD/KRW 환율 로그수익률
- 한국 10년 국채금리 변화량

회귀분석에서는 HAC 표준오차를 사용합니다.  
HAC 표준오차는 시계열 자료에서 발생할 수 있는 자기상관과 이분산성을 보정하기 위해 사용됩니다.

---

#### 4. Granger 인과성 검정

```python
run_granger_ols()
```

Granger 인과성 검정은 한 변수가 다른 변수를 예측하는 데 도움이 되는지 확인하는 분석입니다.

이 코드에서는 두 방향을 검정합니다.

```text
sentiment_causes_market
```

과거 감성지수가 KOSPI 수익률 예측에 도움이 되는지 확인합니다.

```text
market_causes_sentiment
```

과거 KOSPI 수익률이 감성지수 예측에 도움이 되는지 확인합니다.

환율과 금리 변수도 함께 포함하여 통제합니다.

---

#### 5. 정상성 검정

```python
adf_summary()
```

VAR 분석을 수행하기 전에 각 시계열 변수가 정상성을 가지는지 확인합니다.

이를 위해 ADF 검정을 사용합니다.

ADF 검정의 대상 변수는 다음과 같습니다.

- `kospi_log_return`
- `sentiment_index`
- `usdkrw_log_return`
- `korea_10y_yield_change`

---

#### 6. VAR 분석

```python
run_var()
```

VAR(Vector AutoRegression) 모형은 여러 시계열 변수가 서로에게 미치는 영향을 함께 분석하는 모형입니다.

이 코드에서는 다음 변수들을 VAR 모형에 포함합니다.

- KOSPI 로그수익률
- 뉴스 감성지수
- USD/KRW 환율 로그수익률
- 한국 10년 국채금리 변화량

AIC 기준으로 적절한 lag를 선택한 뒤, VAR 기반 causality test를 수행합니다.

---

### 출력 파일

분석 결과는 기본적으로 다음 폴더에 저장됩니다.

```text
data/statistical_tests
```

생성되는 주요 파일은 다음과 같습니다.

#### 분석용 결합 데이터

```text
analysis_dataset_with_controls.csv
```

KOSPI 수익률, 감성지수, 환율, 금리 변수를 결합한 최종 분석 데이터입니다.

#### 교차 상관관계 결과

```text
01_cross_correlations.csv
```

감성지수와 KOSPI 수익률 사이의 lag별 상관관계 결과입니다.

#### 시차 회귀분석 결과

```text
02_lagged_regressions_controls.csv
```

통제변수를 포함한 시차 회귀분석 결과입니다.

#### Granger 인과성 검정 결과

```text
03_granger_controls.csv
```

감성지수와 KOSPI 수익률 사이의 Granger 인과성 검정 결과입니다.

#### VAR 정상성 검정 결과

```text
04_var_adf_stationarity.csv
```

VAR 분석에 사용된 변수들의 ADF 정상성 검정 결과입니다.

#### VAR lag 선택 결과

```text
04_var_lag_order.csv
```

AIC, BIC 등 기준에 따른 VAR lag 선택 결과입니다.

#### VAR causality 검정 결과

```text
04_var_causality.csv
```

VAR 모형 기반 인과성 검정 결과입니다.

#### 요약 보고서

```text
summary.md
```

주요 분석 결과를 Markdown 형식으로 정리한 요약 파일입니다.

---

### 실행 예시

기본 설정으로 실행하려면 다음과 같이 입력합니다.

```bash
python run_statistical_tests.py
```

최대 lag를 5일로 설정하려면 다음과 같이 실행합니다.

```bash
python run_statistical_tests.py --max-lag 5
```

입력 파일과 출력 폴더를 직접 지정할 수도 있습니다.

```bash
python run_statistical_tests.py --market-data data/kospi_sentiment_tradingday_merged_20230510_20260510.csv --output-dir data/statistical_tests
```

---

## 전체 분석 흐름

전체 분석은 다음 순서로 진행됩니다.

```text
1. BigKinds에서 뉴스 제목 데이터 수집
2. sentiment_analysise.py 실행
3. 기사 제목별 감성분석 결과 생성
4. 날짜별 감성지수 생성
5. 감성지수와 KOSPI 거래일 데이터 결합
6. run_statistical_tests.py 실행
7. 상관관계, 회귀분석, Granger 검정, VAR 분석 수행
8. 결과 CSV 파일과 summary.md 생성
```

---

## 필요한 주요 라이브러리

두 코드를 실행하기 위해 필요한 주요 Python 라이브러리는 다음과 같습니다.

```text
pandas
numpy
requests
scipy
statsmodels
torch
transformers
```

라이브러리는 환경에 따라 다음과 같이 설치할 수 있습니다.

```bash
pip install pandas numpy requests scipy statsmodels torch transformers
```

---

## 분석 해석 시 주의사항

이 분석은 뉴스 감성과 KOSPI 수익률 사이의 통계적 관계를 확인하기 위한 것입니다.

따라서 p-value가 유의하게 나오더라도 그것이 반드시 경제적 의미의 직접적인 인과관계를 의미하지는 않습니다. 특히 금융시장은 환율, 금리, 글로벌 증시, 정책, 기업 실적 등 다양한 요인의 영향을 받기 때문에 결과 해석 시 주의가 필요합니다.

또한 Granger 인과성은 “예측에 도움이 되는가”를 검정하는 방법이며, 현실 세계의 원인과 결과를 완전히 증명하는 것은 아닙니다.

---

## 파일 설명 요약

| 파일명 | 설명 |
|---|---|
| `sentiment_analysise.py` | BigKinds 뉴스 제목을 감성분석하고 날짜별 감성지수를 생성하는 코드 |
| `run_statistical_tests.py` | 감성지수와 KOSPI 수익률 사이의 통계적 관계를 분석하는 코드 |
| `README.md` | 코드의 목적, 실행 방법, 입력/출력 파일, 분석 흐름을 설명하는 문서 |
```
