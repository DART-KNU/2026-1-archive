# [2026-1] 모의투자 포트폴리오 - 18기_이진욱

# H-MAS: Hierarchical Multi-Agent System for KOSPI Quantitative Trading

계층적 LLM 에이전트 구조를 활용해 KOSPI 종목의 투자 신호를 생성하는 퀀트 트레이딩 프레임워크입니다.  
Technical / Quant / Qual / News / Macro 5개의 전문 에이전트가 분석을 수행하고, Portfolio Manager 에이전트가 최종 포트폴리오를 결정합니다.

---

## 목차

1. [필수 API 키](#필수-api-키)
2. [환경 설정](#환경-설정)
3. [패키지 설치](#패키지-설치)
4. [Qlib 데이터 빌드](#qlib-데이터-빌드-ml-백테스트-전용)
5. [실행 방법](#실행-방법)
6. [주요 파라미터](#주요-파라미터)
7. [출력 결과](#출력-결과)
8. [파일 구조](#파일-구조)

---

## 필수 API 키

| 환경변수 | 용도 | 발급처 |
|---|---|---|
| `OPENAI_API_KEY` | GPT-4o 에이전트 추론 | https://platform.openai.com |
| `DART_API_KEY` | 한국 기업 공시 수집 (DART) | https://opendart.fss.or.kr |
| `FRED_API_KEY` | 미국 거시경제 데이터 | https://fred.stlouisfed.org/docs/api/api_key.html |
| `ECOS_API_KEY` | 한국은행 경제통계 | https://ecos.bok.or.kr/api |
| `NAVER_CLIENT_ID` | 네이버 뉴스 검색 | https://developers.naver.com |
| `NAVER_CLIENT_SECRET` | 네이버 뉴스 검색 | https://developers.naver.com |
| `ANTHROPIC_API_KEY` | (선택) Claude 모델 사용 시 | https://console.anthropic.com |

---

## 환경 설정

### Windows (PowerShell)

```powershell
$env:OPENAI_API_KEY       = 'sk-proj-...'
$env:DART_API_KEY         = 'caa131...'
$env:FRED_API_KEY         = 'cf58af...'
$env:ECOS_API_KEY         = 'RXFJ5Y...'
$env:NAVER_CLIENT_ID      = 'NOkp5w...'
$env:NAVER_CLIENT_SECRET  = 'XDJHid...'
$env:ANTHROPIC_API_KEY    = 'sk-ant-...'   # 선택
```

현재 세션에서만 유효합니다. 영구 적용이 필요하면 시스템 환경변수에 등록하세요.

### macOS / Linux (bash/zsh)

```bash
export OPENAI_API_KEY="sk-proj-..."
export DART_API_KEY="caa131..."
export FRED_API_KEY="cf58af..."
export ECOS_API_KEY="RXFJ5Y..."
export NAVER_CLIENT_ID="NOkp5w..."
export NAVER_CLIENT_SECRET="XDJHid..."
export ANTHROPIC_API_KEY="sk-ant-..."   # 선택
```

영구 적용은 `~/.bashrc` 또는 `~/.zshrc`에 추가하세요.

---

## 패키지 설치

```bash
pip install openai yfinance pandas numpy scipy requests fredapi qlib
```
더 있을수도 있으니 알아서 깔기 python venv

> **Python 3.9 이상** 권장. Qlib은 별도 의존성이 있으므로 공식 문서를 참고하세요: https://qlib.readthedocs.io

---

## Qlib 데이터 빌드 (ML 백테스트 전용)

`hmas_backtest_ml.py`를 실행하기 전에 Qlib용 KOSPI 데이터를 먼저 빌드해야 합니다.  
`hmas_agent_backtest.py`(에이전트 백테스트)는 이 단계가 **필요 없습니다**.

```bash
python build_qlib_kr_data.py \
  --start 2018-01-01 \
  --end   2024-12-31 \
  --output ~/.qlib/qlib_data/kr_data
```

소요 시간: 약 5~10분. 완료 후 디렉토리 구조:

```
~/.qlib/qlib_data/kr_data/
├── calendars/
│   └── day.txt              ← 거래일 목록
├── instruments/
│   ├── all.txt              ← 전종목
│   └── kospi50.txt          ← KOSPI 50 종목
└── features/
    ├── 005930/              ← 종목별 피처 (close.day.bin, open.day.bin ...)
    ├── 000660/
    └── ...
```

---

## 실행 방법

### 1. 에이전트 백테스트 (`hmas_agent_backtest.py`)

GPT-4o 에이전트 5종이 실제 API를 호출해 신호를 생성합니다.

```powershell
# PowerShell — 로그를 파일로 동시 저장
python hmas_agent_backtest.py `
  --strategy threshold `
  --mode     live `
  --buy      65 `
  --sell     45 `
  --maxpos   5 `
  --start    2024-07-01 `
  --end      2024-12-31 `
  --tickers  005930 000660 005380 035420 105560 `
  | tee output.txt
```

```bash
# bash/zsh
python hmas_agent_backtest.py \
  --strategy threshold \
  --mode     live \
  --buy      65 \
  --sell     45 \
  --maxpos   5 \
  --start    2024-07-01 \
  --end      2024-12-31 \
  --tickers  005930 000660 005380 035420 105560 \
  | tee output.txt
```

API 비용 없이 테스트하려면 `--mode stub`으로 변경하세요.

---

### 2. ML 백테스트 (`hmas_backtest_ml.py`)

Qlib 피처 기반 LightGBM / XGBoost / LSTM 모델 백테스트입니다.  
**Qlib 데이터 빌드를 먼저 완료해야 합니다.**

```powershell
python hmas_backtest_ml.py `
  --model      lgbm `
  --tickers    005930 000660 005380 035420 105560 `
  --train_start 2019-01-01 --train_end 2021-12-31 `
  --valid_start 2022-01-01 --valid_end 2022-12-31 `
  --test_start  2023-01-01 --test_end  2024-12-31 `
  | tee ml_output.txt
```

`--model` 옵션: `lgbm` / `xgb` / `lstm`

---

### 3. 지표 내보내기 (`export_metrics.py`)

백테스트 결과에서 IC, Sharpe, MDD 등 지표를 CSV로 추출합니다.

```powershell
python export_metrics.py `
  --model    all `
  --tickers  005930 000660 005380 035420 105560 `
  --test_start 2024-07-01 `
  --test_end   2024-12-31
```

---

## 주요 파라미터

### `hmas_agent_backtest.py`

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `--strategy` | `threshold` | 전략 유형: `threshold` / `topk` |
| `--mode` | `stub` | `live` = 실제 API 호출, `stub` = 더미 신호 |
| `--tickers` | (내장 목록) | 분석 종목 코드 (공백 구분, 6자리) |
| `--start` | `2022-01-01` | 백테스트 시작일 |
| `--end` | `2024-12-31` | 백테스트 종료일 |
| `--freq` | `monthly` | 리밸런싱 주기: `monthly` / `weekly` |
| `--buy` | `70.0` | 매수 임계 점수 |
| `--sell` | `50.0` | 매도 임계 점수 |
| `--maxpos` | `10` | 최대 보유 종목 수 (`0` = 무제한) |
| `--topk` | `3` | `topk` 전략 시 상위 종목 수 |
| `--cash` | `100,000,000` | 초기 투자금 (원) |
| `--cost` | `15.0` | 거래 비용 (bps) |
| `--out` | `backtest_results` | 결과 저장 디렉토리 |
| `--cache` | `backtest_cache` | 에이전트 캐시 디렉토리 |

### `hmas_backtest_ml.py`

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `--model` | `lgbm` | 모델: `lgbm` / `xgb` / `lstm` |
| `--qlib_data` | `~/.qlib/qlib_data/kr_data` | Qlib 데이터 경로 |
| `--train_start / end` | `2019-01-01 / 2021-12-31` | 학습 기간 |
| `--valid_start / end` | `2022-01-01 / 2022-12-31` | 검증 기간 |
| `--test_start / end` | `2023-01-01 / 2024-12-31` | 테스트 기간 |
| `--topk` | `3` | 매월 편입 상위 종목 수 |
| `--stub` | (flag) | 더미 모드 (API 호출 없음) |

---

## 출력 결과

| 경로 | 내용 |
|---|---|
| `backtest_results/` | 수익률 CSV, 포트폴리오 히스토리 |
| `backtest_results/report.html` | Chart.js 기반 인터랙티브 HTML 리포트 |
| `backtest_cache/` | 에이전트 응답 캐시 (재실행 시 API 비용 절감) |
| `output.txt` | `tee`로 저장한 전체 콘솔 로그 |

주요 지표: **Sharpe Ratio**, **IC / ICIR / Rank IC**, **MDD**, **Calmar Ratio**, **Information Ratio**

---

## 파일 구조

```
D:\last_stock\
├── hmas_agent_backtest.py    ← 에이전트 백테스트 메인
├── hmas_backtest_ml.py       ← ML 모델 백테스트 (Qlib)
├── hmas_pm_agent.py          ← Portfolio Manager 에이전트
├── hmas_base.py              ← 공통 기반 클래스
├── hmas_macro_agent.py       ← 거시경제 에이전트 (FRED / ECOS)
├── hmas_news_agent.py        ← 뉴스 에이전트 (Naver)
├── hmas_qual_agent.py        ← 공시 분석 에이전트 (DART)
├── quant_agent.py            ← 계량 펀더멘털 에이전트
├── technical_agent.py        ← 기술적 분석 에이전트
├── dart_collector.py         ← DART 공시 수집기
├── kospi_pipeline.py         ← KOSPI 데이터 파이프라인
├── kospi_collectors.py       ← 종목 가격·재무 수집기
├── build_qlib_kr_data.py     ← Qlib 데이터 빌더
├── export_metrics.py         ← 지표 CSV 내보내기
└── backtest_results/         ← 백테스트 결과 (자동 생성)
```

---

## 주의사항

- `--mode live` 실행 시 OpenAI API 비용이 발생합니다. 테스트는 `--mode stub`으로 먼저 확인하세요.
- 에이전트 캐시(`backtest_cache/`)가 있으면 동일 종목·날짜에 대해 API를 재호출하지 않습니다.
- DART API는 하루 10,000건 요청 제한이 있습니다.
- Qlib 데이터는 ML 백테스트(`hmas_backtest_ml.py`)에만 필요하며, 에이전트 백테스트는 yfinance를 직접 사용합니다.
