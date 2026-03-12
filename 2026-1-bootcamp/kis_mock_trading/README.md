# KIS 모의투자 프레임워크

한국투자증권(KIS) Open API를 이용한 국내 주식 모의투자 도구입니다.
현재가 조회, 매수/매도/취소 주문, 잔고 조회 기능을 제공하며 추후 자동매매 파이프라인 연결에 활용할 수 있습니다.

## 사전 준비

### 1. KIS 모의투자 API 신청
1. [KIS Developers](https://apiportal.koreainvestment.com/) 회원가입
2. **모의투자** APP 신청 → 앱키(AppKey)와 앱시크릿(AppSecret) 발급
3. HTS/MTS에서 모의투자 계좌 개설 → 계좌번호 10자리 확인

### 2. 가상환경 생성 및 패키지 설치

```bash
python3 -m venv .venv
source .venv/bin/activate        # Mac / Linux
.venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

### 3. 환경 변수 설정
`.env.example`을 복사해서 `.env`로 만들고 발급받은 키를 입력합니다.

```bash
cp .env.example .env
```

```
KIS_APP_KEY=PSxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
KIS_APP_SECRET=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
KIS_ACCOUNT_NO=50000000     # 계좌번호 8자리 (상품코드 포함 시 10자리도 가능)
MOCK_MODE=true              # true=모의투자, false=실전투자
```

> `.env` 파일은 절대 Git에 올리지 마세요. (`.gitignore`에 포함되어 있습니다)

---

## 실행 방법

```bash
python examples/01_get_price.py      # 현재가 조회
python examples/02_buy_order.py      # 매수 주문 (시장가)
python examples/03_sell_order.py     # 매도 주문 (지정가)
python examples/04_check_balance.py  # 잔고 및 보유종목 조회
python examples/05_cancel_order.py   # 주문 취소
```

> 매수/매도/취소 주문은 장 운영 시간(09:00~15:30)에만 가능합니다.
> 취소 시 `05_cancel_order.py`에 `02_buy_order.py` 실행 시 출력된 주문번호를 입력하세요.

---

## 프로젝트 구조

```
kis_mock_trading/
├── .env                    # API 키 (개인, gitignore됨)
├── .env.example            # 환경변수 작성 예시
├── config.py               # BASE_URL, 계좌번호 파싱, 공통 헤더 생성
├── auth.py                 # 토큰 발급 및 캐싱 (.token_cache)
│
├── api/                    # KIS API 함수 모음 (import해서 사용)
│   ├── kr_market.py        # get_stock_price()
│   ├── kr_order.py         # place_buy_order(), place_sell_order(), cancel_order()
│   └── kr_account.py       # get_balance()
│
└── examples/               # 실행 예제 스크립트
    ├── 01_get_price.py
    ├── 02_buy_order.py
    ├── 03_sell_order.py
    ├── 04_check_balance.py
    └── 05_cancel_order.py
```

---

## KIS API 지원 기능

| 기능 | 함수 | TR-ID |
|------|------|-------|
| 현재가 조회 | `get_stock_price(token, stock_code)` | FHKST01010100 |
| 매수 주문 | `place_buy_order(token, stock_code, qty, price, order_type)` | VTTC0012U |
| 매도 주문 | `place_sell_order(token, stock_code, qty, price, order_type)` | VTTC0011U |
| 주문 취소 | `cancel_order(token, org_order_no, exchange_org_no, stock_code)` | VTTC0013U |
| 잔고 조회 | `get_balance(token)` | VTTC8434R |

### 주문 구분 코드
| 코드 | 의미 |
|------|------|
| `"00"` | 지정가 (기본값, price에 원하는 단가 입력) |
| `"01"` | 시장가 (price=0으로 입력) |

---

## 참고
- [KIS Developers 공식 문서](https://apiportal.koreainvestment.com/)
- [한국투자증권 Open Trading API GitHub](https://github.com/koreainvestment/open-trading-api)
