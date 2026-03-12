# 설정 로드: KIS API 접속 정보 및 환경 변수 관리

import os
from dotenv import load_dotenv

load_dotenv()

APP_KEY = os.getenv("KIS_APP_KEY")
APP_SECRET = os.getenv("KIS_APP_SECRET")
_ACCOUNT_NO = os.getenv("KIS_ACCOUNT_NO", "")
MOCK_MODE = os.getenv("MOCK_MODE", "true").lower() == "true"

# 모의투자/실전투자 BASE_URL 자동 선택
if MOCK_MODE:
    BASE_URL = "https://openapivts.koreainvestment.com:29443"
else:
    BASE_URL = "https://openapi.koreainvestment.com:9443"

# CANO: 계좌번호 앞 8자리
# ACNT_PRDT_CD: 상품코드 뒤 2자리 (기본값 "01", 10자리 입력 시 뒤 2자리 사용)
CANO = _ACCOUNT_NO[:8]
ACNT_PRDT_CD = _ACCOUNT_NO[8:] if len(_ACCOUNT_NO) == 10 else "01"


def make_headers(token, tr_id):
    """KIS API 공통 요청 헤더 생성"""
    return {
        "content-type": "application/json",
        "authorization": f"Bearer {token}",
        "appkey": APP_KEY,
        "appsecret": APP_SECRET,
        "tr_id": tr_id,
        "custtype": "P",  # P = 개인
    }
