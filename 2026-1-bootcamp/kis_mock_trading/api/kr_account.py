# 계좌 조회 API: 주식 잔고 및 보유종목 조회

import requests
import config


def get_balance(token):
    """
    주식 잔고 조회

    Args:
        token: 접근 토큰

    Returns:
        {
            "holdings": [
                {
                    "stock_code": 종목코드,
                    "name": 종목명,
                    "qty": 보유수량 (주),
                    "avg_price": 매입 평균가 (원),
                    "eval_amount": 평가금액 (원),
                    "profit_rate": 평가손익률 (%),
                },
                ...
            ],
            "total_eval_amount": 총 평가금액 (원),
            "cash_balance": 예수금 (원),
        }
    """
    url = f"{config.BASE_URL}/uapi/domestic-stock/v1/trading/inquire-balance"
    headers = config.make_headers(token, "VTTC8434R")
    params = {
        "CANO": config.CANO,
        "ACNT_PRDT_CD": config.ACNT_PRDT_CD,
        "AFHR_FLPR_YN": "N",        # 시간외 단일가 여부
        "OFL_YN": "N",              # 오프라인 여부
        "INQR_DVSN": "02",          # 조회 구분: 01=대출일별, 02=종목별
        "UNPR_DVSN": "01",          # 단가 구분
        "FUND_STTL_ICLD_YN": "N",   # 펀드결제분 포함 여부
        "FNCG_AMT_AUTO_RDPT_YN": "N",  # 융자금액 자동상환 여부
        "PRCS_DVSN": "00",          # 처리 구분: 00=전일매매 포함, 01=제외
        "CTX_AREA_FK100": "",       # 연속 조회 키 (첫 호출 시 공백)
        "CTX_AREA_NK100": "",
    }

    resp = requests.get(url, headers=headers, params=params)
    if resp.status_code != 200:
        raise Exception(f"잔고 조회 실패 ({resp.status_code}): {resp.text}")

    data = resp.json()
    if data.get("rt_cd") != "0":
        raise Exception(f"잔고 조회 오류: {data.get('msg1', resp.text)}")

    # 보유종목 목록 (output1)
    holdings = []
    for item in data.get("output1", []):
        if not item.get("pdno"):
            continue
        holdings.append({
            "stock_code": item.get("pdno", ""),
            "name": item.get("prdt_name", ""),
            "qty": int(item.get("hldg_qty", 0)),
            "avg_price": int(float(item.get("pchs_avg_pric", 0))),
            "eval_amount": int(item.get("evlu_amt", 0)),
            "profit_rate": float(item.get("evlu_pfls_rt", 0.0)),
        })

    # 계좌 요약 (output2)
    summary = data.get("output2", [{}])[0] if data.get("output2") else {}

    return {
        "holdings": holdings,
        "total_eval_amount": int(summary.get("tot_evlu_amt", 0)),
        "cash_balance": int(summary.get("dnca_tot_amt", 0)),
    }
