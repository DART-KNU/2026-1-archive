# 시세 조회 API: 국내 주식 현재가 조회

import requests
import config


def get_stock_price(token, stock_code):
    """
    주식 현재가 조회

    Args:
        token: 접근 토큰
        stock_code: 종목코드 (예: "005930" = 삼성전자)

    Returns:
        {
            "name": 종목명,
            "current_price": 현재가 (원),
            "change": 전일 대비 (원),
            "change_rate": 등락률 (%),
            "volume": 누적 거래량 (주),
        }
    """
    url = f"{config.BASE_URL}/uapi/domestic-stock/v1/quotations/inquire-price"
    headers = config.make_headers(token, "FHKST01010100")
    params = {
        "FID_COND_MRKT_DIV_CODE": "J",  # J = KRX 주식
        "FID_INPUT_ISCD": stock_code,
    }

    resp = requests.get(url, headers=headers, params=params)
    if resp.status_code != 200:
        raise Exception(f"현재가 조회 실패 ({resp.status_code}): {resp.text}")

    data = resp.json()
    if data.get("rt_cd") != "0":
        raise Exception(f"현재가 조회 오류: {data.get('msg1', resp.text)}")

    output = data.get("output", {})
    return {
        "name": output.get("hts_kor_isnm", ""),
        "current_price": int(output.get("stck_prpr", 0)),
        "change": int(output.get("prdy_vrss", 0)),
        "change_rate": float(output.get("prdy_ctrt", 0.0)),
        "volume": int(output.get("acml_vol", 0)),
    }
