# 주문 API: 매수, 매도, 취소

import requests
import config


def place_buy_order(token, stock_code, qty, price, order_type="00"):
    """
    주식 매수 주문

    Args:
        token: 접근 토큰
        stock_code: 종목코드 (예: "005930" = 삼성전자)
        qty: 주문 수량 (주)
        price: 주문 단가 (원) / 시장가(order_type="01")일 경우 0
        order_type: "00"=지정가, "01"=시장가

    Returns:
        {"order_no": 주문번호, "exchange_org_no": 거래소주문조직번호, "order_time": 주문시각}
    """
    return _place_order(token, stock_code, qty, price, order_type, tr_id="VTTC0012U")


def place_sell_order(token, stock_code, qty, price, order_type="00"):
    """
    주식 매도 주문

    Args:
        token: 접근 토큰
        stock_code: 종목코드
        qty: 주문 수량 (주)
        price: 주문 단가 (원) / 시장가(order_type="01")일 경우 0
        order_type: "00"=지정가, "01"=시장가

    Returns:
        {"order_no": 주문번호, "exchange_org_no": 거래소주문조직번호, "order_time": 주문시각}
    """
    return _place_order(token, stock_code, qty, price, order_type, tr_id="VTTC0011U")


def cancel_order(token, org_order_no, exchange_org_no, stock_code):
    """
    주문 전량 취소

    Args:
        token: 접근 토큰
        org_order_no: 원주문번호 (매수/매도 응답의 order_no)
        exchange_org_no: 거래소주문조직번호 (매수/매도 응답의 exchange_org_no)
        stock_code: 종목코드

    Returns:
        {"order_no": 취소주문번호}
    """
    url = f"{config.BASE_URL}/uapi/domestic-stock/v1/trading/order-rvsecncl"
    headers = config.make_headers(token, "VTTC0013U")
    body = {
        "CANO": config.CANO,
        "ACNT_PRDT_CD": config.ACNT_PRDT_CD,
        "KRX_FWDG_ORD_ORGNO": exchange_org_no,  # 거래소 주문조직번호
        "ORGN_ODNO": org_order_no,               # 취소할 원주문번호
        "ORD_DVSN": "00",                        # 주문 구분 (취소 시 무관)
        "RVSE_CNCL_DVSN_CD": "02",               # 01=정정, 02=취소
        "ORD_QTY": "0",                          # 수량 (전량 취소 시 0)
        "ORD_UNPR": "0",                         # 단가 (취소 시 0)
        "QTY_ALL_ORD_YN": "Y",                   # Y=전량 취소
        "PDNO": stock_code,
    }

    resp = requests.post(url, headers=headers, json=body)
    if resp.status_code != 200:
        raise Exception(f"취소 주문 실패 ({resp.status_code}): {resp.text}")

    data = resp.json()
    if data.get("rt_cd") != "0":
        raise Exception(f"취소 주문 오류: {data.get('msg1', resp.text)}")

    return {"order_no": data.get("output", {}).get("ODNO", "")}


def _place_order(token, stock_code, qty, price, order_type, tr_id):
    """매수/매도 공통 처리"""
    url = f"{config.BASE_URL}/uapi/domestic-stock/v1/trading/order-cash"
    headers = config.make_headers(token, tr_id)
    body = {
        "CANO": config.CANO,
        "ACNT_PRDT_CD": config.ACNT_PRDT_CD,
        "PDNO": stock_code,          # 종목코드
        "ORD_DVSN": order_type,      # 00=지정가, 01=시장가
        "ORD_QTY": str(qty),         # 주문 수량 (문자열)
        "ORD_UNPR": str(price),      # 주문 단가 (시장가 시 "0")
    }

    resp = requests.post(url, headers=headers, json=body)
    if resp.status_code != 200:
        raise Exception(f"주문 실패 ({resp.status_code}): {resp.text}")

    data = resp.json()
    if data.get("rt_cd") != "0":
        raise Exception(f"주문 오류: {data.get('msg1', resp.text)}")

    output = data.get("output", {})
    return {
        "order_no": output.get("ODNO", ""),
        "exchange_org_no": output.get("KRX_FWDG_ORD_ORGNO", ""),  # 취소 시 필요
        "order_time": output.get("ORD_TMD", ""),
    }
