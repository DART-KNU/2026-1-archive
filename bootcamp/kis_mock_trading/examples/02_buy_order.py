"""
예제 02: 주식 매수 주문
실행: python examples/02_buy_order.py

주의: 실제 모의투자 주문이 실행됩니다.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from auth import get_token
from api import place_buy_order

token = get_token()

stock_code = "005930"  # 삼성전자
qty = 1                # 주문 수량 (주)
price = 0              # 시장가 주문 시 0
order_type = "01"      # "00"=지정가, "01"=시장가

result = place_buy_order(token, stock_code, qty, price, order_type)

print("매수 주문 완료")
print(f"  주문번호           : {result['order_no']}")
print(f"  거래소 주문조직번호 : {result['exchange_org_no']}")
print(f"  주문시각           : {result['order_time']}")
