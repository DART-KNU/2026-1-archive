"""
예제 03: 주식 매도 주문
실행: python examples/03_sell_order.py

주의: 실제 모의투자 주문이 실행됩니다.
보유 종목이 있어야 매도 가능합니다 (04_check_balance.py로 확인).
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from auth import get_token
from api import place_sell_order

token = get_token()

stock_code = "005930"  # 삼성전자
qty = 1                # 주문 수량 (주)
price = 75000          # 주문 단가 (원) - 지정가
order_type = "00"      # "00"=지정가, "01"=시장가

result = place_sell_order(token, stock_code, qty, price, order_type)

print("매도 주문 완료")
print(f"  주문번호           : {result['order_no']}")
print(f"  거래소 주문조직번호 : {result['exchange_org_no']}")
print(f"  주문시각           : {result['order_time']}")
