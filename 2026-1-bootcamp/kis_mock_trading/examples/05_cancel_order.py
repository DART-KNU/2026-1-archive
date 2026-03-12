"""
예제 05: 주문 취소
실행: python examples/05_cancel_order.py

주의: 02_buy_order.py 또는 03_sell_order.py 실행 후
      출력된 주문번호, 거래소 주문조직번호를 아래에 입력하세요.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from auth import get_token
from api import cancel_order

token = get_token()

# 매수/매도 주문 시 출력된 값을 여기에 입력
org_order_no = "0000000000"     # 원주문번호
exchange_org_no = "00000"       # 거래소 주문조직번호
stock_code = "005930"           # 종목코드

result = cancel_order(token, org_order_no, exchange_org_no, stock_code)

print("취소 주문 완료")
print(f"  취소 주문번호: {result['order_no']}")
