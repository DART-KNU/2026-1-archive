"""
예제 01: 주식 현재가 조회
실행: python examples/01_get_price.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from auth import get_token
from api import get_stock_price

token = get_token()

# 원하는 종목코드로 변경하세요
# 005930=삼성전자, 000660=SK하이닉스, 035420=NAVER
stock_code = "005930"

price = get_stock_price(token, stock_code)

print(f"[{stock_code}] {price['name']}")
print(f"  현재가   : {price['current_price']:,}원")
print(f"  전일 대비: {price['change']:+,}원  ({price['change_rate']:+.2f}%)")
print(f"  거래량   : {price['volume']:,}주")
