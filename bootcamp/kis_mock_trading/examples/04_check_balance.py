"""
예제 04: 계좌 잔고 및 보유 종목 조회
실행: python examples/04_check_balance.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from auth import get_token
from api import get_balance

token = get_token()
balance = get_balance(token)

print(f"예수금       : {balance['cash_balance']:,}원")
print(f"총 평가금액  : {balance['total_eval_amount']:,}원")
print()

if not balance["holdings"]:
    print("보유 종목 없음")
else:
    header = f"{'종목명':<14} {'종목코드':<8} {'보유':>5} {'매입단가':>9} {'평가금액':>13} {'손익률':>8}"
    print(header)
    print("-" * len(header))
    for h in balance["holdings"]:
        print(
            f"{h['name']:<14} "
            f"{h['stock_code']:<8} "
            f"{h['qty']:>5,}주  "
            f"{h['avg_price']:>9,}원  "
            f"{h['eval_amount']:>13,}원  "
            f"{h['profit_rate']:>+7.2f}%"
        )
