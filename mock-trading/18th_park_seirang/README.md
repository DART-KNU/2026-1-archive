# [2026-1] 모의투자 포트폴리오 - 18기_박세랑

## 전략
* **핵심 로직:** 유동성·유동비율 필터링 후 시가총액 비례 가중, 개별 종목 15% 상한 및 섹터 중립 제약을 CVXPY로 동시 최적화하여 KOSPI 추적 오차 최소화
* **카페 링크:** [[18기 전략보고서] 수정 시가총액 가중 전략](https://cafe.naver.com/knudart?iframe_url_utf8=%2FArticleRead.nhn%253Fclubid%3D29307656%2526articleid%3D1070%2526menuid%3D55)

## 파일 설명
  - `backtest/engine.py` : 일별 포트폴리오 수익률 시뮬레이션 메인 루프
  - `backtest/portfolio.py` : 포트폴리오 보유 현황 및 NAV 추적
  - `backtest/cost_model.py` : 수수료·증권거래세·슬리피지 비용 모델
