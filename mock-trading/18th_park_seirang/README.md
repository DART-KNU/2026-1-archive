# [2026-1] 모의투자 포트폴리오 - 18기_박세랑

## 전략 (예시 — 실제 내용으로 수정해 주세요)
* **핵심 로직:** 유동성·유동비율 필터링 후 시가총액 비례 가중, 개별 종목 15% 상한 및 섹터 중립 제약을 CVXPY로 동시 최적화하여 KOSPI 추적 오차 최소화
* **카페 링크:** [[18기 전략보고서] 수정 시가총액 가중 전략](https://cafe.naver.com/knudart?iframe_url_utf8=%2FArticleRead.nhn%253Fclubid%3D29307656%2526articleid%3D1070%2526menuid%3D55)

## 파일 설명 (예시 — 파일명·설명 자유)
  - config.py : 전략 파라미터 전역 설정 (캡 비율, 리밸런싱 월, 거래비용 등)
  - strategy/universe.py : ADTV·유동비율 기반 투자 가능 종목 필터링
  - strategy/weights.py : 시가총액 원시 비중 산출, 15% 상한 적용, 섹터 중립 최적화
  - strategy/rebalancer.py : 분기 리밸런싱 날짜 계산
  - backtest/engine.py : 일별 포트폴리오 수익률 시뮬레이션 메인 루프
  - backtest/portfolio.py : 포트폴리오 보유 현황 및 NAV 추적
  - backtest/cost_model.py : 수수료·증권거래세·슬리피지 비용 모델
  - analytics/performance.py : TE, IR, 베타, R², MDD 등 성과 지표 계산
  - analytics/risk_attribution.py : Barra 스타일 팩터 리스크 분해
  - data/loader.py : pykrx 기반 OHLCV·시가총액 데이터 로드 및 캐싱
  - data/preprocessor.py : 데이터 정제, 섹터 매핑, ADTV 전처리
  - visualization/charts.py : 누적 수익률, 롤링 TE, 섹터 배분 등 차트 생성
