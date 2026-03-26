# [2026-1] 모의투자 포트폴리오 - 17기_장연우

## 전략 (정치적 리스크를 반영한 모의투자 전략)
* **핵심 로직:** price > ma50 > ma200 및 GPR Index
* **카페 링크:** [GPR Index를 이용한 모의투자 전략]((https://cafe.naver.com/knudart?iframe_url_utf8=%2FArticleRead.nhn%253Fclubid%3D29307656%2526articleid%3D1060%2526referrerAllArticles%3Dtrue))

## 파일 설명 (예시 — 파일명·설명 자유)
*  `research_screening_backtest.ipynb` : GPR Index를 이용한 KOSPI와 S&P500의 연관성을 보는 그래프 및 스크리닝과 백테스트를 한 파일 안에서 다했습니다.
*   colab을 이용해서 잘 안맞을 수도있습니다. 이제 vscode를 다운로드 했기 때문에 다음부터는 이걸 이용해서 해보겠습니다. * 추가로 GPR Index가 매주 월요일에 업데이트가 되어서 GPR Index를 실시간으로 반영하여 매매하기가 어려운 걸로 판단이 되어서 GDELT 데이터를 이용해서 한번 해보려고 합니다. 기본 매커니즘은 똑같은데 실시간으로 반영할수 있도록 코드를 만들어야 하는데 모의투자가 시작되기 전까지 이것도 한번 올려보겠습니다. 
