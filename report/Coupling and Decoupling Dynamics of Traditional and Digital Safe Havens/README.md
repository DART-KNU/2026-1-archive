# [2026-1] 레포트 - 17기_김수연, 18기_차유성

## 제목
* **핵심 내용:** 비트코인의 자산 정체성 논쟁
* **카페 링크:** [DART 26-1 레포트 [금과 은의 경제적 역할과 비트코인의 자산 정체성 논쟁](https://cafe.naver.com/knudart/1067)

## 파일 설명
* `proprocess.py` : 구리, 금, 은, 비트코인 각각의 일별 종가 데이터들을 하나의 파일로 만들어주는 파이썬 코드입니다. (금, 은, 구리, 비트코인 csv파일은 따로 업로드하지 않았습니다)
* `visualize.py` : 위 파이썬 코드로 만들어진 csv파일을 시각화시켜주는 파이썬 코드입니다.
* `msm.py` : proprocess로 만들어진 csv 파일로 hidden markov model을 돌리는 파이썬 코드입니다.
* `regime_analysis.py` : 위 코드를 실행하면 regime_data.csv 파일이 나옵니다. 그 csv 파일로 각 regime을 분석해주고 t-test까지 하여 결과를 알려주는 파이썬 코드입니다.
