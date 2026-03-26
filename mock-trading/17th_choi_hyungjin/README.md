# [2026-1] 모의투자 포트폴리오 - 17기_최형진

## 전략 
* **핵심 로직:** 유가 변동성 국면 속 안전 배당주 매수
* **카페 링크:** [18기 전략보고서] 유가변동성 확대 국면에 안전자산 투자 (https://cafe.naver.com/f-e/cafes/29307656/articles/1075?boardtype=L&referrerAllArticles=true)

## 파일 설명 (예시 — 파일명·설명 자유)
# Initialize variables for backtest start (moved from earlier cell for re-run robustness)
day = 0
quarter = 0
money = cash # Reset money to initial cash
backtest = pd.DataFrame(columns = df_list) # Reset backtest DataFrame

print('전체 리밸런싱 횟수는 {}'.format(int(len(price_data)/rebalancing)))
for reb in range(int(len(price_data)/rebalancing)): #리밸런싱 횟수를 계산해줍니다. 전체 투자기간/리밸런싱 주기
    print( reb+1, '회 투자')
    print('현금 : ', money)
    inv = price_data.iloc[day:day+rebalancing,:] #전체 주가 데이터에서 투자기간동안의 데이터를 뽑아냅니다.
    inv = inv.replace(np.nan, 0) # 수정주가 데이터에 없는 값들은 0으로 처리해줍니다.
    inv = inv.loc[:,inv.iloc[0,:] > 0 ] # 투자 시작 시점에 데이터가 0인 친구들은 다 날려줍니다.

    ## 스크리닝

    ### 조건 1. 배당수익률 > 6 완

    try:
      div_inv = div.iloc[quarter, :] # 투자 시점의 데이터를 불러와줍니다.
      div_inv = div_inv.dropna() # 투자 시점에 없는 데이터들은 제거해줍니다.
      div_inv = div_inv[div_inv>6] #
      div_inv.dropna(inplace=True)
      div_inv_list = div_inv.index # 리스트로 저장해줍니다.

 ### 조건 2. dps 2년간 상승 기업 완
      dps_inv = pd.DataFrame(dps.iloc[:quarter, :])

      dps_diff_inv = dps_inv.diff()
      dps_diff_inv = dps_diff_inv.iloc[-2:,:]
      dps_diff_inv = dps_diff_inv[dps_diff_inv>0] # 차분된 값 중 0이하를 NaN으로 바꿈
      dps_diff_inv = dps_diff_inv.dropna(axis=1)
      dps_inv_list = dps_diff_inv.columns #리스트로 저장해줍니다.


      ### 조건 1,2 충족 기업 중 NI상승률 상위 10개

      constraint = div_inv_list.intersection(dps_inv_list) # 두 조건 모두에 걸리는 종목들을 선정해줍니다.
      NI_growth = NI[constraint]
      NI_growth = NI_growth.pct_change()
      NI_growth = NI_growth.iloc[quarter,:]

      inv_list = NI_growth.sort_values(ascending=False).head(5).index
    except:
      inv_list = []

    print('투자 후보 갯수는 : ', len(inv_list))

    ## 여기까지 스크리닝 과정

    final_inv_list = []
    for i in range(len(inv_list)):
        if inv_list[i] in inv.columns:
            final_inv_list.append(inv_list[i])
        else:
            print(inv_list[i],' 종목이 없습니다')
    print('투자하는 종목의 수는 : ', len(final_inv_list))
    print('투자종목 : ',final_inv_list)
    # 스크리닝 된 종목 들 중 가격 데이터에 있는 종목들만 투자 리스트에 추가해줍니다.


    # 매수 기준 : 동일 비중
    # 보유한 주식의 가치 평가하기

    if len(final_inv_list)==0:
        allocation=0
    else:
        allocation = money / len(final_inv_list) # 동일 비중이기 때문에 보유 현금 / 투자할 종목 수 로 나눠줍니다.


    print('동일 비중 투자 금액은 : ' , allocation) # 저는 제 코드를 믿지 못하기 때문에 이렇게 중간중간 어디서 에러가 나는지 예상할 수 있게 출력 결과도 함께 프린트합니다.
    final_price_data = inv[final_inv_list].copy()

    vec = pd.DataFrame({'매수수량' : allocation // final_price_data.iloc[0,:]}) # 종목당 매수할 수를 데이터프레임(벡터) 로 만들어줍니다.
    vec = vec.replace(np.nan, 0)


    # 매도(손절 기준) : 10% 이상 떨어질 경우 매도

    # 손절을 할 때 보유하고 있는 주식수를 0으로 만들거나
    # 손절 이후 포트폴리오의 가격을 0으로 만들거나의 방법을 선택할 수 있겠죠?
    # 저는 손절 한 순간부터 포트폴리오에서 주가를 계산하지 않아도 되는 방법이 더 편할 것 같아서 이렇게 선택했지만
    # 저와 다르게 계산하셔도 상관없습니다.
    loss_cut_money_list = []
    loss_cut_money = 0
    for days in final_price_data.index:
        for stocks in final_price_data.columns:
            if final_price_data.loc[days, stocks] < final_price_data[stocks][0] * ( 1 - loss_cut ):
                loss_cut_money = loss_cut_money + (final_price_data.loc[days, stocks] * float(vec.loc[stocks])) * (1 - tax)
                final_price_data.loc[days:, stocks] = 0

        loss_cut_money_list.append(loss_cut_money)


        # 날짜에 저장된 값을 리스트에 추가해줍니다. 이 반복문을 돌면서 포트폴리오 투자기간동안의 현금이 저장될 것입니다.


    product = np.dot(final_price_data, vec) #np.dot 통해 행렬과 벡터의 내적값을 구해줍니다.
    product = pd.DataFrame(product) #데이터프레임으로 만들어서 나중에 붙여주겠습니다.


    balance = pd.DataFrame(index = product.index)
    balance['현금'] = money - product.iloc[0,0] # 현금은 리밸런싱이 이루어지는 날의 현금에서 주식 매수대금을 뺀 값입니다.
    balance['현금'] += loss_cut_money_list # 손절한 금액은 날짜별로 더해줍시다


    _backtest = pd.DataFrame(columns = df_list) # 빈 데이터프레임을 만들어줍니다.
    _backtest['주식총액'] = product # 보유하고 있는 주식의 가치입니다.
    _backtest['현금'] = balance['현금'] # 아까 계산한 일자별 현금 데이터를 넣어줍니다.


    _backtest['포트폴리오가치'] = _backtest['현금'] + _backtest['주식총액'] # 포트폴리오 가치는 현금과 주식총액의 합입니다.
    _backtest['총수익률'] = (_backtest['포트폴리오가치']/cash) # 처음 들고 시작한 돈 대비 수익률
    _backtest.index = price_data.index[day : day + rebalancing]


    backtest = pd.concat([backtest, _backtest], axis = 0, ignore_index=False)
    # 데이터프레임들을 계속 붙여주면서 백테스트 데이터를 만들겠습니다.
    # 반복문을 돌면서 빈데이터프레임에 데이터를 채워가기 때문에 먼저 선언해준 백테스트 데이터프레임에 붙여줍니다.


    # 리밸런싱 이후 시작 금액은 주식 판매대금 - 거래비용 + 현금입니다.
    money = backtest.iloc[-1,0] + (backtest.iloc[-1,1] * ( 1 - tax))
    day = day + rebalancing
    quarter += 1

backtest['일일수익률'] = backtest['포트폴리오가치'].pct_change()
