# 2026-1학기 DART 활동 아카이브

**2026년 1학기 금융데이터과학회 DART의 모든 활동 산출물을 모아두는 공용 저장소입니다.** 학회원 여러분은 아래 가이드에 따라 본인의 결과물을 웹에서 바로 업로드해 주시면 됩니다.

## 📁 폴더 구조
본인의 활동에 맞는 폴더를 클릭해 들어간 후 파일을 올려주세요.

* 📂 **`2026-1-project`** : 프로젝트 활동 관련 코드
* 📂 **`2026-1-quant-strategy`** : 퀀트 및 알고리즘 트레이딩 관련 코드
* 📂 **`2026-1-report`** : 레포트 활동 관련 코드
* 📂 **`2026-1-seminar`** : 세미나 활동 관련 코드

## 📝 파일명 규칙
모든 파일명은 영어 소문자로 작성하며, 단어 사이는 언더바(`_`)로 연결하는 스네이크 케이스(snake_case)를 사용합니다.
단, 이름에는 언더바를 사용하지 않고 모두 붙여서 작성해 주세요.

> 🐍 **스네이크 케이스(snake_case)란?**
> 모든 글자를 **소문자**로 작성하고, 띄어쓰기 대신 언더바(`_`)를 사용하는 프로그래밍 표기법입니다. (예: `my_first_code.ipynb`)

* **기본 형식:** `[name_or_team]_[subject].확장자`
* **작성 가이드:**
  * **개인 과제:** 본인 영문 이름을 언더바 없이 붙여 씁니다. (예: `honggildong`)
  * **팀 프로젝트:** 팀장 이름 뒤에 팀을 붙입니다. (예: `honggildong_team`)

* **폴더 및 상황별 예시:**
  * **seminar:** `honggildong_team_python_basics.ipynb`
  * **project:** `honggildong_team_kospi_volatility.ipynb`
  * **report:** `honggildong_team_samsung_analysis.pdf`
  * **quant-strategy:** `honggildong_team_bollinger_band_strategy.ipynb`

## 🚀 파일 업로드 방법
깃(Git) 설치나 복잡한 명령어 없이 웹에서 바로 올릴 수 있습니다. **단, 커밋 메시지는 반드시 영어로 작성해야 합니다.**

1. 이 저장소에서 본인이 업로드할 폴더(예: `2026-1-seminar`)를 클릭해서 들어갑니다.
2. 우측 상단의 **[Add file]** 버튼을 누르고 **[Upload files]** 를 선택합니다.
3. 작성한 파일(`.ipynb`, `.py` 등)을 화면에 드래그 앤 드롭합니다.
4. 화면 하단의 `Commit changes` 박스에 간단한 메모(예: *Add: honggildong week 1 practice code*)를 적고 초록색 **[Commit changes]** 버튼을 누르면 끝!

### 💡 커밋 메시지 작성 가이드
어떤 작업을 했는지 알기 쉽도록 말머리(Prefix)를 달아 통일해 주세요. 새로운 파일을 올릴 때만 이름/팀명을 적고, 수정할 때는 변경된 내용만 간략히 적습니다.

* `Add:` 새로운 파일이나 실습 코드를 처음 업로드할 때
  * *예시: Add: honggildong_team kospi prediction code*
  * *예시: Add: honggildong pandas practice code*
* `Update:` 이미 올린 파일을 수정하거나 내용을 추가했을 때
  * *예시: Update: revise macd strategy logic*
  * *예시: Update: add parameter tuning*
* `Fix:` 코드의 에러나 오타를 수정했을 때
  * *예시: Fix: data loading error in pandas*
 
## 🚨 주의사항
금융 데이터를 다루는 만큼 아래 내용은 반드시 지켜주셔야 합니다.

1. **유료 데이터 업로드 절대 금지**
   * DataGuide, FnGuide, 유료 애널리스트 리포트 등 저작권이 있는 데이터 원본(`.csv`, `.xlsx`)은 절대 깃허브에 올리지 마세요. **(코드만 업로드)**
2. **개인정보 및 API Key 유출 주의**
   * 증권사 API Key나 비밀번호가 코드에 그대로 적혀있지 않은지 업로드 전에 꼭 확인하세요.
3. **타인 파일 수정 금지**
   * 공용 저장소이므로 본인의 파일만 업로드/수정해 주세요. 다른 학우의 파일을 실수로 덮어쓰지 않도록 주의 부탁드립니다.
4. **Mac 사용자 주의사항**
   * 맥북 환경에서 폴더째로 업로드할 경우 `.DS_Store`라는 숨김 파일이 같이 올라갈 수 있습니다. 이 파일은 제외하고 코드와 리포트 파일만 쏙 골라서 올려주세요.

## 문의
* 깃허브 사용 중 오류가 발생하거나 잘못 올린 파일(API 키 유출 등)이 있다면 즉시 **CS 교육담당: 황지영**에게 알려주세요!
