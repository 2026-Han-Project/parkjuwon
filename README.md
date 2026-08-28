# 🥐 Biz-Forecaster — AI 수요 예측 (빵집 스마트 필터)

> 외부 이벤트 우선·품목간 수요연쇄 연계형 **소상공인 수요예측 POC**
> 2026 한이음 드림업 프로젝트

과거 판매 내역을 바탕으로 동네 빵집의 **품목별 향후 5주 수요를 예측**하는 Streamlit 웹 애플리케이션입니다. 여러 시계열·머신러닝 모델을 **앙상블**로 결합해 예측하고, 품목을 **판매 활성도(최근성)·판매 규모(등급)** 기준으로 필터링해 원하는 대상을 쉽게 분석할 수 있습니다.

---

## ✨ 주요 기능

- **📂 CSV 자동 파싱** — 판매내역 CSV를 업로드하면 품목·날짜·수량을 자동 추출 (소계/누계 행 자동 제외)
- **🏆 품목 등급 자동 분류** — 총 판매량 기준 상위 백분율로 💎시그니처 / 🥇인기 / 🥈스테디셀러 / 🥉일반 4등급 부여
- **🔎 스마트 필터** — 판매 활성도(최근 3·6·12개월)와 판매 등급을 조합해 품목 필터링
- **🤖 앙상블 예측** — 여러 모델의 예측을 평균 내어 향후 5주 수요 산출
- **📊 시각화** — 과거 판매실적 + 예측 추세를 Plotly 인터랙티브 그래프로 표시
- **💾 엑셀 다운로드** — 개별 품목 / Top N 예측 결과를 `.xlsx`로 내보내기
- **📋 Top N 일괄 예측** — 판매량 상위 10·20·30개 품목을 한 번에 예측

## 🧠 예측 모델 (앙상블 구성)

| 모델 | 종류 | 비고 |
|------|------|------|
| 선형 추세 (Linear Trend) | 통계 | 최근 8주 기울기 외삽, 폴백 모델 |
| Holt (지수평활) | 시계열 | 추세 반영 |
| ARIMA(1,1,1) | 시계열 | |
| RandomForest | 머신러닝 | `scikit-learn` 설치 시 |
| XGBoost | 머신러닝 | `xgboost` 설치 시 |
| LSTM | 딥러닝 | `tensorflow` 설치 시 (선택) |

> 설치된 라이브러리에 따라 자동으로 사용 가능한 모델만 앙상블에 포함됩니다.
> 데이터가 부족한 거래처는 선형 추세 모델로 자동 폴백됩니다.

---

## 🚀 실행 방법

### 간편 실행 (Windows)
폴더 안의 **`실행하기.bat`** 을 더블 클릭하면 가상환경 생성 → 라이브러리 설치 → 앱 실행까지 자동으로 진행됩니다.

> ⚠️ **사전 준비:** 컴퓨터에 [Python](https://www.python.org/downloads/)이 설치되어 있어야 합니다.
> 설치 시 첫 화면에서 **`Add Python to PATH`** 체크박스를 반드시 체크하세요.

### 수동 실행
```bash
# 1. 가상환경 생성 및 활성화
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# 2. 라이브러리 설치
pip install -r requirements.txt

# 3. 앱 실행
streamlit run app.py
```
실행하면 자동으로 웹 브라우저가 열립니다. 열리지 않으면 터미널에 표시된 `localhost` 주소로 접속하세요.

---

## 📁 파일 구성

```
수요예측프로그램/
├── app.py                  # Streamlit 메인 애플리케이션
├── requirements.txt        # 의존성 목록
├── 실행하기.bat            # Windows 원클릭 실행 스크립트
├── 설명서.txt              # 사용자용 간단 안내
├── docker-compose.yml      # Grafana 모니터링 컨테이너 구성
├── README.md               # 현재 문서
│
├── data/                   # 이벤트 통합 데이터셋 (H1·H2′·이상탐지·SHAP·What-if 탭 검증용)
│   ├── integrated_dataset.csv   # 합성 POS+이벤트 데이터 (2024~2025, 실거래 정보 아님)
│   │                           #  출처: 2026-Han-Project/dataset @ e3bc984 (2026-08-28)
│   │                           #        data/processed/integrated_dataset.csv 를 그대로 복사
│   └── eda.ipynb                 # 데이터셋 탐색 노트북
│
└── grafana/                # Grafana 프로비저닝 (데이터소스·대시보드·알림)
    ├── provisioning/alerting/   # 알림 규칙·수신처·라우팅 정책
    └── alert_receiver.py        # 알림 수신 확인용 로컬 웹훅 서버 (표준 라이브러리만 사용)
    ├── dashboards/
    └── provisioning/
```

> ℹ️ 실제 판매내역 CSV(사용자가 사이드바로 업로드하는 원본)는 개인정보·영업정보 보호를 위해 저장소에서 제외됩니다 (`.gitignore`의 `*.csv`). 단 `data/integrated_dataset.csv`는 실거래 정보가 아닌 합성 데이터라 예외로 추적됩니다.

## 🗃️ 입력 데이터 형식

판매내역 CSV는 각 줄에 **품목명 + 날짜(YYYY-MM-DD) + 수량**이 포함되어야 합니다.
`상품명`/`품목명`, `판매내역조회` 헤더 행과 `(소계)`, `(누계)` 행은 자동으로 걸러집니다.

---

## 🔧 사용법

1. 왼쪽 사이드바에서 **판매내역 CSV 업로드**
2. **[📊 품목별 상세 분석]** 탭
   - 판매 활성도·등급 필터로 품목 좁히기 → 특정 품목 선택
   - 5주 앙상블 예측 결과표 + 추세 그래프 확인 → 엑셀 다운로드
3. **[🏆 베스트셀러 TOP N]** 탭
   - 판매량 상위 N개 품목을 일괄 예측 → 종합 그래프 + 엑셀 다운로드

---

## 🛣️ 로드맵 (기획 대비 현황)

본 저장소는 전체 기획의 **1차 PoC**입니다. 이벤트 통합 데이터셋(`integrated_dataset.csv`, `date/item/category/sales_qty/is_weekend/is_holiday/is_vacation/season_period/precip_type/temperature` 컬럼 포함)을 업로드하면 아래 기능들을 각각의 탭에서 백테스트로 검증할 수 있습니다.

- [x] **Layer 1 — 시계열 앙상블 기저 수요 예측** (📊 품목별 상세 분석 탭) : 선형추세·Holt·ARIMA·Prophet·TFT·RandomForest·XGBoost 7종 단순평균
- [x] **Layer 1 고도화 — TFT(Temporal Fusion Transformer)** : 품목별 단일 시계열로 학습해 Prophet과 성능 비교(논문용). 일별 H=14는 Prophet 우세, 주간 H=5는 TFT 우세로 해상도에 따라 갈림 (📊 탭 결과표의 TFT 컬럼)
- [x] **Layer 2 — 이벤트 우선(Event-first) 예측 (H1)** : 공휴일·날씨·방학·시즌·경쟁점포 이벤트를 외생변수로 반영, 임계치 게이팅으로 Walk-forward 백테스트 검증 (🌦 탭)
- [x] **Layer 3 — 품목간 수요연쇄 연계 (H2′)** : 카테고리 간 교차상관·Granger 인과검정으로 최적 시차 탐색 및 백테스트 검증 (🔗 탭)
- [x] **이상탐지** : Isolation Forest + 3σ + EWMA 3중 탐지, 전체 매출 동반 vs 품목 단독 이상 구분 (🚨 탭)
- [x] **설명가능성(XAI)** : SHAP TreeExplainer로 예측을 "기저치 + 이벤트별 기여도"로 분해, 자연어 브리핑 생성 (🧮 탭)
- [x] **LangGraph 에이전트** : 조회→예측→해석→브리핑 4단계 워크플로우 (품목별 상세 분석 탭 하단)
- [x] **Grafana 모니터링·알림** : 이상탐지 결과를 SQLite로 내보내 대시보드로 시각화하고, 다중 탐지·전체매출 동반 이상이 감지되면 웹훅으로 알림 발송 (아래 "모니터링 대시보드 실행" 참고)
- [x] **What-if 채팅** : 이벤트 조건을 체크·채팅으로 입력하면 H1 모델로 실계산한 예상 판매량·권장 생산량 응답 (💬 탭, 키워드 매칭 기반이며 LLM 미연동)
- [ ] N노드(체인점·가맹점) 확장 — 본 PoC 로드맵 범위 밖, 향후 계획

> 📄 상세 기획·검증 내용은 한이음 중간보고서에 정리되어 있습니다. 보고서 원문(`docs/`)은 평가 전 공개를 피하기 위해 저장소에서 제외했으며, 필요하시면 팀에 요청해 주세요.

---

## 📈 모니터링 대시보드 실행 (Grafana)

이상탐지 탭에서 "📤 Grafana용 SQLite로 내보내기" 버튼을 누르면 `monitoring.db`가 생성됩니다. 이 폴더에서 Docker로 Grafana를 띄우면 자동으로 데이터소스·대시보드가 구성됩니다.

```bash
docker compose -p biz-forecaster up -d
# http://localhost:3000 접속 (로그인 없이 익명 Admin으로 접속됨)
# "Biz-Forecaster 이상탐지 모니터링" 대시보드가 자동 프로비저닝되어 있음
```

종료하려면:
```bash
docker compose -p biz-forecaster down
```

### 알림 (Alerting)

Grafana가 뜰 때 알림 규칙 2건이 함께 프로비저닝됩니다. 내보낸 데이터의 마지막 날짜 기준
**최근 30일** 구간을 1분마다 평가합니다.

| 규칙 | 발동 조건 | 심각도 |
|---|---|---|
| 다중 탐지 이상 발생 | 3σ·EWMA·Isolation Forest 중 **2개 이상이 동시에** 이상 판정한 날이 있을 때 | warning |
| 전체 매출 동반 이상 발생 | 품목이 아니라 **매장 전체 매출**이 함께 흔들린 이상이 있을 때 | critical |

기본 수신처는 **로컬 웹훅**이라 외부 서비스 계정도 비용도 필요 없습니다. 알림이 실제로
도착하는지 보려면 Grafana를 띄우기 전에 수신 서버를 실행해 두십시오.

```bash
python grafana/alert_receiver.py        # 기본 포트 9099
```

받은 알림은 콘솔에 요약되고 `grafana/alerts_received.log`에 원문이 쌓입니다.
Slack 등 외부로 보내려면 `.env`에 다음을 두면 기본 웹훅 URL을 덮어씁니다.

```
GRAFANA_ALERT_WEBHOOK_URL=https://hooks.slack.com/services/...
```

> `.env`는 `.gitignore`로 차단되어 있으므로 웹훅 URL 같은 비밀값이 저장소에 올라가지 않습니다.


> `grafana/provisioning/`에 데이터소스(SQLite 플러그인: `frser-sqlite-datasource`)와 대시보드가 코드로 정의되어 있어, 컨테이너를 새로 띄워도 수동 설정 없이 그대로 재현됩니다.

---

## 🛠️ 기술 스택

`Python` · `Streamlit` · `pandas` · `numpy` · `plotly` · `statsmodels` · `scikit-learn` · `xgboost` · `prophet` · `pytorch-forecasting`(TFT) · `torch` · `lightning` · `langgraph` · `shap` · `openpyxl` · `Grafana`(Docker)
