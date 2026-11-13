# End-to-End Machine Learning Pipeline: Demand Forecasting & Credit Risk Prediction

실제 도메인 데이터셋을 대상으로 데이터 정제, 도메인 기반 특징 공학(Feature Engineering), 데이터 누수(Data Leakage) 차단 파이프라인 구축, 교차 검증 기반 모델 선별 및 하이퍼파라미터 튜닝까지 전 과정을 수행한 머신러닝 프로젝트입니다.

---

## 📌 프로젝트 개요

* **목표:** 단순 라이브러리 호출을 넘어, 데이터의 구조적 한계(비운영일 노이즈, 타깃 우편향, 클래스 불균형 등)를 이론적으로 진단하고 파이프라인을 설계하여 베이스라인 대비 예측 성능을 극대화합니다.
* **핵심 기술 역량:**
  * `ColumnTransformer` 및 `Pipeline` 캡슐화를 통한 검증 단계 데이터 누수(Data Leakage) 원천 차단
  * 도메인 지식 기반 파생변수 설계 및 Ablation Study(단계별 성능 비교)를 통한 유효성 입증
  * 데이터 분포와 비즈니스 목적에 부합하는 평가지표($R^2$, RMSE, Minority F1-Score, Recall) 중심의 모델 최적화

---

## 📂 디렉터리 구조

```text
ml-end-to-end-project/
├── data/                              # 실험 데이터셋 (로컬 실행 시 배치)
│   ├── SeoulBikeData.csv
│   └── default of credit card clients.xls
├── docs/                              # 상세 분석 및 실험 결과 보고서
│   └── ml_project_report.pdf
├── src/                               # 머신러닝 파이프라인 소스 코드
│   ├── task1_bike_demand.py           # [Task 1] 서울시 공공자전거 대여량 예측 (회귀)
│   └── task2_credit_default.py        # [Task 2] 신용카드 채무 불이행 예측 (분류)
├── .gitignore                         # Git 추적 제외 목록
└── README.md                          # 프로젝트 소개 및 실행 가이드