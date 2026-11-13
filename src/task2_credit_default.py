import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from pathlib import Path

SEED = 77


# 데이터 로드 및 정제
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "default of credit card clients.xls"
df = pd.read_excel(DATA_PATH, header=1)

if 'ID' in df.columns:
    df = df.drop(columns=['ID'])

target_col = [col for col in df.columns if 'default' in col.lower()][0]
df = df.rename(columns={target_col: 'target'})

# 무결성 확보: 정의되지 않은 비공식 범주 값 병합 처리
df['EDUCATION'] = df['EDUCATION'].replace({0: 4, 5: 4, 6: 4}) # 0,5,6 -> 4(Others)
df['MARRIAGE']  = df['MARRIAGE'].replace({0: 3})            # 0 -> 3(Others)

print("\n데이터 로드 완료 (UCI Credit Card Default)\n", sep="")
print("데이터셋 크기:", df.shape)
print("타깃 클래스 분포(0: 정상, 1: 연체):\n", df['target'].value_counts(normalize=True))


# 금융 도메인 피처 엔지니어링 (Feature Engineering)
# [Ablation Study를 위한 피처 엔지니어링 미적용 순수 원본 복사본]
df_base = df.copy()

# 도메인 파생 피처 반영 적용
df_prop = df.copy()
bill_cols = [f'BILL_AMT{i}' for i in range(1, 7)]
pay_amt_cols = [f'PAY_AMT{i}' for i in range(1, 7)]
pay_status_cols = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']

# 파생 변수 1: 6개월간 누적 청구액 및 납부액, 그리고 잔여 실질 채무 계산
df_prop['total_bill'] = df_prop[bill_cols].sum(axis=1)
df_prop['total_pay'] = df_prop[pay_amt_cols].sum(axis=1)
df_prop['remaining_debt'] = df_prop['total_bill'] - df_prop['total_pay']

# 파생 변수 2 [PAY -2, 0 특수 코드 도메인 정밀 분리 패치]:
# -2: 당월 무소비 혹은 청구액 발행 전 선상환 완료 우량 고객 (is_advance_payer)
#  0: 최소 결제금액 상환을 통해 연체 등록을 유예 중인 리볼빙(Revolving) 유저 (is_revolving_user)
df_prop['is_advance_payer'] = (df_prop['PAY_0'] == -2).astype(int)
df_prop['is_revolving_user'] = (df_prop['PAY_0'] == 0).astype(int)

# 파생 변수 3 [한도 소진율 (utilization_rate) 피처 추가]:
# 신용 한도(LIMIT_BAL) 대비 최근 청구액(BILL_AMT1)의 비율을 통해 차주의 금융적 한계 압박 지수를 간접 평가
df_prop['utilization_rate'] = df_prop['BILL_AMT1'] / (df_prop['LIMIT_BAL'] + 1e-5)

# 파생 변수 4: 6개월 동안의 최대 연체 월수 및 연체 누적 횟수
df_prop['max_delay'] = df_prop[pay_status_cols].max(axis=1)
df_prop['delay_months_count'] = (df_prop[pay_status_cols] > 0).sum(axis=1)


# 데이터셋 분할 및 전처리 파이프라인 설계

# 1) 원본 베이스라인 데이터셋 분할
X_b = df_base.drop(columns=['target'])
y_b = df_base['target']
X_tr_b, X_te_b, y_tr_b, y_te_b = train_test_split(X_b, y_b, test_size=0.2, stratify=y_b, random_state=SEED)

# 2) 파생 변수 포함 제안 데이터셋 분할
X_p = df_prop.drop(columns=['target'])
y_p = df_prop['target']
X_tr_p, X_te_p, y_tr_p, y_te_p = train_test_split(X_p, y_p, test_size=0.2, stratify=y_p, random_state=SEED)

cat_cols = ['SEX', 'EDUCATION', 'MARRIAGE']
num_cols_b = [col for col in X_b.columns if col not in cat_cols]
num_cols_p = [col for col in X_p.columns if col not in cat_cols]

# 모델 계열별 전처리기 구성
scaled_pre_b = ColumnTransformer([("num", StandardScaler(), num_cols_b), ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)])
scaled_pre_p = ColumnTransformer([("num", StandardScaler(), num_cols_p), ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)])

tree_pre_b = ColumnTransformer([("num", "passthrough", num_cols_b), ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)])
tree_pre_p = ColumnTransformer([("num", "passthrough", num_cols_p), ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)])

# StratifiedKFold 적용: 불균형 비율(78:22)을 모든 Fold에서 일정하게 제어
cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
cv3 = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)


# Ablation Study & 베이스라인 명시 측정

print("\nAblation Study: 금융 도메인 피처 및 특수코드 분리 전/후 비교 검증\n", sep="")

# 1) 피처 엔지니어링이 전혀 가미되지 않은 순수 원본 베이스라인 RandomForest
rf_base = Pipeline([("pre", tree_pre_b), ("m", RandomForestClassifier(class_weight='balanced', random_state=SEED, n_jobs=-1))])
scores_base = cross_validate(rf_base, X_tr_b, y_tr_b, cv=cv5, scoring=['f1', 'f1_macro'], n_jobs=-1)
print(f"1) 순수 베이스라인 RF : CV F1 (Minority) = {scores_base['test_f1'].mean()*100:.2f}%, F1_Macro = {scores_base['test_f1_macro'].mean()*100:.2f}%")

# 2) 제안 파생 변수(소진율, 선결제, 리볼빙, 잔여채무 등)가 주입된 제안 RandomForest
rf_proposed = Pipeline([("pre", tree_pre_p), ("m", RandomForestClassifier(class_weight='balanced', random_state=SEED, n_jobs=-1))])
scores_proposed = cross_validate(rf_proposed, X_tr_p, y_tr_p, cv=cv5, scoring=['f1', 'f1_macro'], n_jobs=-1)
print(f"2) 제안 특징공학 적용 RF : CV F1 (Minority) = {scores_proposed['test_f1'].mean()*100:.2f}%, F1_Macro = {scores_proposed['test_f1_macro'].mean()*100:.2f}%")


# 전체 모델 교차 검증 다각도 비교 (f1_macro 병행 관측)

print("\n전체 머신러닝 모델 교차 검증 다각도 비교 (F1 & F1-Macro 병행)\n", sep="")

# [학술적 해명 주석]: MLPClassifier는 class_weight='balanced' 옵션을 내장 지원하지 않습니다. 
# 이로 인해 불균형 연체 클래스에 편향적인 가중 전파가 차단되는 한계를 미리 보고서에 방어 설명하기 위해 
# 최종 모델 선정 전 비교군으로 정상 수록해 둡니다.
models = {
    "Logistic": Pipeline([("pre", scaled_pre_p), ("m", LogisticRegression(class_weight='balanced', random_state=SEED))]),
    "DecisionTree": Pipeline([("pre", tree_pre_p), ("m", DecisionTreeClassifier(class_weight='balanced', random_state=SEED))]),
    "RandomForest": rf_proposed,
    "SVC": Pipeline([("pre", scaled_pre_p), ("m", SVC(class_weight='balanced', tol=1e-2, random_state=SEED))]),
    "MLP": Pipeline([("pre", scaled_pre_p), ("m", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, early_stopping=True, random_state=SEED))])
}

base_comparison = []
for name, mdl in models.items():
    # Minority Class F1과 전반적인 클래스 합산 F1-Macro를 병행 평가하여 보고서 신뢰도 극대화
    scores = cross_validate(mdl, X_tr_p, y_tr_p, cv=cv5, scoring=['f1', 'f1_macro'], n_jobs=-1)
    mean_f1 = scores['test_f1'].mean()
    mean_f1_macro = scores['test_f1_macro'].mean()
    
    base_comparison.append({
        "Model": name,
        "CV_F1_Score (Minority)": mean_f1,
        "CV_F1_Macro": mean_f1_macro,
        "Std": scores['test_f1'].std()
    })

print_df = pd.DataFrame(base_comparison)
print_df["CV_F1_Score (Minority)"] = print_df["CV_F1_Score (Minority)"].map(lambda x: f"{x * 100:.2f}%")
print_df["CV_F1_Macro"] = print_df["CV_F1_Macro"].map(lambda x: f"{x * 100:.2f}%")
print(print_df.sort_values("CV_F1_Score (Minority)", ascending=False).to_string(index=False))


# 1위 모델 자동 선별 및 집중 하이퍼파라미터 최적화

print("\n1위 모델 자동 집중 튜닝\n", sep="")

best_base_name = max(base_comparison, key=lambda x: x["CV_F1_Score (Minority)"])["Model"]
print(f"(자동 선별)기본 비교 1위 모델: {best_base_name}")
print(f"--> {best_base_name} 모델의 하이퍼파라미터를 최적화 합니다.\n")


if best_base_name == "Logistic":
    log_pipe = Pipeline([("pre", scaled_pre_p), ("m", LogisticRegression(class_weight='balanced', random_state=SEED))])
    log_param = {"m__C": [0.01, 0.1, 1.0, 10.0, 100.0]}
    grid_search = RandomizedSearchCV(log_pipe, log_param, n_iter=5, cv=cv3, scoring="f1", n_jobs=-1, random_state=SEED)


elif best_base_name == "SVC":
    svc_pipe = Pipeline([("pre", scaled_pre_p), ("m", SVC(class_weight='balanced', tol=1e-2, random_state=SEED))])
    svc_param = {
        "m__C": [1, 10, 50, 100],
        "m__gamma": ["scale", "auto", 0.01, 0.1]
    }
    grid_search = RandomizedSearchCV(svc_pipe, svc_param, n_iter=10, cv=cv3, scoring="f1", n_jobs=-1, random_state=SEED)


else:
    rf_pipe = Pipeline([("pre", tree_pre_p), ("m", RandomForestClassifier(class_weight='balanced', random_state=SEED, n_jobs=-1))])
    rf_param = {
        "m__n_estimators": [100, 200, 300],
        "m__max_depth": [10, 15, 20, None],
        "m__min_samples_split": [2, 5, 10]
    }
    grid_search = RandomizedSearchCV(rf_pipe, rf_param, n_iter=10, cv=cv3, scoring="f1", n_jobs=-1, random_state=SEED)

grid_search.fit(X_tr_p, y_tr_p)
best_model = grid_search.best_estimator_

print(f"[{best_base_name} 튜닝 완료]")
print(f"   - 최적 CV F1-Score: {grid_search.best_score_ * 100:.2f}%")
print(f"   - 최적 설정값: {grid_search.best_params_}")


# 최종 평가 및 Confusion Matrix 출력

print("\n최종 평가 및 Confusion Matrix (미학습된 독립 테스트 데이터 기준)\n",sep="")
best_model.fit(X_tr_p, y_tr_p)
y_pred = best_model.predict(X_te_p)

print("1) Classification Report:")
print(classification_report(y_te_p, y_pred))

print("2) Confusion Matrix (혼동 행렬):")
cm = confusion_matrix(y_te_p, y_pred)
print(f"   - [실제정상, 예측정상] (True Negative) : {cm[0,0]}")
print(f"   - [실제정상, 예측연체] (False Positive): {cm[0,1]}")
print(f"   - [실제연체, 예측정상] (False Negative): {cm[1,0]}")
print(f"   - [실제연체, 예측연체] (True Positive) : {cm[1,1]}")
