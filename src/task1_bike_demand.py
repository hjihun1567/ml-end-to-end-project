import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_validate, RandomizedSearchCV
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

SEED = 77

# 데이터 로드 및 칼럼명 강제 표준화
df_raw = pd.read_csv("SeoulBikeData.csv", encoding="cp949")

safe_columns = [
    'Date', 'target', 'hour', 'temp', 'humidity', 'wind_speed', 
    'visibility', 'dew_point', 'solar_rad', 'rainfall', 'snowfall', 
    'season', 'holiday', 'functioning'
]
df_raw.columns = safe_columns
print("\n데이터 로드 완료 (Seoul 공공자전거 대여량 데이터셋)", sep="")
print("데이터셋 크기:", df_raw.shape)
print("전체 데이터 가동 여부 분포(Yes: 운영일, No: 비운영일):\n", df_raw['functioning'].value_counts())
print("타깃 변수(대여량) 기초 통계량:\n", df_raw['target'].describe())
print("\n")


# 피처 엔지니어링이 배제된 순수 원본 데이터 준비
# 단, 모델 학습이 가능하도록 최소한의 Date 변환과 비운영일 격리(Rule-based)는 동일 적용하여 통제
df_base = df_raw[df_raw["functioning"] == "Yes"].drop(columns=["functioning"])
df_base["Date"] = pd.to_datetime(df_base["Date"], format="%d/%m/%Y")
df_base["month"] = df_base["Date"].dt.month
df_base = df_base.drop(columns=["Date"])

X_b = df_base.drop(columns=["target"])
y_b = df_base["target"]
X_tr_b, X_te_b, y_tr_b, y_te_b = train_test_split(X_b, y_b, test_size=0.2, random_state=SEED)

# 제안 방법론 특징 공학
df_prop = df_raw.copy()

# 파생 변수 1: 체감 불쾌감 지표 (온도와 습도 상호작용 반영)
df_prop["discomfort_index"] = 1.8 * df_prop["temp"] - 0.55 * (1 - df_prop["humidity"]/100) * (1.8 * df_prop["temp"] - 26) + 32

# 파생 변수 2: 날짜 분해 및 주말 여부
df_prop["Date"] = pd.to_datetime(df_prop["Date"], format="%d/%m/%Y")
df_prop["month"]     = df_prop["Date"].dt.month
df_prop["dayofweek"] = df_prop["Date"].dt.dayofweek
df_prop["weekend"]   = (df_prop["dayofweek"] >= 5).astype(int)

# 파생 변수 3: 시간의 원형 주기성 보존 삼각 변환 (sin/cos)
df_prop["hour_sin"]  = np.sin(2 * np.pi * df_prop["hour"] / 24)
df_prop["hour_cos"]  = np.cos(2 * np.pi * df_prop["hour"] / 24)

# 다중공선성 및 중복 정보를 피하기 위해 원본 날짜 및 기존 시간(hour) 변수 제거
df_prop = df_prop.drop(columns=["Date", "hour"])

# 운영일 데이터만 모델 학습용으로 격리 (Rule-based)
df_prop_func = df_prop[df_prop["functioning"] == "Yes"].drop(columns=["functioning"])

X_p = df_prop_func.drop(columns=["target"])
y_p = df_prop_func["target"]
X_tr_p, X_te_p, y_tr_p, y_te_p = train_test_split(X_p, y_p, test_size=0.2, random_state=SEED)

# 공통 검증 설계
cv5 = KFold(n_splits=5, shuffle=True, random_state=SEED)
cv3 = KFold(n_splits=3, shuffle=True, random_state=SEED) # 하이퍼파라미터 튜닝 시 연산 속도 타협안

# 전처리기 정의
cat_cols_b = X_b.select_dtypes(include="object").columns.tolist()
num_cols_b = X_b.select_dtypes(exclude="object").columns.tolist()

cat_cols_p = X_p.select_dtypes(include="object").columns.tolist()
num_cols_p = X_p.select_dtypes(exclude="object").columns.tolist()

scaled_pre_b = ColumnTransformer([("num", StandardScaler(), num_cols_b), ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols_b)])
scaled_pre_p = ColumnTransformer([("num", StandardScaler(), num_cols_p), ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols_p)])

tree_pre_b = ColumnTransformer([("num", "passthrough", num_cols_b), ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols_b)])
tree_pre_p = ColumnTransformer([("num", "passthrough", num_cols_p), ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols_p)])

wrap = lambda pipe: TransformedTargetRegressor(pipe, func=np.log1p, inverse_func=np.expm1)

# Ablation Study 특징 공학 및 Target 로그 변환 효과의 단계적 검증
print("\nAblation Study: 단계별 방법론 추가에 따른 성능 개선 증명\n", sep="")

# 1)  원본 피처 기반 RandomForest (로그 변환 미적용)
rf_base = Pipeline([("pre", tree_pre_b), ("m", RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1))])
scores_base = cross_validate(rf_base, X_tr_b, y_tr_b, cv=cv5, scoring='neg_root_mean_squared_error', n_jobs=-1)
print(f"1) 순수 피처 기반 RF (로그 미적용)   : 5-Fold CV RMSE = {-scores_base['test_score'].mean():.2f}")

# 2) 제안 피처 적용 RandomForest (로그 변환 미적용)
rf_prop_no_log = Pipeline([("pre", tree_pre_p), ("m", RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1))])
scores_prop_no_log = cross_validate(rf_prop_no_log, X_tr_p, y_tr_p, cv=cv5, scoring='neg_root_mean_squared_error', n_jobs=-1)
print(f"2)금융/기상 피처 주입 (No Log) : 5-Fold CV RMSE = {-scores_prop_no_log['test_score'].mean():.2f} ")

# 3) 제안 피처 + Target Log1p 변환 결합 RandomForest
rf_prop_log = wrap(Pipeline([("pre", tree_pre_p), ("m", RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1))]))
scores_prop_log = cross_validate(rf_prop_log, X_tr_p, y_tr_p, cv=cv5, scoring='neg_root_mean_squared_error', n_jobs=-1)
print(f"3)피처 + Target 로그변환 (최종) : 5-Fold CV RMSE = {-scores_prop_log['test_score'].mean():.2f}")

# 전체 머신러닝 알고리즘 교차 검증 비교 (Proposed Features 기준)
print("\n머신러닝 알고리즘 교차 검증 비교 (운영일 기준)\n", sep="")

models = {
    "Linear": wrap(Pipeline([("pre", scaled_pre_p), ("m", LinearRegression())])),
    "DecisionTree": wrap(Pipeline([("pre", tree_pre_p), ("m", DecisionTreeRegressor(random_state=SEED))])),
    "RandomForest": rf_prop_log,
    "SVR": wrap(Pipeline([("pre", scaled_pre_p), ("m", SVR(tol=1e-2))])),
    "MLP": wrap(Pipeline([("pre", scaled_pre_p), ("m", MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, early_stopping=True, random_state=SEED))]))
}

base_comparison = []
for name, mdl in models.items():
    scores = cross_validate(mdl, X_tr_p, y_tr_p, cv=cv5, 
                            scoring={'rmse': 'neg_root_mean_squared_error', 'r2': 'r2'}, 
                            n_jobs=-1)
    
    mean_rmse = -scores['test_rmse'].mean()
    mean_r2 = scores['test_r2'].mean()
    
    base_comparison.append({
        "Model": name,
        "CV_RMSE": round(mean_rmse, 2),
        "CV_R2 (설명력)": mean_r2,
        "Std": round(scores['test_rmse'].std(), 2)
    })

print_df = pd.DataFrame(base_comparison).copy()
print_df["CV_R2 (설명력)"] = print_df["CV_R2 (설명력)"].map(lambda x: f"{x * 100:.2f}%")
print(print_df.sort_values("CV_RMSE").to_string(index=False))

# 1위 모델 자동 선별 및 집중 하이퍼파라미터 튜닝
print("\n1위 모델 자동 집중 튜닝\n", sep="")

best_base_name = min(base_comparison, key=lambda x: x["CV_RMSE"])["Model"]
print(f"(자동 선별) 기본 비교 1위 모델: {best_base_name}")
print(f"--> {best_base_name} 모델의 하이퍼파라미터를 최적화 합니다.\n")

if best_base_name == "SVR":
    svr_pipe = wrap(Pipeline([("pre", scaled_pre_p), ("m", SVR(tol=1e-2))]))
    svr_param = {
        "regressor__m__C": [10, 50, 100, 200],
        "regressor__m__gamma": ["scale", "auto", 0.01, 0.05, 0.1],
        "regressor__m__epsilon": [0.01, 0.05, 0.1]
    }
    grid_search = RandomizedSearchCV(svr_pipe, svr_param, n_iter=10, cv=cv3,
                                     scoring="neg_root_mean_squared_error", n_jobs=-1, random_state=SEED)

elif best_base_name == "MLP":
    mlp_pipe = wrap(Pipeline([("pre", scaled_pre_p), ("m", MLPRegressor(max_iter=1500, early_stopping=True, random_state=SEED))]))
    mlp_param = {
        "regressor__m__hidden_layer_sizes": [(128, 64), (256, 128), (256, 128, 64)],
        "regressor__m__alpha": [1e-4, 1e-3, 1e-2],
        "regressor__m__learning_rate_init": [1e-3, 5e-3, 1e-2]
    }
    grid_search = RandomizedSearchCV(mlp_pipe, mlp_param, n_iter=10, cv=cv3,
                                     scoring="neg_root_mean_squared_error", n_jobs=-1, random_state=SEED)

elif best_base_name == "RandomForest":
    rf_pipe = wrap(Pipeline([("pre", tree_pre_p), ("m", RandomForestRegressor(random_state=SEED, n_jobs=-1))]))
    rf_param = {
        "regressor__m__n_estimators": [100, 200, 300],
        "regressor__m__max_depth": [10, 20, 25, None],
        "regressor__m__min_samples_split": [2, 5, 10]
    }
    grid_search = RandomizedSearchCV(rf_pipe, rf_param, n_iter=10, cv=cv3,
                                     scoring="neg_root_mean_squared_error", n_jobs=-1, random_state=SEED)
else:
    tree_pipe = wrap(Pipeline([("pre", tree_pre_p), ("m", DecisionTreeRegressor(random_state=SEED))]))
    tree_param = {"regressor__m__max_depth": [5, 10, 15, None]}
    grid_search = RandomizedSearchCV(tree_pipe, tree_param, n_iter=5, cv=cv3,
                                     scoring="neg_root_mean_squared_error", n_jobs=-1, random_state=SEED)

# 최종 최적화 피팅 진행
grid_search.fit(X_tr_p, y_tr_p)
best_model = grid_search.best_estimator_

print(f"[{best_base_name} 튜닝 완료]")
print(f"   - 최적 CV RMSE: {-grid_search.best_score_:.2f}")
print(f"   - 최적 설정값: {grid_search.best_params_}")

# 최종 평가
print("\n최종 성능 평가 (전체 원본 데이터 기준)\n", sep="")
best_model.fit(X_tr_p, y_tr_p)

# 운영일 테스트셋 예측 및 평가
test_pred = best_model.predict(X_te_p)
test_rmse = np.sqrt(mean_squared_error(y_te_p, test_pred))
test_r2 = r2_score(y_te_p, test_pred)

print(f"1) 운영일 기준 독립 테스트셋 최종 검증")
print(f"   - RMSE      : {test_rmse:.2f}")
print(f"   - R2 설명력 : {test_r2 * 100:.2f}%")

def evaluate_full(raw_df):
    out = raw_df.copy()
    pred = np.zeros(len(out))
    
    is_functioning = out["functioning"] == "Yes"
    feat = out[is_functioning].drop(columns=["functioning", "target"])
    feat = feat[X_tr_p.columns]
    pred[is_functioning.values] = best_model.predict(feat)
    
    y_actual = out["target"]
    full_rmse = np.sqrt(mean_squared_error(y_actual, pred))
    full_r2 = r2_score(y_actual, pred)
    return full_rmse, full_r2

full_rmse, full_r2 = evaluate_full(df_prop)
print(f"\n2) 비운영일 포함 전체 데이터셋 최종 결합 평가")
print(f"   - 결합 RMSE : {full_rmse:.2f}")
print(f"   - 결합 R2   : {full_r2 * 100:.2f}%")
