from pathlib import Path
import time

import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit

PROJECT_ROOT = Path(r"C:\Users\splen\OneDrive\Desktop\BDAI_\BOOSTMAP\Fraud-FDS-Project")
REPO_ROOT = PROJECT_ROOT / "revision" / "BDAI"   # git clone 받은 폴더

TRAIN_11 = PROJECT_ROOT / "revision" / "train_11features.csv"
TEST_11 = PROJECT_ROOT / "revision" / "test_11features.csv"
RAW_TRAIN = PROJECT_ROOT / "data" / "raw" / "fraudTrain.csv"
RAW_TEST = PROJECT_ROOT / "data" / "raw" / "fraudTest.csv"
FINAL_MODEL_PATH = REPO_ROOT / "webapp" / "backend" / "model" / "final_lightgbm_leakfree.joblib"
CLUSTER_MODEL_PATH = REPO_ROOT / "webapp" / "backend" / "model" / "lightGBM_combination3_K7_cluster_model.joblib"
OUT_PATH = PROJECT_ROOT / "revision" / "tableau_export_leakfree.csv"

FEATURE_COLUMNS = [
    "category", "amt", "trans_hour", "age",
    "recent_24h_high_amt_count", "amt_to_prior_median_ratio",
    "rolling_sum_amt_1h", "amt_zscore_card", "prior_normal_median_amt",
    "count_30min", "high_speed",
]
TARGET = "is_fraud"

print("STEP 1. leak-free 피처 + 최종 모델 로드")
train_df = pd.read_csv(TRAIN_11, parse_dates=["trans_date_trans_time"]).sort_values("trans_date_trans_time").reset_index(drop=True)
test_df = pd.read_csv(TEST_11, parse_dates=["trans_date_trans_time"]).sort_values("trans_date_trans_time").reset_index(drop=True)

pkg = joblib.load(FINAL_MODEL_PATH)
final_model = pkg["model"]
THRESHOLD = pkg["threshold"]
print("최종모델 임계값:", THRESHOLD)

train_df["category"] = train_df["category"].astype("category")
test_df["category"] = test_df["category"].astype(pd.CategoricalDtype(categories=train_df["category"].cat.categories))

# ============================================================
# STEP 2. Train 확률: 3-Fold OOF로 재현 (fold1의 train 구간만 in-sample 최종모델로 대체)
# ============================================================
print("\nSTEP 2. Train 확률 계산 (OOF 우선, 첫 구간만 최종모델로 보완)")
X = train_df[FEATURE_COLUMNS]
y = train_df[TARGET].astype("int8")

BEST_PARAMS = dict(num_leaves=31, max_depth=-1, min_child_samples=50,
                    learning_rate=0.05, reg_alpha=0.1, reg_lambda=1.0)
CV_BASE = dict(objective="binary", boosting_type="gbdt", n_estimators=1000,
               subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
               random_state=42, n_jobs=-1, verbosity=-1)

tscv = TimeSeriesSplit(n_splits=3)
splits = list(tscv.split(X))

train_prob = np.full(len(train_df), np.nan, dtype="float64")
oof_covered = np.zeros(len(train_df), dtype=bool)

t0 = time.perf_counter()
for fold_idx, (tr_idx, va_idx) in enumerate(splits, start=1):
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
    neg, pos = int((y_tr == 0).sum()), int((y_tr == 1).sum())
    spw = neg / pos if pos > 0 else 1.0
    model = lgb.LGBMClassifier(**CV_BASE, **BEST_PARAMS, scale_pos_weight=spw)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="average_precision",
               callbacks=[lgb.early_stopping(stopping_rounds=50, first_metric_only=True, verbose=False)])
    train_prob[va_idx] = model.predict_proba(X_va)[:, 1]
    oof_covered[va_idx] = True
    print(f"  fold {fold_idx} 완료, best_iter={model.best_iteration_}, 커버 {len(va_idx):,}행")

# 맨 앞(첫 fold의 train) 구간은 OOF가 없으므로 최종모델 예측으로 보완
uncovered = ~oof_covered
if uncovered.any():
    train_prob[uncovered] = final_model.predict_proba(X.loc[uncovered])[:, 1]
    print(f"  OOF 미커버 구간 {uncovered.sum():,}행 -> 최종모델 예측으로 보완")

print(f"Train 확률 계산 소요: {time.perf_counter()-t0:.1f}s")

# ============================================================
# STEP 3. Test 확률: 최종모델 (진짜 홀드아웃)
# ============================================================
print("\nSTEP 3. Test 확률 계산 (최종모델)")
X_test = test_df[FEATURE_COLUMNS]
test_prob = final_model.predict_proba(X_test)[:, 1]

# ============================================================
# STEP 4. 5단계 위험등급 부여
# ============================================================
print("\nSTEP 4. 5단계 위험등급 부여")
EMERGENCY_THRESHOLD = 0.9918  # 기존 설계값 유지(참고용, 재설계는 별도 작업 필요)
TIER5_BOUNDARIES = [0.0, 0.01, 0.50, THRESHOLD, EMERGENCY_THRESHOLD, 1.0 + 1e-9]
TIER5_LABELS = ["일반", "관찰", "관심", "위험", "긴급"]

train_df["model_prob"] = train_prob
test_df["model_prob"] = test_prob
train_df["_source"] = "train"
test_df["_source"] = "test"

merged = pd.concat([train_df, test_df], ignore_index=True)
merged["tier_5_final"] = pd.cut(
    merged["model_prob"], bins=TIER5_BOUNDARIES, labels=TIER5_LABELS,
    right=False, include_lowest=True,
).astype(str)

print(merged["tier_5_final"].value_counts().reindex(TIER5_LABELS))

# ============================================================
# STEP 5. K=7 군집 유형 부여 (기존 군집모델 그대로 재사용 - 누수 변수 미의존 확인됨)
# ============================================================
print("\nSTEP 5. K=7 군집 유형 부여")
cluster_pkg = joblib.load(CLUSTER_MODEL_PATH)
cluster_scaler = cluster_pkg["scaler"]
cluster_kmeans = cluster_pkg["kmeans"]
CLUSTER_FEATURES = cluster_pkg["cluster_features"]
CLUSTER_NAME_MAP = cluster_pkg["cluster_name_map"]
CLUSTER_RATIO_MEDIAN = cluster_pkg["ratio_median"]
NO_CLUSTER_LABEL = "해당없음(일반)"


def build_cluster_features(df: pd.DataFrame) -> pd.DataFrame:
    ratio = df["amt_to_prior_median_ratio"].fillna(CLUSTER_RATIO_MEDIAN)
    amt_to_prior_median_ratio_log = np.log1p(ratio)
    prior_1h_sum_amt = (df["rolling_sum_amt_1h"] - df["amt"]).clip(lower=0)
    prior_1h_sum_amt_log = np.log1p(prior_1h_sum_amt)
    recent_24h_high_amt_count_log = np.log1p(df["recent_24h_high_amt_count"])
    count_30min = df["count_30min"]
    radians = 2 * np.pi * df["trans_hour"].astype(float) / 24
    trans_hour_sin = np.sin(radians)
    trans_hour_cos = np.cos(radians)
    out = pd.DataFrame({
        "amt_to_prior_median_ratio_log": amt_to_prior_median_ratio_log,
        "prior_1h_sum_amt_log": prior_1h_sum_amt_log,
        "recent_24h_high_amt_count_log": recent_24h_high_amt_count_log,
        "count_30min": count_30min,
        "trans_hour_sin": trans_hour_sin,
        "trans_hour_cos": trans_hour_cos,
    })
    return out[CLUSTER_FEATURES]


cluster_X = build_cluster_features(merged)
scaled = cluster_scaler.transform(cluster_X)
cluster_ids = cluster_kmeans.predict(scaled)
merged["cluster_type"] = [
    NO_CLUSTER_LABEL if tier == "일반" else CLUSTER_NAME_MAP[int(cid)]
    for tier, cid in zip(merged["tier_5_final"], cluster_ids)
]
print(merged["cluster_type"].value_counts())

# ============================================================
# STEP 6. merchant 컬럼 결합 (원본 raw에서 trans_num 기준 병합)
# ============================================================
print("\nSTEP 6. merchant 컬럼 결합")
raw_train = pd.read_csv(RAW_TRAIN, usecols=["trans_num", "merchant"])
raw_test = pd.read_csv(RAW_TEST, usecols=["trans_num", "merchant"])
raw_merchant = pd.concat([raw_train, raw_test], ignore_index=True).drop_duplicates("trans_num")
merged = merged.merge(raw_merchant, on="trans_num", how="left")
assert merged["merchant"].isna().sum() == 0, "merchant 매칭 누락"

# ============================================================
# STEP 7. 권고대응 매핑 + 최종 저장
# ============================================================
print("\nSTEP 7. 권고대응 매핑 + 최종 저장")
RESPONSE_MAP = {
    "일반": "그대로 승인",
    "관찰": "K-means 유형분류 참고",
    "관심": "K-means 유형분류 → 대응 결정",
    "위험": "추가인증 / 보류",
    "긴급": "즉시차단",
}
merged["response_action"] = merged["tier_5_final"].map(RESPONSE_MAP)
assert merged["response_action"].isna().sum() == 0

export_cols = [
    "trans_date_trans_time", "category", "merchant", "amt", "trans_hour",
    "is_fraud", "model_prob", "tier_5_final", "cluster_type", "response_action",
]
export_df = merged[export_cols].sort_values("trans_date_trans_time").reset_index(drop=True)

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
export_df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

print(f"\n저장 완료: {len(export_df):,}행, {len(export_cols)}컬럼")
print("저장 경로:", OUT_PATH)
print("\n[등급 x 유형별 사기율 (일반 제외, 상위 10)]")
pivot = (
    export_df[export_df["tier_5_final"] != "일반"]
    .groupby(["tier_5_final", "cluster_type"])["is_fraud"]
    .mean().mul(100).round(2)
)
print(pivot.sort_values(ascending=False).head(10))
