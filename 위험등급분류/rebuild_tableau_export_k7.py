# -*- coding: utf-8 -*-
"""
tableau_export.csv 재생성 (K=4 -> K=7 군집 반영)
- 다영님 OOF 재현 파이프라인(위험등급분류_다영.ipynb 1~13번 셀)에 원본 인덱스 추적을 추가
- 민정님의 K=7 최종 군집(lightGBM_조합3_OOF_군집분석_민정.ipynb 132~153번 셀) 로직을 동일하게 적용
"""
import json
import numpy as np
import pandas as pd

from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

RANDOM_STATE = 42

# ============================================================
# 1. 데이터 로드 (다영님 셀 3와 동일)
# ============================================================
print("=" * 70)
print("STEP 1. 데이터 로드")
print("=" * 70)

DATA_PATH = "../data/fraud_full_features.csv"
df = pd.read_csv(DATA_PATH)
df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
df = df.sort_values("trans_date_trans_time").reset_index(drop=True)

with open("../Modeling/final_lightgbm_combination3_config.json", encoding="utf-8") as f:
    FINAL_CONFIG = json.load(f)

COMBO3_FEATURES = FINAL_CONFIG["features"]
HP = FINAL_CONFIG["hyperparameters"]
OFFICIAL_THRESHOLD = FINAL_CONFIG["threshold"]

print("데이터 크기:", df.shape)
print("조합3 변수:", COMBO3_FEATURES)
print("팀 공식 임계값:", OFFICIAL_THRESHOLD)

# ============================================================
# 2. OOF 예측확률 재현 (다영님 셀 5~6과 동일 + 원본 인덱스 추적 추가)
# ============================================================
print("\n" + "=" * 70)
print("STEP 2. OOF 3-Fold 재현 (원본 인덱스 포함)")
print("=" * 70)

boundaries = np.linspace(0, len(df), 7, dtype=int)
time_folds = []
for fold_number in range(1, 4):
    train_end = boundaries[fold_number + 2]
    val_start = train_end
    val_end = boundaries[fold_number + 3]
    time_folds.append({
        "fold": fold_number,
        "train": df.iloc[0:train_end].copy(),
        "val": df.iloc[val_start:val_end].copy(),
    })
    print(f"Fold {fold_number} | Train {train_end:,}건 | Val {val_end - val_start:,}건")


def prepare_lgbm_fold(train_fold, val_fold, features):
    X_train = train_fold[features].copy().reset_index(drop=True)
    X_val = val_fold[features].copy().reset_index(drop=True)
    y_train = train_fold["is_fraud"].astype(np.int8).reset_index(drop=True)
    y_val = val_fold["is_fraud"].astype(np.int8).reset_index(drop=True)

    train_cats = X_train["category"].astype("string").fillna("missing").unique().tolist()
    X_train["category"] = pd.Categorical(X_train["category"].astype("string").fillna("missing"), categories=train_cats)
    X_val["category"] = pd.Categorical(X_val["category"].astype("string").fillna("missing"), categories=train_cats)
    return X_train, X_val, y_train, y_val


oof_true, oof_proba, oof_time, oof_index = [], [], [], []

for tf in time_folds:
    X_train, X_val, y_train, y_val = prepare_lgbm_fold(tf["train"], tf["val"], COMBO3_FEATURES)
    spw = (y_train == 0).sum() / (y_train == 1).sum()

    model = LGBMClassifier(
        objective="binary", n_estimators=3000, learning_rate=HP["learning_rate"],
        num_leaves=HP["num_leaves"], max_depth=HP["max_depth"], min_child_samples=HP["min_child_samples"],
        subsample=HP["subsample"], subsample_freq=HP["subsample_freq"], colsample_bytree=HP["colsample_bytree"],
        reg_alpha=HP["reg_alpha"], reg_lambda=HP["reg_lambda"], min_split_gain=HP["min_split_gain"],
        max_bin=HP["max_bin"], scale_pos_weight=spw, random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1,
    )
    model.fit(
        X_train, y_train, eval_set=[(X_val, y_val)], eval_metric="average_precision",
        categorical_feature=["category"],
        callbacks=[early_stopping(stopping_rounds=100, first_metric_only=True, verbose=False), log_evaluation(period=0)],
    )
    proba_val = model.predict_proba(X_val, num_iteration=model.best_iteration_)[:, 1]
    fold_pr_auc = average_precision_score(y_val, proba_val)
    print(f"Fold {tf['fold']} | best_iter={model.best_iteration_} | PR-AUC={fold_pr_auc:.6f}")

    oof_true.extend(y_val.to_numpy())
    oof_proba.extend(proba_val)
    oof_time.extend(tf["val"]["trans_date_trans_time"].to_numpy())
    oof_index.extend(tf["val"].index.to_numpy())  # <-- 다영님 원래 코드에 없던 추가 부분: 원본 df 인덱스 보존

oof_true = np.array(oof_true)
oof_proba = np.array(oof_proba)
oof_time = np.array(oof_time)
oof_index = np.array(oof_index)

oof_pr_auc = average_precision_score(oof_true, oof_proba)
print(f"\nOOF 통합 PR-AUC = {oof_pr_auc:.6f} (참고: config.json 저장값 {FINAL_CONFIG['oof_results']['pr_auc']:.6f})")
print(f"OOF 전체 건수 = {len(oof_true):,}, 사기 건수 = {int(oof_true.sum()):,}")

# ============================================================
# 3. 5단계 위험등급 부여 (팀 표준 경계값)
# ============================================================
print("\n" + "=" * 70)
print("STEP 3. 5단계 위험등급 부여")
print("=" * 70)

EMERGENCY_THRESHOLD = 0.9918
TIER5_BOUNDARIES = [0.0, 0.01, 0.50, OFFICIAL_THRESHOLD, EMERGENCY_THRESHOLD, 1.0 + 1e-9]
TIER5_LABELS = ["일반", "관찰", "관심", "위험", "긴급"]

oof_df = pd.DataFrame({
    "original_index": oof_index,
    "y_true": oof_true,
    "y_prob": oof_proba,
    "trans_date_trans_time": oof_time,
})

oof_df["tier_5_final"] = pd.cut(
    oof_df["y_prob"], bins=TIER5_BOUNDARIES, labels=TIER5_LABELS,
    right=False, include_lowest=True,
).astype(str)

print(oof_df["tier_5_final"].value_counts().reindex(TIER5_LABELS))

# ============================================================
# 4. K=7 군집 (민정님 132~153번 셀과 동일 로직)
# ============================================================
print("\n" + "=" * 70)
print("STEP 4. K=7 군집분석 (일반 등급 제외 population)")
print("=" * 70)

CLUSTER_CANDIDATES = [
    "amt", "trans_hour", "recent_24h_high_amt_count",
    "amt_to_prior_median_ratio", "rolling_sum_amt_1h",
    "amt_zscore_card", "count_30min",
]
PROFILE_FEATURES = ["category", "high_speed", "age", "prior_normal_median_amt"]

pop_incl = oof_df[oof_df["tier_5_final"] != "일반"].copy().reset_index(drop=True)
print(f"군집 대상(일반 제외): {len(pop_incl):,}건")

raw_features = df.iloc[pop_incl["original_index"].to_numpy()][CLUSTER_CANDIDATES + PROFILE_FEATURES].reset_index(drop=True)
cluster_base_df = pd.concat([pop_incl.reset_index(drop=True), raw_features], axis=1)

# --- 4-1. 결측/파생 변수 (민정님 STEP 7과 동일) ---
cluster_base_df["prior_history_missing"] = cluster_base_df["amt_to_prior_median_ratio"].isna().astype(np.int8)
RATIO_MEDIAN = cluster_base_df["amt_to_prior_median_ratio"].median()
cluster_base_df["amt_to_prior_median_ratio_imputed"] = cluster_base_df["amt_to_prior_median_ratio"].fillna(RATIO_MEDIAN)
cluster_base_df["amt_to_prior_median_ratio_log"] = np.log1p(cluster_base_df["amt_to_prior_median_ratio_imputed"])

cluster_base_df["prior_1h_sum_amt"] = (
    cluster_base_df["rolling_sum_amt_1h"] - cluster_base_df["amt"]
).clip(lower=0)
cluster_base_df["prior_1h_sum_amt_log"] = np.log1p(cluster_base_df["prior_1h_sum_amt"])

cluster_base_df["recent_24h_high_amt_count_log"] = np.log1p(cluster_base_df["recent_24h_high_amt_count"])

cluster_base_df["trans_hour_sin"] = np.sin(2 * np.pi * cluster_base_df["trans_hour"] / 24)
cluster_base_df["trans_hour_cos"] = np.cos(2 * np.pi * cluster_base_df["trans_hour"] / 24)

FINAL_CLUSTER_FEATURES = [
    "amt_to_prior_median_ratio_log", "prior_1h_sum_amt_log",
    "recent_24h_high_amt_count_log", "count_30min",
    "trans_hour_sin", "trans_hour_cos",
]

cluster_feature_df = cluster_base_df[FINAL_CLUSTER_FEATURES].copy()
missing_count = int(cluster_feature_df.isna().sum().sum())
inf_count = int(np.isinf(cluster_feature_df.to_numpy()).sum())
print("결측:", missing_count, "| 무한대:", inf_count)
if missing_count > 0 or inf_count > 0:
    raise ValueError("군집 입력값에 결측 또는 무한대가 있습니다.")

# --- 4-2. 표준화 + KMeans(K=7, seed=42) ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(cluster_feature_df)

kmeans = KMeans(n_clusters=7, init="k-means++", n_init=20, max_iter=500, random_state=RANDOM_STATE)
cluster_base_df["cluster_id"] = kmeans.fit_predict(X_scaled)

CLUSTER_NAME_MAP = {
    0: "주간 소액·평소거래형",
    1: "심야 고액·단시간 누적형",
    2: "심야 초단기 다회·고속이동형",
    3: "단발성 초고액 이탈형",
    4: "중간금액·완만 이탈형",
    5: "심야 소액·평소거래형",
    6: "심야 고액 반복형",
}
cluster_base_df["cluster_type"] = cluster_base_df["cluster_id"].map(CLUSTER_NAME_MAP)

if cluster_base_df["cluster_type"].isna().any():
    raise ValueError("유형명이 연결되지 않은 군집이 있습니다.")

print("\n[K=7 군집 분포]")
counts = cluster_base_df.groupby(["cluster_id", "cluster_type"]).size().reset_index(name="건수")
counts["비율"] = (counts["건수"] / len(cluster_base_df) * 100).round(2)
print(counts.to_string(index=False))

# 실제 이상률(사기율) 확인용 (참고 출력)
fraud_check = cluster_base_df.groupby("cluster_type")["y_true"].mean().sort_values(ascending=False)
print("\n[군집별 실제 사기율]")
print((fraud_check * 100).round(2).astype(str) + "%")

# ============================================================
# 5. 원본 oof_df에 cluster_type 병합
# ============================================================
print("\n" + "=" * 70)
print("STEP 5. cluster_type 전체 데이터에 병합")
print("=" * 70)

oof_df = oof_df.merge(
    cluster_base_df[["original_index", "cluster_type"]],
    on="original_index", how="left",
)
oof_df["cluster_type"] = oof_df["cluster_type"].fillna("해당없음(일반)")

print(oof_df["cluster_type"].value_counts())

# ============================================================
# 6. 권고대응 매핑 + 최종 export
# ============================================================
print("\n" + "=" * 70)
print("STEP 6. tableau_export.csv 생성")
print("=" * 70)

RESPONSE_MAP = {
    "일반": "그대로 승인",
    "관찰": "K-means 유형분류 참고",
    "관심": "K-means 유형분류 → 대응 결정",
    "위험": "추가인증 / 보류",
    "긴급": "즉시차단",
}
oof_df["response_action"] = oof_df["tier_5_final"].map(RESPONSE_MAP)
assert oof_df["response_action"].isna().sum() == 0, "권고대응 매핑 누락"
assert oof_df["tier_5_final"].isna().sum() == 0, "등급 매핑 누락"

# 원본 df에서 필요한 나머지 컬럼(merchant, category, amt, trans_hour) 붙이기
extra_cols = df.iloc[oof_df["original_index"].to_numpy()][["merchant", "category", "amt", "trans_hour"]].reset_index(drop=True)
export_df = pd.concat([oof_df.reset_index(drop=True), extra_cols], axis=1)

export_df = export_df.rename(columns={"y_true": "is_fraud", "y_prob": "model_prob"})

export_cols = [
    "trans_date_trans_time", "category", "merchant", "amt", "trans_hour",
    "is_fraud", "model_prob", "tier_5_final", "cluster_type", "response_action",
]
export_df = export_df[export_cols]

OUT_PATH = "../tableau/tableau_export.csv"
export_df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

print(f"저장 완료: {len(export_df):,}행, {len(export_cols)}컬럼")
print("tier_5_final 분포:")
print(export_df["tier_5_final"].value_counts())
print("\ncluster_type 분포:")
print(export_df["cluster_type"].value_counts())
print("\n저장 경로:", OUT_PATH)
