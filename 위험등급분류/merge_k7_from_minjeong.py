# -*- coding: utf-8 -*-
"""
민정님이 준 2개 파일(OOF 전체 예측 + top4 K=7 군집)을 병합해서
tableau_export.csv를 재생성. LightGBM 재학습 불필요 (순수 데이터 병합).
"""
import pandas as pd
import numpy as np

# ============================================================
# 1. 원본 raw 데이터 (표시용 컬럼: merchant/category/amt/trans_hour/시간)
# ============================================================
print("STEP 1. 원본 데이터 로드")
df = pd.read_csv(r"C:\Users\user\Desktop\bdai부캠\BDAI\data\fraud_full_features.csv")
df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
df = df.sort_values("trans_date_trans_time").reset_index(drop=True)
print("행수:", len(df))

# ============================================================
# 2. 민정님 파일 로드
# ============================================================
print("\nSTEP 2. 민정님 결과 파일 로드")
PRED_PATH = r"C:\Users\user\Documents\카카오톡 받은 파일\lightGBM_combination3_OOF_predictions.csv"
CLUSTER_PATH = r"C:\Users\user\Documents\카카오톡 받은 파일\lightGBM_combination3_OOF_top4_K7_clustered.csv"

pred_df = pd.read_csv(PRED_PATH)
cluster_df = pd.read_csv(CLUSTER_PATH)
print("예측 전체:", len(pred_df), "| 군집(top4):", len(cluster_df))

# ------------------------------------------------------------
# 무결성 확인: original_index 기준 is_fraud/y_true가 df와 일치하는지
# ------------------------------------------------------------
check_sample = pred_df.sample(min(2000, len(pred_df)), random_state=1)
df_fraud_at_index = df.iloc[check_sample["original_index"].to_numpy()]["is_fraud"].to_numpy()
mismatch = int((df_fraud_at_index != check_sample["y_true"].to_numpy()).sum())
print(f"무결성 확인(샘플 {len(check_sample)}건 중 is_fraud 불일치): {mismatch}건")
if mismatch > 0:
    raise ValueError("original_index 정렬이 df와 어긋납니다. 확인이 필요합니다.")

# ============================================================
# 3. 5단계 위험등급 부여
# ============================================================
print("\nSTEP 3. 5단계 위험등급 부여")
OFFICIAL_THRESHOLD = 0.9289392017223173
EMERGENCY_THRESHOLD = 0.9918
TIER5_BOUNDARIES = [0.0, 0.01, 0.50, OFFICIAL_THRESHOLD, EMERGENCY_THRESHOLD, 1.0 + 1e-9]
TIER5_LABELS = ["일반", "관찰", "관심", "위험", "긴급"]

pred_df["tier_5_final"] = pd.cut(
    pred_df["y_probability"], bins=TIER5_BOUNDARIES, labels=TIER5_LABELS,
    right=False, include_lowest=True,
).astype(str)

print(pred_df["tier_5_final"].value_counts().reindex(TIER5_LABELS))

# ============================================================
# 4. cluster_type 병합 (일반 등급 및 top4 밖 행은 "해당없음(일반)")
# ============================================================
print("\nSTEP 4. cluster_type 병합")
merged = pred_df.merge(
    cluster_df[["original_index", "cluster_type"]],
    on="original_index", how="left",
)
merged["cluster_type"] = merged["cluster_type"].fillna("해당없음(일반)")

print(merged["cluster_type"].value_counts())

# 교차 검증: tier_5_final != 일반인 행 수가 cluster_df 행 수와 맞는지
non_general = (merged["tier_5_final"] != "일반").sum()
print(f"\n일반 제외 건수: {non_general:,} (군집 파일 행수: {len(cluster_df):,})")

# ============================================================
# 5. 원본 df에서 표시용 컬럼 붙이기
# ============================================================
print("\nSTEP 5. 표시용 컬럼(merchant/category/amt/trans_hour/시간) 결합")
extra_cols = df.iloc[merged["original_index"].to_numpy()][
    ["trans_date_trans_time", "merchant", "category", "amt", "trans_hour"]
].reset_index(drop=True)

export_df = pd.concat([merged.reset_index(drop=True), extra_cols], axis=1)
export_df = export_df.rename(columns={"y_true": "is_fraud", "y_probability": "model_prob"})

# ============================================================
# 6. 권고대응 매핑
# ============================================================
print("\nSTEP 6. 권고대응 매핑 + 최종 저장")
RESPONSE_MAP = {
    "일반": "그대로 승인",
    "관찰": "K-means 유형분류 참고",
    "관심": "K-means 유형분류 → 대응 결정",
    "위험": "추가인증 / 보류",
    "긴급": "즉시차단",
}
export_df["response_action"] = export_df["tier_5_final"].map(RESPONSE_MAP)
assert export_df["response_action"].isna().sum() == 0, "권고대응 매핑 누락"
assert export_df["tier_5_final"].isna().sum() == 0, "등급 매핑 누락"
assert export_df["cluster_type"].isna().sum() == 0, "군집 매핑 누락"

export_cols = [
    "trans_date_trans_time", "category", "merchant", "amt", "trans_hour",
    "is_fraud", "model_prob", "tier_5_final", "cluster_type", "response_action",
]
export_df = export_df[export_cols]

OUT_PATH = r"C:\Users\user\Desktop\bdai부캠\BDAI\tableau\tableau_export.csv"
export_df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

print(f"\n저장 완료: {len(export_df):,}행, {len(export_cols)}컬럼")
print("\n[tier_5_final 분포]")
print(export_df["tier_5_final"].value_counts())
print("\n[cluster_type 분포]")
print(export_df["cluster_type"].value_counts())
print("\n[등급 x 유형별 사기율 (일반 제외)]")
pivot = (
    export_df[export_df["tier_5_final"] != "일반"]
    .groupby(["tier_5_final", "cluster_type"])["is_fraud"]
    .mean()
    .mul(100)
    .round(2)
    .unstack()
)
print(pivot.to_string())
print("\n저장 경로:", OUT_PATH)
