import io
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "final_lightgbm_combination3.joblib"
CLUSTER_MODEL_PATH = BASE_DIR / "model" / "lightGBM_combination3_K7_cluster_model.joblib"
SAMPLE_PATH = BASE_DIR / "data" / "demo_sample.csv"
FRONTEND_DIR = BASE_DIR.parent / "frontend"

NO_CLUSTER_LABEL = "해당없음(일반)"

app = FastAPI(title="FDS 위험거래 탐지 데모")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------
# 모델 로드 (서버 시작 시 1회)
# ------------------------------------------------------------
package = joblib.load(MODEL_PATH)
model = package["model"]
FEATURES = package["features"]
CATEGORICAL_FEATURES = package["categorical_features"]
CATEGORY_LEVELS = package["category_levels"]
THRESHOLD = package["threshold"]

# 5단계 위험등급 경계값 (Tableau 대시보드 merge_k7_from_minjeong.py와 동일 기준)
EMERGENCY_THRESHOLD = 0.9918
TIER_BOUNDARIES = [0.0, 0.01, 0.50, THRESHOLD, EMERGENCY_THRESHOLD, 1.0 + 1e-9]
TIER_LABELS = ["일반", "관찰", "관심", "위험", "긴급"]
TIER_COLOR = {
    "일반": "#4CAF50",
    "관찰": "#8BC34A",
    "관심": "#FFC107",
    "위험": "#FF7043",
    "긴급": "#E53935",
}

# ------------------------------------------------------------
# K=7 군집 모델 (위험등급 × 유형 28개 분류용)
# ------------------------------------------------------------
cluster_package = joblib.load(CLUSTER_MODEL_PATH)
cluster_scaler = cluster_package["scaler"]
cluster_kmeans = cluster_package["kmeans"]
CLUSTER_FEATURES = cluster_package["cluster_features"]
CLUSTER_NAME_MAP = cluster_package["cluster_name_map"]
CLUSTER_RATIO_MEDIAN = cluster_package["ratio_median"]


def assign_tier(prob: float) -> str:
    for i in range(len(TIER_BOUNDARIES) - 1):
        if TIER_BOUNDARIES[i] <= prob < TIER_BOUNDARIES[i + 1]:
            return TIER_LABELS[i]
    return TIER_LABELS[-1]


def build_cluster_features(df: pd.DataFrame) -> pd.DataFrame:
    """cluster_package['preprocessing']에 적힌 전처리 그대로 재현."""
    ratio = df["amt_to_prior_median_ratio"].fillna(CLUSTER_RATIO_MEDIAN)
    amt_to_prior_median_ratio_log = np.log1p(ratio)

    prior_1h_sum_amt = (df["rolling_sum_amt_1h"] - df["amt"]).clip(lower=0)
    prior_1h_sum_amt_log = np.log1p(prior_1h_sum_amt)

    recent_24h_high_amt_count_log = np.log1p(df["recent_24h_high_amt_count"])

    count_30min = df["count_30min"]

    radians = 2 * np.pi * df["trans_hour"].astype(float) / 24
    trans_hour_sin = np.sin(radians)
    trans_hour_cos = np.cos(radians)

    out = pd.DataFrame(
        {
            "amt_to_prior_median_ratio_log": amt_to_prior_median_ratio_log,
            "prior_1h_sum_amt_log": prior_1h_sum_amt_log,
            "recent_24h_high_amt_count_log": recent_24h_high_amt_count_log,
            "count_30min": count_30min,
            "trans_hour_sin": trans_hour_sin,
            "trans_hour_cos": trans_hour_cos,
        }
    )
    return out[CLUSTER_FEATURES]


def assign_cluster_types(df: pd.DataFrame, tiers: list[str]) -> list[str]:
    """일반 등급은 '해당없음(일반)', 그 외는 K=7 군집 유형명."""
    cluster_X = build_cluster_features(df)
    scaled = cluster_scaler.transform(cluster_X)
    cluster_ids = cluster_kmeans.predict(scaled)

    cluster_types = []
    for tier, cid in zip(tiers, cluster_ids):
        if tier == "일반":
            cluster_types.append(NO_CLUSTER_LABEL)
        else:
            cluster_types.append(CLUSTER_NAME_MAP[int(cid)])
    return cluster_types


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"필요한 컬럼이 없습니다: {missing}",
        )

    X = df[FEATURES].copy()
    for col in CATEGORICAL_FEATURES:
        X[col] = pd.Categorical(
            X[col].astype("string").fillna("missing"),
            categories=CATEGORY_LEVELS[col],
        )
    return X


def run_predictions(df: pd.DataFrame) -> dict:
    X = preprocess(df)
    probs = model.predict_proba(X)[:, 1]
    tiers = [assign_tier(float(p)) for p in probs]
    cluster_types = assign_cluster_types(df, tiers)

    results = []
    for i, prob in enumerate(probs):
        row = df.iloc[i]
        tier = tiers[i]
        cluster_type = cluster_types[i]

        actual = None
        if "is_fraud" in df.columns and pd.notna(row.get("is_fraud")):
            actual = "이상거래" if int(row["is_fraud"]) == 1 else "정상거래"

        results.append(
            {
                "row": i + 1,
                "trans_date_trans_time": (
                    str(row.get("trans_date_trans_time"))
                    if "trans_date_trans_time" in df.columns
                    else None
                ),
                "category": row.get("category"),
                "amt": float(row["amt"]) if pd.notna(row.get("amt")) else None,
                "trans_hour": (
                    int(row["trans_hour"]) if pd.notna(row.get("trans_hour")) else None
                ),
                "probability": round(float(prob) * 100, 2),
                "prediction": "이상거래" if prob >= THRESHOLD else "정상거래",
                "tier": tier,
                "tier_color": TIER_COLOR[tier],
                "cluster_type": cluster_type,
                "actual": actual,
            }
        )

    return {
        "count": len(results),
        "threshold": round(THRESHOLD * 100, 2),
        "results": results,
    }


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="CSV 파일만 업로드할 수 있습니다.")

    raw = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception:
        raise HTTPException(status_code=400, detail="CSV 파일을 읽을 수 없습니다.")

    if len(df) == 0:
        raise HTTPException(status_code=400, detail="빈 파일입니다.")

    return run_predictions(df)


@app.post("/api/predict-sample")
async def predict_sample():
    df = pd.read_csv(SAMPLE_PATH)
    return run_predictions(df)


app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
