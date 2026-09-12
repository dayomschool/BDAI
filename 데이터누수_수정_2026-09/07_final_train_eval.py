from pathlib import Path
import time

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    precision_score, recall_score, f1_score,
    confusion_matrix, precision_recall_curve,
)

FEATURE_COLUMNS = [
    "category", "amt", "trans_hour", "age",
    "recent_24h_high_amt_count", "amt_to_prior_median_ratio",
    "rolling_sum_amt_1h", "amt_zscore_card", "prior_normal_median_amt",
    "count_30min", "high_speed",
]
TARGET_COLUMN = "is_fraud"

BEST_PARAMS = dict(
    num_leaves=31, max_depth=-1, min_child_samples=50,
    learning_rate=0.05, reg_alpha=0.1, reg_lambda=1.0,
)
BASE = dict(
    objective="binary", boosting_type="gbdt", n_estimators=1000,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
    random_state=42, n_jobs=-1, verbosity=-1,
)

train_path = Path(r"C:\Users\user\Desktop\bdai부캠\BDAI\data\train_11features.csv")
test_path = Path(r"C:\Users\user\Desktop\bdai부캠\BDAI\data\test_11features.csv")

train_df = pd.read_csv(train_path, parse_dates=["trans_date_trans_time"]).sort_values("trans_date_trans_time").reset_index(drop=True)
test_df = pd.read_csv(test_path, parse_dates=["trans_date_trans_time"]).sort_values("trans_date_trans_time").reset_index(drop=True)
train_df["category"] = train_df["category"].astype("category")
test_df["category"] = test_df["category"].astype(pd.CategoricalDtype(categories=train_df["category"].cat.categories))

X = train_df[FEATURE_COLUMNS]
y = train_df[TARGET_COLUMN].astype("int8")

# ---------- 1. OOF 예측으로 임계값 선정 ----------
print("=== 1. OOF 예측 수집 (3-Fold) ===")
tscv = TimeSeriesSplit(n_splits=3)
oof_idx = []
oof_prob = []
oof_y = []
t0 = time.perf_counter()
for fold_idx, (tr_idx, va_idx) in enumerate(tscv.split(X), start=1):
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
    neg, pos = int((y_tr == 0).sum()), int((y_tr == 1).sum())
    spw = neg / pos if pos > 0 else 1.0
    model = lgb.LGBMClassifier(**BASE, **BEST_PARAMS, scale_pos_weight=spw)
    model.fit(
        X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="average_precision",
        callbacks=[lgb.early_stopping(stopping_rounds=50, first_metric_only=True, verbose=False)],
    )
    prob = model.predict_proba(X_va)[:, 1]
    oof_idx.append(va_idx)
    oof_prob.append(prob)
    oof_y.append(y_va.to_numpy())
    print(f"  fold {fold_idx} done, best_iter={model.best_iteration_}")

oof_prob = np.concatenate(oof_prob)
oof_y = np.concatenate(oof_y)
print(f"OOF 샘플 수: {len(oof_y):,} (전체 train의 2/3, fold1~3 검증구간)")
print(f"OOF PR-AUC: {average_precision_score(oof_y, oof_prob):.6f}")

precisions, recalls, thresholds = precision_recall_curve(oof_y, oof_prob)
denom = precisions[:-1] + recalls[:-1]
f1s = np.divide(2 * precisions[:-1] * recalls[:-1], denom, out=np.zeros_like(denom), where=denom > 0)
best_idx = int(np.argmax(f1s))
FINAL_THRESHOLD = float(thresholds[best_idx])
print(f"[OOF 기준 F1-max 임계값]: {FINAL_THRESHOLD:.6f}  (F1={f1s[best_idx]:.4f}, P={precisions[best_idx]:.4f}, R={recalls[best_idx]:.4f})")
print(f"OOF 수집 소요: {time.perf_counter()-t0:.1f}s\n")

# ---------- 2. 최종 배포용 모델: 전체 train으로 재학습 (마지막 5%를 early stopping용으로) ----------
print("=== 2. 최종 모델 학습 (전체 Train) ===")
t0 = time.perf_counter()
cut = int(len(train_df) * 0.95)
X_fit, X_es = X.iloc[:cut], X.iloc[cut:]
y_fit, y_es = y.iloc[:cut], y.iloc[cut:]
neg, pos = int((y_fit == 0).sum()), int((y_fit == 1).sum())
spw_final = neg / pos if pos > 0 else 1.0

final_model = lgb.LGBMClassifier(**BASE, **BEST_PARAMS, scale_pos_weight=spw_final)
final_model.fit(
    X_fit, y_fit, eval_set=[(X_es, y_es)], eval_metric="average_precision",
    callbacks=[lgb.early_stopping(stopping_rounds=50, first_metric_only=True, verbose=False)],
)
print(f"best_iteration: {final_model.best_iteration_}")
print(f"최종 학습 소요: {time.perf_counter()-t0:.1f}s\n")

# ---------- 3. Test셋(완전 홀드아웃) 평가 ----------
print("=== 3. Test셋 평가 (진짜 홀드아웃, 한 번도 학습에 안 씀) ===")
X_test = test_df[FEATURE_COLUMNS]
y_test = test_df[TARGET_COLUMN].astype("int8")
test_prob = final_model.predict_proba(X_test)[:, 1]

test_pr_auc = average_precision_score(y_test, test_prob)
test_roc_auc = roc_auc_score(y_test, test_prob)
test_pred = (test_prob >= FINAL_THRESHOLD).astype("int8")
test_precision = precision_score(y_test, test_pred, zero_division=0)
test_recall = recall_score(y_test, test_pred, zero_division=0)
test_f1 = f1_score(y_test, test_pred, zero_division=0)
cm = confusion_matrix(y_test, test_pred)

print(f"Test 건수: {len(test_df):,}, 사기 {int(y_test.sum())}건 ({y_test.mean()*100:.4f}%)")
print(f"Test PR-AUC : {test_pr_auc:.6f}")
print(f"Test ROC-AUC: {test_roc_auc:.6f}")
print(f"Test Precision: {test_precision:.6f} (threshold={FINAL_THRESHOLD:.6f})")
print(f"Test Recall   : {test_recall:.6f}")
print(f"Test F1       : {test_f1:.6f}")
print("Confusion Matrix (Test):")
print(cm)

print("\n=== 4. 과적합 여부 판단 ===")
oof_pr_auc = average_precision_score(oof_y, oof_prob)
gap = oof_pr_auc - test_pr_auc
print(f"OOF(Train) PR-AUC : {oof_pr_auc:.6f}")
print(f"Test PR-AUC       : {test_pr_auc:.6f}")
print(f"차이(gap)         : {gap:+.6f}")
if abs(gap) < 0.02:
    print("=> 차이가 작음: 과적합 징후 없음")
else:
    print("=> 차이가 큼: 과적합 의심, 추가 확인 필요")

import joblib
OUT_MODEL = Path(r"C:\Users\user\Desktop\bdai부캠\BDAI\data\final_lightgbm_leakfree.joblib")
joblib.dump(dict(
    model=final_model,
    features=FEATURE_COLUMNS,
    threshold=FINAL_THRESHOLD,
    hyperparameters=BEST_PARAMS,
    oof_pr_auc=oof_pr_auc,
    test_pr_auc=test_pr_auc,
    test_precision=test_precision,
    test_recall=test_recall,
    test_f1=test_f1,
), OUT_MODEL)
print(f"\n모델 저장 완료: {OUT_MODEL}")
