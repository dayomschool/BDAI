from pathlib import Path
import time

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import average_precision_score

train_path = Path(r"C:\Users\user\Desktop\bdai부캠\BDAI\data\train_11features.csv")
df = pd.read_csv(train_path, parse_dates=["trans_date_trans_time"])
df = df.sort_values("trans_date_trans_time").reset_index(drop=True)

FEATURE_COLUMNS = [
    "category", "amt", "trans_hour", "age",
    "recent_24h_high_amt_count", "amt_to_prior_median_ratio",
    "rolling_sum_amt_1h", "amt_zscore_card", "prior_normal_median_amt",
    "count_30min", "high_speed",
]
TARGET_COLUMN = "is_fraud"
df["category"] = df["category"].astype("category")

X = df[FEATURE_COLUMNS]
y = df[TARGET_COLUMN].astype("int8")

CANDIDATES = {
    "A_cv기본값": dict(num_leaves=31, max_depth=-1, min_child_samples=50,
                      learning_rate=0.05, reg_alpha=0.0, reg_lambda=0.0),
    "B_기존조합3설정": dict(num_leaves=23, max_depth=5, min_child_samples=150,
                        learning_rate=0.03, reg_alpha=0.1, reg_lambda=1.0),
    "C_A에규제추가": dict(num_leaves=31, max_depth=-1, min_child_samples=50,
                       learning_rate=0.05, reg_alpha=0.1, reg_lambda=1.0),
    "D_단순화": dict(num_leaves=15, max_depth=-1, min_child_samples=200,
                    learning_rate=0.05, reg_alpha=0.1, reg_lambda=1.0),
}

BASE = dict(
    objective="binary", boosting_type="gbdt", n_estimators=1000,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
    random_state=42, n_jobs=-1, verbosity=-1,
)

tscv = TimeSeriesSplit(n_splits=3)
splits = list(tscv.split(X))

results = []
for name, params in CANDIDATES.items():
    t0 = time.perf_counter()
    fold_scores = []
    for fold_idx, (train_idx, valid_idx) in enumerate(splits, start=1):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
        neg = int((y_train == 0).sum())
        pos = int((y_train == 1).sum())
        scale_pos_weight = neg / pos if pos > 0 else 1.0

        model = lgb.LGBMClassifier(**BASE, **params, scale_pos_weight=scale_pos_weight)
        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric="average_precision",
            callbacks=[lgb.early_stopping(stopping_rounds=50, first_metric_only=True, verbose=False)],
        )
        prob = model.predict_proba(X_valid)[:, 1]
        pr_auc = average_precision_score(y_valid, prob)
        fold_scores.append(pr_auc)

    avg = float(np.mean(fold_scores))
    std = float(np.std(fold_scores))
    elapsed = time.perf_counter() - t0
    print(f"{name}: folds={['%.4f' % s for s in fold_scores]} avg={avg:.4f} std={std:.4f} ({elapsed:.1f}s)")
    results.append(dict(name=name, avg_pr_auc=avg, std=std, folds=fold_scores, params=params))

best = max(results, key=lambda r: r["avg_pr_auc"])
print("\n=== 최적 조합 ===")
print(best["name"], best["params"], f"avg_pr_auc={best['avg_pr_auc']:.4f}")
