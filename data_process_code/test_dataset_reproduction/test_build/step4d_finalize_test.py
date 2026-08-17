from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "test_build" / "test_step4c_all_features.pkl"
LOCKED_TARGET = ROOT / "test_build" / "test_target_locked.pkl"
TRAIN_FEATURES = ROOT / "upload" / "fraud_full_features(1).csv"
OUT = ROOT / "fraudTest_full_features.csv"


def main():
    features = pd.read_pickle(FEATURES)
    target = pd.read_pickle(LOCKED_TARGET)
    train_columns = pd.read_csv(TRAIN_FEATURES, nrows=0).columns.tolist()

    assert "is_fraud" not in features.columns
    assert target["trans_num"].is_unique
    assert features["trans_num"].is_unique
    assert features["trans_num"].equals(target["trans_num"])
    assert features["_row_order"].equals(target["_row_order"])

    # 모든 파생변수가 끝난 뒤에만 평가 정답을 원래 행 순서대로 복원한다.
    final = features.copy()
    final["is_fraud"] = target["is_fraud"].to_numpy(dtype="int64")
    final = final[train_columns].copy()

    int_columns = [
        "cc_num", "is_fraud", "recent_24h_high_amt_count",
        "category_recent_fraud_rate_missing", "count_30min", "Repeat3",
        "high_speed", "customer_transaction_count", "trans_hour", "age",
        "is_10x_prior_median", "has_prior_normal_transaction",
        "outside_trans_hours_80", "is_online", "risk_time_22_04",
        "merchant_change_count", "is_high_amt",
    ]
    float_columns = [
        "amt", "category_recent_fraud_rate", "speed_2", "customer_mean_amt",
        "customer_std_amt", "amt_ratio_to_mean", "amt_zscore_card",
        "prior_normal_median_amt", "amt_to_prior_median_ratio",
        "interact_repeat_category", "rolling_sum_amt_1h",
    ]
    final[int_columns] = final[int_columns].astype("int64")
    final[float_columns] = final[float_columns].astype("float64")

    assert final.shape == (555_719, 31)
    assert final.columns.tolist() == train_columns
    assert final.columns.tolist().index("is_fraud") == 5
    assert not np.isinf(final.select_dtypes(include="number").to_numpy()).any()
    expected_na = {"prior_normal_median_amt", "amt_to_prior_median_ratio"}
    actual_na = set(final.columns[final.isna().any()])
    assert actual_na == expected_na, (actual_na, expected_na)
    assert set(final["is_fraud"].unique()).issubset({0, 1})
    assert final["trans_date_trans_time"].equals(features["trans_date_trans_time"])

    final.to_csv(OUT, index=False, encoding="utf-8-sig")
    check = pd.read_csv(OUT)
    assert check.shape == final.shape
    assert check.columns.tolist() == train_columns
    assert check[int_columns].dtypes.eq("int64").all()
    assert check["is_fraud"].equals(target["is_fraud"].astype("int64"))

    print(f"output={OUT}")
    print(f"shape={check.shape}")
    print("column_order_matches_train=True")
    print("target_restored_after_features=True")
    print("infinity_count=0")
    print(f"nullable_columns={sorted(actual_na)}")
    print("roundtrip_csv_check=PASS")


if __name__ == "__main__":
    main()
