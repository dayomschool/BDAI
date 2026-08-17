from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "upload" / "fraudTest(1).csv"
OUT_DIR = ROOT / "test_build"
FEATURE_OUT = OUT_DIR / "test_step4a_features.pkl"
TARGET_OUT = OUT_DIR / "test_target_locked.pkl"

ONLINE_CATEGORIES = {"shopping_net", "misc_net", "grocery_net"}
RISK_HOURS = {22, 23, 0, 1, 2, 3}


def exact_age(transaction_date: pd.Series, birth_date: pd.Series) -> pd.Series:
    years = transaction_date.dt.year - birth_date.dt.year
    birthday_not_reached = (
        (transaction_date.dt.month < birth_date.dt.month)
        | (
            (transaction_date.dt.month == birth_date.dt.month)
            & (transaction_date.dt.day < birth_date.dt.day)
        )
    )
    return (years - birthday_not_reached.astype("int64")).astype("int64")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SOURCE)

    required = {
        "is_fraud",
        "trans_num",
        "trans_date_trans_time",
        "dob",
        "category",
        "amt",
    }
    missing = required.difference(df.columns)
    assert not missing, f"필수 열 누락: {sorted(missing)}"
    assert df["trans_num"].notna().all()
    assert df["trans_num"].is_unique

    df.insert(0, "_row_order", np.arange(len(df), dtype="int64"))

    # 정답은 계산 테이블에서 즉시 분리한다. 아래 파생변수 계산에는 접근하지 않는다.
    locked_target = df[["_row_order", "trans_num", "is_fraud"]].copy()
    features = df.drop(columns="is_fraud").copy()

    transaction_dt = pd.to_datetime(features["trans_date_trans_time"], errors="raise")
    birth_dt = pd.to_datetime(features["dob"], errors="raise")

    features["trans_hour"] = transaction_dt.dt.hour.astype("int64")
    features["age"] = exact_age(transaction_dt, birth_dt)
    features["is_online"] = features["category"].isin(ONLINE_CATEGORIES).astype("int64")
    features["risk_time_22_04"] = features["trans_hour"].isin(RISK_HOURS).astype("int64")
    features["is_high_amt"] = features["amt"].ge(500).astype("int64")

    generated = ["trans_hour", "age", "is_online", "risk_time_22_04", "is_high_amt"]
    assert len(features) == len(df) == 555_719
    assert "is_fraud" not in features.columns
    assert len(locked_target) == len(features)
    assert locked_target["trans_num"].equals(features["trans_num"])
    assert features[generated].notna().all().all()
    assert features["trans_hour"].between(0, 23).all()
    assert features["age"].between(0, 120).all()
    for column in ["is_online", "risk_time_22_04", "is_high_amt"]:
        assert set(features[column].unique()).issubset({0, 1})

    features.to_pickle(FEATURE_OUT)
    locked_target.to_pickle(TARGET_OUT)

    # 정답값 자체나 분포는 출력하지 않는다.
    print(f"rows={len(features):,}")
    print(f"feature_columns={features.shape[1]}")
    print(f"generated={generated}")
    print("is_fraud_in_features=False")
    print("generated_missing=0")
    print("range_and_binary_checks=PASS")
    print(f"feature_checkpoint={FEATURE_OUT}")
    print(f"locked_target_checkpoint={TARGET_OUT}")


if __name__ == "__main__":
    main()
