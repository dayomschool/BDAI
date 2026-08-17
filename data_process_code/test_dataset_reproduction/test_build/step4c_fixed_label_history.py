from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_TRAIN = ROOT / "upload" / "fraudTrain(1).csv"
STEP4B = ROOT / "test_build" / "test_step4b_dynamic_features.pkl"
OUT = ROOT / "test_build" / "test_step4c_all_features.pkl"


def active_hour_cutoff(hour_counts: np.ndarray, coverage: float = 0.80) -> int:
    total = int(hour_counts.sum())
    ordered = np.sort(hour_counts)[::-1]
    cumulative = 0
    for count in ordered:
        cumulative += int(count)
        if cumulative >= total * coverage:
            return int(count)
    return 0


def main():
    # is_fraud는 Train의 확정 이력 계산에만 사용한다.
    train = pd.read_csv(
        RAW_TRAIN,
        usecols=["trans_date_trans_time", "cc_num", "category", "amt", "is_fraud"],
    )
    test = pd.read_pickle(STEP4B)
    assert "is_fraud" not in test.columns

    train["calc_dt"] = pd.to_datetime(train["trans_date_trans_time"], errors="raise")
    train_end = train["calc_dt"].max()
    seven_day_start = train_end - pd.Timedelta(days=7)

    # Train 전체 종료시각을 기준으로 직전 7일 [종료-7일, 종료)의 업종별 사기율을 고정한다.
    recent = train[(train["calc_dt"] >= seven_day_start) & (train["calc_dt"] < train_end)]
    category_rate = recent.groupby("category", sort=False)["is_fraud"].mean().to_dict()
    known_categories = set(train["category"].unique())

    test["category_recent_fraud_rate_missing"] = (~test["category"].isin(known_categories)).astype("int64")
    test["category_recent_fraud_rate"] = test["category"].map(category_rate).fillna(0.0).astype("float64")
    test["interact_repeat_category"] = (
        test["recent_24h_high_amt_count"] * test["category_recent_fraud_rate"]
    ).astype("float64")

    # 카드별 Train 정상거래 최종 상태를 고정한다.
    normal = train[train["is_fraud"] == 0].copy()
    median_by_card = normal.groupby("cc_num", sort=False)["amt"].median().to_dict()
    normal_count = normal.groupby("cc_num", sort=False).size().to_dict()
    normal["hour"] = normal["calc_dt"].dt.hour
    hour_tables = {
        cc: np.bincount(group["hour"].to_numpy(dtype="int64"), minlength=24)
        for cc, group in normal.groupby("cc_num", sort=False)
    }
    cutoff_by_card = {
        cc: active_hour_cutoff(counts)
        for cc, counts in hour_tables.items()
        if normal_count[cc] >= 20
    }

    median = test["cc_num"].map(median_by_card).astype("float64")
    test["prior_normal_median_amt"] = median
    test["amt_to_prior_median_ratio"] = test["amt"] / median
    test["is_10x_prior_median"] = (
        median.notna() & median.gt(0) & test["amt_to_prior_median_ratio"].ge(10)
    ).astype("int64")
    test["has_prior_normal_transaction"] = test["cc_num"].isin(median_by_card).astype("int64")

    outside = np.zeros(len(test), dtype="int64")
    cc_values = test["cc_num"].to_numpy()
    hours = test["trans_hour"].to_numpy(dtype="int64")
    for i, (cc, hour) in enumerate(zip(cc_values, hours)):
        cutoff = cutoff_by_card.get(cc)
        if cutoff is not None and hour_tables[cc][hour] < cutoff:
            outside[i] = 1
    test["outside_trans_hours_80"] = outside

    # 값과 누수 방지 정책 검증.
    generated = [
        "category_recent_fraud_rate", "category_recent_fraud_rate_missing",
        "interact_repeat_category", "prior_normal_median_amt",
        "amt_to_prior_median_ratio", "is_10x_prior_median",
        "has_prior_normal_transaction", "outside_trans_hours_80",
    ]
    new_card = ~test["cc_num"].isin(set(train["cc_num"].unique()))
    assert len(test) == 555_719
    assert "is_fraud" not in test.columns
    assert test["trans_num"].is_unique
    assert test[["category_recent_fraud_rate", "category_recent_fraud_rate_missing", "interact_repeat_category",
                 "is_10x_prior_median", "has_prior_normal_transaction", "outside_trans_hours_80"]].notna().all().all()
    assert test.loc[new_card, "prior_normal_median_amt"].isna().all()
    assert test.loc[new_card, "amt_to_prior_median_ratio"].isna().all()
    assert (test.loc[new_card, ["is_10x_prior_median", "has_prior_normal_transaction", "outside_trans_hours_80"]] == 0).all().all()
    assert set(test["category_recent_fraud_rate_missing"].unique()).issubset({0, 1})
    for col in ["is_10x_prior_median", "has_prior_normal_transaction", "outside_trans_hours_80"]:
        assert set(test[col].unique()).issubset({0, 1})
    assert test["category_recent_fraud_rate"].between(0, 1).all()

    test.to_pickle(OUT)
    print(f"rows={len(test):,}")
    print(f"generated_columns={len(generated)}")
    print(f"train_end={train_end}")
    print(f"fixed_category_rates={len(category_rate)}")
    print(f"new_cards={test.loc[new_card, 'cc_num'].nunique()}")
    print("test_is_fraud_used=False")
    print("new_card_policy=PASS")
    print("missing_and_range_checks=PASS")
    print(f"checkpoint={OUT}")


if __name__ == "__main__":
    main()
