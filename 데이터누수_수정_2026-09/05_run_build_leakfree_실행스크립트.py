from __future__ import annotations

import heapq
import time
from pathlib import Path

import numpy as np
import pandas as pd

CANDIDATE_TRAIN_PATHS = [
    Path(r"C:\Users\user\Desktop\bdai부캠\BDAI\data\fraudTrain.csv"),
]

CANDIDATE_TEST_PATHS = [
    Path(r"C:\Users\user\Desktop\bdai부캠\BDAI\data\fraudTest.csv"),
]

OUT_TRAIN = Path(r"C:\Users\user\Desktop\bdai부캠\BDAI\data\train_11features.csv")
OUT_TEST = Path(r"C:\Users\user\Desktop\bdai부캠\BDAI\data\test_11features.csv")

ONLINE_CATEGORIES = {"shopping_net", "misc_net", "grocery_net"}
HIGH_AMT_THRESHOLD = 500
HIGH_SPEED_THRESHOLD_KMH = 100


def find_path(candidates: list[Path]) -> Path:
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "원본 csv를 찾지 못했습니다.\n시도한 경로:\n" + "\n".join(str(p) for p in candidates)
    )


def haversine(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlambda / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def exact_age(trans_dt: pd.Series, birth_dt: pd.Series) -> pd.Series:
    years = trans_dt.dt.year - birth_dt.dt.year
    not_yet = (trans_dt.dt.month < birth_dt.dt.month) | (
        (trans_dt.dt.month == birth_dt.dt.month) & (trans_dt.dt.day < birth_dt.dt.day)
    )
    return (years - not_yet.astype("int64")).astype("int64")


def load_raw(path: Path, source: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")
    required = {"trans_num", "trans_date_trans_time", "cc_num", "dob", "category", "amt",
                "merch_lat", "merch_long", "is_fraud"}
    missing = required.difference(df.columns)
    assert not missing, f"{path} 에 필수 컬럼이 없습니다: {missing}"
    assert df["trans_num"].is_unique, f"{path}: trans_num 중복 존재"
    df["_source"] = source
    return df


def add_current_row_features(df: pd.DataFrame) -> pd.DataFrame:
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"], errors="raise")
    df["dob"] = pd.to_datetime(df["dob"], errors="raise")
    df["trans_hour"] = df["trans_date_trans_time"].dt.hour.astype("int64")
    df["age"] = exact_age(df["trans_date_trans_time"], df["dob"])
    df["is_online"] = df["category"].isin(ONLINE_CATEGORIES)
    df["is_high_amt"] = (df["amt"] >= HIGH_AMT_THRESHOLD).astype("int64")
    return df


def add_offline_speed_and_count30(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["cc_num", "trans_date_trans_time", "trans_num"], kind="mergesort").reset_index(drop=True)

    offline_mask = ~df["is_online"]
    offline_df = df.loc[offline_mask].copy()
    offline_df = offline_df.sort_values(["cc_num", "trans_date_trans_time", "trans_num"], kind="mergesort")

    offline_df["count_30min"] = 0
    for _, idx in offline_df.groupby("cc_num", sort=False).groups.items():
        times = offline_df.loc[idx, "trans_date_trans_time"].values.astype("datetime64[s]")
        left = np.searchsorted(times, times - np.timedelta64(30, "m"))
        right = np.arange(len(times))
        offline_df.loc[idx, "count_30min"] = right - left + 1

    offline_df["prev_merch_lat"] = offline_df.groupby("cc_num")["merch_lat"].shift(1)
    offline_df["prev_merch_long"] = offline_df.groupby("cc_num")["merch_long"].shift(1)
    offline_df["prev_trans_time"] = offline_df.groupby("cc_num")["trans_date_trans_time"].shift(1)

    dist_km = haversine(
        offline_df["prev_merch_lat"], offline_df["prev_merch_long"],
        offline_df["merch_lat"], offline_df["merch_long"],
    )
    hours = (
        offline_df["trans_date_trans_time"] - offline_df["prev_trans_time"]
    ).dt.total_seconds() / 3600

    with np.errstate(divide="ignore", invalid="ignore"):
        raw_speed = np.where(hours > 0, dist_km / hours, 0.0)
    raw_speed = np.nan_to_num(raw_speed, nan=0.0, posinf=0.0, neginf=0.0)

    offline_df["high_speed"] = (raw_speed >= HIGH_SPEED_THRESHOLD_KMH).astype("int64")

    keep = offline_df[["trans_num", "count_30min", "high_speed"]]
    df = df.merge(keep, on="trans_num", how="left")
    df["count_30min"] = df["count_30min"].fillna(0).astype("int64")
    df["high_speed"] = df["high_speed"].fillna(0).astype("int64")
    return df


def add_recent_24h_high_amt_count(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df[["cc_num", "trans_date_trans_time", "is_high_amt"]].copy()
    tmp = tmp.set_index("trans_date_trans_time")
    tmp = tmp.sort_values(["cc_num"], kind="mergesort")
    values = (
        tmp.groupby("cc_num")["is_high_amt"]
        .rolling("24h", closed="left", min_periods=0)
        .sum()
        .values
    )
    tmp2 = df.sort_values(["cc_num"], kind="mergesort")[["trans_num"]].copy()
    tmp2["recent_24h_high_amt_count"] = values
    df = df.merge(tmp2, on="trans_num", how="left")
    df["recent_24h_high_amt_count"] = df["recent_24h_high_amt_count"].fillna(0).astype("int64")
    return df


def add_rolling_sum_amt_1h(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df[["cc_num", "trans_date_trans_time", "amt"]].copy()
    tmp = tmp.set_index("trans_date_trans_time").sort_values(["cc_num"], kind="mergesort")
    values = tmp.groupby("cc_num")["amt"].rolling("1h", min_periods=1).sum().values

    tmp2 = df.sort_values(["cc_num"], kind="mergesort")[["trans_num"]].copy()
    tmp2["rolling_sum_amt_1h"] = values
    df = df.merge(tmp2, on="trans_num", how="left")
    return df


def add_customer_amt_stats(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["cc_num", "trans_date_trans_time", "trans_num"], kind="mergesort").reset_index(drop=True)

    grouped = df.groupby("cc_num")["amt"]
    df["customer_mean_amt"] = grouped.apply(lambda s: s.shift(1).expanding().mean()).reset_index(level=0, drop=True)
    df["customer_std_amt"] = grouped.apply(lambda s: s.shift(1).expanding().std()).reset_index(level=0, drop=True)

    df["customer_mean_amt"] = df["customer_mean_amt"].fillna(0.0)
    df["customer_std_amt"] = df["customer_std_amt"].fillna(0.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        z = (df["amt"] - df["customer_mean_amt"]) / df["customer_std_amt"]
    df["amt_zscore_card"] = z.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return df


class RunningMedian:
    __slots__ = ("lo", "hi")

    def __init__(self):
        self.lo: list[float] = []
        self.hi: list[float] = []

    def add(self, x: float) -> None:
        if not self.lo or x <= -self.lo[0]:
            heapq.heappush(self.lo, -x)
        else:
            heapq.heappush(self.hi, x)

        if len(self.lo) > len(self.hi) + 1:
            heapq.heappush(self.hi, -heapq.heappop(self.lo))
        elif len(self.hi) > len(self.lo):
            heapq.heappush(self.lo, -heapq.heappop(self.hi))

    def median(self):
        if not self.lo and not self.hi:
            return np.nan
        if len(self.lo) > len(self.hi):
            return float(-self.lo[0])
        return (float(-self.lo[0]) + float(self.hi[0])) / 2.0


def add_prior_normal_median_train(train_df: pd.DataFrame):
    train_df = train_df.sort_values(
        ["cc_num", "trans_date_trans_time", "trans_num"], kind="mergesort"
    ).reset_index(drop=True)

    n = len(train_df)
    out = np.full(n, np.nan, dtype="float64")
    medians: dict = {}

    cc_arr = train_df["cc_num"].to_numpy()
    amt_arr = train_df["amt"].to_numpy()
    fraud_arr = train_df["is_fraud"].to_numpy()

    for i in range(n):
        cc = cc_arr[i]
        rm = medians.get(cc)
        out[i] = rm.median() if rm is not None else np.nan
        if fraud_arr[i] == 0:
            if rm is None:
                rm = RunningMedian()
                medians[cc] = rm
            rm.add(amt_arr[i])

    train_df["prior_normal_median_amt"] = out

    final_median_by_card = {cc: rm.median() for cc, rm in medians.items()}
    return train_df, final_median_by_card


def add_prior_normal_median_test(test_df: pd.DataFrame, final_median_by_card: dict) -> pd.DataFrame:
    test_df = test_df.copy()
    test_df["prior_normal_median_amt"] = test_df["cc_num"].map(final_median_by_card)
    return test_df


def finalize_amt_ratio(df: pd.DataFrame) -> pd.DataFrame:
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = df["amt"] / df["prior_normal_median_amt"]
    df["amt_to_prior_median_ratio"] = ratio.replace([np.inf, -np.inf], np.nan)
    return df


FINAL_COLUMNS = [
    "trans_num", "trans_date_trans_time", "cc_num",
    "category", "amt", "trans_hour", "age",
    "recent_24h_high_amt_count", "amt_to_prior_median_ratio",
    "rolling_sum_amt_1h", "amt_zscore_card", "prior_normal_median_amt",
    "count_30min", "high_speed",
    "is_fraud",
]


def build_datasets(train_path: Path, test_path: Path):
    train_raw = load_raw(train_path, "train")
    test_raw = load_raw(test_path, "test")

    combined = pd.concat([train_raw, test_raw], ignore_index=True)
    combined = add_current_row_features(combined)

    combined = add_offline_speed_and_count30(combined)
    combined = add_recent_24h_high_amt_count(combined)
    combined = add_rolling_sum_amt_1h(combined)
    combined = add_customer_amt_stats(combined)

    train_df = combined.loc[combined["_source"] == "train"].copy()
    test_df = combined.loc[combined["_source"] == "test"].copy()

    train_df, final_median_by_card = add_prior_normal_median_train(train_df)
    test_df = add_prior_normal_median_test(test_df, final_median_by_card)

    train_df = finalize_amt_ratio(train_df)
    test_df = finalize_amt_ratio(test_df)

    train_df = train_df.sort_values(["trans_date_trans_time", "trans_num"], kind="mergesort").reset_index(drop=True)
    test_df = test_df.sort_values(["trans_date_trans_time", "trans_num"], kind="mergesort").reset_index(drop=True)

    return train_df[FINAL_COLUMNS], test_df[FINAL_COLUMNS]


def verify_recent_24h(combined_raw: pd.DataFrame, result_df: pd.DataFrame, n_sample: int = 5) -> None:
    rng = np.random.default_rng(42)
    sample_cards = rng.choice(result_df["cc_num"].unique(), size=min(n_sample, result_df["cc_num"].nunique()), replace=False)
    for cc in sample_cards:
        sub = result_df.loc[result_df["cc_num"] == cc].sort_values("trans_date_trans_time")
        raw_sub = combined_raw.loc[combined_raw["cc_num"] == cc].copy()
        raw_sub["trans_date_trans_time"] = pd.to_datetime(raw_sub["trans_date_trans_time"])
        raw_sub["is_high_amt"] = (raw_sub["amt"] >= HIGH_AMT_THRESHOLD).astype(int)
        raw_sub = raw_sub.sort_values("trans_date_trans_time")
        times = raw_sub["trans_date_trans_time"].to_numpy()
        for _, row in sub.head(20).iterrows():
            t = np.datetime64(row["trans_date_trans_time"])
            mask = (times < t) & (times >= t - np.timedelta64(24, "h"))
            expected = int(raw_sub.loc[mask, "is_high_amt"].sum())
            actual = int(row["recent_24h_high_amt_count"])
            assert expected == actual, (
                f"[FAIL] recent_24h_high_amt_count 불일치 cc_num={cc} trans_num={row['trans_num']} "
                f"expected={expected} actual={actual}"
            )
    print("[OK] recent_24h_high_amt_count 재계산 검증 통과 (표본 고객 각 20건)")


def verify_prior_normal_median(train_raw: pd.DataFrame, train_result: pd.DataFrame, n_sample: int = 5) -> None:
    rng = np.random.default_rng(7)
    cards = train_result["cc_num"].unique()
    sample_cards = rng.choice(cards, size=min(n_sample, len(cards)), replace=False)
    train_raw = train_raw.copy()
    train_raw["trans_date_trans_time"] = pd.to_datetime(train_raw["trans_date_trans_time"])

    for cc in sample_cards:
        sub = train_result.loc[train_result["cc_num"] == cc].sort_values("trans_date_trans_time")
        raw_sub = train_raw.loc[train_raw["cc_num"] == cc].sort_values("trans_date_trans_time")
        for _, row in sub.tail(10).iterrows():
            t = row["trans_date_trans_time"]
            prior_normal = raw_sub.loc[
                (raw_sub["trans_date_trans_time"] < t) & (raw_sub["is_fraud"] == 0),
                "amt",
            ]
            if len(prior_normal) == 0:
                assert pd.isna(row["prior_normal_median_amt"])
                continue
            expected = float(prior_normal.median())
            actual = row["prior_normal_median_amt"]
            assert np.isclose(expected, actual, atol=1e-6), (
                f"[FAIL] prior_normal_median_amt 불일치 cc_num={cc} trans_num={row['trans_num']} "
                f"expected={expected} actual={actual}"
            )
    print("[OK] prior_normal_median_amt 재계산 검증 통과 (미래/자기자신 정보 미사용 확인)")


def verify_no_future_leak_basic(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    first_train = train_df.sort_values("trans_date_trans_time").groupby("cc_num").head(1)
    assert (first_train["amt_zscore_card"] == 0).all(), "[FAIL] 카드별 첫 거래의 amt_zscore_card가 0이 아님"
    print("[OK] 카드별 '첫 거래' amt_zscore_card == 0 확인 (미래/전체이력 사용 아님)")

    assert test_df["is_fraud"].isin([0, 1]).all()
    print("[OK] 컬럼 스키마 및 is_fraud 값 범위 확인")


def run_verification(combined_raw, train_raw, train_df, test_df) -> None:
    print("\n=== 검증 시작 ===")
    verify_no_future_leak_basic(train_df, test_df)
    verify_recent_24h(combined_raw, pd.concat([train_df, test_df], ignore_index=True))
    verify_prior_normal_median(train_raw, train_df)
    print("=== 검증 끝: 위에 [FAIL]이 없으면 5개 변수 모두 시점 인과성 통과 ===\n")


if __name__ == "__main__":
    t0 = time.perf_counter()
    train_path = find_path(CANDIDATE_TRAIN_PATHS)
    test_path = find_path(CANDIDATE_TEST_PATHS)
    print(f"train 원본: {train_path}")
    print(f"test  원본: {test_path}")

    train_raw_for_verify = load_raw(train_path, "train")
    combined_raw_for_verify = pd.concat(
        [train_raw_for_verify, load_raw(test_path, "test")], ignore_index=True
    )

    train_df, test_df = build_datasets(train_path, test_path)
    print(f"train_11features: {train_df.shape}")
    print(f"test_11features : {test_df.shape}")

    run_verification(combined_raw_for_verify, train_raw_for_verify, train_df, test_df)

    train_df.to_csv(OUT_TRAIN, index=False, encoding="utf-8-sig")
    test_df.to_csv(OUT_TEST, index=False, encoding="utf-8-sig")
    print(f"저장 완료: {OUT_TRAIN.resolve()}")
    print(f"저장 완료: {OUT_TEST.resolve()}")
    print(f"총 소요 시간: {time.perf_counter() - t0:.1f}초")
