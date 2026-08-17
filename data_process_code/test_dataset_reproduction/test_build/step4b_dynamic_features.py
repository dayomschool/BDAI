from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_TRAIN = ROOT / "upload" / "fraudTrain(1).csv"
STEP4A = ROOT / "test_build" / "test_step4a_features.pkl"
OUT = ROOT / "test_build" / "test_step4b_dynamic_features.pkl"

ONLINE = {"shopping_net", "misc_net", "grocery_net"}
HOUR_NS = 3_600_000_000_000
DAY_NS = 24 * HOUR_NS
MIN30_NS = 30 * 60 * 1_000_000_000


def haversine(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def datetime_ns(series: pd.Series) -> pd.Series:
    """pandas 버전·운영체제와 무관하게 datetime을 나노초 정수로 통일한다."""
    return series.astype("datetime64[ns]").astype("int64")


def main():
    train = pd.read_csv(
        RAW_TRAIN,
        usecols=["trans_num", "trans_date_trans_time", "cc_num", "merchant", "category", "amt", "merch_lat", "merch_long"],
    )
    test = pd.read_pickle(STEP4A)
    assert "is_fraud" not in test.columns

    train["calc_dt"] = pd.to_datetime(train["trans_date_trans_time"], errors="raise")
    test["calc_dt"] = pd.to_datetime(test["trans_date_trans_time"], errors="raise")
    train = train.sort_values(["cc_num", "calc_dt", "trans_num"], kind="mergesort")
    ordered = test.sort_values(["cc_num", "calc_dt", "trans_num"], kind="mergesort").copy()

    # 카드별 시작 상태. 정답과 무관한 입력정보만 Train에서 이어받는다.
    stat = {}
    last_merchant = {}
    last_offline = {}
    high24 = defaultdict(deque)
    offline30 = defaultdict(deque)
    amt1h = defaultdict(deque)

    for cc, group in train.groupby("cc_num", sort=False):
        amounts = group["amt"].to_numpy(dtype="float64")
        stat[cc] = [len(amounts), float(amounts.mean()), float(amounts.std(ddof=1)) if len(amounts) > 1 else 0.0]
        last = group.iloc[-1]
        last_merchant[cc] = last["merchant"]
        end_ns = int(group["calc_dt"].iloc[-1].value)

        group_ns = datetime_ns(group["calc_dt"])
        g24 = group[(group_ns >= end_ns - DAY_NS) & (group["amt"] >= 500)]
        high24[cc].extend(int(x) for x in datetime_ns(g24["calc_dt"]))
        g1h = group[group_ns >= end_ns - HOUR_NS]
        amt1h[cc].extend((int(t), float(a)) for t, a in zip(datetime_ns(g1h["calc_dt"]), g1h["amt"]))

        off = group[~group["category"].isin(ONLINE)]
        if len(off):
            lo = off.iloc[-1]
            last_offline[cc] = (int(lo["calc_dt"].value), float(lo["merch_lat"]), float(lo["merch_long"]))
            cutoff = int(lo["calc_dt"].value) - MIN30_NS
            off_ns = datetime_ns(off["calc_dt"])
            offline30[cc].extend(int(x) for x in datetime_ns(off.loc[off_ns >= cutoff, "calc_dt"]))

    n = len(ordered)
    out = {k: np.zeros(n, dtype="float64") for k in [
        "recent_24h_high_amt_count", "count_30min", "Repeat3", "speed_2", "high_speed",
        "customer_mean_amt", "customer_std_amt", "amt_ratio_to_mean", "amt_zscore_card",
        "customer_transaction_count", "merchant_change_count", "rolling_sum_amt_1h",
    ]}

    for pos, row in enumerate(ordered.itertuples(index=False)):
        cc = row.cc_num
        t = int(row.calc_dt.value)
        amt = float(row.amt)

        # 최근 24시간: 현재시각은 제외(closed='left').
        q24 = high24[cc]
        while q24 and q24[0] < t - DAY_NS:
            q24.popleft()
        out["recent_24h_high_amt_count"][pos] = sum(x < t for x in q24)

        # 고객 과거 통계: 현재 거래를 추가하기 전 값.
        count, mean, std = stat.get(cc, [0, 0.0, 0.0])
        out["customer_transaction_count"][pos] = count
        out["customer_mean_amt"][pos] = mean if count else 0.0
        out["customer_std_amt"][pos] = std if count > 1 else 0.0
        if count and mean != 0:
            out["amt_ratio_to_mean"][pos] = amt / mean
        if count > 1 and std != 0:
            out["amt_zscore_card"][pos] = (amt - mean) / std

        # 모든 거래 기준 직전 가맹점 변경 여부.
        if cc in last_merchant:
            out["merchant_change_count"][pos] = float(row.merchant != last_merchant[cc])

        # 최근 1시간 누적금액: 양쪽 경계와 현재 거래 포함.
        q1 = amt1h[cc]
        while q1 and q1[0][0] < t - HOUR_NS:
            q1.popleft()
        out["rolling_sum_amt_1h"][pos] = sum(a for _, a in q1) + amt

        # 오프라인 거래만 속도와 30분 연속결제 계산.
        if row.category not in ONLINE:
            q30 = offline30[cc]
            while q30 and q30[0] < t - MIN30_NS:
                q30.popleft()
            cnt30 = len(q30) + 1
            out["count_30min"][pos] = cnt30
            out["Repeat3"][pos] = cnt30 if cnt30 >= 3 else 0
            if cc in last_offline:
                pt, plat, plon = last_offline[cc]
                dist = float(haversine(plat, plon, float(row.merch_lat), float(row.merch_long)))
                hours = (t - pt) / HOUR_NS
                speed2 = dist / max(hours, 1 / 60)
                speed_raw = dist / hours if hours > 0 else 0.0
                out["speed_2"][pos] = speed2
                out["high_speed"][pos] = float(speed_raw >= 100)
            q30.append(t)
            last_offline[cc] = (t, float(row.merch_lat), float(row.merch_long))

        # 현재 거래로 정답 비의존 상태 갱신.
        if amt >= 500:
            q24.append(t)
        q1.append((t, amt))
        last_merchant[cc] = row.merchant
        if count == 0:
            stat[cc] = [1, amt, 0.0]
        else:
            new_count = count + 1
            new_mean = mean + (amt - mean) / new_count
            if count == 1:
                new_std = abs(amt - mean) / np.sqrt(2)
            else:
                m2 = std * std * (count - 1)
                m2 += (amt - mean) * (amt - new_mean)
                new_std = np.sqrt(max(m2 / (new_count - 1), 0.0))
            stat[cc] = [new_count, new_mean, new_std]

    for col, values in out.items():
        ordered[col] = values
    for col in ["recent_24h_high_amt_count", "count_30min", "Repeat3", "customer_transaction_count", "merchant_change_count", "high_speed"]:
        ordered[col] = ordered[col].astype("int64")

    result = ordered.sort_values("_row_order", kind="mergesort").drop(columns="calc_dt").reset_index(drop=True)
    generated = list(out)
    assert len(result) == 555_719
    assert result["trans_num"].is_unique
    assert "is_fraud" not in result.columns
    assert result[generated].notna().all().all()
    assert (result["rolling_sum_amt_1h"] + 1e-12 >= result["amt"]).all()
    assert set(result["high_speed"].unique()).issubset({0, 1})
    assert (result.loc[result["is_online"] == 1, ["count_30min", "Repeat3", "speed_2", "high_speed"]] == 0).all().all()
    result.to_pickle(OUT)

    print(f"rows={len(result):,}")
    print(f"generated_columns={len(generated)}")
    print("is_fraud_in_features=False")
    print("missing_in_generated=0")
    print("online_offline_rules=PASS")
    print("range_and_identity_checks=PASS")
    print(f"checkpoint={OUT}")


if __name__ == "__main__":
    main()
