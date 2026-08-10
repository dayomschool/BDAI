# BDAI

# 다영 - " 데이터 전처리 → 파생변수 생성 → 기술통계 → EDA " : 02 노트북에서 완료 !
# 다영 - " 가설 설정 → 통계검정 → p-value/CI 해석 → 후보 변수 선정 " : 03 노트북에서 완료 !
# 다영 - " 상관분석 " : 02,03 노트북에서 완료 !
# 다영 - " 여러 개의 로지스틱 회귀 모델 생성 - 모델별 VIF 확인 - 최종 모델 선택 " : 미완료


# final_train_code.ipynb를 확인한 결과, 최종 df(70:30 / 80:20 분할 및 zip 저장에 쓰이는 데이터셋)에 남는 컬럼은 총 28개입니다.

1. 원본에서 유지된 컬럼 (12개)

fraudTrain.csv 원본 컬럼 중 drop_cols(cell-7: Unnamed: 0, first, last, gender, street, city, state, zip, city_pop, job, unix_time)에 안 걸린 것들.

┌───────────────────────┬───────────────────────┐
│         컬럼          │         설명          │
├───────────────────────┼───────────────────────┤
│ trans_date_trans_time │ 거래 일시             │
├───────────────────────┼───────────────────────┤
│ cc_num                │ 카드번호(고객 식별자) │
├───────────────────────┼───────────────────────┤
│ merchant              │ 가맹점명              │
├───────────────────────┼───────────────────────┤
│ category              │ 거래 카테고리         │
├───────────────────────┼───────────────────────┤
│ amt                   │ 거래 금액             │
├───────────────────────┼───────────────────────┤
│ lat, long             │ 고객 거주지 위경도    │
├───────────────────────┼───────────────────────┤
│ dob                   │ 생년월일              │
├───────────────────────┼───────────────────────┤
│ trans_num             │ 거래 고유 ID          │
├───────────────────────┼───────────────────────┤
│ merch_lat, merch_long │ 가맹점 위경도         │
├───────────────────────┼───────────────────────┤
│ is_fraud              │ 사기 여부(타깃)       │
└───────────────────────┴───────────────────────┘

2. 새로 생성된 파생 컬럼 (16개)

시간/카테고리 기반 (cell-1)
- recent_24h_high_amt_count — 카드별 최근 24시간 내 고액(≥500) 결제 횟수
- category_recent_fraud_rate — 업종별 최근 7일 사
- category_recent_fraud_rate_missing — 위 값이 NaN이었는지(→0으로 채움) 표시 플래그

오프라인 거래 이동/속도 기반 (cell-2~4, offline_df에서 계산 후 trans_num 기준 merge)
- speed_kmh — haversine 기반 이동속도(직전 거래 대
- count_30min — 최근 30분 내 거래 횟수
- Repeat3 — count_30min≥3일 때만 값 유지(그 외 0)
- high_speed — speed_kmh≥100일 때만 값 유지(그 외
- speed_2 — 별도 haversine 함수로 재계산한 이동속도(시간차 최소 1분 clip 적용)

고객별 금액 통계 (cell-5)
- customer_mean_amt — 고객별 과거 누적 평균 결제금
- customer_std_amt — 고객별 과거 누적 표준편차
- amt_ratio_to_mean — 현재 금액 / 과거 평균
- amt_zscore_card — 고객별 금액 Z-score
- customer_transaction_count — 고객별 이전 거래 횟

시간/나이 기반 (cell-6)
- trans_hour — 거래 시각(hour)
- age — 거래 시점 기준 나이
- age_group — 연령대 구간(20세 미만/20대/.../60세 이상)

참고

- is_online, is_high_amt는 cell-1에서 생성되지만 cs_online','is_high_amt'])로 최종 제거됨 (컬럼목록에서 제외).
- offline_df에서만 쓰인 중간 변수(prev_lat, prev_liff_sec, prev_merch_lat, prev_merch_long,prev_trans_time, merchant_shift_km, time_diff_hr)는 merge 시 offline_features에 포함되지 않아 최종 df에는 안 들어감.