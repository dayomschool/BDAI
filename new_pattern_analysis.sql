-- ============================================================
-- 조합3(11개 변수) 밖에서 새로운 사기 패턴 찾기 -- SQL 버전
-- ============================================================
-- 배경: 지금 쓰는 11개 변수는 대부분 "카드의 평소 금액 대비 얼마나
-- 벗어났는가"를 재는 변수다. 하지만 FN(놓친 사기)은 금액이 아니라 다른
-- 형태(위치, 가맹점 다양성, 계정 나이, 거래 간격 등)로 나타날 수 있다.
-- 이 스크립트는 amt 편차 이외의 축으로 사기율을 뜯어보는 탐색용 쿼리 모음.
--
-- 실행 환경: MySQL 8.0 / MariaDB 10.11 이상 (Day4~5에서 쓰던 transactions
-- 테이블 기준). Workbench에 붙여넣고 섹션(A~F) 단위로 실행하면 된다.
-- ============================================================

USE card_db;   -- 본인 DB 이름으로 교체

-- ------------------------------------------------------------------
-- 0. 테이블에 lat/long/merch_lat/merch_long 컬럼이 있는지 먼저 확인
-- ------------------------------------------------------------------
-- Day4 실습 때 "안 쓰는 컬럼(lat, long 등) 체크 해제"로 Import Wizard에서
-- 위경도 컬럼을 뺐을 수 있다. 아래로 확인:
--
--   SHOW COLUMNS FROM transactions LIKE '%lat%';
--
-- 만약 lat/long/merch_lat/merch_long이 안 보이면, B 섹션(지리 패턴)을
-- 쓰기 위해 컬럼을 추가하고 raw csv에서 그 값만 다시 채워 넣어야 한다.
-- 컬럼 추가만 먼저 해두면(값은 나중에 UPDATE 또는 재Import):
--
--   ALTER TABLE transactions
--     ADD COLUMN lat DOUBLE,
--     ADD COLUMN `long` DOUBLE,      -- long은 예약어라 반드시 백틱(`) 필요
--     ADD COLUMN merch_lat DOUBLE,
--     ADD COLUMN merch_long DOUBLE;
--
-- 가장 간단한 방법은 fraudTrain/fraudTest를 위경도 4개 컬럼까지 포함해서
-- 테이블을 통째로 다시 만들고 Import Wizard를 한 번 더 돌리는 것.
-- (A/C/D/E/F 섹션은 위경도가 없어도 그대로 실행된다.)

-- ============================================================
-- A. 시간 · 계정 나이(tenure) 패턴
-- ============================================================

-- A1. 요일 x 업종 조합 중 사기율이 튀는 곳 (단순 "심야"보다 세분화)
SELECT
    DAYNAME(trans_date_trans_time) AS dow,
    category,
    COUNT(*)                       AS tx_cnt,
    SUM(is_fraud)                  AS fraud_cnt,
    ROUND(AVG(is_fraud) * 100, 3)  AS fraud_pct
FROM transactions
GROUP BY dow, category
HAVING tx_cnt >= 30
ORDER BY fraud_pct DESC
LIMIT 15;

-- A2. 카드별 직전 거래와의 시간 간격(초) 구간별 사기율
--     LAG()로 "그 카드의 바로 전 거래 시각"을 끌어와서 간격을 구한다.
WITH gapped AS (
    SELECT
        is_fraud,
        TIMESTAMPDIFF(
            SECOND,
            LAG(trans_date_trans_time) OVER (PARTITION BY cc_num ORDER BY trans_date_trans_time),
            trans_date_trans_time
        ) AS gap_sec
    FROM transactions
)
SELECT
    CASE
        WHEN gap_sec IS NULL      THEN '0_첫거래'
        WHEN gap_sec <= 60        THEN '1_1분이내'
        WHEN gap_sec <= 600       THEN '2_10분이내'
        WHEN gap_sec <= 3600      THEN '3_1시간이내'
        WHEN gap_sec <= 86400     THEN '4_1일이내'
        WHEN gap_sec <= 604800    THEN '5_1주일이내'
        ELSE '6_1주일초과'
    END AS gap_bucket,
    COUNT(*)                       AS tx_cnt,
    SUM(is_fraud)                  AS fraud_cnt,
    ROUND(AVG(is_fraud) * 100, 3)  AS fraud_pct
FROM gapped
GROUP BY gap_bucket
ORDER BY gap_bucket;

-- A3. 카드 "최초 거래" 이후 경과일수 구간별 사기율 (신규 카드일수록 위험한가)
WITH first_tx AS (
    SELECT cc_num, MIN(trans_date_trans_time) AS first_dt
    FROM transactions
    GROUP BY cc_num
),
aged AS (
    SELECT
        t.is_fraud,
        DATEDIFF(t.trans_date_trans_time, f.first_dt) AS days_since_first
    FROM transactions t
    JOIN first_tx f ON t.cc_num = f.cc_num
)
SELECT
    CASE
        WHEN days_since_first = 0  THEN '0_당일(첫거래)'
        WHEN days_since_first <= 7 THEN '1_1주이내'
        WHEN days_since_first <= 30 THEN '2_1개월이내'
        WHEN days_since_first <= 90 THEN '3_3개월이내'
        ELSE '4_3개월초과'
    END AS tenure_bucket,
    COUNT(*)                       AS tx_cnt,
    SUM(is_fraud)                  AS fraud_cnt,
    ROUND(AVG(is_fraud) * 100, 3)  AS fraud_pct
FROM aged
GROUP BY tenure_bucket
ORDER BY tenure_bucket;

-- ============================================================
-- B. 지리 패턴 (lat/long/merch_lat/merch_long 필요 -- 위 0번 참고)
-- ============================================================

-- B1. 자택-가맹점 거리(km) 구간별 사기율. MySQL엔 haversine 내장함수가
--     없어서 구면삼각법 공식을 직접 쓴다. ACOS 인자가 부동소수점 오차로
--     1을 살짝 넘는 걸 막기 위해 LEAST/GREATEST로 [-1, 1]에 clip.
WITH dist AS (
    SELECT
        is_fraud,
        6371 * ACOS(
            LEAST(1, GREATEST(-1,
                COS(RADIANS(lat)) * COS(RADIANS(merch_lat)) * COS(RADIANS(merch_long) - RADIANS(`long`))
                + SIN(RADIANS(lat)) * SIN(RADIANS(merch_lat))
            ))
        ) AS dist_km
    FROM transactions
)
SELECT
    CASE
        WHEN dist_km < 10  THEN '0_10km미만'
        WHEN dist_km < 30  THEN '1_10-30km'
        WHEN dist_km < 50  THEN '2_30-50km'
        WHEN dist_km < 80  THEN '3_50-80km'
        WHEN dist_km < 100 THEN '4_80-100km'
        ELSE '5_100km이상'
    END AS dist_bucket,
    COUNT(*)                       AS tx_cnt,
    SUM(is_fraud)                  AS fraud_cnt,
    ROUND(AVG(is_fraud) * 100, 3)  AS fraud_pct
FROM dist
GROUP BY dist_bucket
ORDER BY dist_bucket;

-- B3. 카드별 "같은 날" 최대 이동거리(km) 구간별 사기(일) 비율.
--     같은 카드 · 같은 날짜의 거래끼리 자기조인(self-join)해서 모든 쌍의
--     거리를 구하고 그중 최댓값을 취한다. (거래가 많은 카드 · 날짜는
--     쌍의 개수가 n^2으로 늘어나므로, 실서비스에선 최근 N일로 범위를
--     좁히는 걸 권장)
WITH same_day AS (
    SELECT
        a.cc_num,
        DATE(a.trans_date_trans_time) AS d,
        MAX(
            6371 * ACOS(
                LEAST(1, GREATEST(-1,
                    COS(RADIANS(a.merch_lat)) * COS(RADIANS(b.merch_lat)) * COS(RADIANS(b.merch_long) - RADIANS(a.merch_long))
                    + SIN(RADIANS(a.merch_lat)) * SIN(RADIANS(b.merch_lat))
                ))
            )
        ) AS max_dist_km,
        MAX(a.is_fraud) AS any_fraud_that_day
    FROM transactions a
    JOIN transactions b
        ON a.cc_num = b.cc_num
        AND DATE(a.trans_date_trans_time) = DATE(b.trans_date_trans_time)
        AND a.trans_num < b.trans_num        -- 같은 쌍을 두 번 세지 않기 위한 트릭
    GROUP BY a.cc_num, d
)
SELECT
    CASE
        WHEN max_dist_km < 50   THEN '0_50km미만'
        WHEN max_dist_km < 150  THEN '1_50-150km'
        WHEN max_dist_km < 300  THEN '2_150-300km'
        ELSE '3_300km이상'
    END AS same_day_span_bucket,
    COUNT(*)                            AS card_day_cnt,
    SUM(any_fraud_that_day)             AS fraud_day_cnt,
    ROUND(AVG(any_fraud_that_day) * 100, 3) AS fraud_day_pct
FROM same_day
GROUP BY same_day_span_bucket
ORDER BY same_day_span_bucket;

-- ============================================================
-- C. 가맹점 · 업종 다양성 패턴
-- ============================================================

-- C1. 카드별 하루 이용 가맹점 수 구간별 사기(일) 비율
--     "여기저기 찔러보는" 패턴이 있는지
WITH daily_div AS (
    SELECT
        cc_num,
        DATE(trans_date_trans_time)      AS d,
        COUNT(DISTINCT merchant)         AS merchant_cnt,
        COUNT(DISTINCT category)         AS category_cnt,
        MAX(is_fraud)                    AS any_fraud
    FROM transactions
    GROUP BY cc_num, d
)
SELECT
    merchant_cnt,
    COUNT(*)                       AS card_day_cnt,
    SUM(any_fraud)                 AS fraud_day_cnt,
    ROUND(AVG(any_fraud) * 100, 3) AS fraud_day_pct
FROM daily_div
GROUP BY merchant_cnt
ORDER BY merchant_cnt
LIMIT 15;

-- C2. 가맹점 최초 이용 여부별 사기율
--     ROW_NUMBER()로 그 카드가 그 가맹점을 몇 번째 쓰는지 매긴다.
WITH ranked AS (
    SELECT
        is_fraud,
        ROW_NUMBER() OVER (PARTITION BY cc_num, merchant ORDER BY trans_date_trans_time) AS rn
    FROM transactions
)
SELECT
    CASE WHEN rn = 1 THEN '최초이용' ELSE '재이용' END AS merchant_novelty,
    COUNT(*)                       AS tx_cnt,
    SUM(is_fraud)                  AS fraud_cnt,
    ROUND(AVG(is_fraud) * 100, 3)  AS fraud_pct
FROM ranked
GROUP BY merchant_novelty;

-- C3. 가맹점별 사기 집중도 랭킹 (최소 20건 거래한 가맹점만, 표본이 너무
--     작은 가맹점의 100% 사기율 같은 착시를 걸러낸다)
SELECT
    merchant,
    COUNT(*)                       AS tx_cnt,
    SUM(is_fraud)                  AS fraud_cnt,
    ROUND(AVG(is_fraud) * 100, 3)  AS fraud_pct
FROM transactions
GROUP BY merchant
HAVING tx_cnt >= 20
ORDER BY fraud_pct DESC
LIMIT 15;

-- C4. 같은 가맹점에서 같은 시간대에 서로 다른 카드가 몰리는 경우
--     (POS 단말기 침해 · 스키밍 의심 신호. distinct_cards가 비정상적으로
--     크면 그 가맹점·시간대를 우선 조사 대상으로)
SELECT
    merchant,
    DATE(trans_date_trans_time)        AS d,
    HOUR(trans_date_trans_time)        AS hh,
    COUNT(DISTINCT cc_num)             AS distinct_cards,
    COUNT(*)                           AS tx_cnt,
    SUM(is_fraud)                      AS fraud_cnt
FROM transactions
GROUP BY merchant, d, hh
HAVING distinct_cards >= 5
ORDER BY distinct_cards DESC
LIMIT 15;

-- ============================================================
-- D. 인구통계 패턴
-- ============================================================

-- D1. 직업(job)별 사기율 랭킹 (최소 100건 -- job은 카디널리티가 높아서
--     표본이 작은 직업이 상위를 왜곡하지 않도록 필터)
SELECT
    job,
    COUNT(*)                       AS tx_cnt,
    SUM(is_fraud)                  AS fraud_cnt,
    ROUND(AVG(is_fraud) * 100, 3)  AS fraud_pct
FROM transactions
GROUP BY job
HAVING tx_cnt >= 100
ORDER BY fraud_pct DESC
LIMIT 15;

-- D2. 도시 인구밀도 구간별 사기율 (도시 vs 농촌)
SELECT
    CASE
        WHEN city_pop < 1000    THEN '0_초농촌(<1천)'
        WHEN city_pop < 10000   THEN '1_농촌(1천-1만)'
        WHEN city_pop < 100000  THEN '2_소도시(1만-10만)'
        WHEN city_pop < 500000  THEN '3_중도시(10만-50만)'
        ELSE '4_대도시(50만+)'
    END AS pop_bucket,
    COUNT(*)                       AS tx_cnt,
    SUM(is_fraud)                  AS fraud_cnt,
    ROUND(AVG(is_fraud) * 100, 3)  AS fraud_pct
FROM transactions
GROUP BY pop_bucket
ORDER BY pop_bucket;

-- D3. 연령대 x 성별 x 업종 3중 코호트 사기율 TOP 15
WITH t AS (
    SELECT
        gender,
        category,
        is_fraud,
        CASE
            WHEN TIMESTAMPDIFF(YEAR, dob, '2020-12-31') < 30 THEN '20s'
            WHEN TIMESTAMPDIFF(YEAR, dob, '2020-12-31') < 40 THEN '30s'
            WHEN TIMESTAMPDIFF(YEAR, dob, '2020-12-31') < 50 THEN '40s'
            WHEN TIMESTAMPDIFF(YEAR, dob, '2020-12-31') < 60 THEN '50s'
            ELSE '60+'
        END AS age_bucket
    FROM transactions
)
SELECT
    age_bucket, gender, category,
    COUNT(*)                       AS tx_cnt,
    SUM(is_fraud)                  AS fraud_cnt,
    ROUND(AVG(is_fraud) * 100, 3)  AS fraud_pct
FROM t
GROUP BY age_bucket, gender, category
HAVING tx_cnt >= 20
ORDER BY fraud_pct DESC
LIMIT 15;

-- ============================================================
-- E. 금액의 절대적 형태 (개인 편차 아닌 방식)
-- ============================================================

-- E1. 라운드넘버 결제(정수 & 10의 배수) 여부별 사기율
--     카드 유효성 확인용 소액/정형 결제 패턴을 잡기 위함
SELECT
    CASE
        WHEN amt = FLOOR(amt) AND MOD(amt, 10) = 0 THEN '라운드(10의배수)'
        WHEN amt = FLOOR(amt)                       THEN '정수'
        ELSE '일반'
    END AS amt_shape,
    COUNT(*)                       AS tx_cnt,
    SUM(is_fraud)                  AS fraud_cnt,
    ROUND(AVG(is_fraud) * 100, 3)  AS fraud_pct
FROM transactions
GROUP BY amt_shape
ORDER BY fraud_pct DESC;

-- E2. "카드테스트 -> 실사용" 시퀀스: 직전 거래가 소액($5 미만)이고
--     5분 이내에 현재 거래가 고액($200 이상)인 경우
WITH seq AS (
    SELECT
        amt,
        is_fraud,
        LAG(amt) OVER (PARTITION BY cc_num ORDER BY trans_date_trans_time) AS prev_amt,
        TIMESTAMPDIFF(
            SECOND,
            LAG(trans_date_trans_time) OVER (PARTITION BY cc_num ORDER BY trans_date_trans_time),
            trans_date_trans_time
        ) AS gap_sec
    FROM transactions
)
SELECT
    CASE WHEN prev_amt < 5 AND amt >= 200 AND gap_sec <= 300
         THEN '카드테스트의심' ELSE '일반' END AS test_pattern,
    COUNT(*)                       AS tx_cnt,
    SUM(is_fraud)                  AS fraud_cnt,
    ROUND(AVG(is_fraud) * 100, 3)  AS fraud_pct
FROM seq
GROUP BY test_pattern;

-- ============================================================
-- F. 연속거래(짧은 시간창) 패턴
-- ============================================================

-- F1. 직전 거래 후 10분 이내 재결제 여부별 사기율
WITH seq2 AS (
    SELECT
        is_fraud,
        TIMESTAMPDIFF(
            SECOND,
            LAG(trans_date_trans_time) OVER (PARTITION BY cc_num ORDER BY trans_date_trans_time),
            trans_date_trans_time
        ) AS gap_sec
    FROM transactions
)
SELECT
    CASE WHEN gap_sec IS NOT NULL AND gap_sec <= 600 THEN '10분이내재결제' ELSE '아님' END AS rapid_repeat,
    COUNT(*)                       AS tx_cnt,
    SUM(is_fraud)                  AS fraud_cnt,
    ROUND(AVG(is_fraud) * 100, 3)  AS fraud_pct
FROM seq2
GROUP BY rapid_repeat;

-- ============================================================
-- 참고: 이 스크립트의 모든 쿼리는 MariaDB 10.11에 fraudTrain/fraudTest
-- 스키마와 동일한 합성 데이터를 실제로 올려서 문법·실행 오류가 없는 것까지
-- 확인했다. 다만 합성 데이터라 merchant/job/city_pop 다양성이 부족해서
-- 결과 자체는 의미 없다 -- 진짜 185만 건 raw 데이터로 돌려야 실제 패턴이
-- 보인다.
-- ============================================================
