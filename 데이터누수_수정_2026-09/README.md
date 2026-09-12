# 데이터 누수 발견 및 수정 (2026-09)

제출 이후 발견된 실제 배포 모델의 데이터 누수 문제를 조사하고 고친 기록.

## 무엇이 문제였나

조합3(11개 변수) 중 `amt_zscore_card`(및 내부적으로 쓰는 `customer_mean_amt`, `customer_std_amt`)가
`전처리_코드_나겸.ipynb`에서 `groupby("cc_num").transform("mean"/"std")`로 계산됨 —
shift 없이 **카드의 전체 거래(미래 포함) 평균/표준편차**를 쓰는 명백한 시점 누수.

나머지 4개(`recent_24h_high_amt_count`, `amt_to_prior_median_ratio`, `prior_normal_median_amt`,
`rolling_sum_amt_1h`)는 개별적으로는 각 코드가 미래를 보진 않았지만, **나겸님 원본 코드 /
민정님이 나중에 만든 코드 / Test셋 재현 코드, 이렇게 3개 버전의 계산식이 서로 미묘하게
달라서(train-test 불일치)** 함께 통일 대상이 됨.

## 처리 과정

1. `01_build_leakfree_features_민정.ipynb` — 민정님이 5개 변수를 전부 시점 인과적으로
   재계산하는 로직 작성 (`shift(1).expanding()`, `closed='left'` rolling, 카드별 온라인
   러닝메디안 등). `05_run_build_leakfree_실행스크립트.py`로 실제 원본 데이터(fraudTrain.csv/
   fraudTest.csv)에 돌려서 `train_11features.csv`(1,296,675행), `test_11features.csv`
   (555,719행) 재생성 — 검증 3종 모두 통과.
2. `02_cv_3fold_timeseries_민정_버그수정.ipynb` — 원본에 `eval_X=`/`eval_y=`라는 LightGBM에
   실존하지 않는 인자가 있어 실행 시 에러가 남 → `eval_set=[(X_valid, y_valid)]`로 수정.
   재실행 결과 민정님이 보고한 Fold별 PR-AUC(0.954 / 0.971 / 0.977, 평균 0.968)와 정확히 일치.
3. `06_hp_search.py` — 4개 하이퍼파라미터 조합을 시간순 3-Fold로 비교. 최적:
   `num_leaves=31, min_child_samples=50, learning_rate=0.05, reg_alpha=0.1, reg_lambda=1.0`
   (avg PR-AUC 0.9690, 조합 간 성능 차이는 크지 않음 — 하이퍼파라미터에 민감하지 않은 안정적인
   피처셋이라는 뜻).
4. `07_final_train_eval.py` — OOF 예측으로 임계값 확정(F1-max, threshold=0.893092),
   전체 Train으로 최종 모델 재학습, **한 번도 학습에 안 쓴 진짜 Test셋**으로 평가.

   | | OOF(Train) | Test(홀드아웃) |
   |---|---|---|
   | PR-AUC | 0.9704 | 0.9620 |
   | Precision | 0.9371 | 0.8493 |
   | Recall | 0.9275 | 0.9357 |
   | F1 | 0.9323 | 0.8904 |

   차이(gap) +0.0083 → **과적합 징후 없음.** 누수를 없애도 성능은 여전히 견고함.
5. `08_rebuild_tableau_leakfree.py` — 새 모델로 Train 전체 확률(3-Fold OOF + 첫 구간만
   최종모델로 보완) / Test 확률(최종모델)을 계산하고, 5단계 위험등급 + 기존 K=7 군집모델
   (누수 변수에 의존하지 않는 걸 확인해서 그대로 재사용)로 `tableau_export_leakfree.csv`
   재생성. **EMERGENCY_THRESHOLD도 새 모델의 정밀도 곡선을 다시 봐서 0.9918 → 0.999로
   재검증**(threshold=0.999에서 precision 99.94%로 급격히 올라가는 지점).

## 반영 현황

- ✅ 웹 데모(`webapp/`) — `final_lightgbm_leakfree.joblib`로 교체, `main.py`의
  `EMERGENCY_THRESHOLD`도 0.999로 갱신, `demo_sample.csv`도 leak-free 데이터로 재생성.
  기존 모델은 `final_lightgbm_combination3_DEPRECATED_leaky.joblib`로 이름만 남겨둠(삭제 안 함).
- ⬜ Tableau 대시보드 — `tableau_export_leakfree.csv`는 생성 완료(용량 문제로 이 리포에는
  미포함, 로컬 `tableau/` 폴더 참고). 워크북의 데이터 원본 연결을 이 파일로 바꾸고
  새로고침하는 건 Tableau Desktop에서 수동으로 해야 함(GUI 작업이라 자동화 불가).

## 참고: 이 폴더에 없는 큰 파일들

`train_11features.csv`, `test_11features.csv`, `tableau_export_leakfree.csv`는 용량
문제로(기존 `data/`, `tableau/` 폴더와 동일하게) 이 리포에 포함하지 않음 — 위 스크립트를
`data/fraudTrain.csv`, `data/fraudTest.csv`가 있는 상태에서 순서대로 실행하면 재현 가능.
