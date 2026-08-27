# FDS 위험거래 탐지 데모 웹

Tableau 대시보드와 별개로, "이 모델이 실제로 잘 작동합니다"를 라이브로 보여주기 위한 아주 단순한 데모 웹앱.
로그인/DB/페이지 이동 없음 — CSV 업로드(or 샘플 버튼) → 로딩 → 결과, 딱 이거 하나만 함.

## 구조

```
webapp/
├── backend/
│   ├── main.py                 # FastAPI: 모델 로드 + /api/predict, /api/predict-sample
│   ├── requirements.txt
│   ├── model/
│   │   ├── final_lightgbm_combination3.joblib
│   │   └── final_lightgbm_combination3_config.json
│   └── data/
│       └── demo_sample.csv     # Test 3구간(홀드아웃)에서 뽑은 40건 샘플 (정상 25 / 이상 15)
└── frontend/
    ├── index.html
    ├── style.css
    └── script.js
```

## 실행 방법

```bash
cd webapp/backend
pip install -r requirements.txt
python -m uvicorn main:app --reload
```

브라우저에서 http://127.0.0.1:8000 접속.

## 지금 되는 것 / 아직 안 되는 것

- ✅ 확률(%) + 정상/이상거래 판정 + 5단계 위험등급(일반/관찰/관심/위험/긴급)
- ✅ 정답(`is_fraud`)이 포함된 CSV면 실제 정답과 비교해서 일치 여부까지 표시
- ✅ **28개 유형(위험등급 × 군집 7유형) 분류까지 완료** — 민정님한테 받은 K=7 군집 모델(`lightGBM_combination3_K7_cluster_model.joblib`) 연동함. 일반 등급은 "해당없음(일반)", 그 외 등급은 군집 유형명(예: "심야 고액 반복형")까지 같이 표시됨.

## 업로드용 CSV 요구 조건

아래 컬럼이 있어야 함 (이미 가공된 피처, `fraudTest_full_features.csv` 계열과 동일):

```
category, amt, trans_hour, age, recent_24h_high_amt_count,
amt_to_prior_median_ratio, rolling_sum_amt_1h, amt_zscore_card,
prior_normal_median_amt, count_30min, high_speed
```

`is_fraud` 컬럼은 선택사항(있으면 정답 비교 표시), `trans_date_trans_time`도 선택사항(있으면 시간 표시).
