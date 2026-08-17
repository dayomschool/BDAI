# Fraud FDS 최종 Test 데이터셋 재현

## 1. 필요 환경

- Python 3.10 이상 권장
- pandas
- numpy

설치 명령:

```bash
python -m pip install pandas numpy
```

## 2. 폴더 구조

압축을 푼 뒤 아래 구조를 유지합니다.

```text
fraud_test_reproduction/
├─ run_reproduce_test_dataset.py
├─ test_build/
│  ├─ step4a_current_features.py
│  ├─ step4b_dynamic_features.py
│  ├─ step4c_fixed_label_history.py
│  └─ step4d_finalize_test.py
└─ upload/
   ├─ fraudTrain(1).csv
   ├─ fraudTest(1).csv
   └─ fraud_full_features(1).csv
```

원본 CSV 3개는 용량 때문에 패키지에 포함하지 않았습니다. 직접 `upload` 폴더에 넣어야 합니다.

`fraud_full_features(1).csv`는 완성 Train 데이터셋이며, 최종 Test의 31개 컬럼명과 순서를 맞추는 기준으로만 사용합니다.

## 3. VS Code에서 실행

1. 압축을 풉니다.
2. VS Code에서 압축을 푼 폴더를 엽니다.
3. 터미널에서 아래 명령을 실행합니다.

```bash
python run_reproduce_test_dataset.py
```

Windows에서 `python` 명령이 연결되지 않으면 다음을 사용합니다.

```bash
py run_reproduce_test_dataset.py
```

## 4. 최종 결과

프로젝트 루트에 다음 파일이 생성됩니다.

```text
fraudTest_full_features.csv
```

정상 결과:

- 행 수: 555,719
- 열 수: 31
- Train 완성본과 컬럼명 및 순서 일치
- 무한대 값 없음
- 결측 허용 열: `prior_normal_median_amt`, `amt_to_prior_median_ratio`

## 5. 단계별 역할

1. `step4a_current_features.py`
   - Test 정답 `is_fraud` 즉시 분리
   - 현재 거래만 사용하는 변수 생성
2. `step4b_dynamic_features.py`
   - Train 상태와 앞선 Test 입력정보만 사용
   - 거래 이력·고객 통계·속도·누적금액 변수 생성
3. `step4c_fixed_label_history.py`
   - Train의 확정된 `is_fraud` 이력만 사용
   - 업종별 사기율과 정상거래 기준 변수를 Test 전체에 적용
   - Test의 `is_fraud`는 사용하지 않음
4. `step4d_finalize_test.py`
   - 모든 파생변수 계산 완료 후 평가용 `is_fraud` 복원
   - Train과 동일한 31개 컬럼 순서로 저장
   - 저장 후 CSV 재로딩 검증

## 6. 주의사항

- 입력 CSV의 파일명을 임의로 변경하지 마세요.
- 단계별 파일만 따로 실행하기보다 메인 실행기를 사용하세요.
- `test_build` 폴더의 `.pkl` 파일은 실행 중 자동 생성되는 중간 체크포인트입니다.
- 최종 모델 입력에서는 `is_fraud`를 반드시 제외해야 합니다.
