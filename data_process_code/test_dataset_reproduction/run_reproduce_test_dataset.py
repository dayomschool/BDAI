"""Fraud FDS 최종 Test 데이터셋 전체 재현 실행기.

실행 전 프로젝트 루트의 upload 폴더에 다음 파일 3개를 둔다.
  - fraudTrain(1).csv
  - fraudTest(1).csv
  - fraud_full_features(1).csv

실행:
  python run_reproduce_test_dataset.py
"""

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
STEPS = [
    ROOT / "test_build" / "step4a_current_features.py",
    ROOT / "test_build" / "step4b_dynamic_features.py",
    ROOT / "test_build" / "step4c_fixed_label_history.py",
    ROOT / "test_build" / "step4d_finalize_test.py",
]


def main():
    required = [
        ROOT / "upload" / "fraudTrain(1).csv",
        ROOT / "upload" / "fraudTest(1).csv",
        ROOT / "upload" / "fraud_full_features(1).csv",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("필수 입력 파일이 없습니다:\n- " + "\n- ".join(missing))

    for number, script in enumerate(STEPS, start=1):
        print(f"\n[{number}/{len(STEPS)}] {script.name} 실행")
        subprocess.run([sys.executable, str(script)], cwd=ROOT, check=True)

    output = ROOT / "final_test_features_31cols.csv"
    print("\n전체 재현 완료")
    print(f"최종 파일: {output}")


if __name__ == "__main__":
    main()
