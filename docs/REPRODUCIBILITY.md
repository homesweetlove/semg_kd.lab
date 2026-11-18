# 재현 가이드

## 저장된 결과 표 재계산

이 저장소에는 결과 표를 검산할 사람별·seed별 오차, 학습 시간 요약, 실제 체크포인트의 추론 반복 로그와 확인 스크립트가 있습니다. 원본 sEMG 특징 배열과 학습 가중치는 전체 재현 묶음에서 사용합니다.

```sh
python -m pip install numpy==2.5.3
python scripts/verify_results.py
```

출력은 사람 30명, 모델 6종, 학생 seed 3개, 직접/증류 2조건의 1,080개 오차와 교사 180개 오차를 확인합니다. 사람 평균 MAE·RMSE, 짝지은 조건부 신뢰구간, 6종 보정 구간, 210개 모델의 추론 시간 집계를 재계산해 저장된 값에 대조합니다.

## 모델과 학습 실행

`semg_distillation_study/`에는 해당 시점의 학습·평가 코드와 입력 정의가 있습니다. 데이터와 부모 실행, 교사 및 학생 가중치까지 확인하려면 **전체 재현 묶음 `SEMG_SIX_FAMILY_FULL.zip`**을 펼친 뒤, 그 안의 `semg_distillation_study` 폴더에서 실행하세요.

```sh
python -B run_six_family.py
python -B audit_six_family.py
python -B report_six_family.py
python -B verify_six_family_examples.py
```

완료된 묶음의 첫 명령은 보존한 학습을 확인하고 종료합니다. 두 번째 명령은 체크포인트와 입력을 확인하고 추론 시간을 다시 측정합니다. 로컬에서 검증한 환경은 Windows, Python 3.14.7, PyTorch 2.14.0+cpu, NumPy 2.5.3입니다. 학습 코드는 실행 환경, 특징 데이터, 부모 실행과 구현 파일의 해시가 프로토콜과 일치하는지 확인합니다.

| 기존 로컬 파일 | 역할 | SHA-256 |
|---|---|---|
| `SEMG_SIX_FAMILY_AI_CONTEXT.zip` | 보고서, 코드, 첫 분할의 교사·직접·증류 가중치 18개와 예제 | `d085f57ae9926f6e7238c782607f0a8d1b3089bed21228926411059282f39e48` |
| `SEMG_SIX_FAMILY_FULL.zip` | 특징 데이터, 부모 실행, 전체 체크포인트 및 재현 기록 | `9ac6424b4798c5a22e4df997d5a155fd34430e92c2e35ec2e3c914612e4fb8e4` |

두 묶음은 실험을 실행한 컴퓨터의 `semg-robustness-followup/outputs`에 있습니다. 실제 파일의 해시와 새 폴더에서 수행한 자동 검사 54개, 완료 상태 재실행, 모델 18개 × 실제 입력 64개의 확인 결과는 [검증 기록](../results/six-family/delivery_verification.json)에 있습니다.
