# analysis_archive

최종 보행 모델(`giukhaji/models/gait_daily_clinical_3feat.joblib`)을 정하기까지 거친 탐색 실험과 이전 모델 자료입니다. 배포 앱과 최종 모델 학습에는 쓰이지 않습니다. 최종 모델 재현 스크립트는 `analysis_scripts/`에 있습니다.

| 경로 | 내용 |
|---|---|
| `scripts/` | 피처·윈도우 길이·도메인 보정·라벨 후보 탐색, smoke 테스트, 속도 없는 이전 라벨 실험 |
| `final__2026/` | 이전 실험실 10초 보행·피처 4개 모델 (train fold Youden 임계값 0.524) |
| `docs/` | 이전 축정렬 모델 요약 문서 (임계값 0.56) |
| `tools/` | 위 요약 문서(docx) 생성 스크립트 |

스크립트 대부분은 이 저장소에 없는 PhysioNet 원본·중간 산출물(`analysis_outputs/`)과 이전 폴더명(`MOCA/`)을 전제로 작성돼 있어 그대로 실행되지 않을 수 있습니다.
