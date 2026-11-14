# 임의로 정한 값·가정 기록

| 날짜 | 무엇 | 왜 | 틀리면 |
|---|---|---|---|
| 2026-11-14 | 폴더 구조를 `backend/`·`frontend/`·`android/`·`analysis/`·`docs/`로 나눔 (backend 안의 `MOCA`·`modeling`·`pengteu`·`core` 패키지 이름은 유지) | 패키지 이름을 바꾸면 import 수정 범위가 커져서 이동만 함 | 기능별 이름(cognitive, gait 등)을 원하면 후속 작업 필요 |
| 2026-11-14 | Render 설정에서 `rootDir`를 빼고 `gunicorn --chdir backend` 로 실행 | `backend/`가 `../frontend`를 읽어야 해서 저장소 전체가 보여야 함 | Render 대시보드에 Root Directory가 따로 설정돼 있으면 배포 시 화면 파일을 못 찾음 |
| 2026-11-14 | `backend/pengteu/knowledge/*.md`는 `docs/`로 옮기지 않음 | RAG가 실행 중에 이 경로에서 직접 읽는 데이터임 | 옮기면 펭트 답변 근거 검색이 비게 됨 |
| 2026-11-14 | 분석 스크립트 2개의 임상 라벨 CSV 경로를 `ROOT.parent` 아래 glob 검색으로 바꿈 | 개인 PC 절대 경로 제거, 다른 스크립트와 같은 방식 | 같은 이름 파일이 여러 개면 첫 번째를 집음 |
| 2026-11-14 | Vosk 음성 모델 zip(83MB)은 올리지 않음 | 저장소가 무거워지고 GitHub 50MB 경고 대상 | 웨이크워드 기능을 쓰려면 모델을 따로 받아 `android/app/src/main/assets/`에 넣어야 함 |
| 2026-11-14 | `OfflineGaitAnalyzer.kt`, `VoskWakeWordEngine.kt`는 파일만 추가하고 `MainActivity`에 연결하지 않음 | 연결 코드는 로컬 7월 버전 `MainActivity`에 있고 main 버전과 달라 합치려면 따로 비교가 필요함 | 지금은 빌드만 되고 앱에서 쓰이지 않음 |
| 2026-11-14 | Vosk·JNA 버전(0.3.75, 5.18.1)은 로컬 `build.gradle.kts` 값을 그대로 씀 | 로컬에서 쓰던 조합 | 버전 문제 생기면 갱신 필요 |
| 2026-11-14 | 해커톤 참가신청서 docx, 발표 PDF는 올리지 않음 | 신청서에 이름 등 개인정보 항목이 있고, PDF(38MB)는 같은 발표 pptx의 중복 | 필요하면 따로 올려야 함 |
| 2026-11-14 | haji 브랜치에서 main과 내용이 다른 파일(`app.py`, 모델 joblib 등)은 가져오지 않고, main에 없는 파일만 가져옴 | main 쪽이 9~10월 최신 작업으로 보임(날짜 기준 추정) | haji 쪽에만 있던 수정이 빠졌을 수 있음 |
