# 임의로 정한 값·가정 기록

| 날짜 | 무엇 | 왜 | 틀리면 |
|---|---|---|---|
| 2026-11-14 | 폴더 구조를 `backend/`·`frontend/`·`android/`·`analysis/`·`docs/`로 나눔 (backend 안의 `MOCA`·`modeling`·`pengteu`·`core` 패키지 이름은 유지) | 패키지 이름을 바꾸면 import 수정 범위가 커져서 이동만 함 | 기능별 이름(cognitive, gait 등)을 원하면 후속 작업 필요 |
| 2026-11-14 | Render 설정에서 `rootDir`를 빼고 `gunicorn --chdir backend` 로 실행 | `backend/`가 `../frontend`를 읽어야 해서 저장소 전체가 보여야 함 | Render 대시보드에 Root Directory가 따로 설정돼 있으면 배포 시 화면 파일을 못 찾음 |
| 2026-11-14 | `backend/pengteu/knowledge/*.md`는 `docs/`로 옮기지 않음 | RAG가 실행 중에 이 경로에서 직접 읽는 데이터임 | 옮기면 펭트 답변 근거 검색이 비게 됨 |
| 2026-11-14 | 분석 스크립트 2개의 임상 라벨 CSV 경로를 `ROOT.parent` 아래 glob 검색으로 바꿈 | 개인 PC 절대 경로 제거, 다른 스크립트와 같은 방식 | 같은 이름 파일이 여러 개면 첫 번째를 집음 |
