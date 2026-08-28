# 발전 기록

## 2026-08-28 인지검사·펭트 음성 경합 방지

### 완료

- 인지검사 STT와 펭트 STT가 Android 마이크를 동시에 점유하지 않도록 상호 중단 처리를 추가했다.
- JavaScript 마이크 소유권 전환이 브라우저 인식기뿐 아니라 Android 네이티브 인식기도 정지하도록 연결했다.
- 사용자가 중단한 펭트 STT에는 오류 안내가 나오지 않게 했고, 인지검사 STT 오류의 자동 재시작 반복을 차단했다.
- 보행 안내가 펭트 TTS를 끊어도 웹의 발화 상태가 해제되도록 네이티브 TTS 중단 콜백을 추가했다.

### 검증 결과

- Python 배포 스모크 테스트, JavaScript 문법 검사와 변경 diff 검사를 수행했다.
- Android STT 상호 중단 브리지와 TTS 중단 콜백 존재 여부를 정적 테스트에 추가했다.

### 제한점

- Android SDK와 실기기가 없어 실제 `SpeechRecognizer` 동시 전환과 제조사별 TTS 콜백은 실행 검증하지 못했다.

### 다음 우선순위

1. Android SDK가 있는 환경에서 APK를 빌드한다.
2. 인지검사 녹음 중 펭트 호출과 펭트 청취 중 검사 재개 시나리오를 에뮬레이터 또는 실기기에서 확인한다.

## 2026-08-28 하이브리드 클라우드 전환

### 완료

- 최신 기능이 포함된 원격 `haji` 브랜치를 기준으로 `deploy-hybrid` 작업 브랜치를 만들었다.
- Android와 센서 앱의 기본 주소를 `https://finalinzi.onrender.com`으로 변경했다.
- Flask 세션 키와 전화번호 해시 salt의 고정 기본값을 제거했다.
- Render 상태 확인 `/health`와 운영 HTTPS 세션 쿠키를 추가했다.
- 펭트 STT뿐 아니라 인지검사 문항 STT도 Android 네이티브 브리지로 연결했다.
- 보행 CSV·DB·RAG·LLM은 서버에 두고 운동 실시간 센서와 동적 UI/UX는 앱에서 처리하는 구조를 유지했다.

### 검증 결과

- Flask·DB·배포 설정 통합 테스트 2개를 통과했다.
- Python 문법 검사, JavaScript `node --check`, `git diff --check`를 통과했다.
- 임시 SQLite DB에서 전화번호 원문이 저장되지 않음을 확인했다.
- Android 네이티브 인지검사 STT 브리지의 Kotlin·JavaScript 연결을 정적 검사했다.

### 제한점

- 실제 Android 컴파일과 마이크 실기기 검증에는 Android SDK와 테스트 단말이 필요하다.
- 기본 SQLite는 Render 무료 인스턴스에서 영구 저장을 보장하지 않는다.
- 서버 RAG·LLM 고급 답변에는 운영 API 키 설정이 필요하다.

### 다음 우선순위

1. 배포 브랜치를 GitHub에 올리고 Render 서비스의 브랜치·환경 변수를 갱신한다.
2. 배포 성공 후 `/health`, 로그인, 보행 업로드 API를 원격에서 검증한다.
3. 운영 URL을 사용하는 release APK를 빌드하고 마이크·센서 전체 흐름을 실기기에서 검증한다.
