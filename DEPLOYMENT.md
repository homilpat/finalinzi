# 클라우드 서버 및 Android 연결

## 운영 서버

- Render 서비스: `https://finalinzi.onrender.com`
- 애플리케이션 루트: `giukhaji`
- 상태 확인: `/health`
- 배포 기준: GitHub의 `deploy-hybrid` 브랜치

운영 환경에는 `SECRET_KEY`, `PHONE_HASH_SALT`, `ACCESS_PASSWORD`를 설정한다.
`OPENAI_API_KEY`는 서버 RAG·LLM 고급 답변을 사용할 때만 설정한다.

기본 SQLite는 무료 Render 인스턴스의 재배포 후 보존을 보장하지 않는다. 실제
사용자 이력을 유지하려면 영구 디스크 또는 외부 PostgreSQL로 전환해야 한다.

## Android

`FinalProjectApp`은 기본적으로 운영 HTTPS URL을 사용한다. 개발 서버를 사용할 때만
Git에서 제외되는 `FinalProjectApp/local.properties`에 다음 값을 설정한다.

```properties
serverUrl=http://10.0.2.2:5000/
```

release 빌드는 HTTPS가 아닌 서버 주소를 거부한다. 펭트 STT와 인지검사 답변 STT는
Android 네이티브 `SpeechRecognizer`를 사용하므로 Render가 마이크 스트림을 직접
처리하거나 저장하지 않는다.
