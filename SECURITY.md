# 보안 및 개인정보 처리 원칙

- 전화번호 원문은 저장하지 않고 운영 환경의 `PHONE_HASH_SALT`를 사용해 해시한다.
- `SECRET_KEY`, `PHONE_HASH_SALT`, `ACCESS_PASSWORD`, API 키는 환경 변수로만 설정한다.
- 로컬 SQLite DB, 업로드 센서 CSV, 로그와 `.env`는 Git에서 제외한다.
- 운영 앱은 HTTPS 서버만 사용하며 마이크 음성 원본은 서버에 저장하지 않는다.
- `giukhaji/data/`의 검사 원응답과 센서 디버그 파일은 공개하지 않는다.

본 서비스 결과는 인지·운동기능 선별 보조 정보이며 진단이나 치료 결정을 대신하지 않는다.
