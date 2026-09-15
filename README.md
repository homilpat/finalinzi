[시연영상](https://youtu.be/FUdJCf0dYKs)
# Finalinzi — AI 시니어 건강 케어 프로토타입

스마트폰 한 대로 인지 선별, 20초 보행 선별, 맞춤형 운동 코칭을 연결하는 Android·Web 통합 프로젝트입니다. Android 앱은 센서와 음성을 담당하고, Flask 서버는 기록 저장, 보행 모델 추론, RAG 검색과 펭트 답변을 담당합니다.

> 본 프로젝트는 질병을 진단하거나 치료를 결정하지 않습니다. 검사 결과는 위험군 선별을 돕는 정보이며, 증상이 지속되거나 악화되면 결과와 관계없이 의료진 상담을 권합니다.

## 프로젝트 개요

| 구분 | 내용 |
|---|---|
| 진행 기간 | 2026.06.01 ~ 2026.07.29 |
| 진행 인원 | 4인 (신인류팀) |
| 담당 역할 | 보행 데이터 전처리·모델링, Flask 서버, 펭트 AI 코치(RAG·규칙 라우터), Android 앱 연동 |

## 주요 기능

- MoCA-K 기반 인지기능 선별 및 교육 수준 보정
- 스마트폰 가속도계·자이로를 이용한 20초 보행 CSV 수집
- 서버의 3개 보행 피처 모델을 이용한 운동기능 저하 선별 보조
- 같은 스마트폰 센서를 이용한 실시간 운동 동작 감지와 피드백
- 펭트 STT·TTS, 운동 재설명, 글자 크기·고대비·음성 속도·볼륨 제어
- 인지검사 STT와 펭트 STT의 마이크 점유 충돌 방지
- 논문 출처·페이지·DOI를 포함하는 로컬 TF-IDF RAG와 OpenAI gpt-4.1-mini 선택 연결
- 사용자·보호자 화면, 검사 기록과 운동 출석 관리

## 시스템 구조

```text
Android APK
├─ WebView: Flask UI
├─ 인지검사·펭트 네이티브 STT/TTS
├─ 보행검사: 20초 IMU → CSV → 서버 업로드
└─ 운동: IMU 실시간 전처리 → 동작 감지 → SensorBridge → UI 피드백
                │ HTTPS
                ▼
Flask / Render
├─ 사용자 기록과 세션
├─ 보행 CSV 전처리·모델 추론
├─ 논문 RAG 검색
└─ 펭트 답변: OpenAI API 또는 안전한 규칙 기반 폴백
```

노트북 서버나 LM Studio를 켜둘 필요는 없습니다. APK는 운영 Render 서버에 연결됩니다. 다만 로그인, 기록 저장, 보행 추론과 RAG/LLM에는 인터넷 연결이 필요합니다.

## 기술 스택

- Android: Kotlin, WebView, SensorManager, SpeechRecognizer, TextToSpeech
- Backend: Python 3.11, Flask, SQLite
- ML: NumPy, SciPy, pandas, scikit-learn 1.5.0
- RAG: Markdown 지식베이스, TF-IDF 검색, 출처·페이지·DOI 메타데이터
- Frontend: Jinja2, HTML, CSS, JavaScript
- Deployment: Render, Gunicorn
- Testing: unittest, Flask test client, Gradle build, 브라우저 운동 흐름 시뮬레이션

## 보행 모델

운영 모델은 `giukhaji/models/gait_daily_clinical_3feat.joblib`입니다. Android 형식의 20초 CSV를 해부학적 축(V/ML/AP)으로 정렬해 100 Hz로 리샘플링하고, 스마트폰–허리 IMU 진폭 차이를 고정 계수 `α = 1.9705`로 보정한 뒤 0.6–3.0 Hz 대역통과 필터와 10초 서브윈도우(2초 스텝) Median·IQR 집계를 수행합니다.

스마트폰과 PhysioNet 허리 IMU 사이의 진폭 분포 차이를 줄이기 위해 피처 추출 전에 V/ML/AP 가속도 시계열 전체에 동일한 단일 보정계수 `α = 1.9705`를 곱합니다. `α`는 정상 보행 수직축 대역통과 RMS 중앙값의 비율(`0.193863 / 0.098382`)로 산출했습니다. 센서의 최대·최소값을 맞추는 Min-Max 정규화나 축마다 서로 다른 값을 적용하는 축별 보정이 아닙니다.

- `v_jerk_rms_median`: 움직임 충격(V Jerk RMS)의 대표값
- `v_jerk_rms_iqr`: 움직임 충격의 변동성
- `v_harmonic_ratio_iqr`: 좌우 보행 대칭성(V ACF Symmetry)의 변동성

피험자 단위 5-fold × 100회 반복 검증(각 학습 fold 안에서 "민감도 0.80 이상 중 특이도 최대" 기준으로 임계값 결정)에서 AUC 0.873 ± 0.007, 민감도 0.835, 특이도 0.731입니다. 서비스 배포 임계값은 0.50으로 고정합니다.

학습 데이터는 PhysioNet 고령자 장기 활동 기록으로, 71명(정상 31 / 저하 40)이 허리에 IMU를 착용하고 약 72시간 생활한 연속 신호입니다. 저하 라벨은 임상 검사 6종의 OR 조건(TUG 12초 이상, FSST 15초 이상, BERG 52점 미만, DGI 19점 이하, 기저·3단계 보행속도 1.0 m/s 미만)으로 정의했습니다. 사전 선별한 20초 보행 구간을 1인당 최대 100개까지 사용하고, 구간별 Median·IQR을 다시 피험자 단위 중앙값으로 접어 1인 1행으로 학습해 같은 사람이 학습과 검증에 동시에 들어가지 않게 했습니다. 세 피처의 VIF는 1.54~2.17입니다.

Optuna로 로지스틱 회귀 하이퍼파라미터를 탐색한 결과 AUC 0.833 ± 0.034, 민감도 0.816, 특이도 0.702로 기본 설정보다 낮고 fold 간 편차도 커서, 71명 규모에서는 기본 설정을 유지했습니다.

결과는 질병 확률이나 확진 결과가 아니라 운동기능 저하 위험군 선별 보조 점수로 사용합니다. 최종 모델의 학습·검증 재현 코드는 `analysis_scripts/`에 있고, 최종 모델에 쓰이지 않은 탐색 실험과 이전 모델 코드는 `analysis_archive/`에 보관했습니다.

## 펭트 RAG

`giukhaji/pengteu/knowledge/`의 문서를 제목 단위로 분할해 검색합니다. 논문 본문 근거를 우선하고 `References`·`참고문헌` 절은 검색 청크에서 제외합니다. 결과에는 출처, 원문 페이지와 DOI가 포함됩니다.

현재 MoCA 선별, 고령자 모바일 보행 측정, 이중과제 균형·보행훈련, 낙상위험 고령자 보행훈련 근거가 포함됩니다. `OPENAI_API_KEY`가 있으면 검색 근거와 사용자 기록을 LLM에 전달하며, 키가 없거나 호출이 실패하면 규칙 기반 안전 답변으로 전환합니다. 검사 중에는 MoCA 정답이나 힌트를 제공하지 않습니다.

### 접근성 명령과 능동 안내

"글씨가 잘 안 보여", "좀 천천히 말해줘"처럼 화면·음성 조절이 필요한 발화는 LLM보다 먼저 클라이언트의 결정론적 라우터(`getPengteuCommand()`)가 처리합니다. 글자 크기·음성 속도·볼륨·고대비·화면 움직임 줄이기를 범위 안에서 조정해 사용자 프로필(`/assistant/profile`)에 저장하고, 채팅 서버와 LLM을 거치지 않고 즉시 반영합니다.

사용자가 먼저 도움을 요청하지 않아도 검사 화면에서 10초간 입력이 없거나 STT가 2회 연속 실패하면 펭트가 먼저 조절 방법을 안내합니다. 문항 음성 재생·답변 녹음 중과 운동 화면에서는 발동하지 않으며, 무입력 안내는 세션당 2회로 제한합니다.

## 로컬 서버 실행

Python 3.11 환경을 권장합니다.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r giukhaji\requirements.txt
Copy-Item giukhaji\.env.example giukhaji\.env
python giukhaji\app.py
```

필수 환경변수:

```text
SECRET_KEY=<충분히 긴 임의 문자열>
PHONE_HASH_SALT=<전화번호 해시용 별도 임의 문자열>
```

선택 환경변수는 `OPENAI_API_KEY`, `OPENAI_ASSISTANT_MODEL`, `MOCA_DB_PATH`입니다. 실제 키와 개인정보는 저장소에 커밋하지 마세요.

## 테스트

```powershell
python -m unittest discover -s tests -v
python -m compileall -q giukhaji tests
node --check giukhaji\static\js\app.js
node --check giukhaji\static\js\exercise.js
```

통합 테스트는 Android 형식 합성 보행 CSV 추론, Flask 업로드, RAG 메타데이터, 참고문헌 제외, 펭트 안전 문구, 마이크 브리지와 개인정보 해시 저장을 검증합니다.

## Android APK 빌드

Android SDK 36과 Java 11 이상이 필요합니다.

```powershell
cd FinalProjectApp
.\gradlew.bat assembleDebug
```

산출물은 `FinalProjectApp/app/build/outputs/apk/debug/app-debug.apk`에 생성됩니다. 디버그 APK는 시연용이며 스토어 배포에는 별도 릴리스 서명이 필요합니다.

## 저장소 구성

```text
FinalProjectApp/       통합 Android 앱
giukhaji/              Flask 서비스, UI, 런타임 모델, RAG
analysis_scripts/      최종 보행 모델 학습·검증 재현 스크립트
analysis_archive/      최종 모델에 쓰이지 않은 탐색 실험, 이전 모델 코드·문서
tests/                 배포·통합 회귀 테스트
render.yaml            Render 배포 설정
DEPLOYMENT.md           배포 절차
DEVELOPMENT_ROADMAP.md  변경 이력, 검증 결과와 제한점
SECURITY.md             보안 및 개인정보 처리 원칙
```

## 검증 상태와 제한점

- Python 통합 테스트 5개 통과
- Android `assembleDebug` 성공
- Render `/health` 응답 확인
- 브라우저에서 운동 유형·단계 전환, TTS와 타이머 흐름 확인
- 합성 센서 신호로 보행 업로드·추론 경로 확인

실기기가 없어 제조사별 STT/TTS, 마이크 권한, 실제 센서 샘플링과 장착 방향에 따른 운동 인식 정확도는 아직 검증하지 못했습니다. 실제 서비스 전에는 목표 사용자 공동 데이터와 실기기를 이용한 외부 검증이 필요합니다.

## 보안

- 전화번호 원문 대신 salt를 적용한 해시를 저장합니다.
- `.env`, DB, CSV, 엑셀, 쿠키, 키스토어와 원본 임상 데이터는 Git에서 제외합니다.
- HTTPS 운영 주소와 보안 세션 쿠키를 사용합니다.
- 테스트 전화번호와 테스트 키 문자열은 실제 개인정보나 운영 비밀값이 아닙니다.

자세한 변경 및 검증 기록은 [DEVELOPMENT_ROADMAP.md](DEVELOPMENT_ROADMAP.md)를 참고하세요.
