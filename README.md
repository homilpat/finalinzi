[시연영상](https://youtu.be/FUdJCf0dYKs)

# Finalinzi — AI 시니어 건강 케어 프로토타입

스마트폰 한 대로 **인지 선별(MoCA-K)**, **20초 보행 선별**, **맞춤 운동 코칭**을 잇는 Android·Web 통합 프로젝트입니다. Android 앱은 센서와 음성을, Flask 서버는 기록 저장·보행 모델 추론·RAG 검색·펭트(AI 코치) 답변을 맡습니다.

> 본 프로젝트는 질병을 진단하거나 치료를 결정하지 않습니다. 검사 결과는 위험군 선별을 돕는 정보이며, 증상이 지속되거나 악화되면 결과와 관계없이 의료진 상담을 권합니다.

| 구분 | 내용 |
|---|---|
| 진행 기간 | 2026.06.01 ~ 2026.07.29 |
| 진행 인원 | 4인 (신인류팀) |
| 담당 역할 | 보행 데이터 전처리·모델링, Flask 서버, 펭트 AI 코치(RAG·규칙 라우터), Android 앱 연동 |
| 운영 주소 | https://finalinzi.onrender.com |

## 최종 보행 모델 한눈에 보기

| 항목 | 값 |
|---|---|
| 모델 | 로지스틱 회귀 (기본 설정) — `backend/models/gait_daily_clinical_3feat.joblib` |
| 입력 피처 | 가속도만 쓰는 3개: `v_jerk_rms_median`, `v_jerk_rms_iqr`, `v_harmonic_ratio_iqr` |
| 학습 데이터 | PhysioNet 고령자 72시간 허리 IMU, **71명** (정상 31 / 저하 40), 1인 1행 |
| 저하 라벨 | TUG ≥ 12초 · FSST ≥ 15초 · BERG < 52 · DGI ≤ 19 · 기저/3단계 보행속도 < 1.0 m/s 중 하나 이상 |
| 임계값 | **0.50 고정** (민감도 0.80 이상 우선) |
| 검증 성능 | 피험자 단위 5-fold × 100회: **AUC 0.873 ± 0.007, 민감도 0.835, 특이도 0.731** |
| 센서 보정 | 스마트폰↔허리 IMU 진폭 차이를 단일 계수 `α = 1.9705`로 보정 |

수치별 출처와 재현 스크립트는 [docs/analysis/results.md](docs/analysis/results.md)에 있습니다.

## 주요 기능

- MoCA-K 기반 인지기능 선별과 교육 수준 보정, 인지검사 STT
- 스마트폰 가속도계·자이로로 20초 보행 CSV 수집 → 서버 모델로 운동기능 저하 선별 보조
- 같은 센서로 실시간 운동 동작 감지와 음성 피드백
- 펭트 AI 코치: STT·TTS, "펭트야" 호출어, 글자 크기·고대비·음성 속도·볼륨 조절
- 논문 출처·페이지·DOI를 포함하는 로컬 RAG + OpenAI 선택 연결 (키가 없으면 규칙 기반 안전 답변)
- 사용자·보호자 화면, 검사 기록과 운동 출석 관리

## 시스템 구조

```text
Android 앱 (online)                     Android 앱 (offline 베타)
├─ WebView: Flask 화면                  ├─ 폰 안의 화면 (assets/offline)
├─ 인지검사·펭트 STT/TTS                 ├─ 폰 안에서 보행 분석 (OfflineGaitAnalyzer)
├─ 보행: 20초 IMU → CSV → 서버 업로드     ├─ Vosk 로컬 "펭트야" 호출어
└─ 운동: IMU 실시간 감지 → 화면 피드백     └─ 인터넷 권한 없음
        │ HTTPS
        ▼
Flask (Render)
├─ 사용자 기록과 세션 (SQLite)
├─ 보행 CSV 전처리·모델 추론
├─ 논문 RAG 검색
└─ 펭트 답변: OpenAI API 또는 규칙 기반 폴백
```

## 저장소 구성

```text
backend/                Flask 서버
  app.py                라우트·앱 진입점
  MOCA/                 인지 선별(MoCA) 채점
  modeling/             보행·운동 센서 처리와 추론
  models/               배포 보행 모델·보정값
  pengteu/              펭트 AI 코치 (RAG, STT, 지식 문서 knowledge/)
  core/                 DB, 세션, 설정 공통 모듈
frontend/               웹 화면
  templates/            Jinja 화면 템플릿
  static/               CSS·JS·이미지·영상
  assets/tts/           안내 음성 파일
android/                Android 앱 (online / offline 두 버전)
  app/src/main/         두 버전 공통 리소스
  app/src/online/       서버 연결 버전 MainActivity
  app/src/offline/      오프라인 베타 MainActivity, 보행 분석기, Vosk 호출어, 오프라인 화면
analysis/
  scripts/              최종 보행 모델 학습·검증 재현 스크립트
  results/              재현 실험 결과 표
  archive/              최종 모델에 쓰이지 않은 탐색 실험, 이전 모델 코드
docs/                   모든 문서 (배포, 로드맵, 보안, 화면 디자인, 분석·보행 기록, 발표 자료)
tests/                  배포·통합 회귀 테스트
render.yaml             Render 배포 설정
```

## 보행 모델 상세

### 전처리
Android 형식 20초 CSV를 해부학적 축(V/ML/AP)으로 정렬하고 100 Hz로 리샘플링합니다. 피처 추출 전에 V/ML/AP 가속도 전체에 같은 계수 `α = 1.9705`를 곱합니다. `α`는 정상 보행 수직축 대역통과 RMS 중앙값 비율(PhysioNet `0.193863` / 스마트폰 `0.098382`)이며, Min-Max 정규화나 축별 보정이 아닙니다. 이후 0.6–3.0 Hz 대역통과 필터와 10초 서브윈도우(2초 스텝) Median·IQR로 집계합니다.

### 피처

| 피처 | 의미 | 정상 중앙값 | 저하 중앙값 |
|---|---|---|---|
| `v_jerk_rms_median` | 움직임 충격(V Jerk RMS)의 대표값 | 1.767 | 1.105 |
| `v_jerk_rms_iqr` | 움직임 충격의 변동성 | 0.132 | 0.073 |
| `v_harmonic_ratio_iqr` | 좌우 보행 대칭성(V ACF Symmetry)의 변동성 | 0.054 | 0.076 |

세 피처의 VIF는 1.54~2.17입니다. 보행속도는 라벨에만 쓰고 입력 피처에는 넣지 않아 직접적인 데이터 누수를 피했습니다.

### 학습 방식
사전 선별한 20초 보행 구간을 1인당 최대 100개 쓰고, 구간별 Median·IQR을 다시 피험자 단위 중앙값으로 접어 1인 1행으로 학습했습니다. 같은 사람이 학습과 검증에 동시에 들어가지 않습니다.

### 왜 이 모델·임계값인가

| 비교 (5-fold × 100회) | 결과 |
|---|---|
| 9개 모델 비교 | LR AUC 0.873 ± 0.008로 1위, 편차도 최소 (Voting 0.864, RF 0.858, GBM·1D-CNN 0.857 …) |
| Optuna 튜닝 LR | AUC 0.833 ± 0.034, 100회 중 90회에서 기본 설정보다 낮음 → 기본 설정 유지 |
| 임계값 후보 | 민감도 0.80 이상 후보 중 고정 0.50이 정확도 0.790·F1 0.817로 가장 높음 |
| 제외한 임계값 | class_weight balanced(민감도 0.789), Youden(민감도 0.730) — 민감도 기준 미달 |
| 피처 조합 385개 | 배포 조합 AUC 0.873 (8위). 1위 4개 조합은 0.879로 차이가 작아 설명하기 쉬운 3개 조합 선택 |

fold 안에서 임계값을 다시 정하는 방식(민감도 0.80 이상 중 특이도 최대)으로 평가하면 AUC 0.873 ± 0.008, 민감도 0.811, 특이도 0.746입니다.

결과는 질병 확률이나 확진이 아니라 **운동기능 저하 위험군 선별 보조 점수**로만 씁니다. 값이 계산되지 않으면 예전 모델로 대체하지 않고 다시 측정하도록 안내합니다.

## 펭트 RAG

`backend/pengteu/knowledge/`의 문서를 제목 단위로 나눠 검색합니다. 논문 본문 근거를 우선하고 `References`·`참고문헌` 절은 검색에서 뺍니다. 결과에는 출처, 원문 페이지와 DOI가 붙습니다.

지식 문서 주제: MoCA·경도인지장애 선별, 고령자 모바일 보행 측정, 허리 IMU 보행분석과 설명가능 AI, 이중과제 운동 효과, 낙상위험 고령자 보행훈련, 음성·언어 인지 지표, 보행 모델 설명.

`OPENAI_API_KEY`가 있으면 검색 근거와 사용자 기록을 LLM에 넘기고, 없거나 실패하면 규칙 기반 안전 답변으로 바뀝니다. 검사 중에는 MoCA 정답이나 힌트를 주지 않습니다.

### 접근성 명령과 능동 안내
"글씨가 잘 안 보여", "좀 천천히 말해줘" 같은 발화는 LLM보다 먼저 클라이언트 라우터(`getPengteuCommand()`)가 처리해 글자 크기·음성 속도·볼륨·고대비·화면 움직임을 바로 바꾸고 사용자 프로필(`/assistant/profile`)에 저장합니다. 검사 화면에서 10초간 입력이 없거나 STT가 2회 연속 실패하면 펭트가 먼저 조절 방법을 안내합니다(세션당 2회까지, 문항 재생·녹음 중과 운동 화면에서는 제외).

## 실행 방법

### 서버

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r backend\requirements.txt
Copy-Item backend\.env.example backend\.env
python backend\app.py
```

필수 환경변수는 `SECRET_KEY`, `PHONE_HASH_SALT`이고, 선택은 `OPENAI_API_KEY`, `OPENAI_ASSISTANT_MODEL`, `MOCA_DB_PATH`, `ACCESS_PASSWORD`입니다. 실제 키와 개인정보는 커밋하지 마세요. 운영(Render)은 Python 3.10.6으로 `gunicorn --chdir backend app:app`을 실행합니다.

### Android

Android SDK 36과 Java 11 이상이 필요합니다.

```powershell
cd android
.\gradlew.bat assembleOnlineDebug    # 서버 연결 버전
.\gradlew.bat assembleOfflineDebug   # 오프라인 베타
```

결과물은 `android/app/build/outputs/apk/<online|offline>/debug/`에 생깁니다. 개발 서버에 붙이려면 `android/local.properties`에 `serverUrl=http://<내 서버 IP>:5000/`을 넣습니다.

offline 베타의 "펭트야" 호출어는 Vosk 한국어 모델(`vosk-model-small-ko-0.22.zip`, 약 83MB)이 필요합니다. 용량 때문에 저장소에 넣지 않았으니 [Vosk 모델 페이지](https://alphacephei.com/vosk/models)에서 받아 `android/app/src/offline/assets/`에 넣으세요. 없으면 호출어만 꺼지고 나머지 기능은 동작합니다.

### 테스트

```powershell
python -m unittest discover -s tests -v
python -m compileall -q backend tests
node --check frontend\static\js\app.js
node --check frontend\static\js\exercise.js
```

테스트는 합성 보행 CSV 추론, Flask 업로드, RAG 메타데이터, 참고문헌 제외, 펭트 안전 문구, 마이크 브리지, 전화번호 해시 저장, 저장소 안 절대 경로 여부를 확인합니다.

## 검증 상태와 제한점

- Python 테스트 6개 통과
- Android online·offline `assembleDebug` 빌드 성공
- Render `/health` 응답 확인
- 브라우저에서 운동 유형·단계 전환, TTS와 타이머 흐름 확인
- 합성 센서 신호로 보행 업로드·추론 경로 확인

제조사별 STT/TTS, 마이크 권한, 실제 센서 샘플링, 장착 방향에 따른 운동 인식 정확도는 실기기로 충분히 검증하지 못했습니다. 보행 모델은 71명 공개 데이터로 학습·내부 검증만 했으므로, 실제 서비스 전에는 목표 사용자 데이터와 실기기로 외부 검증이 필요합니다.

## 보안

- 전화번호 원문 대신 salt를 적용한 해시를 저장합니다.
- `.env`, DB, CSV, 엑셀, 쿠키, 키스토어와 원본 임상 데이터는 Git에서 제외합니다.
- HTTPS 운영 주소와 보안 세션 쿠키를 사용합니다.
- 테스트 전화번호와 테스트 키 문자열은 실제 개인정보나 운영 비밀값이 아닙니다.

## 문서

| 문서 | 내용 |
|---|---|
| [docs/deployment.md](docs/deployment.md) | 배포 절차 |
| [docs/roadmap.md](docs/roadmap.md) | 변경 이력, 검증 결과와 제한점 |
| [docs/SECURITY.md](docs/SECURITY.md) | 보안·개인정보 처리 원칙 |
| [docs/analysis/results.md](docs/analysis/results.md) | 보행 모델 수치 출처 |
| [docs/gait/](docs/gait/) | 보행 피처 공식, 화면 표시 문구 |
| [docs/open-decisions.md](docs/open-decisions.md) | 임의로 정한 값·가정 기록 |
