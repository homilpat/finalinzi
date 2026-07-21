# 인지·운동 평가 기반 맞춤형 이중과제 솔루션 프로토타입

고령자의 인지 및 운동 기능을 함께 평가하고, 평가 결과를 바탕으로 개인별 이중과제 훈련을 추천하기 위한 연구·시연용 프로토타입입니다.

현재 구현된 인지 평가는 한국판 Montreal Cognitive Assessment(MoCA-K / K-MoCA) 설문을 기반으로 하며, 웹브라우저에서 문항 진행, 음성 응답 수집, STT 변환, 자동 점수화, 결과 요약까지 이어지는 흐름을 제공합니다. 운동 평가는 추후 보행, 균형, 반응속도 등 기능평가 모듈과 연동할 수 있도록 확장 구조로 설계되어 있습니다.

본 시스템은 진단 목적이 아닌 연구·시연용이며, 공식 MoCA 검사 또는 공식 채점 시스템을 대체하지 않습니다.

---

## 프로젝트 방향

- **인지 평가**: MoCA-K / K-MoCA 7개 인지 영역 자동 채점 (터치·음성·캔버스 입력)
- **신체 평가**: 스마트폰 IMU 센서 기반 보행 CSV 업로드 → Flask 서버에서 acc-only 3피처 보행 모델 추론
- **AI 코치**: Agentic AI (LLM API + Function Calling) 기반 대화형 코치 — 상황 인식 후 TTS 속도·글자 크기 등 앱 동적 제어
- **결과 활용**: 인지·보행 평가 결과 통합 → 개인별 이중과제 훈련 추천
- **목적**: 의료 진단이 아닌 비상업적 연구, 포트폴리오, 기술 시연

---

## 시스템 기술 요약

### 인지평가 (K-MoCA / MoCA-K, 총 30점)

| 인지능력 영역 | 검사 항목 / 기술 |
|---|---|
| 시공간 / 집행기능 | 길만들기·육면체·시계 그리기 — 컴퓨터 비전 (HoughLines, HoughCircles) + 터치 경로 검증 |
| 이름대기 (어휘력) | 동물 3종 이름 말하기 — STT + Fuzzy Matching |
| 기억력 | 단어 즉각·지연 회상 — STT + Fuzzy Matching |
| 주의력 | 숫자·탭핑·연속계산 — STT + 숫자 파싱 |
| 언어능력 | 따라말하기·단어 유창성 — SequenceMatcher + 화이트리스트 |
| 추상력 | 두 사물 공통점 찾기 — STT + Fuzzy Matching |
| 지남력 | 날짜·요일·장소 말하기 — STT + Regex |

### 신체평가 (보행 스크리닝)

| 평가 항목 | 기술 |
|---|---|
| IMU 기반 보행 분석 | 스마트폰 가속도계 기반 100 Hz 보행 CSV |
| 서버 신호처리 | 100 Hz 리샘플 + 0.6~3.0 Hz Butterworth 밴드패스 |
| 특징 추출 | `v_jerk_rms_median`, `v_jerk_rms_iqr`, `v_harmonic_ratio_iqr` |
| 수집 프로토콜 | 20초 이상 걷기 → 20s/10s 서브윈도우 집계 |
| ML 추론 | 로지스틱 회귀 — acc-only 클린 재학습 모델 (AUC 0.861 ± 0.020, threshold 0.345) |

### 공통 기능

| 기능 | 기술 |
|---|---|
| Agentic AI 코치 | LLM API + Function Calling — 대화 응답과 앱 제어 동시 실행 |
| 실시간 UI 동적 제어 | Tool Use → Android 네이티브 API (TTS 속도·글자 크기 자동 조절) |
| 맥락 인식 자동 대응 | Context-aware — 무응답·발화 감지 → 즉각 처리 or LLM 분기 |
| 지식 검색·주입 | RAG (문항 설명·보행 가이드 청크) |
| 노인 음성 특화 STT | Whisper 파인튜닝 — AI Hub 노인 음성 200h → WER 48%→22% |
| 음성 입출력 | Android STT / TTS |
| 버전 자동 로테이션 | MoCA-K ↔ K-MoCA 6개월 주기 (학습효과 차단) |

---

## 빠른 시작

```bash
cd C:\Users\whdgu\Desktop\MOCA
python app.py
# → http://localhost:5000
```

> 코드 수정 후 반드시 수동 재시작 필요 (`use_reloader=False`)  
> 재시작 시 진행 중 세션 초기화됨 (시연용이므로 허용)

---

## 현재 인지 평가 파이프라인

```
환자 입력 (터치/음성/캔버스)
        │
        ▼
   [Flask 웹앱 app.py]
        │
        ├─ STT (Whisper) ──────────────── 음성 → 텍스트
        ├─ 캔버스 이미지 (base64)
        └─ 터치 좌표 리스트
        │
        ▼
   [채점 모듈 8종]
        │
        ├─ trail_making.py   ← 터치 좌표
        ├─ cube.py           ← 캔버스 이미지 (룰베이스 / CNN 폴백)
        ├─ clock.py          ← 캔버스 이미지 (룰베이스 / CNN 폴백)
        ├─ naming.py         ← STT + fuzzy matching
        ├─ memory.py         ← STT + fuzzy matching
        ├─ attention.py      ← STT / 터치 탭핑
        ├─ language.py       ← STT + SequenceMatcher / 화이트리스트
        ├─ abstraction.py    ← STT + fuzzy matching
        └─ orientation.py    ← STT + 날짜/장소 파싱
        │
        ▼
   [total_scorer.py]
   교육 보정 (+1점) → 인지 저하 위험 신호 참고 지표
```

---

## 파일 구조

```
MOCA/
├─ app.py                    Flask 웹 서버 (시연용, 포트 5000)
│
├─ 채점 모듈
│  ├─ total_scorer.py        통합 채점 진입점 (30점)
│  ├─ trail_making.py        길만들기 (1점)
│  ├─ cube.py                정육면체 (1점)
│  ├─ clock.py               시계 (3점)
│  ├─ naming.py              어휘력 (3점)
│  ├─ memory.py              기억력 (5점)
│  ├─ attention.py           주의력 (6점)
│  ├─ language.py            언어 (3점)
│  ├─ abstraction.py         추상력 (2점)
│  └─ orientation.py         지남력 (6점)
│
├─ 보조 모듈
│  ├─ version_manager.py     MoCA-K / K-MoCA 버전 설정 (단일 진실 공급원)
│  ├─ session_manager.py     검사 흐름 + 5분 대기 관리
│  └─ whisper_stt.py         Whisper STT 래퍼
│
├─ CNN 관련
│  ├─ clock_cnn_inference.py DeepC / DeepH / DeepN 추론
│  ├─ clock_cnn_train.py     시계 U-Net 학습 스크립트
│  ├─ cube_cnn_inference_v2.py KabakusCNN256 추론
│  ├─ cube_cnn_train_v2.py   정육면체 CNN 학습 스크립트
│  ├─ deepc.pth              DeepC 모델 (7.4MB, 시계 원 분할)
│  ├─ deeph.pth              DeepH 모델 (7.4MB, 시계 바늘 분할)
│  ├─ deepn.pth              DeepN 모델 (3.3MB, 숫자 — 미사용)
│  └─ cube_model.pth         KabakusCNN256 (257MB, 학습 중단)
│
├─ 웹앱 정적 파일
│  ├─ templates/             HTML (base/home/item/waiting/result)
│  ├─ static/css/style.css   모바일 퍼스트 CSS
│  ├─ static/js/app.js       TTS체이닝 / STT / 캔버스 / 멀티스텝
│  └─ static/images/         lion.png / rhino.png / camel.png
│
├─ assets/tts/               TTS mp3 51개 (generate_tts.py로 생성)
├─ market_whitelist.txt      시장물건 화이트리스트 (~550개)
├─ Clock Drawing Test.v6i.yolov8/  Roboflow CDT 데이터셋 v6
│
└─ 문서
   ├─ CLAUDE.md              개발 상세 기록 (설계결정/버그/논문근거)
   ├─ README.md              이 파일
   ├─ MoCA_파이프라인_기술문서_v3.docx  제출용 기술문서
   ├─ references.md          참고논문 목록
   ├─ MOCA-K.pdf / K-MOCA.pdf          공식 검사지
   └─ MOCA-K평가기준.pdf / K-MOCA평가기준.pdf  공식 채점기준
```

---

## 모듈별 구현 상태

### 채점 모듈

| 모듈 | 배점 | 상태 | 비고 |
|---|---|---|---|
| `trail_making.py` | 1 | 완료 | 터치 시퀀스 검증, 직후수정 허용 |
| `cube.py` | 1 | 완료 | HoughLinesP 룰베이스, CNN 폴백 |
| `clock.py` | 3 | 완료 | 원/숫자/바늘 룰베이스, CNN 폴백 |
| `naming.py` | 3 | 완료 | fuzzy matching (threshold=0.65) |
| `memory.py` | 5 | 완료 | 즉각회상 기록 + 지연회상 채점 |
| `attention.py` | 6 | 완료 | 숫자파싱 / 탭핑 / serial-7 |
| `language.py` | 3 | 완료 | 따라말하기 + 시장물건/ㄱ초성 유창성 |
| `abstraction.py` | 2 | 완료 | fuzzy 정답 / exact 오답 |
| `orientation.py` | 6 | 완료 | 날짜 regex + 장소 suffix 처리 |

### CNN 모델

| 모델 | 역할 | 상태 |
|---|---|---|
| `deepc.pth` | 시계 원 U-Net 분할 | 존재 (val 6장, 과적합 위험) |
| `deeph.pth` | 시계 바늘 U-Net 분할 | 존재 (Precision≈0.48, 과적합) |
| `deepn.pth` | 숫자 분류 | **미사용** — 10/11/12 인식 불가, 룰베이스로 대체 |
| `cube_model.pth` | 정육면체 이진분류 | 학습 중단 (257MB), 룰베이스 사용 중 |

---

## 주요 기술 결정

### STT 오인식 보정 — Fuzzy Matching
노인 ASR WER 22~48% (JAMIA 2023) 대응.  
`SequenceMatcher.ratio() ≥ 0.65` 로 1음절 오류 허용.  
숫자(`attention`), 날짜(`orientation`)는 별도 파싱으로 처리.

### MoCA-K / K-MoCA 버전 자동 교체
`version_manager.py`가 단일 진실 공급원.  
6개월 주기 버전 교체로 학습 효과 차단.  
기억 단어·문장·추상력 쌍·탭핑 대상 등 모든 차이를 config dict로 관리.

### 시계 숫자 채점 — 룰베이스 (DeepN 대체)
MNIST 기반 DeepN이 10/11/12 인식 불가 → 30° 섹터 분포로 대체.  
12개 구역 중 10개 이상 점유 시 "순서대로 제자리에" 통과.

### 단어 유창성 — 화이트리스트
`market_whitelist.txt` (~550단어) `frozenset` 조회로 ms 수준 응답.  
미등록 단어는 `unknown` 필드로 반환(운영자 검토용).  
`use_llm=True` 시 unknown만 Claude Haiku API로 배치 검증.

---

## 수정 이력 (버그 픽스)

### 2026-06-30

| 버그 | 원인 | 수정 |
|---|---|---|
| STT 채점 전부 0점 | JS가 `{response:{...}}` 래핑 → 서버에서 키 못찾음 | `/submit`에서 `data.get('response', data)` 언래핑 추가 |
| 세션 갑자기 초기화 | Flask auto-reloader가 `_store` 초기화 | `use_reloader=False` 설정 |
| 5분 대기 후 abstraction으로 돌아감 | fetch 방식 + delayed_recall 중복 체크 | `/skip_to_delayed` GET 라우트로 교체 |
| 시계 숫자 채점 항상 0점 | `RETR_EXTERNAL`이 원 내부 컨투어를 1개만 반환 | `RETR_LIST`로 교체 + 마스크 반지름·거리필터 완화 |
| 시침 각도 오류 | 이미지 좌표계 미반영 (60° → 실제 240°) | `score_hands_deeph` 타겟 240°로 수정 |
| HoughLinesP 방향 비결정 | 선분 양 끝점 순서가 랜덤 | 중심에서 끝점 방향으로 normalize |
| session_manager delayed_recall 건너뜀 | 대기 체크를 `item_index += 1` 이후에 함 | 체크를 증가 이전으로 이동 |

---

## 향후 작업

| 항목 | 우선순위 | 비고 |
|---|---|---|
| CNN 재학습 (DeepC/DeepH) | 높음 | 원본 26장 → 증강으로 300~500장 확보 후 재학습 |
| Whisper 파인튜닝 | 중간 | AI Hub 노인 음성 200시간, WER 48%→22% 기대 |
| cube_model.pth 재학습 | 중간 | QuickDraw + 실제 손그림 데이터 확보 필요 |
| market_whitelist 보완 | 낮음 | 운영 중 `unknown` 필드 모니터링 |
| DB 확장 | 낮음 | 시연용 SQLite 저장 이후, 운영 환경용 DB와 평가 이력 조회 기능 확장 |

### CDT 학습 데이터 현황

- **보유**: Roboflow v6 — 원본 26장 (72장은 3배 증강한 것)
- **추가 후보**: `cdt-acaci/cdt-ejwb1` — 2980장이나 분류 전용 여부 미확인
- **권장**: 기존 26장에 회전/반전/밝기/엘라스틱 증강 → 300~500장 확보

---

## 환경 의존성

```
flask
opencv-python
torch
numpy
difflib (표준라이브러리)
openai-whisper        # STT (없으면 STT 기능 비활성)
anthropic             # use_llm=True 시 단어유창성 검증용
python-docx           # generate_doc.py
gTTS                  # generate_tts.py
```

---

## 채점 기준 출처

© Z. Nasreddine MD, JY. Lee 한국판  
www.mocatest.org  
총점 30점 / 참고 기준 23점 미만 / 교육연수 6년 이하 +1점

---

## 고지사항 / Disclaimer

본 프로젝트는 인지·운동 평가 결과를 바탕으로 개인별 이중과제 훈련 추천 구조를 탐색하기 위한 비상업적 연구·시연용 프로토타입입니다. 현재 인지 평가는 한국판 Montreal Cognitive Assessment(MoCA-K / K-MoCA) 설문 기반 자동평가 흐름으로 구현되어 있습니다.

본 프로젝트는 연구, 교육, 포트폴리오, 기술 시연 목적에 한해 작성되었으며, 공식 MoCA 제품이 아니고 MoCA 저작권자 또는 관련 기관의 승인·보증을 받은 것이 아닙니다. 또한 면허를 받은 임상 평가나 전문 의료진의 진단을 대체할 수 없습니다.

MoCA 검사지, 문항, 채점 기준, 명칭 및 관련 자료의 권리는 각 권리자에게 있습니다. 저작권 또는 상표권 관련 문제가 있을 경우 저장소 관리자에게 연락해 주시면 해당 자료를 즉시 삭제하거나 수정하겠습니다.

본 프로젝트는 상업적 이용, 유료 선별검사 서비스, 재판매, 진단 목적 배포를 의도하지 않습니다.

This project is a non-commercial research and demonstration prototype for exploring a personalized dual-task training workflow based on cognitive and motor assessment results. The currently implemented cognitive assessment flow uses Korean Montreal Cognitive Assessment (MoCA-K / K-MoCA) questionnaire-based automated scoring.

It is intended solely for non-commercial research, education, and technical demonstration purposes. It is not an official MoCA product, is not affiliated with or endorsed by the MoCA copyright holders, and must not be used as a substitute for a licensed clinical assessment or professional medical diagnosis.

The MoCA test materials, scoring criteria, names, and related content belong to their respective rights holders. If any copyrighted material or trademarked content is used inappropriately, please contact the repository owner and it will be removed or modified promptly.

No commercial use, resale, paid screening service, or diagnostic deployment is intended.
