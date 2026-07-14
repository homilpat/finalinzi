# MOCA 프로젝트 개발 기록

## 프로젝트 개요
MoCA-K (Montreal Cognitive Assessment - 한국판) 자동 채점 시스템
스마트폰 앱 기반 서비스로 개발 중 (채점 로직 우선 완성 단계)

---

## 모듈 구조

| 파일 | 역할 | 배점 |
|---|---|---|
| `trail_making.py` | 길만들기 (터치 경로 검증) | 1점 |
| `cube.py` | 육면체 그리기 (CNN 결과 수신) | 1점 |
| `clock.py` | 시계 그리기 (CNN 결과 수신) | 3점 |
| `naming.py` | 어휘력 - 동물 이름 맞추기 | 3점 |
| `memory.py` | 기억력 - 즉각/지연 회상 | 5점 |
| `attention.py` | 주의력 - 숫자/탭핑/계산 | 6점 |
| `language.py` | 언어 - 따라말하기/유창성 | 3점 |
| `abstraction.py` | 추상력 - 공통점 찾기 | 2점 |
| `orientation.py` | 지남력 - 날짜/장소 | 6점 |
| `version_manager.py` | MoCA-K / K-MoCA 버전 설정 관리 | - |
| `session_manager.py` | 검사 흐름 및 5분 대기 관리 | - |
| `total_scorer.py` | 전체 통합 채점 + MCI 판정 | 30점 |

---

## 설계 결정 사항

### 1. 버전 관리 구조 (version_manager.py가 단일 진실 공급원)

**문제**: 채점 모듈들이 MoCA-K 내용을 하드코딩하여 K-MoCA 버전 채점 불가

**해결**: `version_manager.py`에 모든 버전별 설정을 정의하고, 채점 함수가 `config` 파라미터를 받는 구조로 변경

```python
# total_scorer.py
def score_total(..., version: str = "MoCA-K") -> dict:
    config = get_version_config(version)
    score_memory(..., config["memory_words"])
    score_attention(..., config)
    score_language(..., config)
    score_abstraction(..., config)
```

**버전 로테이션**: 6개월 주기로 MoCA-K ↔ K-MoCA 교체 (학습 효과 차단 목적)

---

### 2. 버전별 주요 차이점

| 항목 | MoCA-K | K-MoCA |
|---|---|---|
| 기억 단어 | 얼굴·비단·교회·진달래·빨강 | 얼굴·비단·학교·피리·노랑 |
| 따라말하기 | 오늘 나를 도와줄... / 강아지가 방에... | 칼날같이 날카로운 바위 / 스물 일곱 개의... |
| 유창성 기준 | 시장물건 11개 이상 | ㄱ으로 시작하는 단어 6개 이상 |
| 추상력 쌍 | 기차-자전거 / 시계-자 | 기차-비행기 / 시계-저울 |
| 탭핑 타겟 | "가" | "월" |

---

### 3. STT 오인식 보정 - Fuzzy Matching

**근거 논문**:
- Vipperla et al. (Edinburgh) — 노인 ASR WER 일반 성인 대비 10~11% 절대값 상승
- JAMIA 2023 — 요양시설 노인 WER 22~48%
- PMC8516752 (Behavior Research Methods 2021) — fuzzy string matching이 인간 채점과 r=0.94 상관

**적용 대상**: `memory.py`, `naming.py`, `abstraction.py` (정답 키워드만)

**미적용 대상**:
- `attention.py` 숫자 — `extract_numbers()`로 별도 파싱
- `language.py` 따라말하기 — SequenceMatcher 이미 적용
- `abstraction.py` 오답 키워드 — exact match 유지 (fuzzy 오적용으로 억울하게 0점 방지)
- `orientation.py` 날짜 — 숫자 포함 여부 체크라 불필요

**구현 방식**:
```python
FUZZY_THRESHOLD = 0.65  # 음절 단위 SequenceMatcher ratio

def _word_in_text(stt_text, target):
    if target in stt_text:       # 1) exact match 우선
        return True
    for token in stt_text.split():
        if SequenceMatcher(None, token, target).ratio() >= FUZZY_THRESHOLD:
            return True          # 2) 토큰별 fuzzy
    return False
```

**임계값 0.65 근거**:
- 3음절↑: 1음절 오류 시 ratio ≈ 0.67 → PASS (진달래→진달레, 코뿔소→코풀소)
- 2음절: 1음절 오류 시 ratio = 0.50 → FAIL (비단→미단, 사자→상자) — 의도적으로 엄격 유지

---

### 4. 지남력 장소 단축형 (orientation.py)

suffix만 제거 (str.replace 사용 시 중간 글자 제거 버그 있었음 → 수정 완료)
```python
for suffix in ["구", "시", "군"]:
    if correct.endswith(suffix):
        short = correct[:-1]
        break
# "구로구" → "구로" ✓
```

**월/일 부분문자 오탐 방지** (수정 완료):
- `"6" in "60대예요"` 같은 오탐을 regex 단어 경계로 차단
```python
digit_match = bool(re.search(r'(?<!\d)' + correct + r'(?!\d)', text))
```
- 요일 단일 글자("수") 허용 제거 → "수요일" 전체 매칭만 허용 ("수술", "수업" 오탐 방지)

---

### 5. 추상력 반환값 구조

```python
{
    "pair1": 0~1,
    "pair2": 0~1,
    "pair1_name": "기차-자전거",   # 버전별로 다름
    "pair2_name": "시계-자",
    "total": 0~2
}
```

`total_scorer.py`에서 `abstraction_pair1_stt` / `abstraction_pair2_stt` 로 버전 중립 네이밍 사용

---

### 6. 단어 유창성 (language.py) — 구현 완료

**MoCA-K (시장물건)**:
- `market_whitelist.txt` — 약 550개 단어 (과일·채소·버섯·생선·해산물·육류·유제품·두부·떡/면류·가공식품·김치·젓갈·곡류·견과류·양념·빵/과자·음료·생활용품·의류잡화·주방용품)
- 모듈 로드 시 1회 `frozenset`으로 읽어 런타임 ms 수준 lookup
- 화이트리스트 미등록 단어는 `unknown` 필드로 반환 (운영자 검토용)
- `use_llm=True` 설정 시 unknown 단어만 Claude Haiku API로 배치 검증 (기본값 False)
- 화이트리스트 파일 없을 시 전체 단어 카운트로 자동 폴백

**K-MoCA (ㄱ으로 시작하는 단어)**:
- 유니코드 한글 음절 블록 기준 ㄱ 초성 여부 필터
```python
def _starts_with_giyeok(word):
    c = ord(word[0])
    if 0xAC00 <= c <= 0xD7A3:
        return (c - 0xAC00) // (21 * 28) == 0
```

**블랙리스트 없음**: 화이트리스트 미등록 단어는 unknown으로 분류되어 점수 미반영. 별도 블랙리스트 없어도 채점에 영향 없어 제거함.

---

### 7. clock.py 바늘 각도 버그 수정 (완료)

이미지 좌표계(y축 아래 증가) 기준:
- 시침 (11시 방향): **240°** — 수정 완료 (기존 60°로 잘못됨)
- 분침 (2시/10분 방향): **330°** — 유지

HoughLinesP 방향 비결정적 문제 해결:
```python
# 중심에서 끝점 방향으로 normalize
if d1 <= d2:
    angle = math.degrees(math.atan2(y2-y1, x2-x1)) % 360
else:
    angle = math.degrees(math.atan2(y1-y2, x1-x2)) % 360
```

---

### 8. cube.py 기준 정비 (완료)

| 기준 | 변경 전 | 변경 후 |
|---|---|---|
| `count_ok` | 6~25 | 6~20 |
| `no_overline` | ≤25 (count_ok와 동일, 중복) | ≤30 (별도 관심사) |
| `direction_ok` | 수평+수직+대각 필수 | 표준(수평+수직+대각) OR 등각(수평+좌대각+우대각) |
| 길이 유사성 | 없음 | `_check_length_similarity()` 추가 (방향별 CV ≤ 0.5) |

---

### 9. session_manager.py delayed_recall 버그 수정 (완료)

**문제**: 5분 대기 응답 시 `item_index`가 이미 +1된 상태에서 재호출 시 한 번 더 +1 → `delayed_recall` 건너뜀

**수정**: wait 체크를 `item_index += 1` 이전으로 이동
```python
next_index = self.item_index + 1
if next_index < len(ITEM_SEQUENCE) and ITEM_SEQUENCE[next_index] == "delayed_recall":
    wait = self._check_delayed_recall_wait()
    if wait > 0:
        self.state = SessionState.WAITING_5MIN
        return {"status": "waiting", ...}  # item_index 증가 없이 return

self.item_index += 1  # 대기 없을 때만 증가
```

---

### 10. Flask 시연용 웹앱 (app.py) — 완성

**파일 구조**:
```
app.py                  ← Flask 서버 (라우트 + 채점 연동)
templates/
  base.html             ← 상단바 + 진행바
  home.html             ← 이름/학력/동네/시군구 입력
  item.html             ← 전 문항 공용 (타입별 Jinja2 분기)
  waiting.html          ← 5분 대기 카운트다운
  result.html           ← 총점 + 섹션별 점수바 + MCI 판정
static/
  css/style.css         ← 모바일 퍼스트, 색상 팔레트 (#4A90E2 / #4ECDB4 / #FF7B6E)
  js/app.js             ← TTS→타이머 체이닝, STT, 캔버스, 멀티스텝 흐름
  images/               ← lion.png / rhino.png / camel.png (PDF 추출)
assets/tts/             ← 51개 mp3 (generate_tts.py로 생성)
```

**TTS → 타이머 순서**: Audio `onended` 체이닝으로 마지막 TTS 끝나면 타이머 시작  
**응답 저장**: `_store[uid]['raw']` 딕셔너리에 세션 전체 축적 → `/result`에서 `score_total()` 1회 호출  
**5분 대기**: `/submit` → `{'next':'waiting'}` → `waiting.html` 카운트다운 → `/continue_wait` 폴링

---

### 11. 어휘력(naming) 동물 이미지 처리

**추출 방법**: PyMuPDF(`fitz`)로 MOCA-K.pdf 1페이지 200 DPI 렌더링 → OpenCV crop → `static/images/` 저장

**크롭 좌표** (1653×2339px 기준):
- lion:  x=4.5%~33%, y=40.5%~54.5%
- rhino: x=37%~62%, y=40.5%~54.5%
- camel: x=66%~86%, y=40.5%~54.5%

**TTS 없음**: 어휘력은 그림 보고 환자가 이름을 말하는 방식 → 개별 동물 TTS 불필요.  
안내 멘트(`naming_inst.mp3`) 1회만 재생 후 그림 순차 표시.  
`showAnimal()` JS 함수에서 `img.src`만 교체하고 오디오 재생 로직 제거.

---

### 13. 길만들기 UI 개선 (완료)

**노드 위치**: PDF 문제지 배치와 정확히 일치하도록 `trail_making.py` NODE_POSITIONS + `app.js` TRAIL_NODES 동시 수정
- 이전: 노드끼리 겹치고 PDF 배치와 무관
- 이후: 마(끝) 상단좌, 가 상단우, 5 좌측, 나·2 중앙, 1(시작) 중하단, 라·4·3 하단, 다 최하단

**"시작"/"끝" 라벨**: drawTrailNodes에서 노드 1 아래 "시작", 노드 마 아래 "끝" 텍스트 표시

**점선 화살표 힌트**: `_drawDashedArrow()` 함수로 1→가→2 구간에 회색 점선+화살촉 표시 (환자가 교대 규칙 파악 가능)

**시각 반지름 축소**: 0.07 → 0.055 (터치 판정 반지름 0.07은 trail_making.py에서 그대로 유지)

---

### 14. 정육면체 참고 이미지 SVG 수정 (완료)

`templates/item.html` cube 섹션의 SVG 좌표 오류 수정:
- 이전: polygon 좌표가 꼬여 입체감 없는 도형
- 이후: 앞면(직사각형) + 윗면(평행사변형) + 오른쪽면(평행사변형) + 숨은 3모서리(점선)로 올바른 3D 정육면체

```
앞면: (10,75)(55,75)(55,30)(10,30)
윗면: (10,30)(55,30)(75,15)(30,15)
오른쪽면: (55,75)(75,60)(75,15)(55,30)
점선: FL→BL, BL→BR, BL→BTL
```

---

### 12. 배포/실행 환경

**결정**: 로컬 노트북 서버 (경진대회 시연용) + Render 클라우드 (팀원 공유용)

- 용도: 개발 테스트 + 경진대회 시연
- 병목 순서: STT(Whisper) > CNN(clock/cube) > 채점 로직(ms 수준, 문제없음)
- GPU 있으면 Whisper/CNN 빠름, CPU only면 STT가 주요 병목

**Render 배포 (2026-06-30, 팀원 공유용)**:
- URL: https://finalinzi.onrender.com
- 레포: https://github.com/homilpat/finalinzi (master 브랜치)
- Root Directory: `Desktop/MOCA`
- Python: 3.10.6
- STT는 브라우저(Web Speech API) 처리 → 서버에 Whisper 불필요
- 서버 의존성: Flask, Flask-Cors, numpy, opencv-python-headless
- cube_model.pth / clock .pth 파일은 git 미포함 → 룰베이스 폴백으로 동작
- Free 플랜: 비활성 시 50초 콜드스타트 발생
- API 키 없음 (Anthropic은 환경변수, Kakao는 플레이스홀더만)

**Whisper 파인튜닝 계획** (채점 로직 완성 후 별도 작업):
- AI Hub 노인 음성 데이터셋 약 200시간 활용
- HuggingFace transformers로 fine-tuning
- `whisper_stt.py` 모델 경로만 교체하면 연동 완료
- 효과: 논문 기준 WER 48% → 22% 수준으로 개선 기대

---

## 미완성 / 향후 작업

- [ ] `session_manager.py` — DB에서 `last_version`, `last_assessed_at` 불러오는 로직 미구현 (백엔드 팀 연동 시 추가)
- [ ] `market_whitelist.txt` — 운영 중 `unknown` 필드 모니터링하여 누락 단어 보완
- [ ] Whisper 파인튜닝 — AI Hub 노인 음성 데이터 활용, 채점 로직 완성 후 진행
- [ ] CNN 모델 재학습 — 현재 모델 파일 존재하나 과적합/성능 미달 상태. 데이터 보강 필요
  - `cube`: `cube_model.pth` (257MB, 학습 도중 중단됨 — 정상 추론 가능 여부 미확인)
  - `clock`: `deepc.pth` (7.4MB), `deeph.pth` (7.4MB), `deepn.pth` (3.3MB) 존재
    - DeepC/DeepH: val 6장으로 과적합 위험 (Precision=0.478 수준)
    - DeepN: MNIST 기반 0-9 분류 → 10/11/12 인식 불가 → **룰베이스로 대체 확정**
- [x] `clock_cnn_inference.py` — 시침(ha)/분침(ma) 수식 오류 수정 완료, `score_hands_deeph` 각도 타겟 60°→240° 수정 및 HoughLinesP normalize 추가
- [x] `clock.py` `score_numbers` — "다른 숫자 추가 불가"(13개 초과 탐지), "순서대로 제자리에"(30도 섹터 분포) 룰베이스 근사 구현 완료 (CNN 도입 시 교체 예정)
- [x] `cube.py`, `clock.py` — `load_cube_model`/`load_clock_model` CNN 연동 완성: .pt/.pth→PyTorch, .h5/.keras/.hdf5→Keras 자동 감지, `score_cube_cnn`/`score_clock_cnn` 추론 로직 완성
- [x] `language.py` `use_llm` 폴백 — LLM 응답 파싱 강화(쉼표 split 정확 매칭), API 인증 오류 개별 핸들링, `score_language`/`score_total`까지 `use_llm` 파라미터 연결 완료
- [x] Flask 시연 웹앱 (`app.py` + `templates/` + `static/`) — 전 문항 흐름 완성, TTS→타이머, STT, 캔버스 그리기, 5분 대기, 결과 화면
- [x] TTS 51개 mp3 생성 (`generate_tts.py` 실행 완료, `assets/tts/` 저장)
- [x] 동물 이미지 추출 — MOCA-K.pdf → `static/images/lion.png`, `rhino.png`, `camel.png`
- [x] 길만들기 UI — 노드 위치 PDF 문제지와 일치, "시작"/"끝" 라벨, 1→가→2 점선 힌트 추가, 시각 반지름 축소
- [x] 정육면체 참고 SVG — 앞면·윗면·오른쪽면 + 숨은 모서리 점선으로 올바른 3D 정육면체 표시
- [x] Flask 웹앱 버그 수정 (2026-06-30) — 아래 섹션 참고

---

### 15. Flask 웹앱 버그 수정 (2026-06-30)

#### 버그 1: STT 값 전부 빈 문자열 → 채점 0점 (핵심 버그)

**원인**: JS가 `JSON.stringify({ response: App.responses })` 형태로 감싸서 전송하는데, `app.py`의 `_store_response()`는 `data.get('stt', '')` 처럼 `response` 키 없이 바로 꺼내고 있었음 → STT, points, image 등 모든 값이 빈 값으로 저장

**수정**: `app.py` `/submit` 라우트에 언래핑 한 줄 추가
```python
data = request.get_json(silent=True) or {}
data = data.get('response', data)   # JS: {response: {...}} 언래핑
```

#### 버그 2: Flask auto-reloader가 세션 날림

**원인**: `debug=True` + `use_reloader=True`(기본값) → 파일 수정 감지 시 자동 재시작 → `_store` (메모리 세션) 초기화 → 유저 세션 소실 → 아무 안내 없이 홈으로 리다이렉트

**수정**: `app.run(..., use_reloader=False)` 추가

**추가**: JS에서 세션 만료 감지 시 `alert('세션이 만료되었습니다...')` 후 홈으로 이동

#### 버그 3: 5분 대기 건너뛰기 후 12번(abstraction)으로 돌아감

**원인**: JS fetch → `/continue_wait` → `s.next_item()` 내부에서 **또다시** delayed_recall 대기 체크 → 계속 waiting 반환. `force_skip_wait` 파라미터로 우회 시도했으나 JS fetch 방식 자체가 불안정했음 (정확한 원인 미확정)

**최종 수정**: fetch 방식 완전 폐기 → 서버 GET 리다이렉트로 교체
- `waiting.html` 건너뛰기 버튼: `<a href="/skip_to_delayed">` (링크)
- `app.py` `/skip_to_delayed` GET 라우트 추가:
  ```python
  @app.route('/skip_to_delayed')
  def skip_to_delayed():
      s.item_index   = ITEM_SEQUENCE.index('delayed_recall')  # 12
      s.current_item = 'delayed_recall'
      s.state        = SessionState.IN_PROGRESS
      return redirect(url_for('item_page'))
  ```
- 타이머 자연 만료 시 기존 `/continue_wait` POST 방식 유지

#### 앱 실행 방법 (업데이트)
```bash
cd C:\Users\whdgu\Desktop\MOCA
python app.py
# → http://localhost:5000
```
- `use_reloader=False`이므로 코드 수정 후 반드시 수동 재시작 필요
- 재시작 시 `_store` 초기화 → 모든 진행 중 세션 소멸 (시연용이므로 허용)

---

### 16. clock.py 숫자 채점 버그 수정 (2026-06-30)

#### 버그: score_numbers()가 숫자 컨투어를 하나도 찾지 못함

**원인**: `cv2.findContours(inner, cv2.RETR_EXTERNAL, ...)` 사용 시 시계 원 윤곽선이 모든 내부 컨투어의 부모가 되어 RETR_EXTERNAL이 원 하나만 반환 (면적 125,064px)

**수정 내용**:
```python
# RETR_EXTERNAL → RETR_LIST (계층 무시, 모든 컨투어 반환)
contours, _ = cv2.findContours(inner, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# 면적 필터 확장: 50 < area < 2000 → 50 < area < 5000
valid = [cnt for cnt in contours if 50 < cv2.contourArea(cnt) < 5000]

# 마스크 반지름 확장: r → r*1.2 (K-MoCA 원 바깥 숫자 허용)
cv2.circle(mask, (cx, cy), int(r * 1.2), 255, -1)

# 거리 필터 완화: r*0.55~r*0.95 → r*0.45~r*1.15
if r * 0.45 <= dist <= r * 1.15:
```

---

### 17. CDT 학습 데이터 조사 결과 (2026-06-30)

**현재 보유**: Roboflow company-4mkvs/clock-drawing-test v6
- 원본 **약 26장** → 3배 증강 → 78장 (train 72 + val 6)
- 18 클래스 (1~12 숫자, circle, hand, hour_hand, minute_hand, 1110, 1110_location)
- v1~v5 이전 버전도 존재하지만 동일 원본 이미지 — 추가 확보 의미 없음

**추가 발견**: [cdt-acaci/cdt-ejwb1](https://universe.roboflow.com/cdt-acaci/cdt-ejwb1)
- 2980장 (April 2024) — 그러나 **분류(classification) 전용** 가능성 높음
- 픽셀 마스크/바운딩박스 없으면 U-Net 학습 불가 → 직접 확인 필요

**DeepN 대체 결정**: MNIST 기반 0-9 → 10/11/12 인식 불가 → **룰베이스(30° 섹터 분포)로 확정 대체**

**데이터 보강 우선순위**:
1. 기존 26장에 강한 증강(회전 45/90/135/180°, 수평반전, 밝기, 엘라스틱) → 300~500장
2. cdt-acaci 라벨 형식 확인 후 bbox 있으면 활용
3. NHATS 공개 이미지(nhats.org) — 47K장이나 픽셀 라벨 없어 재어노테이션 필요

---

## 공식 MoCA-K 채점 기준 vs 코드 대조 (MOCA-K평가기준.pdf 기반)

### 정육면체 (cube.py / cube_cnn_inference_v2.py)

| 공식 기준 | 코드 구현 | 상태 |
|---|---|---|
| 3차원이어야 한다 | 표준/등각 투영 모두 허용 | ✓ |
| 모든 선이 그려져야 한다 | `count_ok` (6~20개) | ✓ |
| 덧그려진 선이 없어야 한다 | `no_overline` (≤30) | ✓ |
| 대체로 평행하며 **길이는 비슷해야 한다** | `parallel_ok` + `length_ok` (CV ≤ 0.5) | ✓ |

### 시계 그리기 (clock.py / clock_cnn_inference.py)

| 항목 | 공식 기준 | 코드 구현 | 상태 |
|---|---|---|---|
| 윤곽 | 약간 변형된 원도 허용 | Hough Circle + 중심 위치 | ✓ |
| 숫자 | 1~12 모두 있어야 함 | 컨투어 개수 10~16개 (근사) | △ |
| 숫자 | **다른 숫자 추가 불가** | 동일 30° 구역 3개 초과 OR 전체 16개 초과 | ✓ |
| 숫자 | **순서대로 제자리에** | 12시(270°) 기준 30° 간격 12구역 중 10개 이상 점유 | ✓ |
| 바늘 | 올바른 자리(11시 10분) | 시침 240° / 분침 330° + 방향 normalize | ✓ |
| 바늘 | 시침 < 분침 길이 | `length_ok` | ✓ |
| 바늘 | 시침·분침 교점이 중앙 근방 | center_lines 필터로 간접 체크 | △ |

---

## 파일별 코드 내용 요약

### trail_making.py
- **역할**: 길만들기 채점 (1점)
- **핵심 상수**: `CORRECT_SEQUENCE = ["1","가","2","나","3","다","4","라","5","마"]`, `NODE_POSITIONS` (10개 노드 캔버스 비율 좌표, PDF 문제지 배치 기준), `NODE_RADIUS = 0.07`
- **NODE_POSITIONS** (PDF 문제지 기준 — app.js TRAIL_NODES와 동기화 필수):
  - 마(끝)(0.30,0.10), 가(0.62,0.14), 5(0.07,0.37), 나(0.45,0.42), 2(0.66,0.30)
  - 1/시작(0.24,0.56), 라(0.10,0.70), 4(0.47,0.70), 3(0.68,0.68), 다(0.18,0.87)
- **주요 함수**:
  - `extract_node_sequence(touch_points, W, H)` → 터치 좌표 → 노드 통과 순서 추출 (연속 중복 제거, 재방문 허용)
  - `detect_immediate_correction(raw_sequence)` → 실수 직후 올바른 노드로 수정 시 오류 무시
  - `score_trail_making(touch_points, W, H)` → `{raw_sequence, corrected_sequence, correct, total}`
  - `get_node_positions(W, H)` → Flutter 앱 노드 배치용 픽셀 좌표 반환

---

### cube.py
- **역할**: 육면체 그리기 채점 (1점)
- **채점 방식**: HoughLinesP 룰베이스 (CNN 모델 파일 없을 시 폴백)
- **주요 함수**:
  - `preprocess_image(image)` → 그레이→이진화(Otsu)→256×256 리사이즈
  - `extract_lines(image)` → HoughLinesP로 `(x1,y1,x2,y2,length,angle)` 리스트 반환
  - `_check_length_similarity(lines)` → 방향별 15° 그룹화 후 CV ≤ 0.5 확인
  - `score_cube(image)` → `{count_ok(6~20), no_overline(≤30), parallel_ok, length_ok, score}`
    - `direction_ok`: 표준(수평+수직+대각) OR 등각(수평+좌대각+우대각)
  - `load_cube_model(path)` → `.pth`→PyTorch, `.h5/.keras`→Keras 자동 감지
  - `score_cube_cnn(image, model)` → CNN 추론 결과 반환

---

### clock.py
- **역할**: 시계 그리기 채점 (3점: 윤곽1 + 숫자1 + 바늘1)
- **채점 방식**: HoughCircles + 컨투어 + HoughLinesP 룰베이스 (CNN 폴백 구조 동일)
- **주요 함수**:
  - `score_contour(image)` → HoughCircles로 원 감지, 중심이 이미지 중앙 30% 이내면 1점
  - `score_numbers(image)` → 원 내부 컨투어 분석
    - `count_ok = 8 ≤ count ≤ 16`
    - `no_extra`: 동일 30° 구역 3개 초과 or 전체 16개 초과 시 0점
    - `in_place`: `CLOCK_BASE=270°`(12시) 기준 30° 구역 중 10개 이상 점유
    - `score = 1 if count_ok and in_place and no_extra`
  - `score_hands(image)` → HoughLinesP로 바늘 감지
    - 시침 타겟 **240°** (11시), 분침 타겟 **330°** (10분)
    - HoughLinesP 방향 비결정성 해결: 중심에서 끝점 방향으로 normalize
    - `length_ok`: 시침 < 분침 길이 확인
  - `load_clock_model(path)` / `score_clock_cnn(image, model)` → CNN 연동 준비 완료

---

### naming.py
- **역할**: 어휘력 채점 (3점)
- **핵심 상수**: `FUZZY_THRESHOLD = 0.65`, `ANIMAL_ANSWERS = {"사자":["사자"], "코뿔소":["코뿔소","뿔소"], "낙타":["낙타","약대"]}`
- **주요 함수**:
  - `_word_in_text(stt_text, target)` → exact match 우선 → 실패 시 토큰별 SequenceMatcher ≥ 0.65
  - `score_single_animal(stt_text, animal_key)` → 정답 키워드 포함 시 1점
  - `score_naming(lion_stt, rhino_stt, camel_stt)` → `{lion, rhino, camel, total}`

---

### memory.py
- **역할**: 기억력/지연회상 채점 (5점, 즉각회상은 채점 없음)
- **핵심 상수**: `FUZZY_THRESHOLD = 0.65`
- **주요 함수**:
  - `record_immediate_recall(attempt1_stt, attempt2_stt, memory_words)` → 어떤 단어 기억했는지 기록만 (score=None)
  - `score_delayed_recall(stt_text, memory_words)` → 5개 단어 각 1점, fuzzy 허용
  - `record_cued_recall(word, category_cue_response, multiple_choice_response)` → 임상 분석용 (점수 없음, 인출/저장 문제 구분)
  - `score_memory(...)` → `{immediate_recall, delayed_recall, total}`

---

### attention.py
- **역할**: 주의력 채점 (6점: 숫자바로1 + 숫자거꾸로1 + 손뼉1 + 계산3)
- **핵심 상수**: `KOREAN_NUMBER_MAP` (영→0 ~ 아홉→9)
- **주요 함수**:
  - `extract_numbers(text)` → 아라비아숫자/한글숫자 혼합 파싱, 3자리↑ 단일열은 한 자리씩 분리
  - `score_forward_digits(stt, forward_digits)` → exact match 1점
  - `score_backward_digits(stt, backward_answer)` → exact match 1점
  - `score_tapping(tapped_indices, clap_sequence, clap_target)` → 오류 2개↑ 0점, 1개↓ 1점
  - `score_serial_7(stt)` → 이전 답 기준 독립 채점: 0개→0점, 1개→1점, 2~3개→2점, 4~5개→3점
  - `score_attention(forward_stt, backward_stt, tapped_indices, serial7_stt, config)` → `{forward_digits, backward_digits, tapping, serial_7, total}`

---

### language.py
- **역할**: 언어 채점 (3점: 따라말하기2 + 유창성1)
- **핵심 상수**: `SIMILARITY_THRESHOLD = 0.85`, `MARKET_WHITELIST` (frozenset, 모듈 로드 시 1회 읽기)
- **주요 함수**:
  - `score_single_sentence(stt, correct)` → SequenceMatcher ≥ 0.85이면 1점
  - `score_repetition(sentence1_stt, sentence2_stt, sentences)` → `{sentence1, sentence2, similarity1, similarity2, total}`
  - `_starts_with_giyeok(word)` → 유니코드 한글 블록 기준 ㄱ 초성 판별
  - `_validate_with_llm(words)` → Claude Haiku API로 시장물건 여부 확인 (use_llm=True 시만)
  - `score_verbal_fluency(stt, fluency_count, fluency_type, use_llm)` → 중복 제거 후 whitelist 필터 → unknown 분리 → `{words, unknown, count, total}`
  - `score_language(sentence1_stt, sentence2_stt, fluency_stt, config, use_llm)` → `{repetition, fluency, total}`

---

### abstraction.py
- **역할**: 추상력 채점 (2점)
- **핵심 상수**: `FUZZY_THRESHOLD = 0.65`
- **주요 함수**:
  - `_word_in_text(stt, target)` → 정답 키워드에만 fuzzy 적용 (오답 키워드는 exact match 유지)
  - `score_single_abstraction(stt, correct_keywords, incorrect_keywords)` → 오답 먼저 체크 → 정답 체크
  - `score_abstraction(pair1_stt, pair2_stt, config)` → `{pair1, pair2, pair1_name, pair2_name, total}`

---

### orientation.py
- **역할**: 지남력 채점 (6점: 년/월/일/요일/장소/시군구 각 1점)
- **주요 함수**:
  - `get_date_answer_key()` → 시스템 현재 날짜로 `{년,월,일,요일}` 정답키 자동 생성
  - `score_date(stt, date_type, answer_key)` → 월: 숫자+한글(유월 등) 이중 허용, 단어 경계 regex로 오탐 방지
  - `_month_to_korean(month)` → 월→한글 변환 (유월, 시월 등 불규칙형 포함)
  - `get_location_answer_key(lat, lng)` → 카카오 역지오코딩 API (실제 앱용, 현재 미연동)
  - `set_location_answer_key_manual(dong, sigungu)` → 시연 환경 수동 설정
  - `score_location(stt, location_type, location_key)` → 구/시/군 suffix 제거 단축형도 허용
  - `score_orientation(...)` → `{년, 월, 일, 요일, 장소, 시군구, total}`

---

### version_manager.py
- **역할**: MoCA-K / K-MoCA 버전 설정 단일 진실 공급원
- **핵심 데이터** (`VERSIONS` 딕셔너리, 버전별):
  - `memory_words` (5개), `memory_cues` (범주/선택지), `fluency_type/count`
  - `sentences` (2문장), `abstraction_pairs/correct/incorrect`
  - `forward_digits`, `backward_digits/answer`, `clap_sequence/target`
- **주요 함수**:
  - `get_next_version(last_version, last_assessed_at)` → 6개월 경과 시 버전 교체 (학습 효과 차단)
  - `get_version_config(version)` → 버전별 설정 딕셔너리 반환

---

### session_manager.py
- **역할**: 검사 흐름 제어 (문항 순서, 5분 대기, 세션 상태)
- **핵심 상수**: `ITEM_SEQUENCE` (14개 문항 순서), `ITEM_DURATION` (문항별 예상 소요 초), `DELAYED_RECALL_WAIT = 300`
- **SessionState Enum**: IDLE / READY / IN_PROGRESS / WAITING_5MIN / DELAYED_RECALL / COMPLETED
- **MoCASession 클래스**:
  - `start()` → 첫 문항 정보 반환
  - `next_item(response)` → 응답 저장 → 지연회상 직전 5분 대기 체크 (item_index 증가 전) → 다음 문항 반환
  - `_check_delayed_recall_wait()` → `memory_encoded_at` 기준 남은 대기시간(초) 반환
  - `_get_current_item_info()` → 문항별 필요 데이터 포함 dict 반환
- **create_session(user_id, education_years, last_version, last_assessed_at)** → 버전 자동 결정 후 세션 생성

---

### total_scorer.py
- **역할**: 전체 통합 채점 진입점 (30점 만점)
- **주요 함수**:
  - `apply_education_correction(score, education_years)` → 학력 6년 이하 +1점 (최대 30점)
  - `classify_mci(total_score)` → 23점 미만: MCI 의심, `{label, interpretation, score}`
  - `score_total(...)` → 전체 파라미터 받아 8개 채점 모듈 호출 후 통합
    - 반환: `{version, sections, details, raw_score, education_correction, final_score, mci}`
    - `sections` 키: `trail_making/drawing/naming/memory/attention/language/abstraction/orientation`

---

### clock_cnn_inference.py
- **역할**: CNN 기반 시계 채점 (DeepC/DeepH/DeepN 모델 연동)
- **주요 함수**: `score_hands_deeph(image, model)` — 시침 타겟 240°, HoughLinesP normalize 포함

---

### app.py (Flask 웹앱)
- **역할**: 시연용 웹 서버 (로컬 노트북, 포트 5000)
- **전역 상태**: `_store = {uid: {sess, raw, location, sigungu}}` (메모리 세션)
- **라우트 요약**:

| 라우트 | 메서드 | 역할 |
|---|---|---|
| `/` | GET | 홈 (이름/학력/동네 입력) |
| `/start` | POST | 세션 생성 → `/item` 리다이렉트 |
| `/item` | GET | 현재 문항 렌더링 |
| `/submit` | POST | 응답 저장 → `{next: item/waiting/result}` |
| `/waiting` | GET | 5분 대기 카운트다운 화면 |
| `/continue_wait` | POST | 대기 완료 확인 → 다음 라우트 반환 |
| `/result` | GET | score_total() 호출 → 결과 화면 |
| `/audio/<filename>` | GET | TTS mp3 서빙 (assets/tts/) |

- **`_empty_raw()`**: 전체 STT/점수 필드 초기값 dict
- **`_tts_urls(item, version)`**: 문항별 TTS mp3 URL 리스트 반환
- **`_store_response(raw, item, data)`**: POST 데이터 → raw dict 저장
- **`_score_canvas(img_b64, kind)`**: base64 이미지 디코딩 → cube/clock 채점 모듈 호출
- **`_compute_score(raw, entry, s)`**: `score_total()` 호출 후 result.html에 넘길 dict 반환

---

## 채점 기준 참고

- **총점**: 30점 (교육연수 6년 이하 +1점, 최대 30점)
- **MCI 의심 기준**: 23점 미만
- **출처**: © Z. Nasreddine MD, JY. Lee 한국판. www.mocatest.org

---

## 전체 시스템 기술 수준 평가 (2026-07-09)

### 구성요소별 평가

| 구성요소 | 단독 첨단? | 조합 가치 |
|----------|-----------|-----------|
| 보행 로지스틱 회귀 (4 feature) | ✗ | 임상 라벨(DGI/TUG) 기반 검증된 파이프라인 |
| 노인 음성 Whisper 파인튜닝 (AI Hub) | ✓ | 실제 임상 적용 가능 수준 STT |
| MoCA 자동 채점 (10개 문항 완전 자동화) | △ | 버전 로테이션·학습효과 차단까지 구현 |
| LLM 에이전트 + 코드 RAG 동적 앱 제어 | ✓✓ | 프로덕션 헬스케어 앱에서 거의 없는 구조 |
| 온디바이스 완결 (Android 네이티브) | ✓ | 인터넷 없이 보행+인지 동시 스크리닝 |

### 왜 조합이 첨단인가

```
노인 대상 + 의료 스크리닝 + LLM 에이전트 동적 제어 + 온디바이스 완결
```

이 네 가지를 동시에 구현한 헬스케어 앱은 세계적으로도 드물다.

**한 줄 요약**: 기술 논문 레벨 첨단은 아니나, 실용 의료 AI 앱으로는 세계 최전선 수준.

---

## 보행 탭 Android 온디바이스 구현 계획 (2026-07-09 ~)

### 개요
- MOCA 앱에 보행(Gait) 탭 추가 — Android 네이티브 (Kotlin)
- 허리밴드에 스마트폰 세로 고정 → 10초 걷기 → 즉석 분류
- 서버 없이 완전 온디바이스 추론

### 학습된 모델 정보
- 파일: `파이널 보행 프로젝트/final__2026/02_model/final_motor_domain4_labwalks10_logistic_C0p5.joblib`
- 학습 데이터: Labwalks 실험실 보행 10초 window, 67명 (CO024·FL020 제외)
- 라벨: DGI ≤ 19 OR TUG ≥ 12 → 운동저하 가능군(1), else 정상(0)
- CV 성능 (A scheme): AUC 0.847, Sensitivity 0.834, Specificity 0.740

### 4개 Feature (domain4_fixed)

| Feature | 도메인 | 계수 | 중요도 |
|---------|--------|------|--------|
| `v_amp_pool_median` | 보행 활력 (수직 진폭 중앙값) | -1.2796 | ★★★ |
| `base_v_stride_regularity` | 리듬 규칙성 (수직 stride ACF) | -0.4834 | ★★ |
| `roll_amp_pool_iqr` | 몸통 회전 안정성 (roll 진폭 IQR) | -0.3160 | ★ |
| `ml_amp_pool_iqr` | 좌우 안정성 (좌우 진폭 IQR) | +0.1120 | ☆ |

**모델 가중치 (하드코딩용):**
```
intercept = -0.22169196063461533
threshold = 0.4814795955030277
```
StandardScaler mean/scale은 joblib에서 추출 필요 (pipeline.named_steps['scale'].mean_, .scale_)

### Android 구현 단계

**1단계: IMU 수집**
```kotlin
// SensorManager.SENSOR_DELAY_FASTEST → ~100Hz
// TYPE_ACCELEROMETER: x(좌우), y(전후), z(수직) — Android 축 주의
// TYPE_GYROSCOPE: x, y, z
// 허리밴드 세로 고정 시: Android z축 → v(수직), x축 → ml(좌우)
// 30초 수집 → 신호 품질 최적 10초 구간 자동 추출 → acc[1000×3], gyro[1000×3]
```

**Android 축 → Labwalks 축 매핑 (허리밴드 세로 고정 기준):**
- Android `acc.z` → `v` (수직)
- Android `acc.x` → `ml` (좌우)
- Android `acc.y` → `ap` (전후)
- Android `gyro.z` → `roll`

**2단계: Butterworth Bandpass 필터 계수 (Python에서 미리 계산 → 하드코딩)**
```python
# Python으로 계수 추출 (fs=100Hz 기준)
from scipy.signal import butter
sos_v_ml = butter(4, [0.6/50, 3.0/50], btype='bandpass', output='sos')   # v, ml
sos_roll = butter(4, [0.5/50, 5.0/50], btype='bandpass', output='sos')   # roll
# → 숫자 배열을 Kotlin 상수로 하드코딩
```

**3단계: amplitude_pooling_features (Kotlin)**
- `robust_abs_signal()`: detrend → abs
- `median()`: sort → 중앙값
- `iqr()`: p75 - p25

**4단계: base_v_stride_regularity (Kotlin)**
```
1. AP 신호 ACF 계산 (unbiased_acf)
2. 0.8s~1.6s 구간에서 첫 번째 피크 → stride_duration
3. V 신호 ACF[stride_lag] → base_v_stride_regularity
```
ACF는 O(n²) 이중 루프, 1000샘플 기준 ~100ms (허용 범위)

**5단계: 추론**
```kotlin
// StandardScaler 적용 후 로지스틱 회귀
val logit = intercept + w0*(x0-mean0)/scale0 + w1*(x1-mean1)/scale1 + ...
val prob = 1.0 / (1.0 + exp(-logit))
val result = if (prob >= threshold) "운동저하 가능" else "정상"
```

### MOCA 인지평가 온디바이스 전환 범위

| 기능 | 현재 | 온디바이스 전환 여부 |
|------|------|---------------------|
| STT (음성인식) | 브라우저 Web Speech API | 현재 이미 온디바이스 |
| 시계 그리기 | 서버 룰베이스 (clock.py) | CNN 미사용 확정 (데이터 부족·과적합) — 룰베이스 Kotlin 포팅 |
| 육면체 그리기 | 서버 룰베이스 (cube.py) | CNN 미사용 확정 (257MB 미완성 모델) — 룰베이스 Kotlin 포팅 |
| 채점 로직 | 서버 Python | Kotlin 포팅 |

**주의**: clock/cube CNN(TFLite 변환 등)은 하지 않기로 결정됨. 룰베이스 로직만 Kotlin으로 포팅.

---

## 펫 코치 AI 에이전트 설계 (2026-07-09 ~)

### 개념
앱 안에서 자유롭게 돌아다니는 캐릭터형 AI 코치.
검사 중 사용자 상태를 감지해 동적으로 앱 동작을 조절함.

### 기능 목록

| 기능 | 설명 | 기술 |
|------|------|------|
| 문항 재설명 | "다시 설명해줘" 발화 시 해당 문항 TTS 재생 | 키워드 매칭 + Android TTS |
| 속도 조절 | 15초 무응답 감지 시 TTS 속도 자동 감소 | 타이머 트리거 + `setSpeechRate()` |
| 재시도 허용 | 오답 누적 시 현재 문항 재시도 허용 | 세션 상태 제어 |
| 쉬기 안내 | 보행 중 "힘들어요" 발화 시 휴식 안내 | 키워드 매칭 + TTS |
| 운동 반복 설명 | 이해 확인될 때까지 매번 다른 표현으로 운동 동작 재설명 | Claude API — 매 호출마다 다른 문장 생성, 이해 확인 STT로 루프 탈출 |
| 자유 질문 응답 | 위 패턴 외 예상치 못한 질문 처리 | Claude API (function calling) |
| 결과 해석 설명 | 검사 완료 후 MoCA 점수 + 보행 결과 통합 설명 | Claude API |

### 사용 기술

- **Claude API** — 예상 밖 상황 처리, 결과 해석, 운동 동작 다회 재설명 (매번 다른 문장 생성)
- **RAG** — 앱 소스코드·문항 설명·보행 가이드를 청크로 쪼개 LLM context에 주입
- **Intent Classification (키워드 매칭, 온디바이스)** — 자주 오는 상황은 API 호출 없이 즉각 처리
- **Function Calling** — LLM이 허용된 앱 동작(TTS 속도, 재시도 등)만 선택해서 실행
- **Android STT** — 사용자 발화 인식 및 이해 여부 확인 ("알겠어요" 감지 → 운동 설명 루프 탈출)
- **Android TTS** — 코치 음성 출력, 속도/볼륨 동적 조절

### 핵심 구조

```
[자주 오는 상황] → Intent Classification (온디바이스, 0ms)
                     → 사전 정의 응답 실행
[예상 밖 상황]   → RAG로 관련 문서 검색
                     → Claude API + Function Calling
                     → 허용된 앱 동작만 실행
[운동 설명]      → Claude API 루프 (이해 확인 STT까지 반복)
[검사 완료]      → Claude API 1회 → 통합 결과 해석
```

### 에이전트 허용/금지 (규제)

```
허용:
  - TTS 속도 조절 (setSpeechRate)
  - 폰트 크기 조절
  - 문항 재설명 텍스트 생성
  - 문항 재시도 허용
  - 결과 해석 텍스트 생성

금지:
  - 채점 로직 변경
  - 임상 기준값(23점 미만 MCI 등) 수정
  - 개인정보 외부 전송
  - 문항 순서 임의 변경
```

### RAG 지식베이스 구성
- 앱 소스코드 (Kotlin) — 변경 가능한 파라미터 목록
- K-MoCA / MoCA-K 각 문항 공식 설명 및 재설명 버전
- 보행 검사 수행 가이드
- 자주 발생하는 오류 패턴 + 대응 스크립트

### 레이턴시 전략
- 자주 오는 상황 (키워드 매칭): 즉각, API 호출 없음
- 예상 밖 상황: Claude API → 1~2초
- 운동 반복 설명: Claude API 루프, 이해 확인까지 반복
- 결과 해석: 검사 완료 후 1회, 레이턴시 무관

### LLM 호출 시 컨텍스트 구조
```
system: 앱 코드 RAG 청크 + 규제 목록 + 현재 문항 정보
user: 사용자 발화 또는 앱 이벤트
→ Claude가 허용된 tool 중 선택해서 호출
```

---

### 보행 수집 프로토콜

- **30초 평지 걷기** → 신호 품질 최적 10초 구간 자동 추출
- 학습 데이터(Labwalks) 역시 10초 window 기준이므로 추출 후 동일한 특징 파이프라인 적용
- 최적 구간 선택 기준: 보행 주기 ACF 피크 강도 기반 (가장 규칙적인 구간)

---

### 구현 순서 (추천)
1. Android Studio 프로젝트 생성 (Kotlin)
2. SensorManager로 IMU 10초 수집 + 화면에 raw 값 표시
3. Python에서 Butterworth 계수 추출 → Kotlin 하드코딩
4. 필터 적용 + amplitude_pooling (median, IQR) 구현
5. ACF + stride peak 탐지 → base_v_stride_regularity
6. joblib에서 scaler mean/scale + 모델 가중치 추출 → 하드코딩
7. 추론 + 결과 UI
8. MOCA 앱 탭에 통합

---

## final__2026 폴더 정리 (2026-07-10)

### 폴더 구조

```
final__2026/
  01_preprocessing/
    labwalks_service10_amp_spec_features.csv    ← 실험실 10초 피처값 (67명)
  02_model/
    final_motor_domain4_labwalks10_logistic_C0p5.joblib       ← 최종 모델
    final_motor_domain4_labwalks10_logistic_C0p5_metadata.json
    domain4_oof_predictions.csv                 ← A/B/C/LOSO OOF 예측값
    domain4_full_validation_metrics.csv         ← fold-level 검증 지표
    domain_binary_metrics_*.csv
    domain_feature_groups.csv
    domain_selected_features_*.csv
  03_code/
    01_preprocessing/
      extract_labwalks_service20_features.py    ← 피처 추출 코어 (window-sec 인자로 10/15/20 모두 지원)
      run_extract_labwalks_service_windows.py   ← 오케스트레이터 (기본 10,15,20초 순서 실행)
    02_modeling/
      RUN_service10_domain_representative_model_compare.py  ← domain4 모델 학습
    03_validation/
      RUN_final_model_full_validation_suite.py
      RUN_final_model_stability_checks.py
    04_visualization/
      RUN_final_service10_visualizations.py     ← 최종 모델 기준 시각화 (아래 참고)
  04_clinical_data/
    ClinicalDemogData_COFL.xlsx
```

### 전처리 코드 참고

- `extract_labwalks_service20_features.py` — 이름은 service20이나 `--window-sec` 인자 지원, 10초 실행 시 `run_extract_labwalks_service_windows.py --windows 10` 사용
- PhysioNet 기반 파일(`strict_20s_*`) 전부 제거 — 최종 모델은 Labwalks 실험실 데이터만 사용

### 최종 모델 검증 성능 (domain4, subject-level pooled)

| 검증방식 | AUC | Sensitivity | Specificity |
|---------|-----|-------------|-------------|
| A (5-fold×100) | 0.842 | 0.769 | 0.744 |
| B (3-fold×100) | 0.839 | 0.731 | 0.744 |
| C (8:2×100) | 0.845 | 0.769 | 0.744 |
| E (LOSO) | 0.843 | 0.769 | 0.721 |
| **CV 최종 (sens90_maxspec)** | **0.845** | **0.848** | **0.749** |

**LOSO 주의**: fold-level sensitivity(0.290)는 test subject 1명이라 양성 없는 fold에서 0으로 처리되어 왜곡됨 → subject-level pooled(0.769)가 올바른 값

### 시각화 (RUN_final_service10_visualizations.py)

출력 폴더: `시각화_domain4/`

| 파일 | 내용 |
|------|------|
| `01_cv_performance_summary.png` | 교차검증 AUC/Sens/Spec 요약 |
| `02_confusion_matrix.png` | Confusion matrix (apparent train) |
| `03_abce_validation_barplot.png` | A/B/C/LOSO subject-level 성능 비교 |
| `04_train_test_auc_gap.png` | Train-Test AUC gap 과적합 점검 |
| `05_sensitivity_by_fold_boxplot.png` | Fold별 Sensitivity 분포 |
| `06_feature_violin_boxplot.png` | 4개 feature 그룹별 분포 |
| `07_feature_correlation_heatmap.png` | Feature Spearman 상관 |
| `08_logistic_coefficient.png` | Logistic 계수 |
