"""
노인 발화를 고려한 Whisper STT + Silero VAD 모듈 (로컬 설계)

노인 발화의 긴 멈춤과 낮은 음성 에너지를 고려해 Silero VAD 임계값·묵음 길이·
앞뒤 여유 구간을 조정하고, 단답형 문항에 맞춰 Whisper 이전 문맥 참조를 껐다.
문항별 initial_prompt는 정답 단어 없이 문항 맥락만 준다. 배포 앱은 서버 자원
제약으로 Android 기본 STT를 사용하며 이 모듈은 앱에 연결되어 있지 않다.
노인 음성 데이터로 기본 설정 대비 효과를 검증하는 것은 다음 과제다.

참고 문헌 (노인 음성 ASR 문제 배경):
- Challenges in ASR for Adults with Cognitive Impairment (2025)
- Out of the Box, into the Clinic (2025)
- MOPSA (2025)
- Can speech foundation models identify languages in aging populations (2025)
- Silero VAD: github.com/snakers4/silero-vad
"""

import numpy as np
import torch
import os


# ────────────────────────────────────────────
# 노인 음성 특성 (한국 65세 이상)
# ────────────────────────────────────────────
# 1. 느린 말속도: 2~3 음절/초 (정상 5~6 음절/초)
# 2. 불명확한 발음: 자음 약화, 모음 중성화
# 3. 잦은 묵음/멈춤: 단어 사이 1~2초 정지
# 4. 낮은 음성 에너지: 작은 목소리
# 5. 떨리는 목소리: jitter/shimmer 증가


# ────────────────────────────────────────────
# Whisper 디코딩 설정 (노인 발화 고려, 효과 미검증)
# ────────────────────────────────────────────
WHISPER_ELDERLY_PARAMS = {
    "model_size": "small",       # 244M params, 서버 추론 권장
    "language":   "ko",          # 한국어 고정 (자동감지 비활성화)

    # beam_size/best_of=5: Whisper CLI 기본값과 동일 (Python API 기본은 greedy)
    "beam_size":  5,
    "best_of":    5,

    # temperature 폴백: Whisper 기본값 그대로
    "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),

    # no_speech_threshold: Whisper 기본값 그대로
    # no_speech_prob가 이 값을 넘고 avg_logprob가 logprob_threshold 이하이면 구간을 건너뛴다.
    # 값을 낮추면 무음으로 버리는 구간이 늘어나므로 조정하지 않았다 (묵음은 앞단 VAD가 처리)
    "no_speech_threshold": 0.6,

    # Whisper 기본값 그대로
    "compression_ratio_threshold": 2.4,
    "logprob_threshold": -1.0,

    # 기본 True → False (조정): MoCA 단답형 항목 → 이전 컨텍스트 불필요
    "condition_on_previous_text": False,

    "fp16": torch.cuda.is_available(),
}

# 항목별 initial_prompt (문항 맥락 힌트 제공, 효과 미검증)
# 정답 단어는 넣지 않는다: 프롬프트에 정답이 있으면 틀린 응답도 정답으로 전사될 수 있다
ITEM_PROMPTS = {
    "forward_digits":  "숫자를 순서대로 말합니다.",
    "backward_digits": "숫자를 거꾸로 말합니다.",
    "serial_7":        "백에서 칠을 계속 빼서 숫자를 말합니다.",
    "naming":          "그림 속 동물 이름을 말합니다.",
    "memory":          "들은 단어를 따라 말합니다.",
    "sentence_repeat": "문장을 따라 말합니다.",
    "fluency":         "시장 물건 이름.",
    "abstraction":     "공통점을 말합니다.",
    "delayed_recall":  "기억한 단어를 말합니다.",
    "orientation":     "날짜와 장소를 말합니다.",
}


# ────────────────────────────────────────────
# Silero VAD 파라미터 (노인 발화 고려, 효과 미검증)
# ────────────────────────────────────────────
# 기본값은 silero-vad get_speech_timestamps 기준
VAD_ELDERLY_PARAMS = {
    # 발화 판단 임계값: 기본 0.5 → 0.3으로 낮춤
    # 노인 낮은 음성 에너지 → 임계값 낮춰야 음성 감지 가능
    "threshold": 0.3,

    # 최소 발화 길이(ms): 기본 250ms → 100ms
    # 노인 짧은 단어 응답(예: "네", "사자") 감지
    "min_speech_duration_ms": 100,

    # 최소 묵음 길이(ms): 기본 100ms → 1500ms
    # 노인 말 사이 긴 멈춤 허용 → 발화 중간에 잘리지 않게
    "min_silence_duration_ms": 1500,

    # 발화 앞뒤 여유(ms): 기본 30ms → 600ms
    # 노인 발화 시작/끝 부분 잘림 방지
    "speech_pad_ms": 600,

    # 샘플레이트: Whisper 표준
    "sample_rate": 16000,
}


# ────────────────────────────────────────────
# 한글 숫자 표기 (숫자 문항 후처리용)
# ────────────────────────────────────────────
SINO_DIGITS = {"영": 0, "공": 0, "일": 1, "이": 2, "삼": 3, "사": 4,
               "오": 5, "육": 6, "칠": 7, "팔": 8, "구": 9}
NATIVE_DIGITS = {"하나": 1, "둘": 2, "셋": 3, "넷": 4, "다섯": 5,
                 "여섯": 6, "일곱": 7, "여덟": 8, "아홉": 9}


def _sino_to_int(word: str) -> int:
    """'구십삼' → 93, '백' → 100 (앞 숫자가 없으면 1로 본다)"""
    total = 0
    for unit, mult in (("백", 100), ("십", 10)):
        if unit in word:
            head, word = word.split(unit, 1)
            total += (SINO_DIGITS[head] if head else 1) * mult
    if word:
        total += SINO_DIGITS[word]
    return total


# ────────────────────────────────────────────
# Silero VAD 로드
# ────────────────────────────────────────────
def load_silero_vad():
    """
    Silero VAD 모델 로드
    자동 다운로드 (~2MB)
    """
    model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False,
        trust_repo=True
    )
    get_speech_timestamps = utils[0]
    collect_chunks        = utils[3]
    read_audio            = utils[4]
    return model, get_speech_timestamps, collect_chunks, read_audio


# ────────────────────────────────────────────
# VAD 전처리: 음성 구간만 추출
# ────────────────────────────────────────────
def apply_vad(audio: np.ndarray,
              vad_model,
              get_speech_timestamps,
              collect_chunks,
              params: dict = None) -> np.ndarray:
    """
    Silero VAD로 음성 구간만 추출
    노인 묵음/멈춤 구간 제거 (Whisper 환각을 줄이려는 의도)

    Args:
        audio: 16000Hz numpy array
        vad_model: Silero VAD 모델
        params: VAD 파라미터 (기본값: VAD_ELDERLY_PARAMS)

    Returns:
        음성 구간만 포함한 numpy array
    """
    if params is None:
        params = VAD_ELDERLY_PARAMS

    audio_tensor = torch.FloatTensor(audio)

    # 음성 구간 타임스탬프 추출
    speech_timestamps = get_speech_timestamps(
        audio_tensor,
        vad_model,
        threshold             = params["threshold"],
        min_speech_duration_ms= params["min_speech_duration_ms"],
        min_silence_duration_ms=params["min_silence_duration_ms"],
        speech_pad_ms         = params["speech_pad_ms"],
        sampling_rate         = params["sample_rate"],
    )

    if not speech_timestamps:
        # 음성 없음 → 원본 반환
        return audio

    # 음성 구간만 이어붙이기
    speech_audio = collect_chunks(speech_timestamps, audio_tensor)
    return speech_audio.numpy()


# ────────────────────────────────────────────
# 노인 STT 클래스
# ────────────────────────────────────────────
class ElderlySTT:
    """
    노인 발화를 고려한 설정의 Whisper STT + Silero VAD
    MoCA-K 문항별 initial_prompt 적용
    """

    def __init__(self, model_size: str = None, use_vad: bool = True):
        import whisper

        size = model_size or WHISPER_ELDERLY_PARAMS["model_size"]
        print(f"Whisper {size} 로딩...")
        self.model    = whisper.load_model(size)
        self.params   = WHISPER_ELDERLY_PARAMS.copy()
        self.use_vad  = use_vad

        # Silero VAD 로드
        if use_vad:
            print("Silero VAD 로딩...")
            (self.vad_model,
             self.get_speech_timestamps,
             self.collect_chunks,
             self.read_audio) = load_silero_vad()
            print("VAD 준비 완료")

        print("STT 준비 완료")

    def transcribe_file(self, audio_path: str,
                        item_type: str = None) -> dict:
        """
        파일 경로로 STT

        Args:
            audio_path: wav/mp3 파일 경로
            item_type: MoCA 항목 타입

        Returns:
            {"text": str, "vad_applied": bool, ...}
        """
        import librosa
        audio, sr = librosa.load(audio_path, sr=16000)
        return self._transcribe(audio, item_type)

    def transcribe_array(self, audio: np.ndarray,
                         sample_rate: int = 16000,
                         item_type: str = None) -> dict:
        """
        numpy array로 STT (마이크 직접 입력)

        Args:
            audio: 음성 데이터 numpy array
            sample_rate: 샘플레이트
            item_type: MoCA 항목 타입
        """
        import librosa
        if sample_rate != 16000:
            audio = librosa.resample(
                audio, orig_sr=sample_rate, target_sr=16000
            )
        return self._transcribe(audio, item_type)

    def _transcribe(self, audio: np.ndarray,
                    item_type: str = None) -> dict:
        """내부 STT 처리"""
        vad_applied = False

        # VAD 전처리 (노인 묵음 제거)
        if self.use_vad:
            audio_processed = apply_vad(
                audio,
                self.vad_model,
                self.get_speech_timestamps,
                self.collect_chunks,
                VAD_ELDERLY_PARAMS
            )
            vad_applied = len(audio_processed) < len(audio)
        else:
            audio_processed = audio

        # 항목별 초기 프롬프트
        initial_prompt = ITEM_PROMPTS.get(item_type)

        # Whisper 전사
        result = self.model.transcribe(
            audio_processed,
            language                    = self.params["language"],
            beam_size                   = self.params["beam_size"],
            best_of                     = self.params["best_of"],
            temperature                 = self.params["temperature"],
            no_speech_threshold         = self.params["no_speech_threshold"],
            compression_ratio_threshold = self.params["compression_ratio_threshold"],
            logprob_threshold           = self.params["logprob_threshold"],
            condition_on_previous_text  = self.params["condition_on_previous_text"],
            fp16                        = self.params["fp16"],
            initial_prompt              = initial_prompt,
        )

        text = result["text"].strip()
        text = self._postprocess(text, item_type)

        return {
            "text":          text,
            "language":      result.get("language", "ko"),
            "vad_applied":   vad_applied,
            "no_speech_prob": (result["segments"][0].get("no_speech_prob", 0)
                               if result.get("segments") else 0),
        }

    def _postprocess(self, text: str, item_type: str = None) -> str:
        """STT 후처리"""
        import re
        text = re.sub(r'[,\.!?]+', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        if item_type in ["forward_digits", "backward_digits", "serial_7"]:
            text = self._normalize_numbers(text)
        return text

    def _normalize_numbers(self, text: str) -> str:
        """한글 숫자 → 아라비아 숫자"""
        import re
        # 고유어를 먼저 바꾼다: '일곱'의 '일'이 한자어 1로 먼저 바뀌지 않게
        native = "|".join(sorted(NATIVE_DIGITS, key=len, reverse=True))
        text = re.sub(native, lambda m: str(NATIVE_DIGITS[m.group()]), text)

        # 한자어는 백·십 단위까지 한 덩어리로 읽는다: '구십삼' → 93, '일이삼' → 1 2 3
        d = "[" + "".join(SINO_DIGITS) + "]"
        pattern = f"(?:{d}?백)?(?:{d}?십)?{d}?"
        return re.sub(pattern,
                      lambda m: str(_sino_to_int(m.group())) if m.group() else "",
                      text)


# ────────────────────────────────────────────
# 테스트
# ────────────────────────────────────────────
if __name__ == "__main__":
    print("=== 노인 STT 설정 확인 ===")

    print("\n[Whisper 파라미터]")
    for k, v in WHISPER_ELDERLY_PARAMS.items():
        print(f"  {k}: {v}")

    print("\n[Silero VAD 파라미터]")
    for k, v in VAD_ELDERLY_PARAMS.items():
        print(f"  {k}: {v}")

    print("\n[항목별 initial_prompt]")
    for k, v in ITEM_PROMPTS.items():
        print(f"  {k}: {v}")

    print("\n[실제 사용법]")
    print("  stt = ElderlySTT('small', use_vad=True)")
    print("  result = stt.transcribe_file('audio.wav', item_type='naming')")
    print("  print(result['text'])")
    print("  print(result['vad_applied'])")
