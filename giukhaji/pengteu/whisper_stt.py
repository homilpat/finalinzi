"""
노인 대상 Whisper STT + Silero VAD 통합 모듈
논문 근거:
- Challenges in ASR for Adults with Cognitive Impairment (2025)
  → beam_size=5, no_speech_threshold=0.3
- Out of the Box, into the Clinic (2025)
  → 노인 음성 VAD 적용 필수
- Silero VAD (2024): github.com/snakers4/silero-vad
  → 다국어 지원, 경량(1~2MB), 한국어 포함
- MOPSA (2025)
  → 노인 음성 느린 말속도/불명확한 발음 고려
- Can speech foundation models identify languages in aging populations (2025)
  → 노인 음성 ASR 성능 저하 문제 확인
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
# Whisper 노인 최적화 파라미터
# ────────────────────────────────────────────
WHISPER_ELDERLY_PARAMS = {
    "model_size": "small",       # 244M params, 서버 추론 권장
    "language":   "ko",          # 한국어 고정 (자동감지 비활성화)

    # beam_size=5: 노인 불명확 발음 → 더 많은 후보 탐색
    # (Challenges in ASR for Cognitive Impairment, 2025)
    "beam_size":  5,
    "best_of":    5,

    # temperature: 0.0 시작 → 실패 시 0.2씩 증가
    "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),

    # no_speech_threshold=0.3: 기본 0.6 → 낮춤
    # 노인 잦은 멈춤 → 음성을 묵음으로 오인식 방지
    "no_speech_threshold": 0.3,

    "compression_ratio_threshold": 2.4,
    "logprob_threshold": -1.0,

    # False: MoCA 단답형 항목 → 이전 컨텍스트 불필요
    "condition_on_previous_text": False,

    "fp16": torch.cuda.is_available(),
}

# 항목별 initial_prompt (맥락 힌트 → 인식률↑)
ITEM_PROMPTS = {
    "forward_digits":  "숫자를 순서대로 말합니다.",
    "backward_digits": "숫자를 거꾸로 말합니다.",
    "serial_7":        "백에서 칠을 뺍니다. 구십삼, 팔십육.",
    "naming":          "동물 이름. 사자, 코뿔소, 낙타.",
    "memory":          "단어. 얼굴, 비단, 교회, 진달래, 빨강.",
    "sentence_repeat": "문장을 따라 말합니다.",
    "fluency":         "시장 물건 이름.",
    "abstraction":     "공통점을 말합니다.",
    "delayed_recall":  "기억한 단어를 말합니다.",
    "orientation":     "날짜와 장소를 말합니다.",
}


# ────────────────────────────────────────────
# Silero VAD 파라미터 (노인 최적화)
# ────────────────────────────────────────────
# 논문: Silero VAD (2024) - 다국어 지원, 한국어 포함
# 노인 음성 특성 반영하여 파라미터 조정
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

    # 발화 앞뒤 여유(ms): 기본 400ms → 600ms
    # 노인 발화 시작/끝 부분 잘림 방지
    "speech_pad_ms": 600,

    # 샘플레이트: Whisper 표준
    "sample_rate": 16000,
}


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
    노인 묵음/멈춤 구간 제거 → Whisper 환각 감소

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
    노인 최적화 Whisper STT + Silero VAD 통합
    MoCA-K 항목별 최적화
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
        mapping = {
            "영": "0", "일": "1", "이": "2", "삼": "3", "사": "4",
            "오": "5", "육": "6", "칠": "7", "팔": "8", "구": "9",
            "하나": "1", "둘": "2", "셋": "3", "넷": "4", "다섯": "5",
            "여섯": "6", "일곱": "7", "여덟": "8", "아홉": "9",
            "구십삼": "93", "팔십육": "86", "칠십구": "79",
            "칠십이": "72", "육십오": "65",
        }
        for kor, num in mapping.items():
            text = text.replace(kor, num)
        return text


# ────────────────────────────────────────────
# 테스트
# ────────────────────────────────────────────
if __name__ == "__main__":
    print("=== 노인 STT 설정 확인 ===")

    print("\n[Whisper 파라미터]")
    for k, v in WHISPER_ELDERLY_PARAMS.items():
        print(f"  {k}: {v}")

    print("\n[Silero VAD 노인 최적화 파라미터]")
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
