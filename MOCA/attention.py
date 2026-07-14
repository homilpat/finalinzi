"""
MoCA-K 주의력 채점 모듈
항목: 숫자 바로 따라 외우기, 숫자 거꾸로 따라 외우기, 손뼉치기(가), 100-7 계산
총점: 6점 (숫자 2점 + 손뼉 1점 + 계산 3점)
© Z. Nasreddine MD, JY. Lee 한국판. www.mocatest.org
"""
import re

# ────────────────────────────────────────────
# 유틸: 텍스트에서 숫자 추출
# ────────────────────────────────────────────
KOREAN_NUMBER_MAP = {
    "영": 0, "일": 1, "이": 2, "삼": 3, "사": 4,
    "오": 5, "육": 6, "칠": 7, "팔": 8, "구": 9,
    "하나": 1, "둘": 2, "셋": 3, "넷": 4, "다섯": 5,
    "여섯": 6, "일곱": 7, "여덟": 8, "아홉": 9,
}

def extract_numbers(text: str) -> list:
    """
    STT 텍스트에서 숫자 추출
    "이 일 팔 오 사" → [2, 1, 8, 5, 4]
    "21854"          → [2, 1, 8, 5, 4]
    "93 86 79 72 65" → [93, 86, 79, 72, 65]
    """
    arabic = re.findall(r'\d+', text)
    if arabic:
        if len(arabic) > 1:
            # 공백으로 구분된 숫자들 (계산용 2자리 허용)
            return [int(n) for n in arabic]
        # 단일 숫자열이고 3자리 이상이면 한 자리씩 분리
        single = arabic[0]
        if len(single) >= 3:
            return [int(d) for d in single]
        return [int(single)]

    # 한글 숫자 변환
    results = []
    tokens = text.replace(",", " ").split()
    for token in tokens:
        token = token.strip()
        if token in KOREAN_NUMBER_MAP:
            results.append(KOREAN_NUMBER_MAP[token])
    return results


# ────────────────────────────────────────────
# 1. 숫자 바로 따라 외우기 (1점)
# ────────────────────────────────────────────
def score_forward_digits(stt_text: str, forward_digits: list) -> int:
    numbers = extract_numbers(stt_text)
    return 1 if numbers == forward_digits else 0


# ────────────────────────────────────────────
# 2. 숫자 거꾸로 따라 외우기 (1점)
# ────────────────────────────────────────────
def score_backward_digits(stt_text: str, backward_answer: list) -> int:
    numbers = extract_numbers(stt_text)
    return 1 if numbers == backward_answer else 0


# ────────────────────────────────────────────
# 3. 화면 탭하기 (1점)
# clap_target 글자가 나올 때만 탭, 오류 2개 이상이면 0점
# ────────────────────────────────────────────
def score_tapping(tapped_indices: list, clap_sequence: list, clap_target: str) -> int:
    """
    tapped_indices: 탭 발생 시점의 글자 인덱스 리스트
    프론트엔드에서 오디오/화면 표시와 동기화하여 탭 발생 시점의 글자 인덱스를 전송.
    """
    errors = 0
    for i, char in enumerate(clap_sequence):
        tapped = (i in tapped_indices)
        if char == clap_target and not tapped:
            errors += 1
        elif char != clap_target and tapped:
            errors += 1
    return 0 if errors >= 2 else 1


# ────────────────────────────────────────────
# 4. 100에서 7씩 빼기 (3점)
# 정답: 93, 86, 79, 72, 65
# 각 뺄셈 독립 평가 (연속 오류 허용)
# ────────────────────────────────────────────
def score_serial_7(stt_text: str) -> int:
    """
    각 답이 이전 답에서 정확히 7을 뺐으면 정답
    연속 오류 허용: 92→85→78→71→64 (4개 정답 → 3점)
    0개: 0점 / 1개: 1점 / 2~3개: 2점 / 4~5개: 3점
    """
    numbers = extract_numbers(stt_text)
    if not numbers:
        return 0

    numbers = numbers[:5]
    correct = 0
    prev = 100

    for num in numbers:
        if prev - num == 7:
            correct += 1
        prev = num  # 정답/오답 무관하게 다음 기준점으로 사용

    if correct >= 4:
        return 3
    elif correct >= 2:
        return 2
    elif correct == 1:
        return 1
    else:
        return 0


# ────────────────────────────────────────────
# 통합 채점 함수
# ────────────────────────────────────────────
def score_attention(
    forward_stt: str,
    backward_stt: str,
    tapped_indices: list,
    serial7_stt: str,
    config: dict,
) -> dict:
    forward  = score_forward_digits(forward_stt,  config["forward_digits"])
    backward = score_backward_digits(backward_stt, config["backward_answer"])
    tapping  = score_tapping(tapped_indices, config["clap_sequence"], config["clap_target"])
    serial7  = score_serial_7(serial7_stt)

    return {
        "forward_digits":  forward,
        "backward_digits": backward,
        "tapping":         tapping,
        "serial_7":        serial7,
        "total":           forward + backward + tapping + serial7,
    }


# ────────────────────────────────────────────
# 테스트
# ────────────────────────────────────────────
if __name__ == "__main__":
    from version_manager import get_version_config

    for ver in ("MoCA-K", "K-MoCA"):
        cfg = get_version_config(ver)
        print(f"\n=== {ver} ===")
        correct_taps = [i for i, c in enumerate(cfg["clap_sequence"]) if c == cfg["clap_target"]]
        result = score_attention(
            forward_stt="21854",
            backward_stt="247",
            tapped_indices=correct_taps,
            serial7_stt="93 86 79 72 65",
            config=cfg,
        )
        print(result)

    print("\n=== 계산 케이스 ===")
    print(score_serial_7("93 86 79 72 65"))   # 3 (5개 정답)
    print(score_serial_7("92 85 78 71 64"))   # 3 (4개 연속정답)
    print(score_serial_7("90 83 76 69 62"))   # 3 (시작 틀려도 이후 4개 정답 — 독립 채점)
