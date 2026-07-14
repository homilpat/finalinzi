"""
MoCA-K 언어 채점 모듈
항목1: 문장 따라 말하기 (2점)
항목2: 단어 유창성 (1점)
총점: 3점
© Z. Nasreddine MD, JY. Lee 한국판. www.mocatest.org
"""

import os
import re
from difflib import SequenceMatcher

# STT 오인식 감안 유사도 임계값
SIMILARITY_THRESHOLD = 0.85

# ────────────────────────────────────────────
# 시장물건 화이트리스트 / 블랙리스트 로드 (모듈 초기화 시 1회)
# ────────────────────────────────────────────
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def _load_word_set(filename: str) -> frozenset:
    path = os.path.join(_BASE_DIR, filename)
    if not os.path.exists(path):
        return frozenset()
    with open(path, encoding="utf-8") as f:
        return frozenset(
            line.strip() for line in f
            if line.strip() and not line.startswith("#")
        )

MARKET_WHITELIST: frozenset = _load_word_set("market_whitelist.txt")


# ────────────────────────────────────────────
# 따라말하기 채점
# ────────────────────────────────────────────
def _normalize(text: str) -> str:
    text = re.sub(r'[^\w]', '', text)
    return text.strip()


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, _normalize(a), _normalize(b)).ratio()


def score_single_sentence(stt_text: str, correct_sentence: str) -> int:
    """
    STT 결과와 정답 문장 유사도 비교
    threshold 이상이면 1점

    원본 기준: 빠진 부분, 줄이거나 덧붙인 부분 없어야 함
    → 유사도 0.85 이상으로 처리
    """
    sim = _similarity(stt_text, correct_sentence)
    return 1 if sim >= SIMILARITY_THRESHOLD else 0


def score_repetition(
    sentence1_stt: str,
    sentence2_stt: str,
    sentences: list,
) -> dict:
    """
    따라말하기 2문장 채점

    Args:
        sentences: 버전별 정답 문장 목록 (config["sentences"])

    Returns:
        {
            "sentence1": 0~1,
            "sentence2": 0~1,
            "similarity1": float,
            "similarity2": float,
            "total": 0~2
        }
    """
    s1 = score_single_sentence(sentence1_stt, sentences[0])
    s2 = score_single_sentence(sentence2_stt, sentences[1])

    return {
        "sentence1":   s1,
        "sentence2":   s2,
        "similarity1": round(_similarity(sentence1_stt, sentences[0]), 2),
        "similarity2": round(_similarity(sentence2_stt, sentences[1]), 2),
        "total":       s1 + s2,
    }


# ────────────────────────────────────────────
# 단어 유창성 채점
# MoCA-K: 시장물건 1분간 11개 이상 → 1점
# K-MoCA: ㄱ으로 시작하는 단어 1분간 6개 이상 → 1점
# ────────────────────────────────────────────
def _starts_with_giyeok(word: str) -> bool:
    """유니코드 한글 음절 블록 기준 ㄱ 초성 여부 확인"""
    if not word:
        return False
    c = ord(word[0])
    if 0xAC00 <= c <= 0xD7A3:
        return (c - 0xAC00) // (21 * 28) == 0
    return False


def _is_proper_noun(word: str) -> bool:
    """
    K-MoCA 유창성 규칙: 고유명사 제외
    한글 자모 범위를 벗어난 대문자·숫자 혼합이나 명백한 지명/인명 패턴 감지.
    완전한 판별은 어렵고 STT 결과에는 고유명사가 드물므로 보수적으로 적용.
    """
    # 숫자만으로 이뤄진 경우
    if re.fullmatch(r'\d+', word):
        return True
    return False


def _deduplicate_stem(words: list) -> list:
    """
    K-MoCA 유창성 규칙: 접미사만 다른 낱말 중복 제거
    예) 가위, 가위질, 가위질하다 → 가위만 인정 (가장 짧은 것 기준)
    동일 어간(stem)이면 하나만 카운트.
    """
    accepted = []
    for word in words:
        # 기존 accepted 단어 중 현재 word가 접두어인 것이 있으면 중복
        is_stem_dup = any(
            (word != acc and (word.startswith(acc) or acc.startswith(word)))
            for acc in accepted
        )
        if not is_stem_dup:
            accepted.append(word)
    return accepted


def _validate_with_llm(words: list) -> list:
    """화이트리스트 미등록 단어를 Claude API로 검증 (use_llm=True 시만 호출)"""
    if not words:
        return []
    try:
        import anthropic
        client = anthropic.Anthropic()
        word_list = ", ".join(words)
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=300,
            messages=[{
                "role": "user",
                "content": (
                    f"다음 단어들 중 시장(마트·전통시장)에서 살 수 있는 물건인 단어만 "
                    f"쉼표로 구분해 정확히 답하세요. 없으면 '없음'이라고만 하세요.\n단어: {word_list}"
                ),
            }],
        )
        result = msg.content[0].text.strip()
        if result == "없음":
            return []
        # 쉼표로 split 후 strip하여 정확히 매칭 (부분 문자열 오탐 방지)
        approved = {w.strip() for w in result.split(",") if w.strip()}
        return [w for w in words if w in approved]
    except anthropic.AuthenticationError:
        print("[경고] Claude API 인증 실패 — ANTHROPIC_API_KEY 확인 필요")
        return []
    except Exception as e:
        print(f"[경고] LLM 검증 실패: {e}")
        return []


def score_verbal_fluency(
    stt_text: str,
    fluency_count: int,
    fluency_type: str = "시장물건",
    use_llm: bool = False,
) -> dict:
    """
    1분 동안 말한 단어 카운팅, fluency_count 이상이면 1점

    - MoCA-K(시장물건): MARKET_WHITELIST 필터 → 미등록 단어는 unknown으로 분리
      use_llm=True 이면 unknown을 Claude API로 추가 검증
    - K-MoCA(ㄱ으로 시작하는 단어): 유니코드 ㄱ 초성 필터
    """
    text   = re.sub(r'[^\w\s]', '', stt_text)
    tokens = text.split()

    # 중복 제거 (순서 보존)
    unique = []
    seen_set = set()
    for token in tokens:
        token = token.strip()
        if token and token not in seen_set:
            unique.append(token)
            seen_set.add(token)

    if fluency_type == "ㄱ으로 시작하는 단어":
        # 고유명사(숫자 등) 제외 후 ㄱ 초성 필터
        giyeok = [w for w in unique if _starts_with_giyeok(w) and not _is_proper_noun(w)]
        # 접미사만 다른 변형어 중복 제거 (가위/가위질 → 가위만)
        valid   = _deduplicate_stem(giyeok)
        unknown = []

    elif fluency_type == "시장물건" and MARKET_WHITELIST:
        valid   = [w for w in unique if w in MARKET_WHITELIST]
        unknown = [w for w in unique if w not in MARKET_WHITELIST]
        if use_llm and unknown:
            valid += _validate_with_llm(unknown)
            unknown = []

    else:
        # 화이트리스트 미로드 시 전체 카운트 (폴백)
        valid   = unique
        unknown = []

    count = len(valid)

    return {
        "words":   valid,
        "unknown": unknown,
        "count":   count,
        "total":   1 if count >= fluency_count else 0,
    }


# ────────────────────────────────────────────
# 통합 채점 함수
# ────────────────────────────────────────────
def score_language(
    sentence1_stt: str,
    sentence2_stt: str,
    fluency_stt: str,
    config: dict,
    use_llm: bool = False,
) -> dict:
    """
    언어 전체 채점

    Returns:
        {
            "repetition": {sentence1, sentence2, total},
            "fluency": {words, count, total},
            "total": 0~3
        }
    """
    repetition = score_repetition(sentence1_stt, sentence2_stt, config["sentences"])
    fluency    = score_verbal_fluency(fluency_stt, config["fluency_count"], config["fluency_type"], use_llm=use_llm)

    return {
        "repetition": repetition,
        "fluency":    fluency,
        "total":      repetition["total"] + fluency["total"],
    }


# ────────────────────────────────────────────
# 테스트
# ────────────────────────────────────────────
if __name__ == "__main__":
    from version_manager import get_version_config

    print("=== MoCA-K 따라말하기 ===")
    cfg = get_version_config("MoCA-K")
    print(score_repetition(
        "오늘 나를 도와줄 사람은 철수뿐이다",
        "강아지가 방에 들어오면 고양이는 의자 밑에 숨는다",
        cfg["sentences"],
    ))  # 2점

    print("\n=== K-MoCA 따라말하기 ===")
    cfg2 = get_version_config("K-MoCA")
    print(score_repetition(
        "칼날같이 날카로운 바위",
        "스물 일곱 개의 찬 맥주병이 냉장고에 있다",
        cfg2["sentences"],
    ))  # 2점

    print("\n=== 단어 유창성 ===")
    # MoCA-K: 시장물건 11개 이상 (화이트리스트 필터)
    r = score_verbal_fluency(
        "사과 배 감 포도 수박 참외 딸기 당근 양파 마늘 고추 오이 호박",
        cfg["fluency_count"],
        cfg["fluency_type"],
    )
    print(f"MoCA-K 유창성: {r['count']}개 → {r['total']}점 | valid={r['words']} | unknown={r['unknown']}")

    # 오염 케이스: 사람·장소 포함
    r2 = score_verbal_fluency(
        "사과 배 학교 사람 아이 당근 양파 마늘 자동차 고추 오이 호박",
        cfg["fluency_count"],
        cfg["fluency_type"],
    )
    print(f"오염 케이스: {r2['count']}개 → {r2['total']}점 | unknown={r2['unknown']}")

    # K-MoCA: ㄱ 단어 6개 이상
    r3 = score_verbal_fluency(
        "가위 가방 가구 기차 고양이 구두 나비 어머니",
        cfg2["fluency_count"],
        cfg2["fluency_type"],
    )
    print(f"K-MoCA 유창성: {r3['count']}개 → {r3['total']}점 | valid={r3['words']}")

    print("\n=== 통합 ===")
    result = score_language(
        sentence1_stt="오늘 나를 도와줄 사람은 철수뿐이다",
        sentence2_stt="강아지가 방에 들어오면 고양이는 의자 밑에 숨는다",
        fluency_stt="사과 배 감 포도 수박 참외 딸기 당근 양파 마늘 고추 오이",
        config=cfg,
    )
    print(result)
