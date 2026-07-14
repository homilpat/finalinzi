"""
MoCA-K 기억력/지연회상 채점 모듈
항목: 지연회상 (버전별 5개 단어)
즉각회상은 채점 없음 (시행만)
총점: 5점
© Z. Nasreddine MD, JY. Lee 한국판. www.mocatest.org
"""

from difflib import SequenceMatcher

# 노인 ASR WER 10~22% 수준 (JAMIA 2023, Edinburgh ASR) 근거
# 음절 단위 edit distance 1 허용: 3음절↑ ratio ≥ 0.65
# 2음절 단어(비단·빨강)는 0.5 < 0.65 이므로 사실상 exact match 유지
FUZZY_THRESHOLD = 0.65


def _word_in_text(stt_text: str, target: str) -> bool:
    """
    1) 정확한 포함(in) 우선
    2) 실패 시 토큰별 음절 유사도 비교 (Levenshtein 기반 SequenceMatcher)
    예: "진달래"→"진달레"(0.67), "교회"→"교이"(0.5, 탈락) 처리
    """
    if target in stt_text:
        return True
    for token in stt_text.split():
        if SequenceMatcher(None, token, target).ratio() >= FUZZY_THRESHOLD:
            return True
    return False


# ────────────────────────────────────────────
# 즉각회상 (채점 없음 - 시행 기록만)
# ────────────────────────────────────────────
def record_immediate_recall(
    attempt1_stt: str,
    attempt2_stt: str,
    memory_words: list,
) -> dict:
    """
    즉각회상은 점수 없음. 어떤 단어 기억했는지만 기록.
    5분 후 지연회상에 활용.
    """
    recalled_1 = [w for w in memory_words if _word_in_text(attempt1_stt, w)]
    recalled_2 = [w for w in memory_words if _word_in_text(attempt2_stt, w)]

    return {
        "attempt1": recalled_1,
        "attempt2": recalled_2,
        "score":    None,  # 즉각회상은 채점 없음
    }


# ────────────────────────────────────────────
# 지연회상 채점 (5점)
# 단서 없이 자발적으로 회상한 단어만 점수
# ────────────────────────────────────────────
def score_delayed_recall(stt_text: str, memory_words: list) -> dict:
    """
    STT 텍스트에서 버전별 5개 단어 포함 여부 확인
    단서 없이 말한 것만 점수 (1점/단어)
    """
    recalled = []
    scores   = {}

    for word in memory_words:
        if _word_in_text(stt_text, word):
            recalled.append(word)
            scores[word] = 1
        else:
            scores[word] = 0

    return {
        "recalled": recalled,
        "scores":   scores,
        "total":    len(recalled),
    }


# ────────────────────────────────────────────
# 단서 후 회상 (선택사항 - 채점 안됨)
# 임상 정보용으로만 기록
# ────────────────────────────────────────────
def record_cued_recall(
    word: str,
    category_cue_response: str = None,
    multiple_choice_response: str = None,
) -> dict:
    """
    범주단서/다중선택 후 회상 기록
    점수는 주지 않음 - 임상 분석용 데이터만 수집

    인출 문제 → 단서로 향상됨
    저장 문제 → 단서로도 향상 안됨
    """
    correct_with_category = (
        category_cue_response is not None and
        word in category_cue_response
    )
    correct_with_multiple = (
        multiple_choice_response is not None and
        word in multiple_choice_response
    )

    return {
        "word":                    word,
        "category_cue_correct":    correct_with_category,
        "multiple_choice_correct": correct_with_multiple,
        "score":                   0,  # 단서 후 회상은 항상 0점
        "clinical_note": (
            "인출 문제 의심" if (correct_with_category or correct_with_multiple)
            else "저장 문제 의심"
        ),
    }


# ────────────────────────────────────────────
# 통합 채점 함수
# ────────────────────────────────────────────
def score_memory(
    immediate_attempt1_stt: str,
    immediate_attempt2_stt: str,
    delayed_recall_stt: str,
    memory_words: list,
) -> dict:
    """
    기억력 전체 채점

    Args:
        memory_words: 버전별 기억 단어 목록 (config["memory_words"])

    Returns:
        {
            "immediate_recall": {attempt1, attempt2, score: None},
            "delayed_recall": {recalled, scores, total},
            "total": 0~5
        }
    """
    immediate = record_immediate_recall(
        immediate_attempt1_stt,
        immediate_attempt2_stt,
        memory_words,
    )
    delayed = score_delayed_recall(delayed_recall_stt, memory_words)

    return {
        "immediate_recall": immediate,
        "delayed_recall":   delayed,
        "total":            delayed["total"],  # 즉각회상 점수 없음
    }


# ────────────────────────────────────────────
# 테스트
# ────────────────────────────────────────────
if __name__ == "__main__":
    from version_manager import get_version_config

    for ver in ("MoCA-K", "K-MoCA"):
        cfg   = get_version_config(ver)
        words = cfg["memory_words"]
        print(f"\n=== {ver} 기억단어: {words} ===")

        result = score_memory(
            immediate_attempt1_stt=" ".join(words[:3]),
            immediate_attempt2_stt=" ".join(words[:4]),
            delayed_recall_stt=" ".join(words),
            memory_words=words,
        )
        print(f"즉각회상: {result['immediate_recall']}")
        print(f"지연회상: {result['delayed_recall']}")
        print(f"총점: {result['total']}/5")

    print("\n=== 단서 후 회상 (임상정보용) ===")
    words = get_version_config("MoCA-K")["memory_words"]
    print(record_cued_recall(words[3], category_cue_response=words[3]))
    print(record_cued_recall(words[4], category_cue_response="모르겠어요"))
