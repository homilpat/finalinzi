"""
MoCA-K 추상력 채점 모듈
항목: 두 단어의 공통점 찾기 (버전별 2쌍)
  MoCA-K: 기차-자전거 / 시계-자
  K-MoCA: 기차-비행기 / 시계-저울
총점: 2점
© Z. Nasreddine MD, JY. Lee 한국판. www.mocatest.org
"""

from difflib import SequenceMatcher

FUZZY_THRESHOLD = 0.65


def _word_in_text(stt_text: str, target: str) -> bool:
    """
    정답 키워드 매칭에만 사용 (오답 키워드는 exact match 유지)
    예: "운송수다"→"운송수단"(0.75), "측량"→"측량"(1.0)
    """
    if target in stt_text:
        return True
    for token in stt_text.split():
        if SequenceMatcher(None, token, target).ratio() >= FUZZY_THRESHOLD:
            return True
    return False


# ────────────────────────────────────────────
# 개별 항목 채점
# ────────────────────────────────────────────
def score_single_abstraction(
    stt_text: str,
    correct_keywords: list,
    incorrect_keywords: list,
) -> int:
    """
    STT 텍스트에서 정답/오답 키워드 검사
    오답 키워드 있으면 0점 (원본 기준)
    정답 키워드 있으면 1점
    """
    text = stt_text.strip()

    # 오답 키워드: exact match (fuzzy 오적용으로 억울하게 0점 받으면 안됨)
    for kw in incorrect_keywords:
        if kw in text:
            return 0

    # 정답 키워드: fuzzy 허용 (STT 오인식 보정)
    for kw in correct_keywords:
        if _word_in_text(text, kw):
            return 1

    return 0


# ────────────────────────────────────────────
# 통합 채점 함수
# ────────────────────────────────────────────
def score_abstraction(
    pair1_stt: str,
    pair2_stt: str,
    config: dict,
) -> dict:
    """
    버전별 추상력 2쌍 채점

    config 키:
        abstraction_pairs:    [("기차","자전거"), ("시계","자")]
        abstraction_correct:  {"기차-자전거": [...], "시계-자": [...]}
        abstraction_incorrect: {"기차-자전거": [...], "시계-자": [...]}

    Returns:
        {
            "pair1": 0~1,
            "pair2": 0~1,
            "pair1_name": "기차-자전거",
            "pair2_name": "시계-자",
            "total": 0~2
        }
    """
    pairs = config["abstraction_pairs"]
    key1  = f"{pairs[0][0]}-{pairs[0][1]}"
    key2  = f"{pairs[1][0]}-{pairs[1][1]}"

    score1 = score_single_abstraction(
        pair1_stt,
        config["abstraction_correct"][key1],
        config["abstraction_incorrect"][key1],
    )
    score2 = score_single_abstraction(
        pair2_stt,
        config["abstraction_correct"][key2],
        config["abstraction_incorrect"][key2],
    )

    return {
        "pair1":      score1,
        "pair2":      score2,
        "pair1_name": key1,
        "pair2_name": key2,
        "total":      score1 + score2,
    }


# ────────────────────────────────────────────
# 테스트
# ────────────────────────────────────────────
if __name__ == "__main__":
    from version_manager import get_version_config

    print("=== MoCA-K (기차-자전거 / 시계-자) ===")
    cfg = get_version_config("MoCA-K")
    print(score_abstraction("둘 다 운송수단이에요", "측정도구예요",  cfg))  # 2점
    print(score_abstraction("바퀴가 있어요",        "숫자가 있어요", cfg))  # 0점 (명시적 오답)
    print(score_abstraction("모르겠어요",           "모르겠어요",    cfg))  # 0점

    print("\n=== K-MoCA (기차-비행기 / 시계-저울) ===")
    cfg2 = get_version_config("K-MoCA")
    print(score_abstraction("둘 다 교통수단이에요", "측량도구예요",  cfg2))  # 2점
    print(score_abstraction("날개가 있어요",        "눈금이 있어요", cfg2))  # 0점 (시계-저울 오답)
