"""
MoCA 어휘력 채점 모듈
항목: 동물 이름 맞추기 (버전별 3마리)
  MoCA-K: 사자 / 코뿔소 / 낙타
  K-MoCA: 사자 / 박쥐   / 낙타
총점: 3점
© Z. Nasreddine MD, JY. Lee 한국판. www.mocatest.org
"""

from difflib import SequenceMatcher

FUZZY_THRESHOLD = 0.65


def _word_in_text(stt_text: str, target: str) -> bool:
    if target in stt_text:
        return True
    for token in stt_text.split():
        if SequenceMatcher(None, token, target).ratio() >= FUZZY_THRESHOLD:
            return True
    return False


def score_single_animal(stt_text: str, answers: list) -> int:
    """answers 리스트 중 하나라도 매칭되면 1점"""
    text = stt_text.strip()
    for keyword in answers:
        if _word_in_text(text, keyword):
            return 1
    return 0


def score_naming(stt_results: list, animals: list) -> dict:
    """
    버전별 동물 3마리 채점

    Args:
        stt_results: STT 결과 리스트 (동물 순서대로, 3개)
        animals:     config["animals"] — [{"key":..., "label":..., "answers":[...]}]

    Returns:
        {
            "<key1>": 0~1,
            "<key2>": 0~1,
            "<key3>": 0~1,
            "total":  0~3
        }
    """
    result = {}
    total = 0
    for i, animal in enumerate(animals):
        stt = stt_results[i] if i < len(stt_results) else ""
        s = score_single_animal(stt, animal["answers"])
        result[animal["key"]] = s
        total += s
    result["total"] = total
    return result


# ────────────────────────────────────────────
# 테스트
# ────────────────────────────────────────────
if __name__ == "__main__":
    from version_manager import get_version_config

    print("=== MoCA-K (사자/코뿔소/낙타) ===")
    cfg = get_version_config("MoCA-K")
    print(score_naming(["사자", "코뿔소", "낙타"], cfg["animals"]))   # 3점
    print(score_naming(["사자", "뿔소",   "약대"], cfg["animals"]))   # 3점 (유사표현)
    print(score_naming(["호랑이", "코끼리", "말"], cfg["animals"]))   # 0점

    print("\n=== K-MoCA (사자/박쥐/낙타) ===")
    cfg2 = get_version_config("K-MoCA")
    print(score_naming(["사자", "박쥐", "낙타"], cfg2["animals"]))    # 3점
    print(score_naming(["사자", "코뿔소", "낙타"], cfg2["animals"]))  # 2점 (코뿔소는 오답)
    print(score_naming(["호랑이", "박쥐", "말"],  cfg2["animals"]))   # 1점
