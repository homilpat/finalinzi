"""
MoCA-K 총점 계산 모듈
모든 채점 모듈 통합 → 총점 계산 → MCI 판정
총점: 30점 (교육 6년 이하 +1점)
정상 기준: 23점 이상
© Z. Nasreddine MD, JY. Lee 한국판. www.mocatest.org
"""

from attention     import score_attention
from naming        import score_naming
from memory        import score_memory
from orientation   import score_orientation
from language      import score_language
from abstraction   import score_abstraction
from trail_making  import score_trail_making
from version_manager import get_version_config


# ────────────────────────────────────────────
# 교육수준 보정
# ────────────────────────────────────────────
def apply_education_correction(score: int, education_years: int) -> int:
    """
    학력 6년 이하 → +1점
    단, 30점 초과 불가
    """
    if education_years <= 6:
        return min(score + 1, 30)
    return score


# ────────────────────────────────────────────
# MCI 판정
# ────────────────────────────────────────────
def classify_mci(total_score: int) -> dict:
    """
    총점 기반 MCI 판정
    23점 이상: 정상
    23점 미만: MCI 의심

    Returns:
        {
            "label": 0 or 1,  # 0=정상, 1=MCI의심
            "interpretation": "정상" or "MCI 의심",
            "score": 총점
        }
    """
    is_mci = total_score < 23
    return {
        "label":          1 if is_mci else 0,
        "interpretation": "MCI 의심" if is_mci else "정상",
        "score":          total_score,
    }


# ────────────────────────────────────────────
# 통합 채점 함수
# ────────────────────────────────────────────
def score_total(
    # 길만들기
    trail_touch_points: list,
    canvas_width: int,
    canvas_height: int,
    # 드로잉 (CNN 결과 직접 받음)
    cube_score: int,        # 0~1
    clock_contour: int,     # 0~1
    clock_numbers: int,     # 0~1
    clock_hands: int,       # 0~1
    # 어휘력 (버전별 동물 3마리 STT 순서대로)
    naming_stts: list,
    # 기억력
    immediate1_stt: str,
    immediate2_stt: str,
    delayed_recall_stt: str,
    # 주의력
    forward_stt: str,
    backward_stt: str,
    tapped_indices: list,
    serial7_stt: str,
    # 언어
    sentence1_stt: str,
    sentence2_stt: str,
    fluency_stt: str,
    # 추상력 (버전 중립 네이밍: pair1=첫 번째 쌍, pair2=두 번째 쌍)
    abstraction_pair1_stt: str,
    abstraction_pair2_stt: str,
    # 지남력
    year_stt: str,
    month_stt: str,
    day_stt: str,
    weekday_stt: str,
    place_stt: str,
    sigungu_stt: str,
    location_key: dict,
    # 교육수준 및 버전
    education_years: int,
    version: str = "MoCA-K",
    use_llm: bool = False,
) -> dict:
    """
    전체 MoCA-K 채점

    Args:
        version: "MoCA-K" (기본) 또는 "K-MoCA"

    Returns:
        {
            "version": str,
            "sections": {각 섹션별 점수},
            "details": {각 섹션별 상세},
            "raw_score": 보정 전 총점,
            "final_score": 보정 후 총점,
            "education_correction": 0 or 1,
            "mci": {label, interpretation, score},
        }
    """
    config = get_version_config(version)

    # 1. 길만들기 (1점)
    trail = score_trail_making(trail_touch_points, canvas_width, canvas_height)

    # 2. 드로잉 (4점: 육면체1 + 시계3)
    drawing = {
        "cube":          cube_score,
        "clock_contour": clock_contour,
        "clock_numbers": clock_numbers,
        "clock_hands":   clock_hands,
        "total":         cube_score + clock_contour + clock_numbers + clock_hands,
    }

    # 3. 어휘력 (3점)
    naming = score_naming(naming_stts, config["animals"])

    # 4. 기억력/지연회상 (5점)
    memory = score_memory(
        immediate1_stt, immediate2_stt, delayed_recall_stt,
        config["memory_words"],
    )

    # 5. 주의력 (6점)
    attention = score_attention(
        forward_stt, backward_stt,
        tapped_indices,
        serial7_stt,
        config,
    )

    # 6. 언어 (3점)
    language = score_language(sentence1_stt, sentence2_stt, fluency_stt, config, use_llm=use_llm)

    # 7. 추상력 (2점)
    abstraction = score_abstraction(abstraction_pair1_stt, abstraction_pair2_stt, config)

    # 8. 지남력 (6점)
    orientation = score_orientation(
        year_stt, month_stt, day_stt, weekday_stt,
        place_stt, sigungu_stt, location_key,
    )

    # 총점 계산
    raw_score = (
        trail["total"] +
        drawing["total"] +
        naming["total"] +
        memory["total"] +
        attention["total"] +
        language["total"] +
        abstraction["total"] +
        orientation["total"]
    )

    # 교육수준 보정
    education_correction = 1 if education_years <= 6 else 0
    final_score = apply_education_correction(raw_score, education_years)

    # MCI 판정
    mci = classify_mci(final_score)

    return {
        "version": version,
        "sections": {
            "trail_making": trail["total"],       # /1
            "drawing":      drawing["total"],      # /4
            "naming":       naming["total"],       # /3
            "memory":       memory["total"],       # /5
            "attention":    attention["total"],    # /6
            "language":     language["total"],     # /3
            "abstraction":  abstraction["total"],  # /2
            "orientation":  orientation["total"],  # /6
        },
        "details": {
            "trail_making": trail,
            "drawing":      drawing,
            "naming":       naming,
            "memory":       memory,
            "attention":    attention,
            "language":     language,
            "abstraction":  abstraction,
            "orientation":  orientation,
        },
        "raw_score":            raw_score,
        "education_correction": education_correction,
        "final_score":          final_score,
        "mci":                  mci,
    }


# ────────────────────────────────────────────
# 테스트
# ────────────────────────────────────────────
if __name__ == "__main__":
    from trail_making import NODE_POSITIONS
    from datetime import datetime

    W, H = 400, 600
    def pos(node):
        rx, ry = NODE_POSITIONS[node]
        return (int(rx * W), int(ry * H))

    CORRECT_SEQ = ["1", "가", "2", "나", "3", "다", "4", "라", "5", "마"]
    now     = datetime.now()
    loc_key = {"장소": "역삼동", "시군구": "강남구"}

    _weekday_map = ["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"]
    correct_weekday = _weekday_map[now.weekday()]

    cfg_k = get_version_config("MoCA-K")
    taps_k = [i for i, c in enumerate(cfg_k["clap_sequence"]) if c == cfg_k["clap_target"]]

    print("=== MoCA-K 만점 케이스 ===")
    result = score_total(
        trail_touch_points=[pos(n) for n in CORRECT_SEQ],
        canvas_width=W, canvas_height=H,
        cube_score=1, clock_contour=1, clock_numbers=1, clock_hands=1,
        naming_stts=["사자", "코뿔소", "낙타"],
        immediate1_stt="얼굴 비단 교회 진달래 빨강",
        immediate2_stt="얼굴 비단 교회 진달래 빨강",
        delayed_recall_stt="얼굴 비단 교회 진달래 빨강",
        forward_stt="21854", backward_stt="247",
        tapped_indices=taps_k,
        serial7_stt="93 86 79 72 65",
        sentence1_stt="오늘 나를 도와줄 사람은 철수뿐이다",
        sentence2_stt="강아지가 방에 들어오면 고양이는 의자 밑에 숨는다",
        fluency_stt="사과 배 감 포도 수박 참외 딸기 당근 양파 마늘 고추",
        abstraction_pair1_stt="운송수단이에요",
        abstraction_pair2_stt="측정도구예요",
        year_stt=f"{now.year}년", month_stt=f"{now.month}월",
        day_stt=f"{now.day}일", weekday_stt=correct_weekday,
        place_stt="역삼동", sigungu_stt="강남구",
        location_key=loc_key,
        education_years=12,
        version="MoCA-K",
    )
    print(f"버전: {result['version']}")
    print(f"섹션별: {result['sections']}")
    print(f"원점수: {result['raw_score']}/30")
    print(f"교육보정: +{result['education_correction']}")
    print(f"최종점수: {result['final_score']}/30")
    print(f"MCI 판정: {result['mci']}")

    print("\n=== K-MoCA 만점 케이스 ===")
    cfg_km = get_version_config("K-MoCA")
    taps_km = [i for i, c in enumerate(cfg_km["clap_sequence"]) if c == cfg_km["clap_target"]]
    result2 = score_total(
        trail_touch_points=[pos(n) for n in CORRECT_SEQ],
        canvas_width=W, canvas_height=H,
        cube_score=1, clock_contour=1, clock_numbers=1, clock_hands=1,
        naming_stts=["사자", "박쥐", "낙타"],
        immediate1_stt="얼굴 비단 학교 피리 노랑",
        immediate2_stt="얼굴 비단 학교 피리 노랑",
        delayed_recall_stt="얼굴 비단 학교 피리 노랑",
        forward_stt="21854", backward_stt="247",
        tapped_indices=taps_km,
        serial7_stt="93 86 79 72 65",
        sentence1_stt="칼날같이 날카로운 바위",
        sentence2_stt="스물 일곱 개의 찬 맥주병이 냉장고에 있다",
        fluency_stt="가위 가방 가구 기차 고양이 구두",
        abstraction_pair1_stt="교통수단이에요",
        abstraction_pair2_stt="측량도구예요",
        year_stt=f"{now.year}년", month_stt=f"{now.month}월",
        day_stt=f"{now.day}일", weekday_stt=correct_weekday,
        place_stt="역삼동", sigungu_stt="강남구",
        location_key=loc_key,
        education_years=12,
        version="K-MoCA",
    )
    print(f"버전: {result2['version']}")
    print(f"섹션별: {result2['sections']}")
    print(f"원점수: {result2['raw_score']}/30")
    print(f"최종점수: {result2['final_score']}/30")
    print(f"MCI 판정: {result2['mci']}")
    print(f"추상력 상세: {result2['details']['abstraction']}")
