"""
MoCA-K 지남력 채점 모듈
항목: 년, 월, 일, 요일, 장소(동읍면), 시군구
총점: 6점
GPS 역지오코딩으로 장소/시군구 정답키 자동 설정
© Z. Nasreddine MD, JY. Lee 한국판. www.mocatest.org
"""

import re
from datetime import datetime

# ────────────────────────────────────────────
# 날짜 정답키 (시스템 자동)
# ────────────────────────────────────────────
def get_date_answer_key() -> dict:
    """
    시스템 현재 날짜로 정답키 생성
    """
    now = datetime.now()
    weekday_map = {
        0: "월요일", 1: "화요일", 2: "수요일",
        3: "목요일", 4: "금요일", 5: "토요일", 6: "일요일"
    }
    return {
        "년": str(now.year),
        "월": str(now.month),
        "일": str(now.day),
        "요일": weekday_map[now.weekday()]
    }


# ────────────────────────────────────────────
# 날짜 채점
# ────────────────────────────────────────────
def score_date(stt_text: str, date_type: str, answer_key: dict) -> int:
    """
    date_type: "년" | "월" | "일" | "요일"
    날짜/요일은 하나라도 틀리면 0점 (원본 기준)
    """
    text = stt_text.strip()
    correct = answer_key[date_type]

    # 년도: "2025년" or "2025"
    if date_type == "년":
        return 1 if correct in text else 0

    # 월: "6월" or "6" or "유월" or "육월"
    # 단독 숫자 매칭: "6"이 "60"이나 "16" 속 숫자에 걸리지 않도록 단어 경계 체크
    if date_type == "월":
        month_korean = _month_to_korean(int(correct))
        digit_match = bool(re.search(r'(?<!\d)' + correct + r'(?!\d)', text))
        return 1 if (digit_match or month_korean in text) else 0

    # 일: "24일" or "24" (단어 경계 체크)
    if date_type == "일":
        return 1 if bool(re.search(r'(?<!\d)' + correct + r'(?!\d)', text)) else 0

    # 요일: "수요일" or "수요일" 전체 단어 매칭 (단일 글자 "수"는 오탐 위험으로 미사용)
    if date_type == "요일":
        return 1 if correct in text else 0

    return 0


def _month_to_korean(month: int) -> str:
    month_map = {
        1: "일월", 2: "이월", 3: "삼월", 4: "사월",
        5: "오월", 6: "유월", 7: "칠월", 8: "팔월",
        9: "구월", 10: "시월", 11: "십일월", 12: "십이월"
    }
    return month_map.get(month, "")


# ────────────────────────────────────────────
# GPS 역지오코딩 정답키 설정
# 카카오 API or 네이버 API 연동
# ────────────────────────────────────────────
def get_location_answer_key(lat: float, lng: float) -> dict:
    """
    GPS 좌표 → 동읍면 + 시군구 추출
    실제 앱에서는 카카오 역지오코딩 API 호출

    Returns:
        {"장소": "역삼동", "시군구": "강남구"}
    """
    try:
        import requests
        url = "https://dapi.kakao.com/v2/local/geo/coord2regioncode.json"
        headers = {"Authorization": "KakaoAK YOUR_API_KEY"}
        params = {"x": lng, "y": lat}
        res = requests.get(url, headers=headers, params=params, timeout=3)
        data = res.json()

        region = data["documents"][0]
        return {
            "장소": region.get("region_3depth_name", ""),  # 동읍면
            "시군구": region.get("region_2depth_name", "") # 시군구
        }
    except Exception:
        # API 실패 시 빈값 반환 (시연 환경에서 수동 설정 가능)
        return {"장소": "", "시군구": ""}


def set_location_answer_key_manual(dong: str, sigungu: str) -> dict:
    """
    GPS 없을 때 수동으로 정답키 설정
    시연 환경에서 사용
    """
    return {"장소": dong, "시군구": sigungu}


# ────────────────────────────────────────────
# 장소 채점
# ────────────────────────────────────────────
def score_location(stt_text: str, location_type: str, location_key: dict) -> int:
    """
    location_type: "장소" (동읍면) | "시군구"
    """
    text = stt_text.strip()
    correct = location_key.get(location_type, "")

    if not correct:
        return 0

    # "강남구" → "강남", "구로구" → "구로" (suffix만 제거, 중간 글자 보존)
    short = correct
    for suffix in ["구", "시", "군"]:
        if correct.endswith(suffix):
            short = correct[:-1]
            break
    return 1 if (correct in text or short in text) else 0


# ────────────────────────────────────────────
# 통합 채점 함수
# ────────────────────────────────────────────
def score_orientation(
    year_stt: str,
    month_stt: str,
    day_stt: str,
    weekday_stt: str,
    place_stt: str,
    sigungu_stt: str,
    location_key: dict = None
) -> dict:
    """
    지남력 전체 채점

    Args:
        year_stt ~ sigungu_stt: 각 항목 STT 결과
        location_key: GPS 역지오코딩 결과 {"장소": "역삼동", "시군구": "강남구"}
                      None이면 장소/시군구 0점 처리

    Returns:
        {
            "년": 0~1, "월": 0~1, "일": 0~1, "요일": 0~1,
            "장소": 0~1, "시군구": 0~1,
            "total": 0~6
        }
    """
    answer_key = get_date_answer_key()

    year    = score_date(year_stt,    "년",   answer_key)
    month   = score_date(month_stt,   "월",   answer_key)
    day     = score_date(day_stt,     "일",   answer_key)
    weekday = score_date(weekday_stt, "요일", answer_key)

    if location_key:
        place   = score_location(place_stt,   "장소",  location_key)
        sigungu = score_location(sigungu_stt, "시군구", location_key)
    else:
        place   = 0
        sigungu = 0

    return {
        "년":   year,
        "월":   month,
        "일":   day,
        "요일": weekday,
        "장소": place,
        "시군구": sigungu,
        "total": year + month + day + weekday + place + sigungu
    }


# ────────────────────────────────────────────
# 테스트
# ────────────────────────────────────────────
if __name__ == "__main__":
    now = datetime.now()
    print(f"현재 날짜: {now.year}년 {now.month}월 {now.day}일")

    loc_key = set_location_answer_key_manual("역삼동", "강남구")

    print("\n=== 정답 케이스 ===")
    result = score_orientation(
        year_stt=f"{now.year}년",
        month_stt=f"{now.month}월",
        day_stt=f"{now.day}일",
        weekday_stt="수요일",
        place_stt="역삼동",
        sigungu_stt="강남구",
        location_key=loc_key
    )
    print(result)

    print("\n=== 오답 케이스 ===")
    result2 = score_orientation(
        year_stt="2022년",
        month_stt="3월",
        day_stt="15일",
        weekday_stt="월요일",
        place_stt="신촌동",
        sigungu_stt="마포구",
        location_key=loc_key
    )
    print(result2)

    print("\n=== GPS 없는 경우 ===")
    result3 = score_orientation(
        year_stt=f"{now.year}년",
        month_stt=f"{now.month}월",
        day_stt=f"{now.day}일",
        weekday_stt="수요일",
        place_stt="역삼동",
        sigungu_stt="강남구",
        location_key=None
    )
    print(result3)
