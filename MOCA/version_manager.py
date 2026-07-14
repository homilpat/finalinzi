"""
MoCA 버전 관리 모듈
MoCA-K ↔ K-MoCA 6개월 로테이션
© Z. Nasreddine MD, JY. Lee 한국판. www.mocatest.org
"""

from datetime import datetime, timedelta

# ────────────────────────────────────────────
# 버전별 문항 정의
# ────────────────────────────────────────────
VERSIONS = {
    "MoCA-K": {
        "animals": [
            {"key": "lion",  "label": "사자",   "answers": ["사자"]},
            {"key": "rhino", "label": "코뿔소", "answers": ["코뿔소", "뿔소"]},
            {"key": "camel", "label": "낙타",   "answers": ["낙타", "약대"]},
        ],
        "memory_words":    ["얼굴", "비단", "교회", "진달래", "빨강"],
        "memory_cues": {
            "얼굴":   {"범주": "신체의 일부분", "선택지": ["코", "얼굴", "손"]},
            "비단":   {"범주": "옷감",          "선택지": ["나일론", "면", "비단"]},
            "교회":   {"범주": "건물",          "선택지": ["교회", "학교", "병원"]},
            "진달래": {"범주": "꽃",            "선택지": ["장미", "진달래", "동백"]},
            "빨강":   {"범주": "색깔",          "선택지": ["빨강", "파랑", "초록"]},
        },
        "fluency_type":    "시장물건",
        "fluency_count":   11,
        "sentences": [
            "오늘 나를 도와줄 사람은 철수뿐이다",
            "강아지가 방에 들어오면 고양이는 의자 밑에 숨는다"
        ],
        "abstraction_pairs": [
            ("기차", "자전거"),
            ("시계", "자")
        ],
        "abstraction_correct": {
            "기차-자전거": ["운송수단", "교통수단", "탈것", "탈 것", "이동수단", "여행", "교통", "굴러간다"],
            "시계-자":     ["측량", "측정", "재는", "측량도구", "측정도구"]
        },
        "abstraction_incorrect": {
            "기차-자전거": ["바퀴"],
            "시계-자":     ["숫자", "눈금"]
        },
        "forward_digits":  [2, 1, 8, 5, 4],
        "backward_digits": [7, 4, 2],
        "backward_answer": [2, 4, 7],
        "clap_sequence":   list("바나가다차파가가사아자나가바가아라마가가가사가차하바가가나"),
        "clap_target":     "가",
        "serial7_start":   100,
    },
    "K-MoCA": {
        "animals": [
            {"key": "lion",  "label": "사자", "answers": ["사자"]},
            {"key": "bat",   "label": "박쥐", "answers": ["박쥐"]},
            {"key": "camel", "label": "낙타", "answers": ["낙타", "약대"]},
        ],
        "memory_words":    ["얼굴", "비단", "학교", "피리", "노랑"],
        "memory_cues": {
            "얼굴": {"범주": "신체의 일부", "선택지": ["코", "얼굴", "손"]},
            "비단": {"범주": "옷감",        "선택지": ["면", "삼베", "비단"]},
            "학교": {"범주": "건물",        "선택지": ["학교", "교회", "병원"]},
            "피리": {"범주": "악기",        "선택지": ["가야금", "피리", "장고"]},
            "노랑": {"범주": "색깔",        "선택지": ["노랑", "빨강", "파랑"]},
        },
        "fluency_type":    "ㄱ으로 시작하는 단어",
        "fluency_count":   6,
        "sentences": [
            "칼날같이 날카로운 바위",
            "스물 일곱 개의 찬 맥주병이 냉장고에 있다"
        ],
        "abstraction_pairs": [
            ("기차", "비행기"),
            ("시계", "저울")
        ],
        "abstraction_correct": {
            "기차-비행기": ["교통수단", "운송수단", "여행수단", "탈것", "탈 것", "이동수단", "타고 여행"],
            "시계-저울":   ["측정도구", "측량도구", "측정", "측량", "재는", "가지고 잰다"]
        },
        "abstraction_incorrect": {
            "기차-비행기": ["바퀴"],
            "시계-저울":   ["숫자"]
        },
        "forward_digits":  [2, 1, 8, 5, 4],
        "backward_digits": [7, 4, 2],
        "backward_answer": [2, 4, 7],
        "clap_sequence":   list("토화월수일화월월목금토화월토월금목금월월월목월일화토월월화"),
        "clap_target":     "월",
        "serial7_start":   100,
    }
}

ROTATION_MONTHS = 6  # 버전 교체 주기


# ────────────────────────────────────────────
# 버전 결정
# ────────────────────────────────────────────
def get_next_version(last_version: str = None, last_assessed_at: datetime = None) -> str:
    """
    다음 검사 버전 결정
    - 첫 검사: MoCA-K
    - 이전 버전 있으면: 6개월 경과 시 교체
    """
    if last_version is None:
        return "MoCA-K"

    if last_assessed_at:
        months_passed = (datetime.now() - last_assessed_at).days / 30
        if months_passed < ROTATION_MONTHS:
            return last_version  # 6개월 안됐으면 같은 버전 유지

    # 6개월 이상 경과 → 교체
    return "K-MoCA" if last_version == "MoCA-K" else "MoCA-K"


def get_version_config(version: str) -> dict:
    """버전별 문항 설정 반환"""
    return VERSIONS.get(version, VERSIONS["MoCA-K"])


# ────────────────────────────────────────────
# 테스트
# ────────────────────────────────────────────
if __name__ == "__main__":
    print("=== 첫 검사 ===")
    print(get_next_version())  # MoCA-K

    print("\n=== 6개월 미만 ===")
    recent = datetime.now() - timedelta(days=90)
    print(get_next_version("MoCA-K", recent))  # MoCA-K 유지

    print("\n=== 6개월 이상 ===")
    old = datetime.now() - timedelta(days=200)
    print(get_next_version("MoCA-K", old))   # K-MoCA
    print(get_next_version("K-MoCA", old))   # MoCA-K

    print("\n=== MoCA-K 설정 ===")
    config = get_version_config("MoCA-K")
    print(f"기억단어: {config['memory_words']}")
    print(f"유창성: {config['fluency_type']} {config['fluency_count']}개↑")

    print("\n=== K-MoCA 설정 ===")
    config2 = get_version_config("K-MoCA")
    print(f"기억단어: {config2['memory_words']}")
    print(f"유창성: {config2['fluency_type']} {config2['fluency_count']}개↑")
