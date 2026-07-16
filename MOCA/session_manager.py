"""
MoCA 검사 세션 관리 모듈
검사 진행 흐름, 타이머, 문항 순서, 5분 대기 관리
"""

import time
from datetime import datetime
from enum import Enum
from version_manager import get_version_config, get_next_version


# ────────────────────────────────────────────
# 검사 상태 정의
# ────────────────────────────────────────────
class SessionState(Enum):
    IDLE            = "idle"
    READY           = "ready"
    IN_PROGRESS     = "in_progress"
    WAITING_5MIN    = "waiting_5min"   # 지연회상 대기
    DELAYED_RECALL  = "delayed_recall"
    COMPLETED       = "completed"


# ────────────────────────────────────────────
# 문항 순서 정의
# ────────────────────────────────────────────
ITEM_SEQUENCE = [
    "trail_making",       # 1. 길만들기
    "cube",               # 2. 육면체
    "clock",              # 3. 시계그리기
    "naming",             # 4. 어휘력
    "memory_immediate",   # 5. 기억력 즉각회상 (채점없음)
    "forward_digits",     # 6. 숫자 바로
    "backward_digits",    # 7. 숫자 거꾸로
    "clapping",           # 8. 손뼉치기
    "serial_7",           # 9. 100-7 계산
    "sentence_repeat",    # 10. 따라말하기
    "verbal_fluency",     # 11. 단어유창성
    "abstraction",        # 12. 추상력
    "delayed_recall",     # 13. 지연회상 (5분 후)
    "orientation",        # 14. 지남력
]

# 문항별 예상 소요시간 (초)
ITEM_DURATION = {
    "trail_making":     60,
    "cube":             45,
    "clock":            60,
    "naming":           30,
    "memory_immediate": 60,
    "forward_digits":   20,
    "backward_digits":  20,
    "clapping":         35,
    "serial_7":         45,
    "sentence_repeat":  30,
    "verbal_fluency":   70,   # 1분 + 여유
    "abstraction":      30,
    "delayed_recall":   60,
    "orientation":      40,
}

DELAYED_RECALL_WAIT = 300  # 5분 (초)


# ────────────────────────────────────────────
# 세션 클래스
# ────────────────────────────────────────────
class MoCASession:
    def __init__(self, user_id: str, version: str, education_years: int):
        self.user_id         = user_id
        self.version         = version
        self.config          = get_version_config(version)
        self.education_years = education_years
        self.state           = SessionState.IDLE
        self.current_item    = None
        self.item_index      = 0
        self.responses       = {}       # 항목별 응답 저장
        self.scores          = {}       # 항목별 점수 저장
        self.started_at      = None
        self.completed_at    = None
        self.memory_encoded_at = None   # 기억력 인코딩 시각
        self._timer          = None


    # ────────────────────────────────────────
    # 세션 시작
    # ────────────────────────────────────────
    def start(self):
        self.state      = SessionState.IN_PROGRESS
        self.started_at = datetime.now()
        self.item_index = 0
        self.current_item = ITEM_SEQUENCE[0]
        print(f"[세션 시작] 버전: {self.version} | 사용자: {self.user_id}")
        return self._get_current_item_info()


    # ────────────────────────────────────────
    # 다음 문항으로
    # ────────────────────────────────────────
    def next_item(self, response=None, force_skip_wait=False):
        """
        현재 문항 응답 저장 후 다음 문항 반환
        force_skip_wait=True: 5분 대기 조건 무시하고 즉시 진행 (시연용)
        """
        if response is not None:
            self.responses[self.current_item] = response

        # 기억력 즉각회상 완료 → 인코딩 시각 기록
        if self.current_item == "memory_immediate":
            self.memory_encoded_at = datetime.now()

        # 지연회상 직전 → 5분 경과 확인 (item_index 증가 전에 체크)
        # 버그 방지: 대기 중 return 시 item_index를 늘리지 않아야 재호출 때 정상 진행
        next_index = self.item_index + 1
        if not force_skip_wait and next_index < len(ITEM_SEQUENCE) and ITEM_SEQUENCE[next_index] == "delayed_recall":
            wait = self._check_delayed_recall_wait()
            if wait > 0:
                self.state = SessionState.WAITING_5MIN
                return {
                    "status": "waiting",
                    "wait_seconds": int(wait),
                    "message": f"지연회상까지 {int(wait)}초 남았습니다."
                }

        self.item_index += 1
        self.state = SessionState.IN_PROGRESS

        # 검사 완료
        if self.item_index >= len(ITEM_SEQUENCE):
            return self._complete()

        self.current_item = ITEM_SEQUENCE[self.item_index]
        return self._get_current_item_info()


    # ────────────────────────────────────────
    # 5분 대기 확인
    # ────────────────────────────────────────
    def _check_delayed_recall_wait(self) -> float:
        """
        기억력 인코딩 후 5분 경과 여부 확인
        반환값: 남은 대기 시간 (초), 0이면 바로 진행 가능
        """
        if self.memory_encoded_at is None:
            return 0
        elapsed = (datetime.now() - self.memory_encoded_at).total_seconds()
        remaining = DELAYED_RECALL_WAIT - elapsed
        return max(0, remaining)


    # ────────────────────────────────────────
    # 현재 문항 정보 반환
    # ────────────────────────────────────────
    def _get_current_item_info(self) -> dict:
        item = self.current_item
        config = self.config

        item_info = {
            "item":     item,
            "index":    self.item_index + 1,
            "total":    len(ITEM_SEQUENCE),
            "duration": ITEM_DURATION.get(item, 30),
            "status":   "in_progress"
        }

        # 문항별 필요 데이터 추가
        if item == "memory_immediate":
            item_info["words"] = config["memory_words"]

        elif item == "forward_digits":
            item_info["digits"]     = config["forward_digits"]
            item_info["tts_script"] = " ".join(str(d) for d in config["forward_digits"])

        elif item == "backward_digits":
            item_info["digits"]     = config["backward_digits"]
            item_info["tts_script"] = " ".join(str(d) for d in config["backward_digits"])

        elif item == "clapping":
            item_info["sequence"]   = config["clap_sequence"]
            item_info["target"]     = config["clap_target"]
            item_info["tts_script"] = " ".join(config["clap_sequence"])

        elif item == "sentence_repeat":
            item_info["sentences"]  = config["sentences"]

        elif item == "verbal_fluency":
            item_info["fluency_type"]  = config["fluency_type"]
            item_info["fluency_count"] = config["fluency_count"]
            item_info["timer"]         = 60

        elif item == "abstraction":
            item_info["pairs"] = config["abstraction_pairs"]

        elif item == "delayed_recall":
            item_info["words"] = config["memory_words"]

        return item_info


    # ────────────────────────────────────────
    # 세션 완료
    # ────────────────────────────────────────
    def _complete(self) -> dict:
        self.state        = SessionState.COMPLETED
        self.completed_at = datetime.now()
        duration          = (self.completed_at - self.started_at).total_seconds()

        print(f"[세션 완료] 소요시간: {int(duration)}초")

        return {
            "status":    "completed",
            "version":   self.version,
            "responses": self.responses,
            "duration":  int(duration),
        }


    # ────────────────────────────────────────
    # 세션 상태 요약
    # ────────────────────────────────────────
    def get_status(self) -> dict:
        return {
            "user_id":      self.user_id,
            "version":      self.version,
            "state":        self.state.value,
            "current_item": self.current_item,
            "progress":     f"{self.item_index}/{len(ITEM_SEQUENCE)}",
        }


# ────────────────────────────────────────────
# 세션 팩토리
# ────────────────────────────────────────────
def create_session(
    user_id: str,
    education_years: int,
    last_version: str = None,
    last_assessed_at: datetime = None
) -> MoCASession:
    """
    사용자 정보 기반 세션 생성
    버전 자동 결정
    """
    version = get_next_version(last_version, last_assessed_at)
    return MoCASession(user_id, version, education_years)


# ────────────────────────────────────────────
# 테스트
# ────────────────────────────────────────────
if __name__ == "__main__":
    print("=== 세션 생성 ===")
    session = create_session(
        user_id="user_001",
        education_years=12,
        last_version=None
    )

    print("\n=== 세션 시작 ===")
    item = session.start()
    print(f"첫 문항: {item}")

    print("\n=== 문항 순서 진행 ===")
    for i in range(5):
        result = session.next_item(response=f"테스트응답_{i}")
        print(f"→ {result.get('item', result.get('status'))}")

    print("\n=== 세션 상태 ===")
    print(session.get_status())

    print("\n=== 5분 대기 확인 ===")
    wait = session._check_delayed_recall_wait()
    print(f"남은 대기: {wait:.0f}초")
