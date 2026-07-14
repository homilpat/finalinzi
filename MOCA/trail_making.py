"""
MoCA-K 길만들기 채점 모듈
항목: 1-가-2-나-3-다-4-라-5-마 순서대로 선 잇기
총점: 1점
터치 캔버스에서 노드 통과 순서 검증
© Z. Nasreddine MD, JY. Lee 한국판. www.mocatest.org
"""

import math

CORRECT_SEQUENCE = ["1", "가", "2", "나", "3", "다", "4", "라", "5", "마"]

# 노드 위치 (캔버스 비율 0~1) — PDF 문제지 배치 기준
NODE_POSITIONS = {
    "마": (0.30, 0.10),  # 끝
    "가": (0.62, 0.14),
    "5":  (0.07, 0.37),
    "나": (0.45, 0.42),
    "2":  (0.66, 0.30),
    "1":  (0.24, 0.56),  # 시작
    "라": (0.10, 0.70),
    "4":  (0.47, 0.70),
    "3":  (0.68, 0.68),
    "다": (0.18, 0.87),
}
NODE_RADIUS = 0.07


# ────────────────────────────────────────────
# 터치 경로 → 노드 통과 순서 추출
# ────────────────────────────────────────────
def extract_node_sequence(touch_points: list, canvas_width: int, canvas_height: int) -> list:
    """
    연속 중복만 제거 (같은 노드 연달아 = 1개)
    재방문 허용 (즉각수정 처리 위해)
    """
    sequence = []
    last = None

    for (px, py) in touch_points:
        rx, ry = px / canvas_width, py / canvas_height
        for node, (nx, ny) in NODE_POSITIONS.items():
            if math.sqrt((rx - nx) ** 2 + (ry - ny) ** 2) <= NODE_RADIUS:
                if node != last:
                    sequence.append(node)
                    last = node
                break

    return sequence


# ────────────────────────────────────────────
# 즉각 수정 감지
# ────────────────────────────────────────────
def detect_immediate_correction(raw_sequence: list) -> list:
    """
    잘못된 노드 직후 올바른 노드로 가면 오류 무시
    ex) 1→나(실수)→가(수정)→2→나→3... → 정답
    """
    result = []
    correct_idx = 0

    for i, node in enumerate(raw_sequence):
        if correct_idx >= len(CORRECT_SEQUENCE):
            break
        expected = CORRECT_SEQUENCE[correct_idx]

        if node == expected:
            result.append(node)
            correct_idx += 1
        else:
            next_node = raw_sequence[i + 1] if i + 1 < len(raw_sequence) else None
            if next_node != expected:
                result.append(node)  # 실제 오류

    return result


# ────────────────────────────────────────────
# 채점
# ────────────────────────────────────────────
def score_trail_making(touch_points: list, canvas_width: int, canvas_height: int) -> dict:
    """
    Returns:
        {
            "raw_sequence": [...],
            "corrected_sequence": [...],
            "correct": True/False,
            "total": 0~1
        }
    """
    raw       = extract_node_sequence(touch_points, canvas_width, canvas_height)
    corrected = detect_immediate_correction(raw)
    correct   = (corrected == CORRECT_SEQUENCE)

    return {
        "raw_sequence":       raw,
        "corrected_sequence": corrected,
        "correct":            correct,
        "total":              1 if correct else 0
    }


# ────────────────────────────────────────────
# 노드 픽셀 좌표 반환 (Flutter 앱 배치용)
# ────────────────────────────────────────────
def get_node_positions(canvas_width: int, canvas_height: int) -> dict:
    return {
        node: (int(rx * canvas_width), int(ry * canvas_height))
        for node, (rx, ry) in NODE_POSITIONS.items()
    }


# ────────────────────────────────────────────
# 테스트
# ────────────────────────────────────────────
if __name__ == "__main__":
    W, H = 400, 600

    def pos(node):
        rx, ry = NODE_POSITIONS[node]
        return (int(rx * W), int(ry * H))

    print("=== 정답 ===")
    print(score_trail_making([pos(n) for n in CORRECT_SEQUENCE], W, H))  # 1점

    print("\n=== 오답 (순서 틀림) ===")
    print(score_trail_making([pos(n) for n in ["1","나","2","가","3","다","4","라","5","마"]], W, H))  # 0점

    print("\n=== 즉각수정 허용 ===")
    print(score_trail_making([pos(n) for n in ["1","나","가","2","나","3","다","4","라","5","마"]], W, H))  # 1점

    print("\n=== 노드 좌표 ===")
    print(get_node_positions(W, H))
