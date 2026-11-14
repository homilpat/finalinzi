"""
MoCA-K 육면체 그리기 채점 모듈
데이터: QuickDraw cube 데이터셋
모델: CNN (정상/비정상 이진 분류)
총점: 1점
채점기준:
  - 3차원이어야 함
  - 모든 선이 그려져야 함
  - 덧그린 선 없어야 함
  - 선들이 대체로 평행하고 길이 비슷
© Z. Nasreddine MD, JY. Lee 한국판. www.mocatest.org
"""

import numpy as np
import cv2
import math
import os


# ────────────────────────────────────────────
# 1. 전처리
# ────────────────────────────────────────────
def preprocess_image(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    resized = cv2.resize(binary, (256, 256))
    return resized


# ────────────────────────────────────────────
# 2. 선분 추출
# ────────────────────────────────────────────
def extract_lines(image: np.ndarray) -> list:
    """
    Hough Line Transform으로 선분 추출
    Returns: [(x1,y1,x2,y2,length,angle), ...]
    """
    processed = preprocess_image(image)
    edges = cv2.Canny(processed, 50, 150)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi/180,
        threshold=40,
        minLineLength=30,
        maxLineGap=8
    )

    if lines is None:
        return []

    result = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = math.sqrt((x2-x1)**2 + (y2-y1)**2)
        angle  = math.degrees(math.atan2(y2-y1, x2-x1)) % 180
        result.append((x1, y1, x2, y2, length, angle))

    return result


def _check_length_similarity(lines: list) -> bool:
    """같은 방향(15도 단위 그룹) 선분끼리 길이 변동계수(CV)가 0.5 이하인지 확인"""
    groups: dict = {}
    for _, _, _, _, length, angle in lines:
        bucket = round(angle / 15) * 15
        groups.setdefault(bucket, []).append(length)

    for lengths in groups.values():
        if len(lengths) < 2:
            continue
        mean = sum(lengths) / len(lengths)
        if mean == 0:
            continue
        std = math.sqrt(sum((l - mean) ** 2 for l in lengths) / len(lengths))
        if std / mean > 0.5:
            return False
    return True


# ────────────────────────────────────────────
# 3. 룰베이스 채점
# ────────────────────────────────────────────
def score_cube_rulebased(image: np.ndarray) -> dict:
    """
    육면체 룰베이스 채점
    채점 기준:
    1. 선분 개수: 정상 6~20개, 30개 초과 시 덧그리기
    2. 방향 다양성: 표준(수평+수직+대각) 또는 등각 투영(수평+좌대각+우대각)
    3. 평행선 쌍 존재 + 같은 방향 선분 길이 유사성
    4. 덧그린 선 없음 (30개 초과 방지)
    """
    lines = extract_lines(image)

    if not lines:
        return {
            "score": 0,
            "line_count": 0,
            "details": "선분 미감지"
        }

    line_count = len(lines)

    # 기준 1: 선분 개수 (육면체 = 12개 모서리, 오차 허용)
    count_ok = 6 <= line_count <= 20

    # 기준 2: 방향 다양성 (표준 투영 또는 등각 투영 허용)
    angles = [l[5] for l in lines]
    horizontal = sum(1 for a in angles if a < 20 or a > 160)
    vertical   = sum(1 for a in angles if 70 < a < 110)
    left_diag  = sum(1 for a in angles if 20 <= a <= 70)
    right_diag = sum(1 for a in angles if 110 <= a <= 160)
    diagonal   = left_diag + right_diag

    # 표준(사각형 투영): 수평+수직+대각 / 등각 투영: 수평+좌대각+우대각
    direction_ok = (
        horizontal >= 2 and (
            (vertical >= 2 and diagonal >= 1) or
            (left_diag >= 2 and right_diag >= 2)
        )
    )

    # 기준 3: 평행선 쌍 + 같은 방향 선분 길이 유사성
    angle_groups = {}
    for a in angles:
        bucket = round(a / 15) * 15
        angle_groups[bucket] = angle_groups.get(bucket, 0) + 1
    parallel_ok = any(v >= 2 for v in angle_groups.values())
    length_ok   = _check_length_similarity(lines)

    # 기준 4: 과도한 선분 없음 (덧그리기 방지, count_ok와 분리)
    no_overline = line_count <= 30

    # 종합 판단
    score = 1 if (count_ok and direction_ok and parallel_ok and length_ok and no_overline) else 0

    return {
        "score":        score,
        "line_count":   line_count,
        "horizontal":   horizontal,
        "vertical":     vertical,
        "diagonal":     diagonal,
        "direction_ok": direction_ok,
        "parallel_ok":  parallel_ok,
        "length_ok":    length_ok,
        "count_ok":     count_ok,
        "details":      f"선분 {line_count}개 | 수평:{horizontal} 수직:{vertical} 대각:{diagonal} | 길이유사:{length_ok}"
    }


# ────────────────────────────────────────────
# 4. CNN 모델 기반 채점 (향후 확장)
# ────────────────────────────────────────────
def load_cube_model(model_path: str):
    """
    QuickDraw cube 데이터로 학습한 CNN 모델 로드
    확장자로 프레임워크 자동 감지:
      .pt / .pth  → PyTorch
      .h5 / .keras / .hdf5 → Keras/TensorFlow
    Returns: (framework, model) tuple, or None if unavailable
    """
    if not os.path.exists(model_path):
        print(f"[경고] 모델 파일 없음: {model_path} → 룰베이스로 처리")
        return None

    ext = os.path.splitext(model_path)[1].lower()

    if ext in (".pt", ".pth"):
        try:
            import torch
            model = torch.load(model_path, map_location="cpu")
            model.eval()
            print(f"[정보] PyTorch 육면체 모델 로드 완료: {model_path}")
            return ("torch", model)
        except Exception as e:
            print(f"[오류] PyTorch 로드 실패: {e} → 룰베이스로 처리")
            return None

    elif ext in (".h5", ".keras", ".hdf5"):
        try:
            import tensorflow as tf
            model = tf.keras.models.load_model(model_path)
            print(f"[정보] Keras 육면체 모델 로드 완료: {model_path}")
            return ("keras", model)
        except Exception as e:
            print(f"[오류] Keras 로드 실패: {e} → 룰베이스로 처리")
            return None

    else:
        print(f"[경고] 지원하지 않는 모델 형식: {ext} → 룰베이스로 처리")
        return None


def score_cube_cnn(image: np.ndarray, model) -> dict:
    """
    CNN 모델로 육면체 채점
    model이 None이면 룰베이스로 fallback
    model은 (framework, net) tuple
    """
    if model is None:
        return score_cube_rulebased(image)

    framework, net = model
    processed = preprocess_image(image)

    if framework == "torch":
        try:
            import torch
            tensor = (
                torch.tensor(processed / 255.0, dtype=torch.float32)
                .unsqueeze(0).unsqueeze(0)
            )
            with torch.no_grad():
                pred = net(tensor)
                prob = float(torch.sigmoid(pred).item())
            score = 1 if prob >= 0.5 else 0
            return {"score": score, "confidence": round(prob, 3), "method": "CNN(PyTorch)"}
        except Exception as e:
            print(f"[오류] CNN 추론 실패: {e} → 룰베이스로 처리")
            return score_cube_rulebased(image)

    elif framework == "keras":
        try:
            inp = processed.reshape(1, 256, 256, 1) / 255.0
            prob = float(net.predict(inp, verbose=0)[0][0])
            score = 1 if prob >= 0.5 else 0
            return {"score": score, "confidence": round(prob, 3), "method": "CNN(Keras)"}
        except Exception as e:
            print(f"[오류] CNN 추론 실패: {e} → 룰베이스로 처리")
            return score_cube_rulebased(image)

    return score_cube_rulebased(image)


# ────────────────────────────────────────────
# 5. 통합 채점
# ────────────────────────────────────────────
def score_cube(image: np.ndarray, model=None) -> dict:
    """
    육면체 그리기 채점 (1점)

    Args:
        image: 캔버스 이미지
        model: CNN 모델 (None이면 룰베이스)

    Returns:
        {"score": 0~1, "total": 0~1, "details": {...}}
    """
    if model is not None:
        result = score_cube_cnn(image, model)
    else:
        result = score_cube_rulebased(image)

    result["total"] = result["score"]
    return result


# ────────────────────────────────────────────
# 테스트 (합성 육면체)
# ────────────────────────────────────────────
if __name__ == "__main__":
    img = np.ones((256, 256), dtype=np.uint8) * 255

    # 앞면 사각형
    pts_front = np.array([[80,100],[160,100],[160,180],[80,180]], np.int32)
    cv2.polylines(img, [pts_front], True, 0, 2)

    # 뒷면 사각형 (오른쪽 위로 이동)
    pts_back = np.array([[110,70],[190,70],[190,150],[110,150]], np.int32)
    cv2.polylines(img, [pts_back], True, 0, 2)

    # 연결선 4개 (앞면-뒷면)
    cv2.line(img, (80,100),  (110,70),  0, 2)
    cv2.line(img, (160,100), (190,70),  0, 2)
    cv2.line(img, (160,180), (190,150), 0, 2)
    cv2.line(img, (80,180),  (110,150), 0, 2)

    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    print("=== 육면체 채점 테스트 ===")
    result = score_cube(img_bgr)
    print(f"점수: {result['score']}점")
    print(f"상세: {result['details']}")
    print(f"총점: {result['total']}/1")

    print("\n=== 빈 캔버스 (0점) ===")
    empty = np.ones((256, 256, 3), dtype=np.uint8) * 255
    result2 = score_cube(empty)
    print(f"점수: {result2['score']}점 | {result2['details']}")
