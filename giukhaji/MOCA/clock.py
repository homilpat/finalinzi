"""
MoCA-K 시계 그리기 채점 모듈
데이터: Roboflow CDT (윤곽/바늘) + MNIST (숫자)
모델: U-Net (segmentation) + CNN (분류)
총점: 3점 (윤곽1 + 숫자1 + 바늘1)
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
    """
    캔버스 이미지 전처리
    - 그레이스케일 변환
    - 이진화 (흰 배경, 검정 선)
    - 리사이즈 512x512
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # 이진화 (Otsu)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 리사이즈
    resized = cv2.resize(binary, (512, 512))
    return resized


# ────────────────────────────────────────────
# 2. 윤곽 채점 (룰베이스 + CNN 보조)
# ────────────────────────────────────────────
def score_contour(image: np.ndarray) -> dict:
    """
    원(윤곽) 감지
    - Hough Circle Transform으로 원 감지
    - 원이 감지되면 1점
    - 약간 변형된 원도 허용 (원본 기준)

    Returns:
        {"score": 0~1, "circle_detected": bool, "circle_info": dict}
    """
    processed = preprocess_image(image)

    # 가우시안 블러
    blurred = cv2.GaussianBlur(processed, (9, 9), 2)

    # Hough Circle 감지
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=100,
        param1=50,
        param2=30,
        minRadius=80,
        maxRadius=250
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        best = circles[0][0]  # 가장 큰 원
        cx, cy, r = int(best[0]), int(best[1]), int(best[2])

        # 원의 중심이 이미지 중앙 근처인지 확인
        h, w = processed.shape
        center_dist = math.sqrt((cx - w//2)**2 + (cy - h//2)**2)
        is_centered = center_dist < w * 0.3

        return {
            "score": 1 if is_centered else 0,
            "circle_detected": True,
            "circle_info": {"cx": cx, "cy": cy, "radius": r}
        }

    return {
        "score": 0,
        "circle_detected": False,
        "circle_info": None
    }


# ────────────────────────────────────────────
# 3. 숫자 채점
# ────────────────────────────────────────────
def score_numbers(image: np.ndarray, circle_info: dict = None) -> dict:
    """
    시계 숫자 (1~12) 채점
    공식 기준:
      - 다른 숫자 추가 불가: 동일 30° 구역에 3개 초과 컨투어, 또는 전체 16개 초과
      - 순서대로 제자리에: 12시(270°) 기준 30° 간격 12구역 중 10개 이상 점유
    이미지 좌표계 각도: 12시=270°, 3시=0°, 6시=90°, 9시=180°
    """
    processed = preprocess_image(image)

    if circle_info is None:
        contour_result = score_contour(image)
        circle_info = contour_result.get("circle_info")

    if circle_info is None:
        return {"score": 0, "number_count": 0, "details": "원 미감지"}

    cx, cy, r = circle_info["cx"], circle_info["cy"], circle_info["radius"]

    # 원 내부 + 바깥 20% 여유 (K-MoCA: 숫자가 외곽선 바깥도 허용)
    mask = np.zeros_like(processed)
    cv2.circle(mask, (cx, cy), int(r * 1.2), 255, -1)
    inner = cv2.bitwise_and(processed, mask)

    # RETR_LIST: 원 테두리가 외부 컨투어로 묻히는 문제 방지
    contours, _ = cv2.findContours(inner, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    valid = [cnt for cnt in contours if 50 < cv2.contourArea(cnt) < 5000]
    count = len(valid)

    # 12시(270°) 기준 30° 간격 시계 구역에 무게중심 매핑
    # sector 0 = 12시, sector 1 = 1시, ..., sector 11 = 11시
    CLOCK_BASE = 270
    sector_counts = [0] * 12

    for cnt in valid:
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        ccx = int(M["m10"] / M["m00"])
        ccy = int(M["m01"] / M["m00"])
        dist = math.sqrt((ccx - cx) ** 2 + (ccy - cy) ** 2)
        if r * 0.45 <= dist <= r * 1.15:
            angle = math.degrees(math.atan2(ccy - cy, ccx - cx)) % 360
            sector = int(((angle - CLOCK_BASE + 15) % 360) / 30) % 12
            sector_counts[sector] += 1

    # ── 기준 1: 다른 숫자 추가 불가 ──
    # 동일 구역 3개 초과: 10·11·12 같은 두 자리 숫자는 2개까지 허용, 3개면 여분 숫자 존재
    # 전체 16개 초과: 두 자리 숫자(10,11,12) 고려해도 지나치게 많으면 위반
    no_extra = not any(c > 2 for c in sector_counts) and count <= 16

    # ── 기준 2: 순서대로 제자리에 ──
    # 시계 위치 12구역(12시~11시) 중 10개 이상에 숫자 존재
    sectors_filled = sum(1 for c in sector_counts if c >= 1)
    in_place = sectors_filled >= 10

    count_ok = 8 <= count <= 16  # 두 자리 숫자(10,11,12) 고려해 상한 완화
    score = 1 if (count_ok and in_place and no_extra) else 0

    return {
        "score": score,
        "number_count": count,
        "no_extra_digits": no_extra,
        "in_place": in_place,
        "sectors_filled": sectors_filled,
        "details": (
            f"감지 {count}개 | 구역:{sectors_filled}/12 "
            f"| 추가숫자없음:{no_extra} | 제자리:{in_place}"
        )
    }


# ────────────────────────────────────────────
# 4. 바늘 채점
# ────────────────────────────────────────────
def score_hands(image: np.ndarray, circle_info: dict = None) -> dict:
    """
    시계바늘 채점 (11시 10분)
    - 원 중심에서 뻗은 선분 감지 (Hough Line)
    - 두 바늘 감지 + 각도 검증
    - 시침(짧음) < 분침(김) 확인
    - 11시 10분: 시침 약 240도, 분침 약 330도 (이미지 좌표계, 중심→끝점 기준)

    Returns:
        {"score": 0~1, "hands_detected": int, "angle_info": dict}
    """
    processed = preprocess_image(image)

    if circle_info is None:
        contour_result = score_contour(image)
        circle_info = contour_result.get("circle_info")

    if circle_info is None:
        return {"score": 0, "hands_detected": 0, "angle_info": None}

    cx, cy, r = circle_info["cx"], circle_info["cy"], circle_info["radius"]

    # 원 내부만 사용
    mask = np.zeros_like(processed)
    cv2.circle(mask, (cx, cy), int(r * 0.9), 255, -1)
    inner = cv2.bitwise_and(processed, mask)

    # Hough Line 감지
    lines = cv2.HoughLinesP(
        inner,
        rho=1,
        theta=np.pi/180,
        threshold=30,
        minLineLength=int(r * 0.3),
        maxLineGap=10
    )

    if lines is None or len(lines) < 2:
        return {"score": 0, "hands_detected": 0, "angle_info": None}

    # 중심 근처에서 시작하는 선분만 필터링
    center_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # 선분의 한 끝이 중심 근처인지 확인
        d1 = math.sqrt((x1-cx)**2 + (y1-cy)**2)
        d2 = math.sqrt((x2-cx)**2 + (y2-cy)**2)
        if min(d1, d2) < r * 0.3:
            length = math.sqrt((x2-x1)**2 + (y2-y1)**2)
            # 중심에서 끝점 방향으로 normalize (HoughLinesP 방향 비결정적 문제 해결)
            if d1 <= d2:
                angle = math.degrees(math.atan2(y2-y1, x2-x1)) % 360
            else:
                angle = math.degrees(math.atan2(y1-y2, x1-x2)) % 360
            center_lines.append({
                "line": (x1, y1, x2, y2),
                "length": length,
                "angle": angle
            })

    if len(center_lines) < 2:
        return {"score": 0, "hands_detected": len(center_lines), "angle_info": None}

    # 길이 기준 정렬 (짧은=시침, 긴=분침)
    center_lines.sort(key=lambda x: x["length"])
    hour_hand   = center_lines[0]   # 시침 (짧음)
    minute_hand = center_lines[-1]  # 분침 (김)

    # 11시 10분 검증 (중심→끝점 방향 기준, 이미지 좌표계)
    # 시침: 240° (11시 방향) / 분침: 330° (2시/10분 방향)
    hour_ok   = _check_angle(hour_hand["angle"],   target=240, tolerance=35)
    minute_ok = _check_angle(minute_hand["angle"], target=330, tolerance=35)
    length_ok = hour_hand["length"] < minute_hand["length"]

    score = 1 if (hour_ok and minute_ok and length_ok) else 0

    return {
        "score": score,
        "hands_detected": len(center_lines),
        "angle_info": {
            "hour_angle":   round(hour_hand["angle"], 1),
            "minute_angle": round(minute_hand["angle"], 1),
            "hour_length":  round(hour_hand["length"], 1),
            "minute_length":round(minute_hand["length"], 1),
        }
    }


def _check_angle(angle: float, target: float, tolerance: float) -> bool:
    """각도 허용 범위 내 여부 확인 (360도 wrap 처리)"""
    diff = abs(angle - target) % 360
    if diff > 180:
        diff = 360 - diff
    return diff <= tolerance


# ────────────────────────────────────────────
# 5. 통합 채점
# ────────────────────────────────────────────
def score_clock(image: np.ndarray) -> dict:
    """
    시계 그리기 전체 채점 (3점)

    Args:
        image: 캔버스 이미지 (numpy array, BGR or Gray)

    Returns:
        {
            "contour": {"score": 0~1, ...},
            "numbers": {"score": 0~1, ...},
            "hands":   {"score": 0~1, ...},
            "total":   0~3
        }
    """
    # 윤곽 먼저 (원 정보를 숫자/바늘 채점에 재사용)
    contour = score_contour(image)
    circle_info = contour.get("circle_info")

    numbers = score_numbers(image, circle_info)
    hands   = score_hands(image, circle_info)

    total = contour["score"] + numbers["score"] + hands["score"]

    return {
        "contour": contour,
        "numbers": numbers,
        "hands":   hands,
        "total":   total
    }


# ────────────────────────────────────────────
# 6. 모델 기반 채점 (CNN - 향후 확장)
# ────────────────────────────────────────────
def load_clock_model(model_path: str):
    """
    Roboflow CDT 데이터로 학습한 CNN/U-Net 모델 로드
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
            print(f"[정보] PyTorch 시계 모델 로드 완료: {model_path}")
            return ("torch", model)
        except Exception as e:
            print(f"[오류] PyTorch 로드 실패: {e} → 룰베이스로 처리")
            return None

    elif ext in (".h5", ".keras", ".hdf5"):
        try:
            import tensorflow as tf
            model = tf.keras.models.load_model(model_path)
            print(f"[정보] Keras 시계 모델 로드 완료: {model_path}")
            return ("keras", model)
        except Exception as e:
            print(f"[오류] Keras 로드 실패: {e} → 룰베이스로 처리")
            return None

    else:
        print(f"[경고] 지원하지 않는 모델 형식: {ext} → 룰베이스로 처리")
        return None


def score_clock_cnn(image: np.ndarray, model) -> dict:
    """
    CNN 모델로 시계 전체 채점 (3점)
    model이 None이면 룰베이스 fallback
    model은 (framework, net) tuple
    """
    if model is None:
        return score_clock(image)

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
                # 출력이 (contour, numbers, hands) 3채널이면 각각, 아니면 단일 점수
                if pred.shape[-1] == 3:
                    scores = (pred.squeeze() >= 0.5).int().tolist()
                    return {
                        "contour": {"score": scores[0]},
                        "numbers": {"score": scores[1]},
                        "hands":   {"score": scores[2]},
                        "total":   sum(scores),
                        "method":  "CNN(PyTorch)",
                    }
                else:
                    prob = float(torch.sigmoid(pred).item())
                    total = round(prob * 3)
                    return {"score": total, "total": total, "confidence": round(prob, 3), "method": "CNN(PyTorch)"}
        except Exception as e:
            print(f"[오류] CNN 추론 실패: {e} → 룰베이스로 처리")
            return score_clock(image)

    elif framework == "keras":
        try:
            inp = processed.reshape(1, 512, 512, 1) / 255.0
            pred = net.predict(inp, verbose=0)[0]
            if len(pred) == 3:
                scores = [1 if p >= 0.5 else 0 for p in pred]
                return {
                    "contour": {"score": scores[0]},
                    "numbers": {"score": scores[1]},
                    "hands":   {"score": scores[2]},
                    "total":   sum(scores),
                    "method":  "CNN(Keras)",
                }
            else:
                prob = float(pred[0])
                total = round(prob * 3)
                return {"score": total, "total": total, "confidence": round(prob, 3), "method": "CNN(Keras)"}
        except Exception as e:
            print(f"[오류] CNN 추론 실패: {e} → 룰베이스로 처리")
            return score_clock(image)

    return score_clock(image)


# ────────────────────────────────────────────
# 테스트 (합성 이미지)
# ────────────────────────────────────────────
if __name__ == "__main__":
    # 테스트용 합성 시계 이미지 생성
    img = np.ones((512, 512), dtype=np.uint8) * 255  # 흰 배경

    # 원 그리기
    cv2.circle(img, (256, 256), 200, 0, 3)

    # 숫자 위치에 점 찍기 (실제 숫자 대신)
    for i in range(1, 13):
        angle = math.radians(i * 30 - 90)
        x = int(256 + 170 * math.cos(angle))
        y = int(256 + 170 * math.sin(angle))
        cv2.circle(img, (x, y), 8, 0, -1)

    # 11시 방향 시침
    hour_angle = math.radians(330 - 90)
    cv2.line(img, (256, 256),
             (int(256 + 120 * math.cos(hour_angle)),
              int(256 + 120 * math.sin(hour_angle))), 0, 4)

    # 2시(10분) 방향 분침
    min_angle = math.radians(60 - 90)
    cv2.line(img, (256, 256),
             (int(256 + 170 * math.cos(min_angle)),
              int(256 + 170 * math.sin(min_angle))), 0, 3)

    # BGR로 변환
    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    print("=== 시계 채점 테스트 ===")
    result = score_clock(img_bgr)
    print(f"윤곽: {result['contour']['score']}점 | {result['contour']}")
    print(f"숫자: {result['numbers']['score']}점 | {result['numbers']}")
    print(f"바늘: {result['hands']['score']}점  | {result['hands']}")
    print(f"총점: {result['total']}/3")
