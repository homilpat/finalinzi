package com.example.finalprojectapp

import kotlin.math.abs

/**
 * 정지 상태 감지기
 */
class StopDetector {
    private var lastMotionTime = 0L
    private val motionThreshold = 1.0f // 선형 가속도 임계값
    private val gyroThreshold = 0.5f   // 자이로 임계값
    private val durationRequired = 1000L // 1초 유지

    fun update(timestamp: Long, linearMag: Float, gyroMag: Float): Boolean {
        if (linearMag > motionThreshold || gyroMag > gyroThreshold) {
            lastMotionTime = timestamp
            return false
        }
        return (timestamp - lastMotionTime) >= durationRequired
    }
}

/**
 * 체중 이동 감지기 (중력 기반)
 */
class WeightShiftDetector(private val calib: CalibrationProfile) {
    private val weightThreshold = 3.0f // 측면 중력 성분 임계값
    private val holdDuration = 500L    // 유지 시간

    private var startTime = 0L

    fun check(timestamp: Long, gravity: FloatArray, expectRight: Boolean): Boolean {
        val lateralVal = when (calib.lateralAxis) {
            0 -> gravity[0] * calib.lateralSign
            1 -> gravity[1] * calib.lateralSign
            else -> gravity[2] * calib.lateralSign
        }

        val isShifting = if (expectRight) lateralVal > weightThreshold else lateralVal < -weightThreshold

        if (isShifting) {
            if (startTime == 0L) startTime = timestamp
            return (timestamp - startTime) >= holdDuration
        } else {
            startTime = 0L
            return false
        }
    }
}

/**
 * 스텝 감지기 (충격 및 방향 기반)
 */
class StepDetector(private val calib: CalibrationProfile) {
    private val impactThreshold = 4.0f // 충격(선형 가속도) 임계값
    
    fun check(linearMag: Float, lx: Float, ly: Float, lz: Float, direction: String): Boolean {
        if (linearMag < impactThreshold) return false

        // 방향 판정 (매우 단순화된 로직)
        return when (direction) {
            "step_right" -> (lx * calib.lateralSign) > 1.5f
            "step_left" -> (lx * calib.lateralSign) < -1.5f
            "step_forward_right" -> (lz * calib.forwardSign) > 1.5f // 전진 방향 (Z축 가정)
            "step_backward" -> (lz * calib.forwardSign) < -1.5f
            else -> false
        }
    }
}

/**
 * 무릎 펴기 감지기 (앉은 자세)
 * 허리에 폰이 있으므로, 무릎을 펼 때 발생하는 미세한 진동이나 기울기 변화를 감지해야 함.
 * 여기서는 단순 움직임 발생으로 대체 (힌트에 따라)
 */
class KneeExtensionDetector {
    fun check(linearMag: Float, gyroMag: Float): Boolean {
        return linearMag > 2.0f || gyroMag > 1.5f
    }
}

/**
 * 아무 반응이나 감지
 */
class AnyReactionDetector {
    private val threshold = 1.5f
    fun check(linearMag: Float, gyroMag: Float): Boolean {
        return linearMag > threshold || gyroMag > threshold
    }
}
