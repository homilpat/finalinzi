package com.example.ttstest

import kotlin.math.abs

/**
 * 정지 상태 감지기
 */
class StopDetector {
    private var lastMotionTime = 0L
    private val motionThreshold = 1.0f // 선형 가속도 임계값
    private val gyroThreshold = 0.5f   // 자이로 임계값
    private val durationRequired = 1000L // 1초 유지
    private var hasFired = false

    fun update(timestamp: Long, linearMag: Float, gyroMag: Float): Boolean {
        if (linearMag > motionThreshold || gyroMag > gyroThreshold) {
            lastMotionTime = timestamp
            hasFired = false
            return false
        }
        val isStopped = (timestamp - lastMotionTime) >= durationRequired
        if (isStopped && !hasFired) {
            hasFired = true
            return true
        }
        return false
    }

    fun reset() {
        hasFired = false
        lastMotionTime = 0L
    }
}

/**
 * 체중 이동 감지기 (중력 기반)
 */
class WeightShiftDetector(private val calib: CalibrationProfile) {
    private val weightThreshold = 0.55f // 측면 중력 성분 임계값 (실측 기준 0.55f로 정밀 매칭)
    private val holdDuration = 1000L   // 유지 시간 (1초로 갱신)
    private val graceDuration = 200L   // 200ms 흔들림 유예 기간 추가

    private var startTime = 0L
    private var lastShiftTime = 0L
    private var hasFired = false

    fun check(timestamp: Long, gravityMl: Float, expectRight: Boolean, threshold: Float = weightThreshold): Boolean {
        val isShifting = if (expectRight) gravityMl > threshold else gravityMl < -threshold

        if (isShifting) {
            lastShiftTime = timestamp
            if (startTime == 0L) startTime = timestamp
            val isHeld = (timestamp - startTime) >= holdDuration
            if (isHeld && !hasFired) {
                hasFired = true
                return true
            }
            return false
        } else {
            // 유예 기간 내에 있으면 타이머 유지
            if (startTime != 0L && (timestamp - lastShiftTime) < graceDuration) {
                return false
            }
            startTime = 0L
            hasFired = false
            return false
        }
    }

    fun reset() {
        startTime = 0L
        lastShiftTime = 0L
        hasFired = false
    }
}

/**
 * 전후 자세 유지 감지기 (AP축 중력 기반)
 * 오른발 앞으로/뒤로 한 발 뻗고 최소 1초 유지하는 동작 감지
 */
class ApStanceDetector {
    private val holdDuration = 1000L // 1초 유지
    private val graceDuration = 200L // 200ms 흔들림 유예 기간 추가
    private var startTime = 0L
    private var lastShiftTime = 0L
    private var hasFired = false

    fun check(timestamp: Long, gravityAp: Float, expectForward: Boolean, threshold: Float): Boolean {
        // 전진은 AP > threshold, 후진은 AP < -threshold
        val isStance = if (expectForward) gravityAp > threshold else gravityAp < -threshold

        if (isStance) {
            lastShiftTime = timestamp
            if (startTime == 0L) startTime = timestamp
            val isHeld = (timestamp - startTime) >= holdDuration
            if (isHeld && !hasFired) {
                hasFired = true
                return true
            }
            return false
        } else {
            // 유예 기간 내에 있으면 타이머 유지
            if (startTime != 0L && (timestamp - lastShiftTime) < graceDuration) {
                return false
            }
            startTime = 0L
            hasFired = false
            return false
        }
    }

    fun reset() {
        startTime = 0L
        lastShiftTime = 0L
        hasFired = false
    }
}

/**
 * 스텝 감지기 (충격 및 방향 기반)
 * 걷기 동작 감지 (1회성 충격 + 최근 300ms 방향 이력 분석)
 */
class StepDetector(private val calib: CalibrationProfile) {
    private val impactThreshold = 1.8f // 충격(선형 가속도) 임계값 (실측 분석을 토대로 1.8f로 하향 조정)
    private val historySize = 30       // 100Hz 기준 300ms 이력 저장
    private val mlHistory = FloatArray(historySize)
    private var writeIdx = 0
    
    fun addSample(linearMl: Float) {
        mlHistory[writeIdx] = linearMl
        writeIdx = (writeIdx + 1) % historySize
    }

    fun check(linearMag: Float, direction: String): Boolean {
        if (linearMag < impactThreshold) return false

        // 300ms 이력 창에서 방향 최대/최소 피크 탐색
        var minMl = 99f
        var maxMl = -99f
        for (v in mlHistory) {
            if (v < minMl) minMl = v
            if (v > maxMl) maxMl = v
        }

        return when (direction) {
            "step_right" -> minMl < -0.6f  // 실측 분석 기준 0.6f로 감도 향상 (우측 스텝 안전선 확보)
            "step_left" -> maxMl > 0.6f   // 실측 분석 기준 0.6f로 감도 향상 (좌측 스텝 감지 성능 극대화)
            else -> false
        }
    }
}

/**
 * 무릎 펴기 감지기 (앉은 자세)
 * 허리에 폰이 있으므로, 무릎을 펼 때 발생하는 미세한 진동이나 기울기 변화를 감지해야 함.
 * 실측 데이터 기반 임계값 하향 조정 (기존 linearMag > 2.0 || gyroMag > 1.5)
 */
class KneeExtensionDetector {
    fun check(linearMag: Float, gyroMag: Float): Boolean {
        return linearMag > 1.8f || gyroMag > 0.5f
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
