package com.example.finalprojectapp

import kotlin.math.abs

/**
 * 3초 정지 캘리브레이션 → 중력 방향으로 축 매핑 자동 결정
 * Python gait_axis_aligned_core.align_to_vmlap() 동일 방식
 *
 * Android 표준 좌표계 (TYPE_ACCELEROMETER):
 *   X+ = 오른쪽 (portrait 기준)
 *   Y+ = 위
 *   Z+ = 화면 밖 (사용자 방향)
 *
 * 정지 시 가속도계 = 중력 반력 → 위 방향 축이 +9.81 읽힘
 */
class AxisCalibrator(private val durationMs: Long = 3000L) {

    private val samples = mutableListOf<FloatArray>()
    private var startTime = 0L
    private var running = false

    val isRunning get() = running

    fun start() {
        samples.clear()
        startTime = System.currentTimeMillis()
        running = true
    }

    /** 가속도계 원시값 입력 (정지 상태에서 ≈ 중력벡터) */
    fun addSample(ax: Float, ay: Float, az: Float): CalibrationProfile? {
        if (!running) return null
        samples.add(floatArrayOf(ax, ay, az))

        if (System.currentTimeMillis() - startTime < durationMs) return null
        running = false
        return compute()
    }

    /** 경과 시간 (0.0~1.0) — 진행바 표시용 */
    fun progress(): Float {
        if (!running) return 1f
        return ((System.currentTimeMillis() - startTime).toFloat() / durationMs).coerceIn(0f, 1f)
    }

    private fun compute(): CalibrationProfile {
        // 각 축의 평균 중력 성분
        val mean = FloatArray(3) { i -> samples.map { it[i] }.average().toFloat() }

        // up axis: |mean| 가장 큰 축
        val absVal = mean.map { abs(it) }
        val upAxis = absVal.indexOf(absVal.max()!!)
        val upSign = if (mean[upAxis] >= 0f) 1f else -1f

        // 나머지 두 수평 축 → 표준 Android 배치로 lateral/forward 할당
        // upAxis=1(Y 위): X=lateral(오른쪽+), Z=forward(화면밖 방향)
        // upAxis=2(Z 위): X=lateral, Y=forward
        // upAxis=0(X 위): Y=lateral, Z=forward
        val (lateralAxis, forwardAxis) = when (upAxis) {
            1    -> Pair(0, 2)
            2    -> Pair(0, 1)
            else -> Pair(1, 2)
        }

        return CalibrationProfile(
            upAxis      = upAxis,
            lateralAxis = lateralAxis,
            forwardAxis = forwardAxis,
            upSign      = upSign,
            lateralSign = 1f,
            forwardSign = 1f,
        )
    }
}
