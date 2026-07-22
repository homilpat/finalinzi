package com.example.ttstest

import kotlin.math.sqrt

/**
 * 중력과 선형 가속도를 분리하기 위한 전처리 클래스 (Low-Pass Filter 적용)
 */
class SensorPreprocessor {
    private val alpha = 0.8f
    private val gravity = floatArrayOf(0f, 0f, 0f)

    data class ProcessedSample(
        val gx: Float, val gy: Float, val gz: Float, // 중력 성분
        val lx: Float, val ly: Float, val lz: Float, // 선형 가속도 성분
        val linearMag: Float,
        val gravityMag: Float
    )

    fun process(ax: Float, ay: Float, az: Float): ProcessedSample {
        // LPF를 사용하여 중력 성분 추출
        gravity[0] = alpha * gravity[0] + (1 - alpha) * ax
        gravity[1] = alpha * gravity[1] + (1 - alpha) * ay
        gravity[2] = alpha * gravity[2] + (1 - alpha) * az

        // 선형 가속도 = 원시값 - 중력
        val lx = ax - gravity[0]
        val ly = ay - gravity[1]
        val lz = az - gravity[2]

        val lMag = sqrt(lx * lx + ly * ly + lz * lz)
        val gMag = sqrt(gravity[0] * gravity[0] + gravity[1] * gravity[1] + gravity[2] * gravity[2])

        return ProcessedSample(
            gravity[0], gravity[1], gravity[2],
            lx, ly, lz,
            lMag, gMag
        )
    }
}
