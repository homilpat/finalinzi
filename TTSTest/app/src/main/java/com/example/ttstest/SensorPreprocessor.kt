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
        val gravityMag: Float,
        val verticalG: Float, // 수직축 전체 가속도 (g 단위)
        val gravityMl: Float, // 중력 성분 ML 투영 (m/s2)
        val gravityAp: Float, // 중력 성분 AP 투영 (m/s2)
        val linearMl: Float,  // 선형 가속도 ML 투영 (m/s2)
        val linearAp: Float   // 선형 가속도 AP 투영 (m/s2)
    )

    fun process(ax: Float, ay: Float, az: Float, calib: CalibrationProfile): ProcessedSample {
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

        // 그람-슈미트 정렬 기저 벡터 투영
        val verticalG = (ax * calib.uVertX + ay * calib.uVertY + az * calib.uVertZ) / 9.80665f
        val gravityMl = gravity[0] * calib.uMlX + gravity[1] * calib.uMlY + gravity[2] * calib.uMlZ
        val gravityAp = gravity[0] * calib.uApX + gravity[1] * calib.uApY + gravity[2] * calib.uApZ
        val linearMl = lx * calib.uMlX + ly * calib.uMlY + lz * calib.uMlZ
        val linearAp = lx * calib.uApX + ly * calib.uApY + lz * calib.uApZ

        return ProcessedSample(
            gravity[0], gravity[1], gravity[2],
            lx, ly, lz,
            lMag, gMag,
            verticalG, gravityMl, gravityAp, linearMl, linearAp
        )
    }
}
