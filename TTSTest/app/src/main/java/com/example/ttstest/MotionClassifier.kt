package com.example.ttstest

import kotlin.math.sqrt

class MotionClassifier(private val calib: CalibrationProfile) {
    private val preprocessor = SensorPreprocessor()
    private var expectedAction: String? = null
    private var lastFiredAt = 0L
    private val globalRefractoryMs = 800  // 디바운스 간격

    private val stopDetector = StopDetector()
    private val weightShiftDetector = WeightShiftDetector(calib)
    private val stepDetector = StepDetector(calib)
    private val kneeDetector = KneeExtensionDetector()
    private val anyReactionDetector = AnyReactionDetector()

    fun setExpectedAction(action: String?) {
        expectedAction = action
    }

    fun onSensorData(timestamp: Long, ax: Float, ay: Float, az: Float, gx: Float, gy: Float, gz: Float): String? {
        val sample = preprocessor.process(ax, ay, az)
        val gyroMag = sqrt(gx * gx + gy * gy + gz * gz)

        // 디바운스: 최근에 동작이 감지되었다면 무시
        if (System.currentTimeMillis() - lastFiredAt < globalRefractoryMs) return null

        val gravityArray = floatArrayOf(sample.gx, sample.gy, sample.gz)

        val actionToTest = expectedAction
        if (actionToTest != null) {
            val matched = when (actionToTest) {
                "stop" -> stopDetector.update(timestamp, sample.linearMag, gyroMag)
                "weight_right" -> weightShiftDetector.check(timestamp, gravityArray, expectRight = true)
                "weight_left" -> weightShiftDetector.check(timestamp, gravityArray, expectRight = false)
                "weight_right_sit" -> weightShiftDetector.check(timestamp, gravityArray, expectRight = true) // 일단 서서 하는 것과 동일 로직
                "weight_left_sit" -> weightShiftDetector.check(timestamp, gravityArray, expectRight = false)
                "step_forward_right" -> stepDetector.check(sample.linearMag, sample.lx, sample.ly, sample.lz, "step_forward_right")
                "step_backward" -> stepDetector.check(sample.linearMag, sample.lx, sample.ly, sample.lz, "step_backward")
                "step_right" -> stepDetector.check(sample.linearMag, sample.lx, sample.ly, sample.lz, "step_right")
                "step_left" -> stepDetector.check(sample.linearMag, sample.lx, sample.ly, sample.lz, "step_left")
                "knee_extension" -> kneeDetector.check(sample.linearMag, gyroMag)
                "any_reaction" -> anyReactionDetector.check(sample.linearMag, gyroMag)
                else -> false
            }

            if (matched) {
                lastFiredAt = System.currentTimeMillis()
                expectedAction = null // 한 번 감지되면 초기화 (중복 방지)
                return actionToTest
            }
        } else {
            // expectedAction이 null인 경우, 모든 감지기를 검사하여 감지된 첫 동작을 반환
            // 1. 체중 이동 감지 (서기/앉기 공통)
            if (weightShiftDetector.check(timestamp, gravityArray, expectRight = true)) {
                lastFiredAt = System.currentTimeMillis()
                return "weight_right"
            }
            if (weightShiftDetector.check(timestamp, gravityArray, expectRight = false)) {
                lastFiredAt = System.currentTimeMillis()
                return "weight_left"
            }

            // 2. 스텝 감지 (충격이 동반되는 동작)
            if (sample.linearMag >= 4.0f) {
                if (stepDetector.check(sample.linearMag, sample.lx, sample.ly, sample.lz, "step_forward_right")) {
                    lastFiredAt = System.currentTimeMillis()
                    return "step_forward_right"
                }
                if (stepDetector.check(sample.linearMag, sample.lx, sample.ly, sample.lz, "step_backward")) {
                    lastFiredAt = System.currentTimeMillis()
                    return "step_backward"
                }
                if (stepDetector.check(sample.linearMag, sample.lx, sample.ly, sample.lz, "step_right")) {
                    lastFiredAt = System.currentTimeMillis()
                    return "step_right"
                }
                if (stepDetector.check(sample.linearMag, sample.lx, sample.ly, sample.lz, "step_left")) {
                    lastFiredAt = System.currentTimeMillis()
                    return "step_left"
                }
            }

            // 3. 무릎 펴기 감지
            if (kneeDetector.check(sample.linearMag, gyroMag)) {
                lastFiredAt = System.currentTimeMillis()
                return "knee_extension"
            }

            // 4. 임의 반응 감지
            if (anyReactionDetector.check(sample.linearMag, gyroMag)) {
                lastFiredAt = System.currentTimeMillis()
                return "any_reaction"
            }

            // 5. 정지 감지 (가장 마지막에 체크하여 움직임이 없을 때만 반환)
            if (stopDetector.update(timestamp, sample.linearMag, gyroMag)) {
                lastFiredAt = System.currentTimeMillis()
                return "stop"
            }
        }
        return null
    }
}
