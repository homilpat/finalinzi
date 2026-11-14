package com.example.finalprojectapp

import kotlin.math.sqrt

class MotionClassifier(calib: CalibrationProfile = CalibrationProfile()) {
    private val preprocessor = SensorPreprocessor()
    private var expectedAction: String? = null
    private var lastFiredAt = 0L
    private val globalRefractoryMs = 800  // 디바운스 간격

    private var stopDetector = StopDetector()
    private var weightShiftDetector = WeightShiftDetector(calib)
    private var stepDetector = StepDetector(calib)
    private val kneeDetector = KneeExtensionDetector()
    private val anyReactionDetector = AnyReactionDetector()

    /** 캘리브레이션 완료 후 새 프로파일로 축 의존 감지기 재생성 */
    fun updateCalibration(calib: CalibrationProfile) {
        weightShiftDetector = WeightShiftDetector(calib)
        stepDetector = StepDetector(calib)
    }

    fun setExpectedAction(action: String?) {
        expectedAction = action
    }

    fun onSensorData(timestamp: Long, ax: Float, ay: Float, az: Float, gx: Float, gy: Float, gz: Float): String? {
        val sample = preprocessor.process(ax, ay, az)
        val gyroMag = sqrt(gx * gx + gy * gy + gz * gz)

        if (expectedAction == null) return null
        
        // 디바운스: 최근에 동작이 감지되었다면 무시
        if (System.currentTimeMillis() - lastFiredAt < globalRefractoryMs) return null

        val gravityArray = floatArrayOf(sample.gx, sample.gy, sample.gz)

        val matched = when (expectedAction) {
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
            val action = expectedAction
            expectedAction = null // 한 번 감지되면 초기화 (중복 방지)
            return action
        }
        return null
    }
}
