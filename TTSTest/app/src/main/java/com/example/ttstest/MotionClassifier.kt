package com.example.ttstest

import android.util.Log
import kotlin.math.sqrt

class MotionClassifier(private val calib: CalibrationProfile) {
    private val preprocessor = SensorPreprocessor()
    private var expectedAction: String? = null
    private var lastFiredAt = 0L
    private val globalRefractoryMs = 800  // 디바운스 간격

    var latestDebugText = ""
        private set

    private val stopDetector = StopDetector()
    private val weightShiftDetector = WeightShiftDetector(calib)
    private val weightShiftSitDetector = WeightShiftDetector(calib) // 앉기 전용 추가
    private val stepForwardDetector = ApStanceDetector()            // 전방 스텝 유지 추가
    private val stepBackwardDetector = ApStanceDetector()           // 후방 스텝 유지 추가
    private val stepDetector = StepDetector(calib)
    private val kneeDetector = KneeExtensionDetector()
    private val anyReactionDetector = AnyReactionDetector()

    fun setExpectedAction(action: String?) {
        expectedAction = action
    }

    fun onSensorData(timestamp: Long, ax: Float, ay: Float, az: Float, gx: Float, gy: Float, gz: Float): String? {
        val sample = preprocessor.process(ax, ay, az, calib)
        
        latestDebugText = "Expected: $expectedAction\n" +
                "Tilt_ML: ${String.format("%.2f", sample.gravityMl)} | Tilt_AP: ${String.format("%.2f", sample.gravityAp)}\n" +
                "Dyn_ML: ${String.format("%.2f", sample.linearMl)} | Dyn_AP: ${String.format("%.2f", sample.linearAp)}\n" +
                "Impact: ${String.format("%.2f", sample.linearMag)}"
        val gyroMag = sqrt(gx * gx + gy * gy + gz * gz)

        // 매 프레임 스텝 감지용 버퍼에 방향성 가속도 축적
        stepDetector.addSample(sample.linearMl)

        // 500ms 주기로 실시간 센서값 디버깅 출력
        if (System.currentTimeMillis() % 500 < 20) {
            Log.d("SensorVal", "Expected: $expectedAction | ML(좌우)=${String.format("%.2f", sample.gravityMl)}, AP(전후)=${String.format("%.2f", sample.gravityAp)}, Mag(충격)=${String.format("%.2f", sample.linearMag)}")
        }

        // 디바운스: 최근에 동작이 감지되었다면 무시
        if (System.currentTimeMillis() - lastFiredAt < globalRefractoryMs) return null

        val actionToTest = expectedAction
        if (actionToTest != null) {
            val matched = when (actionToTest) {
                "stop" -> stopDetector.update(timestamp, sample.linearMag, gyroMag)
                "weight_right" -> weightShiftDetector.check(timestamp, sample.gravityMl, expectRight = true, threshold = 0.55f)
                "weight_left" -> weightShiftDetector.check(timestamp, sample.gravityMl, expectRight = false, threshold = 0.55f)
                "weight_right_sit" -> weightShiftSitDetector.check(timestamp, sample.gravityMl, expectRight = false, threshold = 2.0f) // 앉기 부호 반전, 실측치 매치
                "weight_left_sit" -> weightShiftSitDetector.check(timestamp, sample.gravityMl, expectRight = true, threshold = 2.0f)  // 앉기 부호 반전, 실측치 매치
                "step_forward_right" -> stepForwardDetector.check(timestamp, sample.gravityAp, expectForward = true, threshold = 0.5f)
                "step_backward" -> stepBackwardDetector.check(timestamp, sample.gravityAp, expectForward = false, threshold = 0.05f)
                "step_right" -> stepDetector.check(sample.linearMag, "step_right")
                "step_left" -> stepDetector.check(sample.linearMag, "step_left")
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
            // 1. 스텝 감지 (충격이 동반되는 동작, 1회성 걷기를 최우선 판정)
            if (stepDetector.check(sample.linearMag, "step_right")) {
                lastFiredAt = System.currentTimeMillis()
                return "step_right"
            }
            if (stepDetector.check(sample.linearMag, "step_left")) {
                lastFiredAt = System.currentTimeMillis()
                return "step_left"
            }

            // 2. 무릎 펴기 감지 (앉아서 하는 dynamic 동작)
            if (kneeDetector.check(sample.linearMag, gyroMag)) {
                lastFiredAt = System.currentTimeMillis()
                return "knee_extension"
            }

            // 3. 체중 이동 감지 (서기/앉기 각각 분리된 인스턴스 및 정적 기울기 검사)
            if (weightShiftDetector.check(timestamp, sample.gravityMl, expectRight = true, threshold = 0.55f)) {
                lastFiredAt = System.currentTimeMillis()
                return "weight_right"
            }
            if (weightShiftDetector.check(timestamp, sample.gravityMl, expectRight = false, threshold = 0.55f)) {
                lastFiredAt = System.currentTimeMillis()
                return "weight_left"
            }
            if (weightShiftSitDetector.check(timestamp, sample.gravityMl, expectRight = false, threshold = 2.0f)) {
                lastFiredAt = System.currentTimeMillis()
                return "weight_right_sit"
            }
            if (weightShiftSitDetector.check(timestamp, sample.gravityMl, expectRight = true, threshold = 2.0f)) {
                lastFiredAt = System.currentTimeMillis()
                return "weight_left_sit"
            }

            // 4. 전후 자세 유지 감지 (AP축 기울임 1초 유지)
            if (stepForwardDetector.check(timestamp, sample.gravityAp, expectForward = true, threshold = 0.5f)) {
                lastFiredAt = System.currentTimeMillis()
                return "step_forward_right"
            }
            if (stepBackwardDetector.check(timestamp, sample.gravityAp, expectForward = false, threshold = 0.05f)) {
                lastFiredAt = System.currentTimeMillis()
                return "step_backward"
            }

            // 5. 임의 반응 감지
            if (anyReactionDetector.check(sample.linearMag, gyroMag)) {
                lastFiredAt = System.currentTimeMillis()
                return "any_reaction"
            }

            // 6. 정지 감지 (가장 마지막에 체크하여 움직임이 없을 때만 반환)
            if (stopDetector.update(timestamp, sample.linearMag, gyroMag)) {
                lastFiredAt = System.currentTimeMillis()
                return "stop"
            }
            return null
        }
        return null
    }
}
