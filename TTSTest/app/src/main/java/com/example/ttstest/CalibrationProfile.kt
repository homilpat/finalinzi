package com.example.ttstest

/**
 * 센서 축 매핑 및 오프셋 정보를 담는 클래스.
 * 기본적으로 안드로이드 표준 좌표계를 따르되, 착용 방식에 따라 축을 재정의할 수 있음.
 */
data class CalibrationProfile(
    val upAxis: Int = 1,      // 보통 Y축 (0:X, 1:Y, 2:Z)
    val lateralAxis: Int = 0, // 보통 X축
    val forwardAxis: Int = 2, // 보통 Z축
    val upSign: Float = 1f,
    val lateralSign: Float = 1f,
    val forwardSign: Float = 1f
)
