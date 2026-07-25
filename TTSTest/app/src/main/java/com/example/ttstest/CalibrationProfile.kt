package com.example.ttstest

/**
 * 센서 축 매핑 및 오프셋 정보를 담는 클래스.
 * 기본적으로 안드로이드 표준 좌표계를 따르되, 착용 방식에 따라 축을 재정의할 수 있음.
 */
data class CalibrationProfile(
    var isCalibrated: Boolean = false,
    var gyroBiasX: Float = 0f,
    var gyroBiasY: Float = 0f,
    var gyroBiasZ: Float = 0f,
    var gravityMeanX: Float = 0f,
    var gravityMeanY: Float = 0f,
    var gravityMeanZ: Float = 0f,
    var uVertX: Float = 0f,
    var uVertY: Float = 0f,
    var uVertZ: Float = 1f,
    var uMlX: Float = 1f,
    var uMlY: Float = 0f,
    var uMlZ: Float = 0f,
    var uApX: Float = 0f,
    var uApY: Float = 1f,
    var uApZ: Float = 0f,
    
    // 호환성을 위해 유지
    val upAxis: Int = 1,
    val lateralAxis: Int = 0,
    val forwardAxis: Int = 2,
    val upSign: Float = 1f,
    val lateralSign: Float = 1f,
    val forwardSign: Float = 1f
)
