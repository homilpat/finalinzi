package com.example.finalprojectapp

import android.annotation.SuppressLint
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Bundle
import android.view.View
import android.webkit.*
import android.widget.ProgressBar
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat

class MainActivity : AppCompatActivity(), SensorEventListener {

    private lateinit var webView: WebView
    private lateinit var progressBar: ProgressBar
    
    private lateinit var sensorManager: SensorManager
    private var accelerometer: Sensor? = null
    private var gyroscope: Sensor? = null
    
    private val classifier = MotionClassifier()
    private val axisCalibrator = AxisCalibrator(durationMs = 3000L)
    private var isSensorRegistered = false
    private var isCalibrating = false

    private var ax = 0f
    private var ay = 0f
    private var az = 0f
    private var gx = 0f
    private var gy = 0f
    private var gz = 0f

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_main)

        webView = findViewById(R.id.webView)
        progressBar = findViewById(R.id.progressBar)

        setupWebView()
        setupSensors()

        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main)) { v, insets ->
            val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom)
            insets
        }
    }

    @SuppressLint("SetJavaScriptEnabled")
    private fun setupWebView() {
        webView.settings.javaScriptEnabled = true
        webView.settings.domStorageEnabled = true
        webView.settings.mediaPlaybackRequiresUserGesture = false // 자동 재생 허용
        webView.settings.mixedContentMode = WebSettings.MIXED_CONTENT_ALWAYS_ALLOW

        webView.webViewClient = object : WebViewClient() {
            override fun onPageFinished(view: WebView?, url: String?) {
                progressBar.visibility = View.GONE
            }
        }

        webView.webChromeClient = WebChromeClient()
        
        // 브릿지 등록
        webView.addJavascriptInterface(WebAppInterface(), "AndroidBridge")
        
        webView.loadUrl("https://finalinzi.onrender.com/")
    }

    private fun setupSensors() {
        sensorManager = getSystemService(SENSOR_SERVICE) as SensorManager
        accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        gyroscope = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
    }

    private fun registerSensors() {
        if (!isSensorRegistered) {
            accelerometer?.let {
                sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME)
            }
            gyroscope?.let {
                sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME)
            }
            isSensorRegistered = true
        }
    }

    private fun unregisterSensors() {
        if (isSensorRegistered) {
            sensorManager.unregisterListener(this)
            isSensorRegistered = false
        }
    }

    override fun onSensorChanged(event: SensorEvent?) {
        if (event == null) return

        when (event.sensor.type) {
            Sensor.TYPE_ACCELEROMETER -> {
                ax = event.values[0]
                ay = event.values[1]
                az = event.values[2]
            }
            Sensor.TYPE_GYROSCOPE -> {
                gx = event.values[0]
                gy = event.values[1]
                gz = event.values[2]
            }
        }

        // 캘리브레이션 진행 중: 중력벡터 수집 → 완료 시 JS 알림
        if (isCalibrating) {
            val profile = axisCalibrator.addSample(ax, ay, az)
            if (profile != null) {
                isCalibrating = false
                classifier.updateCalibration(profile)
                notifyCalibrationDone(profile)
            }
            return
        }

        val detectedAction = classifier.onSensorData(
            System.currentTimeMillis(),
            ax, ay, az, gx, gy, gz
        )

        detectedAction?.let { action ->
            sendActionToWebView(action)
        }
    }

    private fun notifyCalibrationDone(profile: CalibrationProfile) {
        val json = "{\"type\":\"calibration_done\"," +
            "\"upAxis\":${profile.upAxis}," +
            "\"upSign\":${profile.upSign}," +
            "\"lateralAxis\":${profile.lateralAxis}," +
            "\"forwardAxis\":${profile.forwardAxis}}"
        val script = "window.SensorBridge && window.SensorBridge.onSensorEvent('$json')"
        webView.post { webView.evaluateJavascript(script, null) }
    }

    private fun sendActionToWebView(action: String) {
        val json = "{\"action\":\"$action\"}"
        val script = "window.SensorBridge.onSensorEvent('$json')"
        webView.post {
            webView.evaluateJavascript(script, null)
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}

    override fun onDestroy() {
        unregisterSensors()
        super.onDestroy()
    }

    /**
     * WebView에서 호출할 인터페이스
     */
    inner class WebAppInterface {
        @JavascriptInterface
        fun startMeasurement(stage: String) {
            registerSensors()
            // 첫 측정 시 3초 캘리브레이션 자동 시작
            // JS에서 calibration_done 이벤트 받은 뒤 실제 동작 감지 시작
            axisCalibrator.start()
            isCalibrating = true
        }

        @JavascriptInterface
        fun stopMeasurement() {
            unregisterSensors()
            classifier.setExpectedAction(null)
        }

        @JavascriptInterface
        fun setExpectedAction(action: String) {
            classifier.setExpectedAction(action)
        }
    }
}
