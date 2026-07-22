package com.example.finalprojectapp

import android.annotation.SuppressLint
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.content.Context
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.view.View
import android.webkit.CookieManager
import android.webkit.JavascriptInterface
import android.webkit.WebChromeClient
import android.webkit.WebSettings
import android.webkit.WebView
import android.webkit.WebViewClient
import android.widget.ProgressBar
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import org.json.JSONObject
import java.io.DataOutputStream
import java.net.HttpURLConnection
import java.net.URL
import kotlin.concurrent.thread

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

    private val mainHandler = Handler(Looper.getMainLooper())
    private val gaitSamples = mutableListOf<GaitSample>()
    private var isGaitMeasuring = false
    private var gaitUploadUrl = ""
    private var gaitMemberPhone = ""
    private var gaitDurationMs = 20_000L
    private var gaitStartTimestampNs = 0L
    private var rememberedLoginAttempted = false

    private var ax = 0f
    private var ay = 0f
    private var az = 0f
    private var gx = 0f
    private var gy = 0f
    private var gz = 0f

    private data class GaitSample(
        val timestampNs: Long,
        val ax: Float,
        val ay: Float,
        val az: Float,
        val gx: Float,
        val gy: Float,
        val gz: Float
    )

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
        webView.settings.mediaPlaybackRequiresUserGesture = false
        webView.settings.mixedContentMode = WebSettings.MIXED_CONTENT_ALWAYS_ALLOW
        CookieManager.getInstance().setAcceptCookie(true)

        webView.webViewClient = object : WebViewClient() {
            override fun onPageFinished(view: WebView?, url: String?) {
                progressBar.visibility = View.GONE
                attemptRememberedLogin(url)
            }
        }

        webView.webChromeClient = WebChromeClient()
        webView.addJavascriptInterface(WebAppInterface(), "AndroidBridge")

        webView.loadUrl(DEFAULT_SERVER_URL)
    }

    private fun attemptRememberedLogin(url: String?) {
        if (rememberedLoginAttempted) return
        if (url == null || !url.startsWith(DEFAULT_SERVER_URL.trimEnd('/'))) return

        val prefs = getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        val phone = prefs.getString(PREF_MEMBER_PHONE, "") ?: ""
        val educationLevel = prefs.getString(PREF_EDUCATION_LEVEL, "high") ?: "high"
        if (phone.isBlank()) return

        rememberedLoginAttempted = true
        val payload = JSONObject()
            .put("member_phone", phone)
            .put("education_level", educationLevel)
            .toString()
            .replace("\\", "\\\\")
            .replace("'", "\\'")

        val script = """
            (async function() {
              try {
                const res = await fetch('/api/mobile/remember-login', {
                  method: 'POST',
                  headers: {'Content-Type': 'application/json'},
                  body: '$payload'
                });
                const data = await res.json();
                if (data && data.ok && data.redirect_url) {
                  window.location.replace(data.redirect_url);
                }
              } catch (e) {}
            })();
        """.trimIndent()
        webView.post { webView.evaluateJavascript(script, null) }
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
        if (isSensorRegistered && !isGaitMeasuring) {
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
                if (isGaitMeasuring) {
                    val timestamp = if (gaitStartTimestampNs == 0L) {
                        gaitStartTimestampNs = event.timestamp
                        0L
                    } else {
                        event.timestamp - gaitStartTimestampNs
                    }
                    gaitSamples.add(GaitSample(timestamp, ax, ay, az, gx, gy, gz))
                }
            }
            Sensor.TYPE_GYROSCOPE -> {
                gx = event.values[0]
                gy = event.values[1]
                gz = event.values[2]
            }
        }

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

    private fun startGaitMeasurement(configJson: String) {
        val config = runCatching { JSONObject(configJson) }.getOrElse { JSONObject() }
        gaitUploadUrl = config.optString("uploadUrl", "${DEFAULT_SERVER_URL.trimEnd('/')}/gait/upload-csv")
        gaitMemberPhone = config.optString("memberPhone", "")
        gaitDurationMs = (config.optDouble("durationSec", 20.0) * 1000.0).toLong().coerceIn(5_000L, 60_000L)
        gaitSamples.clear()
        gaitStartTimestampNs = 0L
        isGaitMeasuring = true
        registerSensors()
        notifyGaitEvent("started", JSONObject().put("durationSec", gaitDurationMs / 1000))

        mainHandler.removeCallbacksAndMessages(GAIT_STOP_TOKEN)
        mainHandler.postAtTime({ finishGaitMeasurementAndUpload() }, GAIT_STOP_TOKEN, android.os.SystemClock.uptimeMillis() + gaitDurationMs)
    }

    private fun finishGaitMeasurementAndUpload() {
        if (!isGaitMeasuring) return
        isGaitMeasuring = false
        val samples = gaitSamples.toList()
        notifyGaitEvent("uploading", JSONObject().put("sampleCount", samples.size))

        thread {
            try {
                if (samples.size < 80) {
                    throw IllegalStateException("not enough gait samples")
                }
                val response = uploadGaitCsv(samples)
                notifyGaitEvent("complete", JSONObject(response))
            } catch (e: Exception) {
                notifyGaitEvent("error", JSONObject().put("message", e.message ?: "gait upload failed"))
            } finally {
                if (!isCalibrating) {
                    sensorManager.unregisterListener(this)
                    isSensorRegistered = false
                }
            }
        }
    }

    private fun uploadGaitCsv(samples: List<GaitSample>): String {
        val boundary = "----FinalinziGait${System.currentTimeMillis()}"
        val lineEnd = "\r\n"
        val uploadUrl = URL(gaitUploadUrl)
        val connection = (uploadUrl.openConnection() as HttpURLConnection).apply {
            requestMethod = "POST"
            doInput = true
            doOutput = true
            useCaches = false
            connectTimeout = 15_000
            readTimeout = 60_000
            setRequestProperty("Content-Type", "multipart/form-data; boundary=$boundary")
            setRequestProperty("Accept", "application/json")
            CookieManager.getInstance().getCookie(gaitUploadUrl)?.let { cookie ->
                setRequestProperty("Cookie", cookie)
            }
        }

        DataOutputStream(connection.outputStream).use { out ->
            fun writeField(name: String, value: String) {
                out.writeBytes("--$boundary$lineEnd")
                out.writeBytes("Content-Disposition: form-data; name=\"$name\"$lineEnd$lineEnd")
                out.writeBytes(value)
                out.writeBytes(lineEnd)
            }

            if (gaitMemberPhone.isNotBlank()) {
                writeField("member_phone", gaitMemberPhone)
            }

            out.writeBytes("--$boundary$lineEnd")
            out.writeBytes("Content-Disposition: form-data; name=\"file\"; filename=\"apk_gait.csv\"$lineEnd")
            out.writeBytes("Content-Type: text/csv; charset=utf-8$lineEnd$lineEnd")
            out.writeBytes(buildGaitCsv(samples))
            out.writeBytes(lineEnd)
            out.writeBytes("--$boundary--$lineEnd")
            out.flush()
        }

        val status = connection.responseCode
        val stream = if (status in 200..299) connection.inputStream else connection.errorStream
        val body = stream.bufferedReader(Charsets.UTF_8).use { it.readText() }
        connection.disconnect()
        if (status !in 200..299) {
            throw IllegalStateException(body.ifBlank { "server returned HTTP $status" })
        }
        return body
    }

    private fun buildGaitCsv(samples: List<GaitSample>): String {
        val builder = StringBuilder()
        builder.append("# Source: FinalProjectApp APK WebView bridge\n")
        builder.append("# Accel_Maximum_Range_m_s2: 78.4532\n")
        builder.append("# Gyro_Maximum_Range_rad_s: 34.9066\n")
        builder.append("Timestamp_ns,Acc_X,Acc_Y,Acc_Z,Gyro_Clean_X,Gyro_Clean_Y,Gyro_Clean_Z\n")
        samples.forEach { s ->
            builder.append(s.timestampNs).append(',')
                .append(s.ax).append(',')
                .append(s.ay).append(',')
                .append(s.az).append(',')
                .append(s.gx).append(',')
                .append(s.gy).append(',')
                .append(s.gz).append('\n')
        }
        return builder.toString()
    }

    private fun notifyGaitEvent(status: String, payload: JSONObject) {
        val event = payload.put("status", status)
        val script = "window.onGaitApkEvent && window.onGaitApkEvent($event)"
        webView.post { webView.evaluateJavascript(script, null) }
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
        isGaitMeasuring = false
        sensorManager.unregisterListener(this)
        super.onDestroy()
    }

    inner class WebAppInterface {
        @JavascriptInterface
        fun startMeasurement(stage: String) {
            registerSensors()
            axisCalibrator.start()
            isCalibrating = true
        }

        @JavascriptInterface
        fun stopMeasurement() {
            isGaitMeasuring = false
            unregisterSensors()
            classifier.setExpectedAction(null)
        }

        @JavascriptInterface
        fun setExpectedAction(action: String) {
            classifier.setExpectedAction(action)
        }

        @JavascriptInterface
        fun startGaitMeasurement(configJson: String) {
            this@MainActivity.startGaitMeasurement(configJson)
        }

        @JavascriptInterface
        fun rememberMember(phone: String, educationLevel: String) {
            getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
                .edit()
                .putString(PREF_MEMBER_PHONE, phone.filter { it.isDigit() })
                .putString(PREF_EDUCATION_LEVEL, educationLevel.ifBlank { "high" })
                .apply()
        }
    }

    companion object {
        private const val DEFAULT_SERVER_URL = "http://192.168.0.251:5000/"
        private const val PREFS_NAME = "finalinzi_member"
        private const val PREF_MEMBER_PHONE = "member_phone"
        private const val PREF_EDUCATION_LEVEL = "education_level"
        private val GAIT_STOP_TOKEN = Any()
    }
}
