package com.example.finalprojectapp

import android.Manifest
import android.annotation.SuppressLint
import android.content.Intent
import android.content.pm.PackageManager
import android.location.Geocoder
import android.location.Location
import android.location.LocationListener
import android.location.LocationManager
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.content.Context
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.view.View
import android.webkit.CookieManager
import android.webkit.JavascriptInterface
import android.webkit.PermissionRequest
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
import java.util.Locale
import kotlin.concurrent.thread

class MainActivity : AppCompatActivity(), SensorEventListener, TextToSpeech.OnInitListener {

    private lateinit var webView: WebView
    private lateinit var progressBar: ProgressBar

    private lateinit var sensorManager: SensorManager
    private lateinit var locationManager: LocationManager
    private var accelerometer: Sensor? = null
    private var gyroscope: Sensor? = null

    private val classifier = MotionClassifier()
    private val axisCalibrator = AxisCalibrator(durationMs = 3000L)
    private var isSensorRegistered = false
    private var isCalibrating = false
    private var gaitTts: TextToSpeech? = null
    private var gaitTtsReady = false
    private var pengteuRecognizer: SpeechRecognizer? = null

    private val mainHandler = Handler(Looper.getMainLooper())
    private val gaitSamples = mutableListOf<GaitSample>()
    private var isGaitSessionActive = false
    private var isGaitMeasuring = false
    private var gaitUploadUrl = ""
    private var gaitMemberPhone = ""
    private var gaitWearMs = 7_000L
    private var gaitReadyMs = 3_000L
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
        setupLocation()
        gaitTts = TextToSpeech(this, this)

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

        webView.webChromeClient = object : WebChromeClient() {
            override fun onPermissionRequest(request: PermissionRequest?) {
                request?.grant(request.resources)
            }
        }
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

    private fun setupLocation() {
        locationManager = getSystemService(LOCATION_SERVICE) as LocationManager
    }

    override fun onInit(status: Int) {
        if (status != TextToSpeech.SUCCESS) return
        gaitTtsReady = true
        gaitTts?.language = Locale.KOREAN
        gaitTts?.setSpeechRate(0.9f)
        gaitTts?.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
            override fun onStart(utteranceId: String?) {}

            override fun onDone(utteranceId: String?) {
                if (utteranceId?.startsWith("pengteu-") == true) {
                    notifyPengteuNative("onTtsEnd")
                }
            }

            @Deprecated("Deprecated in Java")
            override fun onError(utteranceId: String?) {
                if (utteranceId?.startsWith("pengteu-") == true) {
                    notifyPengteuNative("onTtsEnd")
                }
            }
        })
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
        if (isSensorRegistered && !isGaitSessionActive && !isGaitMeasuring) {
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
        gaitWearMs = (config.optDouble("wearSec", 7.0) * 1000.0).toLong().coerceIn(0L, 15_000L)
        gaitReadyMs = (config.optDouble("readySec", 3.0) * 1000.0).toLong().coerceIn(0L, 10_000L)
        gaitDurationMs = (config.optDouble("durationSec", 20.0) * 1000.0).toLong().coerceIn(5_000L, 60_000L)
        gaitSamples.clear()
        gaitStartTimestampNs = 0L
        isGaitSessionActive = true
        isGaitMeasuring = false
        registerSensors()
        notifyGaitEvent(
            "started",
            JSONObject()
                .put("wearSec", gaitWearMs / 1000)
                .put("readySec", gaitReadyMs / 1000)
                .put("durationSec", gaitDurationMs / 1000)
        )

        val preparationMs = gaitWearMs + gaitReadyMs
        mainHandler.removeCallbacksAndMessages(GAIT_STOP_TOKEN)
        mainHandler.postAtTime(
            { beginGaitCollection() },
            GAIT_STOP_TOKEN,
            android.os.SystemClock.uptimeMillis() + preparationMs
        )
        mainHandler.postAtTime(
            { finishGaitMeasurementAndUpload() },
            GAIT_STOP_TOKEN,
            android.os.SystemClock.uptimeMillis() + preparationMs + gaitDurationMs
        )
    }

    private fun beginGaitCollection() {
        if (!isGaitSessionActive) return
        gaitSamples.clear()
        gaitStartTimestampNs = 0L
        isGaitMeasuring = true
        notifyGaitEvent("measuring", JSONObject().put("durationSec", gaitDurationMs / 1000))
    }

    private fun finishGaitMeasurementAndUpload() {
        if (!isGaitSessionActive) return
        isGaitSessionActive = false
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
            throw IllegalStateException("gait upload failed")
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

    private fun speakGaitCue(text: String) {
        if (!gaitTtsReady || text.isBlank()) return
        gaitTts?.stop()
        gaitTts?.speak(text, TextToSpeech.QUEUE_FLUSH, null, "gait-${System.currentTimeMillis()}")
    }

    private fun speakPengteu(text: String, rate: Float, volume: Float) {
        if (!gaitTtsReady || text.isBlank()) {
            notifyPengteuNative("onTtsEnd")
            return
        }
        val params = Bundle().apply {
            putFloat(TextToSpeech.Engine.KEY_PARAM_VOLUME, volume.coerceIn(0f, 1f))
        }
        gaitTts?.stop()
        gaitTts?.setSpeechRate(rate.coerceIn(0.65f, 1.15f))
        gaitTts?.speak(text, TextToSpeech.QUEUE_FLUSH, params, "pengteu-${System.currentTimeMillis()}")
    }

    private fun stopPengteuTts() {
        gaitTts?.stop()
        notifyPengteuNative("onTtsEnd")
    }

    private fun notifyPengteuNative(functionName: String, argument: String? = null) {
        val script = if (argument == null) {
            "window.PengteuAssistantNative && window.PengteuAssistantNative.$functionName()"
        } else {
            "window.PengteuAssistantNative && window.PengteuAssistantNative.$functionName(${JSONObject.quote(argument)})"
        }
        webView.post { webView.evaluateJavascript(script, null) }
    }

    private fun startPengteuStt() {
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(arrayOf(Manifest.permission.RECORD_AUDIO), AUDIO_PERMISSION_REQUEST)
            return
        }
        if (!SpeechRecognizer.isRecognitionAvailable(this)) {
            notifyPengteuNative("onSttError", "이 기기에서는 음성 인식을 사용할 수 없어요. 글자로 입력해 주세요.")
            return
        }
        stopPengteuTts()
        val recognizer = pengteuRecognizer ?: SpeechRecognizer.createSpeechRecognizer(this).also {
            pengteuRecognizer = it
            it.setRecognitionListener(object : RecognitionListener {
                override fun onReadyForSpeech(params: Bundle?) {
                    notifyPengteuNative("onSttStart")
                }

                override fun onBeginningOfSpeech() {}
                override fun onRmsChanged(rmsdB: Float) {}
                override fun onBufferReceived(buffer: ByteArray?) {}
                override fun onEndOfSpeech() {
                    notifyPengteuNative("onSttEnd")
                }

                override fun onError(error: Int) {
                    notifyPengteuNative("onSttEnd")
                    notifyPengteuNative("onSttError", "음성을 잘 듣지 못했어요. 마이크를 다시 눌러 말해 주세요.")
                }

                override fun onResults(results: Bundle?) {
                    notifyPengteuNative("onSttEnd")
                    val matches = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                    val text = matches?.firstOrNull().orEmpty()
                    if (text.isBlank()) {
                        notifyPengteuNative("onSttError", "음성을 잘 듣지 못했어요. 다시 말해 주세요.")
                    } else {
                        notifyPengteuNative("onSttResult", text)
                    }
                }

                override fun onPartialResults(partialResults: Bundle?) {}
                override fun onEvent(eventType: Int, params: Bundle?) {}
            })
        }
        val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
            putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
            putExtra(RecognizerIntent.EXTRA_LANGUAGE, "ko-KR")
            putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, false)
            putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 1)
        }
        recognizer.startListening(intent)
    }

    private fun requestOrientationLocation() {
        if (
            checkSelfPermission(Manifest.permission.ACCESS_FINE_LOCATION) != PackageManager.PERMISSION_GRANTED &&
            checkSelfPermission(Manifest.permission.ACCESS_COARSE_LOCATION) != PackageManager.PERMISSION_GRANTED
        ) {
            requestPermissions(
                arrayOf(
                    Manifest.permission.ACCESS_FINE_LOCATION,
                    Manifest.permission.ACCESS_COARSE_LOCATION
                ),
                LOCATION_PERMISSION_REQUEST
            )
            return
        }

        val providers = listOf(LocationManager.GPS_PROVIDER, LocationManager.NETWORK_PROVIDER)
            .filter { provider -> runCatching { locationManager.isProviderEnabled(provider) }.getOrDefault(false) }
        val last = providers
            .mapNotNull { provider -> runCatching { locationManager.getLastKnownLocation(provider) }.getOrNull() }
            .maxByOrNull { it.time }
        if (last != null) {
            sendOrientationLocation(last)
            return
        }

        val provider = providers.firstOrNull()
        if (provider == null) {
            notifyOrientationLocation(false, "위치 서비스를 켜 주세요.")
            return
        }

        val listener = object : LocationListener {
            override fun onLocationChanged(location: Location) {
                runCatching { locationManager.removeUpdates(this) }
                sendOrientationLocation(location)
            }

            override fun onProviderDisabled(provider: String) {
                notifyOrientationLocation(false, "위치 서비스를 켜 주세요.")
            }
        }
        runCatching {
            locationManager.requestSingleUpdate(provider, listener, Looper.getMainLooper())
        }.onFailure {
            notifyOrientationLocation(false, "현재 위치를 가져오지 못했어요.")
        }
    }

    private fun regionFromLocation(location: Location): JSONObject {
        val json = JSONObject()
            .put("latitude", location.latitude)
            .put("longitude", location.longitude)
            .put("accuracy", location.accuracy.toDouble())
        val address = runCatching {
            Geocoder(this, Locale.KOREAN).getFromLocation(location.latitude, location.longitude, 1)
                ?.firstOrNull()
        }.getOrNull()
        val addressLine = address?.getAddressLine(0).orEmpty()
        val parts = addressLine.split(" ").map { it.trim() }.filter { it.isNotBlank() }
        val sigungu = listOfNotNull(
            address?.subLocality,
            address?.locality,
            address?.subAdminArea,
            parts.firstOrNull { it.endsWith("구") || it.endsWith("군") || it.endsWith("시") }
        ).firstOrNull { it.isNotBlank() }.orEmpty()
        val dong = listOfNotNull(
            address?.thoroughfare,
            address?.featureName,
            parts.firstOrNull { it.endsWith("동") || it.endsWith("읍") || it.endsWith("면") || it.endsWith("리") }
        ).firstOrNull { it.isNotBlank() && it != sigungu }.orEmpty()
        json.put("sigungu", sigungu)
        json.put("location", dong)
        json.put("address", addressLine)
        return json
    }

    private fun sendOrientationLocation(location: Location) {
        val payload = regionFromLocation(location)
        thread {
            try {
                val url = URL("${DEFAULT_SERVER_URL.trimEnd('/')}/api/orientation/location")
                val body = payload.toString().toByteArray(Charsets.UTF_8)
                val connection = (url.openConnection() as HttpURLConnection).apply {
                    requestMethod = "POST"
                    doOutput = true
                    connectTimeout = 8_000
                    readTimeout = 8_000
                    setRequestProperty("Content-Type", "application/json; charset=utf-8")
                    setRequestProperty("Accept", "application/json")
                    CookieManager.getInstance().getCookie(DEFAULT_SERVER_URL)?.let { cookie ->
                        setRequestProperty("Cookie", cookie)
                    }
                }
                connection.outputStream.use { it.write(body) }
                val ok = connection.responseCode in 200..299
                connection.disconnect()
                notifyOrientationLocation(ok, if (ok) "위치를 확인했어요." else "위치를 저장하지 못했어요.")
            } catch (e: Exception) {
                notifyOrientationLocation(false, "위치를 저장하지 못했어요.")
            }
        }
    }

    private fun notifyOrientationLocation(ok: Boolean, message: String) {
        val payload = JSONObject()
            .put("ok", ok)
            .put("message", message)
        val script = "window.onOrientationLocationEvent && window.onOrientationLocationEvent($payload)"
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
        isGaitSessionActive = false
        isGaitMeasuring = false
        mainHandler.removeCallbacksAndMessages(GAIT_STOP_TOKEN)
        gaitTts?.stop()
        gaitTts?.shutdown()
        pengteuRecognizer?.destroy()
        sensorManager.unregisterListener(this)
        super.onDestroy()
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == AUDIO_PERMISSION_REQUEST) {
            if (grantResults.firstOrNull() == PackageManager.PERMISSION_GRANTED) {
                startPengteuStt()
            } else {
                notifyPengteuNative("onSttError", "마이크 권한이 필요해요. 앱 설정에서 마이크 권한을 허용해 주세요.")
            }
            return
        }
        if (requestCode == LOCATION_PERMISSION_REQUEST) {
            if (grantResults.any { it == PackageManager.PERMISSION_GRANTED }) {
                requestOrientationLocation()
            } else {
                notifyOrientationLocation(false, "위치 권한이 필요해요. 동네와 시군구는 직접 말씀해 주세요.")
            }
        }
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
            isGaitSessionActive = false
            isGaitMeasuring = false
            mainHandler.removeCallbacksAndMessages(GAIT_STOP_TOKEN)
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
        fun speakGaitCue(text: String) {
            this@MainActivity.speakGaitCue(text)
        }

        @JavascriptInterface
        fun speakPengteu(text: String, rate: Double, volume: Double) {
            this@MainActivity.speakPengteu(text, rate.toFloat(), volume.toFloat())
        }

        @JavascriptInterface
        fun stopPengteuTts() {
            this@MainActivity.stopPengteuTts()
        }

        @JavascriptInterface
        fun startPengteuStt() {
            this@MainActivity.startPengteuStt()
        }

        @JavascriptInterface
        fun requestOrientationLocation() {
            this@MainActivity.requestOrientationLocation()
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
        private const val AUDIO_PERMISSION_REQUEST = 1101
        private const val LOCATION_PERMISSION_REQUEST = 1201
        private val GAIT_STOP_TOKEN = Any()
    }
}
