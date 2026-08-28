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
import java.net.Inet4Address
import java.net.NetworkInterface
import java.net.URL
import java.util.Locale
import java.util.concurrent.ExecutorCompletionService
import java.util.concurrent.Executors
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
    private var testRecognizer: SpeechRecognizer? = null

    // "펭트야" always-on 웨이크워드 (네이티브 전용 — WebView엔 Web Speech 없음)
    private var wakeRecognizer: SpeechRecognizer? = null
    private var wakeWantsRun = false        // 웨이크 리스너를 계속 돌려야 하는 상태
    private var isWakeListening = false      // 현재 실제로 듣고 있는지
    private var isPengteuSttActive = false   // 명령 STT 중(마이크 점유) → 웨이크 정지
    private var isTestSttActive = false      // 인지검사 답변 STT 중
    private var testSttStopRequested = false
    private var isTtsSpeaking = false        // 펭트/보행 TTS 발화 중 → 자기목소리 오탐 방지
    private val wakeRestartToken = Any()     // 웨이크 재시작 예약 취소용 핸들러 토큰

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

    // 실제로 접속할 서버 주소. 앱 시작 시 같은 WiFi 서브넷을 자동탐색해 결정한다.
    // 탐색 전/실패 시에는 마지막으로 저장된 주소(없으면 빌드 기본값)를 쓴다.
    @Volatile
    private var serverBaseUrl: String = DEFAULT_SERVER_URL

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

        // 저장된(마지막 성공) 서버 주소를 우선 폴백으로 둔다. 자동탐색이 새로 찾으면 갱신된다.
        serverBaseUrl = getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .getString(PREF_SERVER_URL, null) ?: DEFAULT_SERVER_URL

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
        // 뷰포트(width=device-width)를 존중해 폰 화면에 맞게 렌더 + 로드시 화면맞춤
        webView.settings.useWideViewPort = true
        webView.settings.loadWithOverviewMode = true
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

        discoverServerAndLoad()
    }

    /**
     * 서버 주소를 자동으로 정한다. 발표장 WiFi가 바뀌어도 재빌드 없이 붙게 하기 위함.
     * 1) 마지막으로 쓰던(또는 기본) 주소가 살아있으면 그대로 사용
     * 2) 아니면 같은 WiFi /24 서브넷을 병렬 스캔해 /ping 응답 서버를 찾는다
     * 3) 다 실패하면 폴백 주소로 로드한다(연결 실패는 웹 화면에서 드러남)
     */
    private fun discoverServerAndLoad() {
        thread {
            val fallback = serverBaseUrl
            var target = if (pingServer(fallback)) fallback else null
            if (target == null) target = scanSubnetForServer()
            val finalUrl = target ?: fallback
            if (target != null && target != fallback) saveServerUrl(target)
            runOnUiThread {
                serverBaseUrl = finalUrl
                webView.loadUrl(finalUrl)
            }
        }
    }

    /** 해당 주소의 /ping 이 giukhaji 서버인지 짧은 타임아웃으로 확인한다. */
    private fun pingServer(baseUrl: String): Boolean {
        return try {
            val url = URL("${baseUrl.trimEnd('/')}/ping")
            val conn = (url.openConnection() as HttpURLConnection).apply {
                requestMethod = "GET"
                connectTimeout = 600
                readTimeout = 600
            }
            val ok = conn.responseCode == 200 &&
                conn.inputStream.bufferedReader().use { it.readText() }.contains("giukhaji")
            conn.disconnect()
            ok
        } catch (e: Exception) {
            false
        }
    }

    /** 현재 WiFi IPv4의 /24 서브넷(1~254)을 병렬로 훑어 giukhaji 서버를 찾는다. */
    private fun scanSubnetForServer(): String? {
        val localIp = localIpv4() ?: return null
        val prefix = localIp.substringBeforeLast('.')      // 예: 192.168.50
        val port = runCatching {
            URL(serverBaseUrl).port.let { if (it > 0) it else 5001 }
        }.getOrDefault(5001)

        val pool = Executors.newFixedThreadPool(40)
        val ecs = ExecutorCompletionService<String?>(pool)
        var submitted = 0
        for (i in 1..254) {
            val candidate = "http://$prefix.$i:$port/"
            ecs.submit { if (pingServer(candidate)) candidate else null }
            submitted++
        }
        var found: String? = null
        try {
            for (n in 0 until submitted) {
                val r = ecs.take().get()
                if (r != null) { found = r; break }
            }
        } catch (e: Exception) {
            // 무시하고 폴백
        } finally {
            pool.shutdownNow()
        }
        return found
    }

    /** 사이트 로컬(사설망) IPv4 주소를 반환한다. 별도 권한 불필요. */
    private fun localIpv4(): String? {
        return runCatching {
            NetworkInterface.getNetworkInterfaces().toList()
                .filter { runCatching { it.isUp && !it.isLoopback }.getOrDefault(false) }
                .flatMap { it.inetAddresses.toList() }
                .filterIsInstance<Inet4Address>()
                .firstOrNull { it.isSiteLocalAddress }
                ?.hostAddress
        }.getOrNull()
    }

    private fun saveServerUrl(url: String) {
        getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .edit()
            .putString(PREF_SERVER_URL, url)
            .apply()
    }

    private fun attemptRememberedLogin(url: String?) {
        if (rememberedLoginAttempted) return
        if (url == null || !url.startsWith(serverBaseUrl.trimEnd('/'))) return

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
            override fun onStart(utteranceId: String?) {
                // 발화 중엔 웨이크 인식을 멈춰 TTS 소리로 자기 자신을 깨우지 않게 한다.
                isTtsSpeaking = true
                mainHandler.post { pauseWakeForOther() }
            }

            override fun onDone(utteranceId: String?) {
                isTtsSpeaking = false
                if (utteranceId?.startsWith("pengteu-") == true) {
                    notifyPengteuNative("onTtsEnd")
                }
                mainHandler.post { resumeWake() }
            }

            @Deprecated("Deprecated in Java")
            override fun onError(utteranceId: String?) {
                isTtsSpeaking = false
                if (utteranceId?.startsWith("pengteu-") == true) {
                    notifyPengteuNative("onTtsEnd")
                }
                mainHandler.post { resumeWake() }
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
        gaitUploadUrl = config.optString("uploadUrl", "${serverBaseUrl.trimEnd('/')}/gait/upload-csv")
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

    private fun notifyTestSpeech(functionName: String, argument: String? = null) {
        val script = if (argument == null) {
            "window.TestSpeechNative && window.TestSpeechNative.$functionName()"
        } else {
            "window.TestSpeechNative && window.TestSpeechNative.$functionName(${JSONObject.quote(argument)})"
        }
        webView.post { webView.evaluateJavascript(script, null) }
    }

    private fun startTestStt() {
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(arrayOf(Manifest.permission.RECORD_AUDIO), TEST_AUDIO_PERMISSION_REQUEST)
            return
        }
        if (!SpeechRecognizer.isRecognitionAvailable(this)) {
            notifyTestSpeech("onError", "이 기기에서는 음성 인식을 사용할 수 없어요.")
            return
        }
        if (isTestSttActive) return
        stopPengteuTts()
        runCatching { pengteuRecognizer?.cancel() }
        isPengteuSttActive = false
        pauseWakeForOther()
        testSttStopRequested = false

        val recognizer = testRecognizer ?: SpeechRecognizer.createSpeechRecognizer(this).also {
            testRecognizer = it
            it.setRecognitionListener(object : RecognitionListener {
                override fun onReadyForSpeech(params: Bundle?) = notifyTestSpeech("onStart")
                override fun onBeginningOfSpeech() {}
                override fun onRmsChanged(rmsdB: Float) {}
                override fun onBufferReceived(buffer: ByteArray?) {}
                override fun onEndOfSpeech() {}

                override fun onError(error: Int) {
                    val requested = testSttStopRequested
                    isTestSttActive = false
                    notifyTestSpeech("onEnd")
                    if (!requested) {
                        notifyTestSpeech("onError", "음성을 잘 듣지 못했어요. 다시 말씀해 주세요.")
                    }
                    resumeWake()
                }

                override fun onResults(results: Bundle?) {
                    val text = results
                        ?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                        ?.firstOrNull()
                        .orEmpty()
                    isTestSttActive = false
                    if (text.isNotBlank()) notifyTestSpeech("onResult", text)
                    else notifyTestSpeech("onError", "음성을 잘 듣지 못했어요. 다시 말씀해 주세요.")
                    notifyTestSpeech("onEnd")
                    resumeWake()
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
        isTestSttActive = true
        recognizer.startListening(intent)
    }

    private fun stopTestStt() {
        testSttStopRequested = true
        isTestSttActive = false
        runCatching { testRecognizer?.cancel() }
        notifyTestSpeech("onEnd")
        resumeWake()
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
                    isPengteuSttActive = false
                    resumeWake()
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
                    isPengteuSttActive = false
                    resumeWake()
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
        // 명령 STT가 마이크를 점유하는 동안 웨이크 리스너 정지(안드로이드는 SR 동시 1개만)
        isPengteuSttActive = true
        pauseWakeForOther()
        recognizer.startListening(intent)
    }

    // ── "펭트야" always-on 웨이크워드 (네이티브 엔진) ─────────────
    // 웹의 wakeShouldPause() 경합 로직이 startWakeWord/stopWakeWord로 큰 스위치를 제어하고,
    // 네이티브는 발화가 끝날 때마다 스스로 재시작하며(SR은 1발화 후 정지) 자기 STT/TTS 중엔 즉시 자가정지.
    private fun isWakeWord(raw: String): Boolean {
        val t = raw.replace(Regex("\\s+"), "")
        if (t.isEmpty()) return false
        if (Regex("[펭팽펜]트[야아님씨이]").containsMatchIn(t)) return true          // 펭트야/펜트야/펭트님…
        if (t.length <= 3 && Regex("[펭팽펜]트").containsMatchIn(t)) return true      // 짧게 "펭트"만 불러도
        return false
    }

    private fun ensureWakeRecognizer(): SpeechRecognizer? {
        if (!SpeechRecognizer.isRecognitionAvailable(this)) return null
        return wakeRecognizer ?: SpeechRecognizer.createSpeechRecognizer(this).also {
            wakeRecognizer = it
            it.setRecognitionListener(object : RecognitionListener {
                override fun onReadyForSpeech(params: Bundle?) { isWakeListening = true }
                override fun onBeginningOfSpeech() {}
                override fun onRmsChanged(rmsdB: Float) {}
                override fun onBufferReceived(buffer: ByteArray?) {}
                override fun onEndOfSpeech() {}

                override fun onError(error: Int) {
                    isWakeListening = false
                    scheduleWakeRestart()
                }

                override fun onResults(results: Bundle?) {
                    isWakeListening = false
                    val matches = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                    if (matches?.any { isWakeWord(it) } == true) onWakeDetected() else scheduleWakeRestart()
                }

                override fun onPartialResults(partialResults: Bundle?) {
                    // 부분 결과로 빠르게 감지(발화 종료 전 반응)
                    val matches = partialResults?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                    if (matches?.any { isWakeWord(it) } == true) onWakeDetected()
                }

                override fun onEvent(eventType: Int, params: Bundle?) {}
            })
        }
    }

    private fun onWakeDetected() {
        if (isPengteuSttActive) return
        mainHandler.removeCallbacksAndMessages(wakeRestartToken)
        isWakeListening = false
        runCatching { wakeRecognizer?.cancel() }
        // 웹의 onWakeWord(=activateByWake): 패널 열고 응대 후 명령 청취로 전환.
        notifyPengteuNative("onWakeWord")
    }

    private fun pumpWake() {
        if (!wakeWantsRun || isWakeListening || isPengteuSttActive || isTestSttActive || isTtsSpeaking) return
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) return
        val recognizer = ensureWakeRecognizer() ?: return
        val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
            putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
            putExtra(RecognizerIntent.EXTRA_LANGUAGE, "ko-KR")
            putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, true)
            putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 3)
        }
        isWakeListening = true
        runCatching { recognizer.startListening(intent) }.onFailure {
            isWakeListening = false
            scheduleWakeRestart()
        }
    }

    private fun scheduleWakeRestart() {
        if (!wakeWantsRun) return
        mainHandler.removeCallbacksAndMessages(wakeRestartToken)
        // 짧은 텀을 둬 busy-loop/발열을 피하고 마이크를 잠깐 놓아준다.
        mainHandler.postAtTime(
            { pumpWake() },
            wakeRestartToken,
            android.os.SystemClock.uptimeMillis() + 600L
        )
    }

    private fun startWakeWord() {
        wakeWantsRun = true
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(arrayOf(Manifest.permission.RECORD_AUDIO), WAKE_AUDIO_PERMISSION_REQUEST)
            return
        }
        pumpWake()
    }

    private fun stopWakeWord() {
        wakeWantsRun = false
        mainHandler.removeCallbacksAndMessages(wakeRestartToken)
        isWakeListening = false
        runCatching { wakeRecognizer?.cancel() }
    }

    // 다른 마이크 사용자(명령 STT·TTS)를 위해 일시정지하되, 재개 의사(wakeWantsRun)는 유지.
    private fun pauseWakeForOther() {
        mainHandler.removeCallbacksAndMessages(wakeRestartToken)
        isWakeListening = false
        runCatching { wakeRecognizer?.cancel() }
    }

    private fun resumeWake() {
        if (!wakeWantsRun || isPengteuSttActive || isTestSttActive || isTtsSpeaking) return
        scheduleWakeRestart()
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
                val url = URL("${serverBaseUrl.trimEnd('/')}/api/orientation/location")
                val body = payload.toString().toByteArray(Charsets.UTF_8)
                val connection = (url.openConnection() as HttpURLConnection).apply {
                    requestMethod = "POST"
                    doOutput = true
                    connectTimeout = 8_000
                    readTimeout = 8_000
                    setRequestProperty("Content-Type", "application/json; charset=utf-8")
                    setRequestProperty("Accept", "application/json")
                    CookieManager.getInstance().getCookie(serverBaseUrl)?.let { cookie ->
                        setRequestProperty("Cookie", cookie)
                    }
                }
                connection.outputStream.use { it.write(body) }
                val ok = connection.responseCode in 200..299
                connection.disconnect()
                notifyOrientationLocation(
                    ok,
                    if (ok) "위치를 확인했어요." else "위치를 저장하지 못했어요.",
                    payload.optString("sigungu", ""),
                    payload.optString("location", "")
                )
            } catch (e: Exception) {
                notifyOrientationLocation(false, "위치를 저장하지 못했어요.")
            }
        }
    }

    private fun notifyOrientationLocation(
        ok: Boolean,
        message: String,
        sigungu: String = "",
        location: String = ""
    ) {
        val payload = JSONObject()
            .put("ok", ok)
            .put("message", message)
            .put("sigungu", sigungu)
            .put("location", location)
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

    override fun onPause() {
        super.onPause()
        // 백그라운드에선 마이크를 놓아준다(재개 의사 wakeWantsRun은 유지).
        pauseWakeForOther()
    }

    override fun onResume() {
        super.onResume()
        resumeWake()
    }

    override fun onDestroy() {
        isGaitSessionActive = false
        isGaitMeasuring = false
        mainHandler.removeCallbacksAndMessages(GAIT_STOP_TOKEN)
        mainHandler.removeCallbacksAndMessages(wakeRestartToken)
        wakeWantsRun = false
        gaitTts?.stop()
        gaitTts?.shutdown()
        pengteuRecognizer?.destroy()
        testRecognizer?.destroy()
        wakeRecognizer?.destroy()
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
        if (requestCode == TEST_AUDIO_PERMISSION_REQUEST) {
            if (grantResults.firstOrNull() == PackageManager.PERMISSION_GRANTED) {
                startTestStt()
            } else {
                notifyTestSpeech("onError", "마이크 권한이 필요해요. 앱 설정에서 허용해 주세요.")
            }
            return
        }
        if (requestCode == WAKE_AUDIO_PERMISSION_REQUEST) {
            if (grantResults.firstOrNull() == PackageManager.PERMISSION_GRANTED) {
                pumpWake()
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
        fun startTestStt() {
            mainHandler.post { this@MainActivity.startTestStt() }
        }

        @JavascriptInterface
        fun stopTestStt() {
            mainHandler.post { this@MainActivity.stopTestStt() }
        }

        @JavascriptInterface
        fun startWakeWord() {
            mainHandler.post { this@MainActivity.startWakeWord() }
        }

        @JavascriptInterface
        fun stopWakeWord() {
            mainHandler.post { this@MainActivity.stopWakeWord() }
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
        // 실제 서버 주소는 local.properties(gitignore)의 serverUrl로 빌드 시 주입된다.
        private val DEFAULT_SERVER_URL = BuildConfig.SERVER_URL
        private const val PREFS_NAME = "finalinzi_member"
        private const val PREF_SERVER_URL = "server_url"
        private const val PREF_MEMBER_PHONE = "member_phone"
        private const val PREF_EDUCATION_LEVEL = "education_level"
        private const val AUDIO_PERMISSION_REQUEST = 1101
        private const val WAKE_AUDIO_PERMISSION_REQUEST = 1102
        private const val TEST_AUDIO_PERMISSION_REQUEST = 1103
        private const val LOCATION_PERMISSION_REQUEST = 1201
        private val GAIT_STOP_TOKEN = Any()
    }
}
