package com.example.finalprojectapp

import android.Manifest
import android.annotation.SuppressLint
import android.content.Intent
import android.content.pm.PackageManager
import android.location.Geocoder
import android.location.Location
import android.location.LocationListener
import android.location.LocationManager
import android.media.AudioAttributes
import android.media.AudioManager
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
import android.util.Log
import android.view.View
import android.webkit.CookieManager
import android.webkit.JavascriptInterface
import android.webkit.PermissionRequest
import android.webkit.WebChromeClient
import android.webkit.WebSettings
import android.webkit.WebView
import android.webkit.WebViewClient
import android.webkit.WebResourceError
import android.webkit.WebResourceRequest
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
import kotlin.math.PI
import kotlin.math.sqrt

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
    @Volatile private var exerciseTtsText = ""
    @Volatile private var exerciseTtsResumeOffset = 0
    @Volatile private var exerciseTtsChunkBaseOffset = 0
    @Volatile private var exerciseTtsPaused = false
    private var exerciseTtsRate = 0.9f
    private var exerciseTtsVolume = 0.8f
    private var pengteuRecognizer: SpeechRecognizer? = null
    private var pengteuSttRetryCount = 0

    // 무료 로컬 Vosk 한국어 STT로 "펭트야"만 상시 감지한다.
    private lateinit var wakeWordEngine: VoskWakeWordEngine
    private var wakeWantsRun = false        // 웨이크 리스너를 계속 돌려야 하는 상태
    private var isPengteuSttActive = false   // 명령 STT 중(마이크 점유) → 웨이크 정지
    private var isTtsSpeaking = false        // 펭트/보행 TTS 발화 중 → 자기목소리 오탐 방지

    private val mainHandler = Handler(Looper.getMainLooper())
    private val gaitSamples = mutableListOf<GaitSample>()
    private val gaitCalibrationSamples = mutableListOf<GaitSample>()
    private var isGaitSessionActive = false
    private var isGaitCalibrationActive = false
    private var isGaitMeasuring = false
    private var gaitUploadUrl = ""
    private var gaitMemberPhone = ""
    private var gaitWearMs = 7_000L
    private var gaitReadyMs = 3_000L
    private var gaitDurationMs = 20_000L
    private var gaitStartTimestampNs = 0L
    private var rememberedLoginAttempted = false

    // 레거시 서버용 메서드는 기존 발표 빌드와의 diff를 작게 유지하기 위해 남겨 두되,
    // 오프라인 진입점에서는 호출하지 않으며 WebView 네트워크도 차단한다.
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

    private data class GaitBasis(
        val vertical: DoubleArray,
        val ml: DoubleArray,
        val ap: DoubleArray
    )

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_main)
        volumeControlStream = AudioManager.STREAM_MUSIC

        webView = findViewById(R.id.webView)
        progressBar = findViewById(R.id.progressBar)
        wakeWordEngine = VoskWakeWordEngine(
            this,
            onWakeWord = { transcript -> onWakeDetected(transcript) },
            onStatus = { status, message -> notifyWakeStatus(status, message) },
        )

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
        webView.settings.blockNetworkLoads = true
        webView.settings.allowFileAccess = true
        webView.settings.allowContentAccess = false
        webView.settings.javaScriptCanOpenWindowsAutomatically = false
        // 뷰포트(width=device-width)를 존중해 폰 화면에 맞게 렌더 + 로드시 화면맞춤
        webView.settings.useWideViewPort = true
        webView.settings.loadWithOverviewMode = true
        CookieManager.getInstance().setAcceptCookie(true)

        webView.webViewClient = object : WebViewClient() {
            override fun onPageFinished(view: WebView?, url: String?) {
                progressBar.visibility = View.GONE
            }

            override fun onReceivedError(
                view: WebView?,
                request: WebResourceRequest?,
                error: WebResourceError?,
            ) {
                if (request?.isForMainFrame == true && request.url.toString() != OFFLINE_ERROR_URL) {
                    view?.loadUrl(OFFLINE_ERROR_URL)
                }
            }
        }

        webView.webChromeClient = object : WebChromeClient() {
            override fun onPermissionRequest(request: PermissionRequest?) {
                // 로컬 화면은 Web API 권한이 필요 없다. 센서/음성은 명시적인
                // AndroidBridge만 사용하므로 WebView의 포괄 권한 요청은 거부한다.
                request?.deny()
            }
        }
        webView.addJavascriptInterface(WebAppInterface(), "AndroidBridge")

        webView.loadUrl(OFFLINE_HOME_URL)
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
        if (status != TextToSpeech.SUCCESS) {
            gaitTtsReady = false
            return
        }

        val audioAttributes = AudioAttributes.Builder()
            .setUsage(AudioAttributes.USAGE_MEDIA)
            .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
            .build()
        gaitTts?.setAudioAttributes(audioAttributes)

        var languageResult = gaitTts?.setLanguage(Locale.KOREA)
        if (languageResult == TextToSpeech.LANG_MISSING_DATA ||
            languageResult == TextToSpeech.LANG_NOT_SUPPORTED
        ) {
            languageResult = gaitTts?.setLanguage(Locale.KOREAN)
        }
        gaitTtsReady = languageResult != TextToSpeech.LANG_MISSING_DATA &&
            languageResult != TextToSpeech.LANG_NOT_SUPPORTED
        if (!gaitTtsReady) return

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
                } else if (utteranceId?.startsWith("exercise-") == true) {
                    if (exerciseTtsPaused) return
                    clearExerciseTtsState()
                    notifyExerciseTtsEnd()
                }
                mainHandler.post { resumeWake() }
            }

            override fun onRangeStart(
                utteranceId: String?,
                start: Int,
                end: Int,
                frame: Int
            ) {
                if (utteranceId?.startsWith("exercise-") == true) {
                    exerciseTtsResumeOffset =
                        (exerciseTtsChunkBaseOffset + start).coerceAtMost(exerciseTtsText.length)
                }
            }

            override fun onStop(utteranceId: String?, interrupted: Boolean) {
                isTtsSpeaking = false
            }

            @Deprecated("Deprecated in Java")
            override fun onError(utteranceId: String?) {
                isTtsSpeaking = false
                if (utteranceId?.startsWith("pengteu-") == true) {
                    notifyPengteuNative("onTtsEnd")
                } else if (utteranceId?.startsWith("exercise-") == true) {
                    if (exerciseTtsPaused) return
                    clearExerciseTtsState()
                    notifyExerciseTtsEnd()
                }
                mainHandler.post { resumeWake() }
            }
        })
    }

    private fun registerSensors() {
        if (!isSensorRegistered) {
            accelerometer?.let {
                sensorManager.registerListener(this, it, SENSOR_PERIOD_US)
            }
            gyroscope?.let {
                sensorManager.registerListener(this, it, SENSOR_PERIOD_US)
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
                if (isGaitCalibrationActive) {
                    gaitCalibrationSamples.add(
                        GaitSample(event.timestamp, ax, ay, az, gx, gy, gz)
                    )
                }
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
        gaitCalibrationSamples.clear()
        gaitStartTimestampNs = 0L
        isGaitSessionActive = true
        isGaitCalibrationActive = false
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
        if (gaitReadyMs > 0L) {
            mainHandler.postAtTime(
                { beginGaitCalibration() },
                GAIT_STOP_TOKEN,
                android.os.SystemClock.uptimeMillis() + gaitWearMs
            )
        }
        mainHandler.postAtTime(
            { beginGaitCollection() },
            GAIT_STOP_TOKEN,
            android.os.SystemClock.uptimeMillis() + preparationMs
        )
        mainHandler.postAtTime(
                { finishGaitMeasurementAndAnalyze() },
            GAIT_STOP_TOKEN,
            android.os.SystemClock.uptimeMillis() + preparationMs + gaitDurationMs
        )
    }

    private fun beginGaitCalibration() {
        if (!isGaitSessionActive) return
        gaitCalibrationSamples.clear()
        isGaitCalibrationActive = true
        isGaitMeasuring = false
        notifyGaitEvent(
            "calibrating",
            JSONObject().put("durationSec", gaitReadyMs / 1000)
        )
    }

    private fun beginGaitCollection() {
        if (!isGaitSessionActive) return
        isGaitCalibrationActive = false
        gaitSamples.clear()
        gaitStartTimestampNs = 0L
        isGaitMeasuring = true
        notifyGaitEvent("measuring", JSONObject().put("durationSec", gaitDurationMs / 1000))
    }

    private fun finishGaitMeasurementAndAnalyze() {
        if (!isGaitSessionActive) return
        isGaitSessionActive = false
        isGaitCalibrationActive = false
        isGaitMeasuring = false
        val samples = gaitSamples.toList()
        val calibrationSamples = gaitCalibrationSamples.toList()
        notifyGaitEvent("analyzing", JSONObject().put("sampleCount", samples.size))

        thread {
            try {
                if (samples.size < 80) {
                    throw IllegalStateException("not enough gait samples")
                }
                if (gaitReadyMs > 0L && calibrationSamples.size < 20) {
                    throw IllegalStateException("not enough calibration samples")
                }
                val calibrationSource = calibrationSamples.ifEmpty {
                    samples.take(minOf(samples.size, 50))
                }
                val gravityMean = meanVector(calibrationSource, acceleration = true)
                val basis = buildGaitBasis(gravityMean)
                val input = samples.map { sample ->
                    val acceleration = doubleArrayOf(
                        sample.ax.toDouble(), sample.ay.toDouble(), sample.az.toDouble()
                    )
                    OfflineGaitAnalyzer.InputSample(
                        timestampNs = sample.timestampNs,
                        verticalG = vectorDot(acceleration, basis.vertical) / GRAVITY_M_S2,
                    )
                }
                val result = OfflineGaitAnalyzer.analyze(input)
                notifyGaitEvent("complete", result.toJson())
            } catch (e: Exception) {
                Log.e(TAG, "offline gait analysis failed", e)
                notifyGaitEvent("error", JSONObject().put("message", e.message ?: "온디바이스 분석에 실패했습니다."))
            } finally {
                if (!isCalibrating) {
                    sensorManager.unregisterListener(this)
                    isSensorRegistered = false
                }
            }
        }
    }

    private fun uploadGaitCsv(
        samples: List<GaitSample>,
        calibrationSamples: List<GaitSample>
    ): String {
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
            out.write(
                buildGaitCsv(samples, calibrationSamples)
                    .toByteArray(Charsets.UTF_8)
            )
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

    private fun buildGaitCsv(
        samples: List<GaitSample>,
        calibrationSamples: List<GaitSample>
    ): String {
        val calibrationSource = calibrationSamples.ifEmpty {
            samples.take(minOf(samples.size, 50))
        }
        val gravityMean = meanVector(calibrationSource, acceleration = true)
        val gyroBias = meanVector(calibrationSource, acceleration = false)
        val accStd = standardDeviationVector(
            calibrationSource, gravityMean, acceleration = true
        )
        val gyroStd = standardDeviationVector(
            calibrationSource, gyroBias, acceleration = false
        )
        val basis = buildGaitBasis(gravityMean)
        val gravityNorm = vectorNorm(gravityMean)

        val builder = StringBuilder()
        builder.append("# [FinalProjectApp] 14-Column Anatomical Gait Dataset\n")
        builder.append("# Source: FinalProjectApp APK WebView bridge\n")
        builder.append("# Accel_Calibration: Android factory-calibrated TYPE_ACCELEROMETER; no custom per-axis scale\n")
        builder.append("# Gyro_Calibration: Android factory calibration plus stationary zero-bias subtraction\n")
        builder.append("# Anatomical_Alignment: 3-second stationary gravity estimate and fixed waist-device orientation\n")
        builder.append("# Calibration_Samples: ${calibrationSamples.size}\n")
        builder.append("# Gyro_Zero_Bias_rad_s: Gx=${csvNumber(gyroBias[0])}, Gy=${csvNumber(gyroBias[1])}, Gz=${csvNumber(gyroBias[2])}\n")
        builder.append("# Gyro_Static_Std_rad_s: Gx=${csvNumber(gyroStd[0])}, Gy=${csvNumber(gyroStd[1])}, Gz=${csvNumber(gyroStd[2])}\n")
        builder.append("# Gravity_Mean_m_s2: Ax=${csvNumber(gravityMean[0])}, Ay=${csvNumber(gravityMean[1])}, Az=${csvNumber(gravityMean[2])}\n")
        builder.append("# Gravity_Norm_m_s2: ${csvNumber(gravityNorm)}\n")
        builder.append("# Accel_Static_Std_m_s2: Ax=${csvNumber(accStd[0])}, Ay=${csvNumber(accStd[1])}, Az=${csvNumber(accStd[2])}\n")
        builder.append("# Basis_Vertical_Unit: ${formatVector(basis.vertical)}\n")
        builder.append("# Basis_ML_Unit: ${formatVector(basis.ml)}\n")
        builder.append("# Basis_AP_Unit: ${formatVector(basis.ap)}\n")
        appendSensorMetadata(builder, "Accel", accelerometer)
        appendSensorMetadata(builder, "Gyro", gyroscope)
        builder.append(
            "Timestamp_ns,Acc_X,Acc_Y,Acc_Z," +
                "Gyro_Raw_X,Gyro_Raw_Y,Gyro_Raw_Z," +
                "Gyro_Clean_X,Gyro_Clean_Y,Gyro_Clean_Z," +
                "Acc_Vertical_g,Acc_ML_g,Acc_AP_g,Gyro_Roll_deg_s\n"
        )
        samples.forEach { s ->
            val acceleration = doubleArrayOf(
                s.ax.toDouble(), s.ay.toDouble(), s.az.toDouble()
            )
            val cleanGyro = doubleArrayOf(
                s.gx.toDouble() - gyroBias[0],
                s.gy.toDouble() - gyroBias[1],
                s.gz.toDouble() - gyroBias[2]
            )
            val verticalG = vectorDot(acceleration, basis.vertical) / GRAVITY_M_S2
            val mlG = vectorDot(acceleration, basis.ml) / GRAVITY_M_S2
            val apG = vectorDot(acceleration, basis.ap) / GRAVITY_M_S2
            val rollDegS = vectorDot(cleanGyro, basis.ap) * 180.0 / PI
            builder.append(s.timestampNs).append(',')
                .append(csvNumber(acceleration[0])).append(',')
                .append(csvNumber(acceleration[1])).append(',')
                .append(csvNumber(acceleration[2])).append(',')
                .append(csvNumber(s.gx.toDouble())).append(',')
                .append(csvNumber(s.gy.toDouble())).append(',')
                .append(csvNumber(s.gz.toDouble())).append(',')
                .append(csvNumber(cleanGyro[0])).append(',')
                .append(csvNumber(cleanGyro[1])).append(',')
                .append(csvNumber(cleanGyro[2])).append(',')
                .append(csvNumber(verticalG)).append(',')
                .append(csvNumber(mlG)).append(',')
                .append(csvNumber(apG)).append(',')
                .append(csvNumber(rollDegS)).append('\n')
        }
        return builder.toString()
    }

    private fun meanVector(
        samples: List<GaitSample>,
        acceleration: Boolean
    ): DoubleArray {
        if (samples.isEmpty()) {
            return if (acceleration) {
                doubleArrayOf(0.0, GRAVITY_M_S2, 0.0)
            } else {
                doubleArrayOf(0.0, 0.0, 0.0)
            }
        }
        val sum = DoubleArray(3)
        samples.forEach { sample ->
            val values = if (acceleration) {
                doubleArrayOf(
                    sample.ax.toDouble(),
                    sample.ay.toDouble(),
                    sample.az.toDouble()
                )
            } else {
                doubleArrayOf(
                    sample.gx.toDouble(),
                    sample.gy.toDouble(),
                    sample.gz.toDouble()
                )
            }
            for (index in 0..2) sum[index] += values[index]
        }
        return DoubleArray(3) { index -> sum[index] / samples.size }
    }

    private fun standardDeviationVector(
        samples: List<GaitSample>,
        mean: DoubleArray,
        acceleration: Boolean
    ): DoubleArray {
        if (samples.isEmpty()) return DoubleArray(3)
        val squared = DoubleArray(3)
        samples.forEach { sample ->
            val values = if (acceleration) {
                doubleArrayOf(
                    sample.ax.toDouble(),
                    sample.ay.toDouble(),
                    sample.az.toDouble()
                )
            } else {
                doubleArrayOf(
                    sample.gx.toDouble(),
                    sample.gy.toDouble(),
                    sample.gz.toDouble()
                )
            }
            for (index in 0..2) {
                val delta = values[index] - mean[index]
                squared[index] += delta * delta
            }
        }
        return DoubleArray(3) { index ->
            sqrt(squared[index] / samples.size)
        }
    }

    private fun buildGaitBasis(gravityMean: DoubleArray): GaitBasis {
        val vertical = normalizeVector(
            gravityMean, doubleArrayOf(0.0, 1.0, 0.0)
        )
        val references = listOf(
            doubleArrayOf(0.0, 0.0, 1.0),
            doubleArrayOf(0.0, 1.0, 0.0),
            doubleArrayOf(1.0, 0.0, 0.0)
        )
        var ap = doubleArrayOf(0.0, 0.0, 1.0)
        for (reference in references) {
            val projection = DoubleArray(3) { index ->
                reference[index] -
                    vertical[index] * vectorDot(reference, vertical)
            }
            if (vectorNorm(projection) > 0.05) {
                ap = normalizeVector(
                    projection, doubleArrayOf(0.0, 0.0, 1.0)
                )
                break
            }
        }
        val ml = normalizeVector(
            vectorCross(vertical, ap), doubleArrayOf(1.0, 0.0, 0.0)
        )
        return GaitBasis(vertical = vertical, ml = ml, ap = ap)
    }

    private fun vectorDot(left: DoubleArray, right: DoubleArray): Double {
        return left[0] * right[0] +
            left[1] * right[1] +
            left[2] * right[2]
    }

    private fun vectorNorm(vector: DoubleArray): Double {
        return sqrt(vectorDot(vector, vector))
    }

    private fun normalizeVector(
        vector: DoubleArray,
        fallback: DoubleArray
    ): DoubleArray {
        val norm = vectorNorm(vector)
        if (!norm.isFinite() || norm < 1e-9) return fallback.copyOf()
        return DoubleArray(3) { index -> vector[index] / norm }
    }

    private fun vectorCross(
        left: DoubleArray,
        right: DoubleArray
    ): DoubleArray {
        return doubleArrayOf(
            left[1] * right[2] - left[2] * right[1],
            left[2] * right[0] - left[0] * right[2],
            left[0] * right[1] - left[1] * right[0]
        )
    }

    private fun csvNumber(value: Double): String {
        if (!value.isFinite()) return ""
        return String.format(Locale.US, "%.9g", value)
    }

    private fun formatVector(vector: DoubleArray): String {
        return "(${csvNumber(vector[0])}, ${csvNumber(vector[1])}, " +
            "${csvNumber(vector[2])})"
    }

    private fun appendSensorMetadata(
        builder: StringBuilder,
        prefix: String,
        sensor: Sensor?
    ) {
        if (sensor == null) {
            builder.append("# ${prefix}_Available: false\n")
            return
        }
        val name = sensor.name.replace('\n', ' ').replace('\r', ' ')
        val vendor = sensor.vendor.replace('\n', ' ').replace('\r', ' ')
        val rangeKey = if (prefix == "Accel") {
            "Accel_Maximum_Range_m_s2"
        } else {
            "Gyro_Maximum_Range_rad_s"
        }
        val resolutionKey = if (prefix == "Accel") {
            "Accel_Resolution_m_s2"
        } else {
            "Gyro_Resolution_rad_s"
        }
        builder.append("# ${prefix}_Available: true\n")
        builder.append("# ${prefix}_Name: $name\n")
        builder.append("# ${prefix}_Vendor: $vendor\n")
        builder.append(
            "# $rangeKey: " +
                "${csvNumber(sensor.maximumRange.toDouble())}\n"
        )
        builder.append(
            "# $resolutionKey: " +
                "${csvNumber(sensor.resolution.toDouble())}\n"
        )
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

    private fun speakExercise(text: String, rate: Float, volume: Float) {
        if (!gaitTtsReady || text.isBlank()) {
            notifyExerciseTtsEnd()
            return
        }
        exerciseTtsText = text
        exerciseTtsResumeOffset = 0
        exerciseTtsRate = rate.coerceIn(0.65f, 1.15f)
        exerciseTtsVolume = volume.coerceIn(0f, 1f)
        exerciseTtsPaused = false
        speakExerciseFromOffset(0)
    }

    private fun speakExerciseFromOffset(offset: Int) {
        val safeOffset = offset.coerceIn(0, exerciseTtsText.length)
        val remainingText = exerciseTtsText.substring(safeOffset)
        if (remainingText.isBlank()) {
            clearExerciseTtsState()
            notifyExerciseTtsEnd()
            return
        }
        exerciseTtsChunkBaseOffset = safeOffset
        val params = Bundle().apply {
            putFloat(TextToSpeech.Engine.KEY_PARAM_VOLUME, exerciseTtsVolume)
        }
        gaitTts?.stop()
        gaitTts?.setSpeechRate(exerciseTtsRate)
        gaitTts?.speak(
            remainingText,
            TextToSpeech.QUEUE_FLUSH,
            params,
            "exercise-${System.currentTimeMillis()}"
        )
    }

    private fun pauseExerciseTts() {
        if (exerciseTtsText.isBlank()) return
        exerciseTtsPaused = true
        gaitTts?.stop()
    }

    private fun resumeExerciseTts() {
        if (!exerciseTtsPaused || exerciseTtsText.isBlank()) return
        exerciseTtsPaused = false
        speakExerciseFromOffset(exerciseTtsResumeOffset)
    }

    private fun clearExerciseTtsState() {
        exerciseTtsText = ""
        exerciseTtsResumeOffset = 0
        exerciseTtsChunkBaseOffset = 0
        exerciseTtsPaused = false
    }

    private fun notifyExerciseTtsEnd() {
        val script = "window.ExerciseTtsNative && window.ExerciseTtsNative.onTtsEnd()"
        webView.post { webView.evaluateJavascript(script, null) }
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

    private fun notifyWakeStatus(status: String, message: String? = null) {
        val script =
            "window.PengteuAssistantNative && window.PengteuAssistantNative.onWakeStatus && " +
                "window.PengteuAssistantNative.onWakeStatus(" +
                "${JSONObject.quote(status)}, ${JSONObject.quote(message.orEmpty())})"
        webView.post { webView.evaluateJavascript(script, null) }
    }

    private fun startPengteuStt(isRetry: Boolean = false) {
        if (!isRetry) pengteuSttRetryCount = 0
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(arrayOf(Manifest.permission.RECORD_AUDIO), AUDIO_PERMISSION_REQUEST)
            return
        }
        if (!SpeechRecognizer.isRecognitionAvailable(this)) {
            notifyPengteuNative("onSttError", "이 기기에서는 음성 인식을 사용할 수 없어요. 글자로 입력해 주세요.")
            return
        }
        if (isPengteuSttActive) return

        // TTS 종료 콜백이 resumeWake()를 호출해도 명령 인식 직전에
        // Vosk가 마이크를 다시 가져가지 못하도록 상태를 먼저 잠근다.
        isPengteuSttActive = true
        pauseWakeForOther()
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
                    isPengteuSttActive = false
                    Log.w("PengteuStt", "Recognition error code=$error")
                    if (
                        (error == SpeechRecognizer.ERROR_RECOGNIZER_BUSY ||
                            error == SpeechRecognizer.ERROR_AUDIO) &&
                        pengteuSttRetryCount < 1
                    ) {
                        pengteuSttRetryCount += 1
                        runCatching { pengteuRecognizer?.cancel() }
                        runCatching { pengteuRecognizer?.destroy() }
                        pengteuRecognizer = null
                        mainHandler.postDelayed(
                            { startPengteuStt(isRetry = true) },
                            STT_RETRY_DELAY_MS
                        )
                        return
                    }
                    val message = when (error) {
                        SpeechRecognizer.ERROR_INSUFFICIENT_PERMISSIONS ->
                            "마이크 권한이 필요해요. 앱 설정에서 권한을 허용해 주세요."
                        SpeechRecognizer.ERROR_NETWORK,
                        SpeechRecognizer.ERROR_NETWORK_TIMEOUT ->
                            "음성 인식 서버 연결이 불안정해요. 인터넷 연결을 확인해 주세요."
                        SpeechRecognizer.ERROR_SPEECH_TIMEOUT ->
                            "말소리가 들리지 않았어요. 마이크를 누른 뒤 바로 말씀해 주세요."
                        else ->
                            "음성을 잘 듣지 못했어요. 마이크를 다시 눌러 말해 주세요."
                    }
                    notifyPengteuNative("onSttError", message)
                    mainHandler.postDelayed({ resumeWake() }, AUDIO_HANDOFF_DELAY_MS)
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
                    pengteuSttRetryCount = 0
                    isPengteuSttActive = false
                    mainHandler.postDelayed({ resumeWake() }, AUDIO_HANDOFF_DELAY_MS)
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
        // Vosk의 AudioRecord 스레드가 실제 장치를 반납할 시간을 준다.
        // 즉시 시작하면 일부 기기에서 RECOGNIZER_BUSY/AUDIO 오류가 난다.
        mainHandler.postDelayed({
            if (!isPengteuSttActive || isFinishing || isDestroyed) return@postDelayed
            try {
                recognizer.startListening(intent)
            } catch (error: RuntimeException) {
                Log.e("PengteuStt", "Unable to start command recognition", error)
                runCatching { recognizer.cancel() }
                runCatching { recognizer.destroy() }
                if (pengteuRecognizer === recognizer) pengteuRecognizer = null
                isPengteuSttActive = false
                notifyPengteuNative(
                    "onSttError",
                    "마이크를 시작하지 못했어요. 잠시 후 다시 눌러 주세요."
                )
                mainHandler.postDelayed({ resumeWake() }, AUDIO_HANDOFF_DELAY_MS)
            }
        }, AUDIO_HANDOFF_DELAY_MS)
    }

    // ── "펭트야" always-on 웨이크워드 (무료 로컬 Vosk 엔진) ──────
    // Vosk가 마이크를 계속 읽다가 호출어를 감지하면, 웹의 기존 펭트 대화
    // 흐름으로 넘긴다. 질문 STT/TTS 중에는 반드시 Vosk 마이크를 놓는다.

    private fun onWakeDetected(transcript: String) {
        if (isPengteuSttActive) return
        // 전체 발화를 웹에 전달한다. "펭트야 + 명령"이면 뒤 문장을 바로
        // 처리하고, 호출어만 있으면 기존처럼 추가 명령 STT를 시작한다.
        notifyPengteuNative("onWakeWord", transcript)
    }

    private fun startWakeWord() {
        wakeWantsRun = true
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(arrayOf(Manifest.permission.RECORD_AUDIO), WAKE_AUDIO_PERMISSION_REQUEST)
            return
        }
        resumeWake()
    }

    private fun stopWakeWord() {
        wakeWantsRun = false
        wakeWordEngine.stop()
    }

    // 다른 마이크 사용자(명령 STT·TTS)를 위해 일시정지하되, 재개 의사(wakeWantsRun)는 유지.
    private fun pauseWakeForOther() {
        wakeWordEngine.stop()
    }

    private fun resumeWake() {
        if (!wakeWantsRun || isPengteuSttActive || isTtsSpeaking) return
        wakeWordEngine.start()
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
                val responseText = runCatching {
                    val stream = if (ok) connection.inputStream else connection.errorStream
                    stream?.bufferedReader(Charsets.UTF_8)?.use { it.readText() }.orEmpty()
                }.getOrDefault("")
                val responseJson = runCatching { JSONObject(responseText) }.getOrNull()
                val resolvedSigungu = responseJson?.optString("sigungu").orEmpty()
                    .ifBlank { payload.optString("sigungu", "") }
                val resolvedLocation = responseJson?.optString("location").orEmpty()
                    .ifBlank { payload.optString("location", "") }
                connection.disconnect()
                notifyOrientationLocation(
                    ok,
                    if (ok) "위치를 확인했어요." else "위치를 저장하지 못했어요.",
                    resolvedSigungu,
                    resolvedLocation
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
        isGaitCalibrationActive = false
        isGaitMeasuring = false
        mainHandler.removeCallbacksAndMessages(GAIT_STOP_TOKEN)
        wakeWantsRun = false
        wakeWordEngine.shutdown()
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
        if (requestCode == WAKE_AUDIO_PERMISSION_REQUEST) {
            if (grantResults.firstOrNull() == PackageManager.PERMISSION_GRANTED) {
                resumeWake()
            } else {
                notifyWakeStatus("error", "마이크 권한이 필요합니다")
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
        fun startCalibration() {
            registerSensors()
            axisCalibrator.start()
            isCalibrating = true
        }

        @JavascriptInterface
        fun startMeasurement(stage: String) {
            registerSensors()
        }

        @JavascriptInterface
        fun stopMeasurement() {
            isGaitSessionActive = false
            isGaitCalibrationActive = false
            isGaitMeasuring = false
            isCalibrating = false
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
        fun speakExercise(text: String, rate: Double, volume: Double) {
            this@MainActivity.speakExercise(text, rate.toFloat(), volume.toFloat())
        }

        @JavascriptInterface
        fun pauseExerciseTts() {
            this@MainActivity.pauseExerciseTts()
        }

        @JavascriptInterface
        fun resumeExerciseTts() {
            this@MainActivity.resumeExerciseTts()
        }

        @JavascriptInterface
        fun speak(text: String) {
            this@MainActivity.speakPengteu(text, 0.9f, 0.8f)
        }

        @JavascriptInterface
        fun isTtsReady(): Boolean = gaitTtsReady

        @JavascriptInterface
        fun stopPengteuTts() {
            this@MainActivity.stopPengteuTts()
        }

        @JavascriptInterface
        fun startPengteuStt() {
            // SpeechRecognizer 생성과 호출은 반드시 앱 메인 스레드에서 한다.
            mainHandler.post { this@MainActivity.startPengteuStt() }
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
        private const val OFFLINE_HOME_URL = "file:///android_asset/offline/index.html"
        private const val OFFLINE_ERROR_URL = "file:///android_asset/offline/error.html"
        private const val TAG = "GiukhajiOffline"
        // 실제 서버 주소는 local.properties(gitignore)의 serverUrl로 빌드 시 주입된다.
        private val DEFAULT_SERVER_URL = BuildConfig.SERVER_URL
        private const val PREFS_NAME = "finalinzi_member"
        private const val PREF_SERVER_URL = "server_url"
        private const val PREF_MEMBER_PHONE = "member_phone"
        private const val PREF_EDUCATION_LEVEL = "education_level"
        private const val AUDIO_PERMISSION_REQUEST = 1101
        private const val WAKE_AUDIO_PERMISSION_REQUEST = 1102
        private const val LOCATION_PERMISSION_REQUEST = 1201
        private const val AUDIO_HANDOFF_DELAY_MS = 400L
        private const val STT_RETRY_DELAY_MS = 800L
        private const val GRAVITY_M_S2 = 9.80665
        private const val SENSOR_PERIOD_US = 10_000
        private val GAIT_STOP_TOKEN = Any()
    }
}
