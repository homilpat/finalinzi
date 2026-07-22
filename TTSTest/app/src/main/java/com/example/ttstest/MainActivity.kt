package com.example.ttstest

import android.annotation.SuppressLint
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.speech.tts.TextToSpeech
import android.util.Log
import android.view.View
import android.webkit.*
import android.widget.ProgressBar
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import android.media.MediaPlayer
import android.media.AudioAttributes
import android.media.AudioManager
import android.net.Uri
import java.util.Locale

class MainActivity : AppCompatActivity(), SensorEventListener, TextToSpeech.OnInitListener {

    private lateinit var webView: WebView
    private lateinit var progressBar: ProgressBar

    private lateinit var sensorManager: SensorManager
    private var accelerometer: Sensor? = null
    private var gyroscope: Sensor? = null

    private val classifier = MotionClassifier(CalibrationProfile())
    private var isSensorRegistered = false
    private var sensorThread: HandlerThread? = null
    private var sensorHandler: Handler? = null

    private var ax = 0f
    private var ay = 0f
    private var az = 0f
    private var gx = 0f
    private var gy = 0f
    private var gz = 0f

    private var tts: TextToSpeech? = null
    private var isTtsReady = false
    private var activeMediaPlayer: MediaPlayer? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_main)

        // 볼륨 조절 버튼이 미디어 볼륨(STREAM_MUSIC)을 조절하도록 설정
        volumeControlStream = AudioManager.STREAM_MUSIC

        webView = findViewById(R.id.webView)
        progressBar = findViewById(R.id.progressBar)

        setupWebView()
        setupSensors()

        // TTS 초기화
        tts = TextToSpeech(this, this)

        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main)) { v, insets ->
            val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom)
            insets
        }
    }

    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            // TTS 출력을 미디어(STREAM_MUSIC) 스트림으로 설정하여 무음 모드 등에서도 볼륨 연동되도록 함
            val audioAttributes = AudioAttributes.Builder()
                .setUsage(AudioAttributes.USAGE_MEDIA)
                .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                .build()
            tts?.setAudioAttributes(audioAttributes)

            var result = tts?.setLanguage(Locale.KOREA) // ko_KR 시도
            if (result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED) {
                result = tts?.setLanguage(Locale.KOREAN) // ko 시도
            }

            if (result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED) {
                Log.e("TTS", "Korean Language not supported on this TTS engine")
                isTtsReady = false
            } else {
                Log.d("TTS", "TTS Initialization Successful with Korean")
                isTtsReady = true
            }
        } else {
            Log.e("TTS", "Initialization failed")
            isTtsReady = false
        }
    }

    @SuppressLint("SetJavaScriptEnabled")
    private fun setupWebView() {
        // 캐시를 완전히 삭제하여 이전 빌드의 웹뷰 캐싱 문제 차단
        webView.clearCache(true)

        webView.settings.apply {
            javaScriptEnabled = true
            domStorageEnabled = true
            databaseEnabled = true
            allowFileAccess = true
            allowContentAccess = true
            mediaPlaybackRequiresUserGesture = false
            mixedContentMode = WebSettings.MIXED_CONTENT_ALWAYS_ALLOW

            // 캐시를 타지 않고 항상 최신 서버 데이터를 가져오도록 설정
            cacheMode = WebSettings.LOAD_NO_CACHE

            // Google TTS 차단 방지를 위한 User Agent 설정
            userAgentString =
                "Mozilla/5.0 (Linux; Android 13; Pixel 7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Mobile Safari/537.36"
        }

        webView.webViewClient = object : WebViewClient() {
            override fun onPageFinished(view: WebView?, url: String?) {
                progressBar.visibility = View.GONE
            }

            // Google TTS URL이 호출될 때 Native TTS가 준비된 경우에만 가로채서 재생합니다.
            override fun shouldInterceptRequest(
                view: WebView?,
                request: WebResourceRequest?
            ): WebResourceResponse? {
                val url = request?.url.toString()
                if (url.contains("translate.google.com/translate_tts")) {
                    val text = request?.url?.getQueryParameter("q")
                    if (text != null && isTtsReady) {
                        speakText(text)
                        // 빈 응답을 반환하여 WebView가 직접 재생하지 않도록 함
                        return WebResourceResponse("audio/mpeg", "UTF-8", null)
                    }
                }
                return super.shouldInterceptRequest(view, request)
            }
        }

        webView.webChromeClient = object : WebChromeClient() {
            override fun onConsoleMessage(consoleMessage: ConsoleMessage?): Boolean {
                consoleMessage?.let {
                    Log.d(
                        "WebViewConsole",
                        "${it.message()} -- From line ${it.lineNumber()} of ${it.sourceId()}"
                    )
                }
                return true
            }
        }

        webView.addJavascriptInterface(WebAppInterface(), "AndroidBridge")
        webView.loadUrl("https://apptest-tvig.onrender.com")
    }

    private fun playWithMediaPlayer(text: String) {
        try {
            // 이전 재생 중이던 MediaPlayer가 있으면 해제
            activeMediaPlayer?.let {
                if (it.isPlaying) {
                    it.stop()
                }
                it.release()
            }
            activeMediaPlayer = null

            val url = "https://translate.google.com/translate_tts?ie=UTF-8&tl=ko&client=tw-ob&q=${
                Uri.encode(text)
            }"
            val player = MediaPlayer().apply {
                // 오디오 스트림 속성을 미디어로 설정
                val audioAttributes = AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_MEDIA)
                    .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                    .build()
                setAudioAttributes(audioAttributes)

                // Google TTS 차단 회피용 User Agent 헤더 설정
                val headers = HashMap<String, String>()
                headers["User-Agent"] =
                    "Mozilla/5.0 (Linux; Android 13; Pixel 7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Mobile Safari/537.36"

                setDataSource(applicationContext, Uri.parse(url), headers)
                prepareAsync()
                setOnPreparedListener { start() }
                setOnCompletionListener {
                    release()
                    if (activeMediaPlayer == this) {
                        activeMediaPlayer = null
                    }
                }
                setOnErrorListener { mp, what, extra ->
                    Log.e("TTS", "MediaPlayer error: $what, $extra")
                    mp.release()
                    if (activeMediaPlayer == this) {
                        activeMediaPlayer = null
                    }
                    true
                }
            }
            // GC(가비지 컬렉션)에 의해 릴리즈되지 않도록 클래스 멤버 변수에 대입하여 참조 유지
            activeMediaPlayer = player
        } catch (e: Exception) {
            Log.e("TTS", "MediaPlayer fallback failed", e)
        }
    }

    private fun speakText(text: String) {
        Log.d("TTS", "speakText called: $text (Ready: $isTtsReady)")
        runOnUiThread {
            var spokenSuccessfully = false

            if (isTtsReady) {
                val params = Bundle()
                val result = tts?.speak(text, TextToSpeech.QUEUE_FLUSH, params, "TTS_ID")
                if (result == TextToSpeech.SUCCESS) {
                    spokenSuccessfully = true
                } else {
                    Log.e("TTS", "tts.speak returned ERROR")
                }
            }

            if (!spokenSuccessfully) {
                Log.w(
                    "TTS",
                    "Native TTS failed or not ready. Using MediaPlayer fallback with custom headers."
                )
                playWithMediaPlayer(text)
            }
        }
    }

    private fun setupSensors() {
        sensorManager = getSystemService(SENSOR_SERVICE) as SensorManager
        accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        gyroscope = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
    }

    private fun startSensorThread() {
        if (sensorThread == null) {
            sensorThread = HandlerThread("SensorThread").apply {
                start()
                sensorHandler = Handler(looper)
            }
        }
    }

    private fun stopSensorThread() {
        sensorThread?.quitSafely()
        sensorThread = null
        sensorHandler = null
    }

    private fun registerSensors() {
        if (!isSensorRegistered) {
            startSensorThread()
            val handler = sensorHandler
            accelerometer?.let {
                sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_UI, handler)
            }
            gyroscope?.let {
                sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_UI, handler)
            }
            isSensorRegistered = true
        }
    }

    private fun unregisterSensors() {
        if (isSensorRegistered) {
            sensorManager.unregisterListener(this)
            stopSensorThread()
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

        val detectedAction = classifier.onSensorData(
            System.currentTimeMillis(),
            ax, ay, az, gx, gy, gz
        )

        detectedAction?.let { action ->
            sendActionToWebView(action)
        }
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
        tts?.stop()
        tts?.shutdown()
        super.onDestroy()
    }

    inner class WebAppInterface {
        @JavascriptInterface
        fun startCalibration() {
            Log.d("Sensor", "startCalibration called")
            registerSensors()
        }

        @JavascriptInterface
        fun startMeasurement(stage: String) {
            registerSensors()
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

        @JavascriptInterface
        fun speak(text: String) {
            speakText(text)
        }

        @JavascriptInterface
        fun isTtsReady(): Boolean {
            return isTtsReady
        }
    }
}