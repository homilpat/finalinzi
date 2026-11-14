package com.example.finalprojectapp

import android.content.Context
import android.os.Handler
import android.os.Looper
import android.util.Log
import org.json.JSONObject
import org.vosk.Model
import org.vosk.Recognizer
import org.vosk.android.RecognitionListener
import org.vosk.android.SpeechService
import java.io.BufferedInputStream
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.util.concurrent.Executors
import java.util.zip.ZipInputStream

/**
 * Offline, continuous Korean speech recognition used only to detect "펭트야".
 *
 * Android's SpeechRecognizer is intentionally not used here: it is a one-shot
 * API and some devices play a system sound every time it starts or stops.
 */
class VoskWakeWordEngine(
    context: Context,
    private val onWakeWord: (String) -> Unit,
    private val onStatus: (String, String?) -> Unit = { _, _ -> },
) : RecognitionListener {

    companion object {
        private const val TAG = "PengteuWake"
        private const val MODEL_ZIP_ASSET = "vosk-model-small-ko-0.22.zip"
        private const val MODEL_DIR = "vosk-model-small-ko-0.22"
        private const val MODEL_VERSION = "ko-0.22-pengteu-1"
        private const val SAMPLE_RATE = 16_000.0f
        private const val DETECTION_COOLDOWN_MS = 1_500L
        private const val CAPTURE_RETRY_DELAY_MS = 1_500L
        private const val PARTIAL_WAKE_FALLBACK_MS = 1_600L

        // "펭트" is a coined name, so a general Korean STT can return a
        // phonetically close common word. Keep this list deliberately small.
        private val WAKE_FORMS = listOf(
            "펭트야", "펭트",
            "팽트야", "팽트",
            "펜트야", "펜트",
            "팬트야", "팬트",
            "펭뜨야", "팽뜨야", "펜뜨야",
            "벤트야", "밴트야", "뱅트야",
            "텐트야",
            "팩트야",
        )
    }

    private val appContext = context.applicationContext
    private val mainHandler = Handler(Looper.getMainLooper())
    private val modelExecutor = Executors.newSingleThreadExecutor()

    private var model: Model? = null
    private var recognizer: Recognizer? = null
    private var speechService: SpeechService? = null
    private var wantsListening = false
    private var loading = false
    private var destroyed = false
    private var detectionLatched = false
    @Volatile private var pendingWakeText = ""
    private val partialWakeFallback = Runnable {
        val text = pendingWakeText
        if (text.isNotBlank() && wantsListening && !destroyed && !detectionLatched) {
            triggerWake(text)
        }
    }
    private val hearingStatusReset = Runnable {
        if (wantsListening && !destroyed && !detectionLatched) {
            onStatus("listening", null)
        }
    }

    fun start() {
        if (destroyed) return
        wantsListening = true
        if (model == null) {
            loadModel()
        } else {
            startCapture()
        }
    }

    fun stop() {
        wantsListening = false
        clearPendingWake()
        mainHandler.removeCallbacks(hearingStatusReset)
        stopCapture()
    }

    fun shutdown() {
        destroyed = true
        wantsListening = false
        clearPendingWake()
        stopCapture()
        model?.close()
        model = null
        mainHandler.removeCallbacksAndMessages(null)
        modelExecutor.shutdownNow()
    }

    private fun loadModel() {
        if (loading || destroyed) return
        loading = true
        Log.i(TAG, "Preparing Korean Vosk model")
        onStatus("loading", null)
        modelExecutor.execute {
            try {
                val modelDirectory = unpackModelIfNeeded()
                val loadedModel = Model(modelDirectory.absolutePath)
                mainHandler.post {
                    loading = false
                    if (destroyed) {
                        loadedModel.close()
                        return@post
                    }
                    model = loadedModel
                    Log.i(TAG, "Korean Vosk model ready")
                    onStatus("ready", null)
                    if (wantsListening) startCapture()
                }
            } catch (error: Exception) {
                mainHandler.post {
                    loading = false
                    if (!destroyed) {
                        Log.e(TAG, "Failed to prepare Korean Vosk model", error)
                        onStatus("error", error.message)
                    }
                }
            }
        }
    }

    /**
     * The official Korean model contains a file larger than GitHub's 100 MB
     * limit, so the app ships the official ZIP and expands it once on device.
     */
    private fun unpackModelIfNeeded(): File {
        val externalRoot = appContext.getExternalFilesDir(null)
            ?: throw IOException("App storage is unavailable")
        val modelRoot = File(externalRoot, "vosk")
        val modelDirectory = File(modelRoot, MODEL_DIR)
        val readyMarker = File(modelDirectory, ".pengteu-model-version")
        if (readyMarker.isFile && readyMarker.readText().trim() == MODEL_VERSION) {
            return modelDirectory
        }

        val stagingRoot = File(modelRoot, ".staging-$MODEL_VERSION")
        if (stagingRoot.exists() && !stagingRoot.deleteRecursively()) {
            throw IOException("Could not clear an incomplete Vosk model")
        }
        if (!stagingRoot.mkdirs()) {
            throw IOException("Could not create Vosk model directory")
        }

        val canonicalStaging = stagingRoot.canonicalPath + File.separator
        try {
            ZipInputStream(BufferedInputStream(appContext.assets.open(MODEL_ZIP_ASSET))).use { zip ->
                val buffer = ByteArray(64 * 1024)
                while (true) {
                    if (Thread.currentThread().isInterrupted) throw IOException("Model setup interrupted")
                    val entry = zip.nextEntry ?: break
                    val output = File(stagingRoot, entry.name)
                    val canonicalOutput = output.canonicalPath
                    if (!canonicalOutput.startsWith(canonicalStaging)) {
                        throw IOException("Unsafe model ZIP entry: ${entry.name}")
                    }
                    if (entry.isDirectory) {
                        if (!output.exists() && !output.mkdirs()) {
                            throw IOException("Could not create ${entry.name}")
                        }
                    } else {
                        output.parentFile?.let {
                            if (!it.exists() && !it.mkdirs()) {
                                throw IOException("Could not create ${it.name}")
                            }
                        }
                        FileOutputStream(output).use { file ->
                            while (true) {
                                val count = zip.read(buffer)
                                if (count < 0) break
                                file.write(buffer, 0, count)
                            }
                        }
                    }
                    zip.closeEntry()
                }
            }

            val stagedModel = File(stagingRoot, MODEL_DIR)
            if (!File(stagedModel, "am/final.mdl").isFile) {
                throw IOException("Korean Vosk model is incomplete")
            }
            File(stagedModel, ".pengteu-model-version").writeText(MODEL_VERSION)
            if (modelDirectory.exists() && !modelDirectory.deleteRecursively()) {
                throw IOException("Could not replace the previous Vosk model")
            }
            if (!modelRoot.exists() && !modelRoot.mkdirs()) {
                throw IOException("Could not create Vosk storage")
            }
            if (!stagedModel.renameTo(modelDirectory)) {
                throw IOException("Could not activate the Korean Vosk model")
            }
            return modelDirectory
        } finally {
            if (stagingRoot.exists()) stagingRoot.deleteRecursively()
        }
    }

    private fun startCapture() {
        if (destroyed || !wantsListening || speechService != null) return
        val loadedModel = model ?: return
        try {
            val nextRecognizer = Recognizer(loadedModel, SAMPLE_RATE)
            val nextService = SpeechService(nextRecognizer, SAMPLE_RATE)
            recognizer = nextRecognizer
            speechService = nextService
            nextService.startListening(this)
            Log.i(TAG, "Wake-word listening started")
            onStatus("listening", null)
        } catch (error: Exception) {
            Log.e(TAG, "Unable to start wake-word listening", error)
            onStatus("error", error.message)
            stopCapture()
            // 웹은 네이티브 웨이크가 이미 켜졌다고 기억한다. 여기서 자체
            // 재시도하지 않으면 일시적인 마이크 busy 뒤 영구 정지한다.
            if (wantsListening && !destroyed) {
                mainHandler.postDelayed(
                    { if (wantsListening && !destroyed) startCapture() },
                    CAPTURE_RETRY_DELAY_MS,
                )
            }
        }
    }

    private fun stopCapture() {
        val service = speechService
        val activeRecognizer = recognizer
        speechService = null
        recognizer = null
        runCatching { service?.cancel() }
        runCatching { service?.shutdown() }
        runCatching { activeRecognizer?.close() }
        if (service != null) Log.i(TAG, "Wake-word listening stopped")
    }

    override fun onPartialResult(hypothesis: String?) {
        val text = hypothesisText(hypothesis, "partial")
        if (text.isBlank()) return
        Log.d(TAG, "Vosk partial: $text")
        onStatus("hearing", text)
        mainHandler.removeCallbacks(hearingStatusReset)
        mainHandler.postDelayed(hearingStatusReset, 2_000L)
        if (!isWakeWord(text) || detectionLatched) return

        // 호출어는 중간 결과에서 기억하되 즉시 마이크를 끊지 않는다.
        // 이후 partial이 더 길어지면 "펭트야 + 전체 명령"으로 계속 갱신한다.
        pendingWakeText = text
        mainHandler.removeCallbacks(partialWakeFallback)
        mainHandler.postDelayed(partialWakeFallback, PARTIAL_WAKE_FALLBACK_MS)
    }

    override fun onResult(hypothesis: String?) {
        inspectHypothesis(hypothesis, "text")
    }

    override fun onFinalResult(hypothesis: String?) {
        inspectHypothesis(hypothesis, "text")
    }

    override fun onError(exception: Exception?) {
        Log.e(TAG, "Wake-word recognition error", exception)
        onStatus("error", exception?.message)
        stopCapture()
        if (wantsListening && !destroyed) {
            mainHandler.postDelayed({ startCapture() }, 1_500L)
        }
    }

    override fun onTimeout() {
        stopCapture()
        if (wantsListening && !destroyed) startCapture()
    }

    private fun inspectHypothesis(hypothesis: String?, field: String) {
        if (detectionLatched) return
        val text = hypothesisText(hypothesis, field)
        Log.d(TAG, "Vosk $field: $text")
        val remembered = pendingWakeText
        val detectedText = when {
            text.isNotBlank() && isWakeWord(text) -> text
            remembered.isNotBlank() && text.isBlank() -> remembered
            remembered.isNotBlank() && text.isNotBlank() ->
                if (text.startsWith(remembered)) text else "$remembered $text"
            else -> return
        }
        triggerWake(detectedText)
    }

    private fun triggerWake(text: String) {
        if (detectionLatched) return
        detectionLatched = true
        clearPendingWake()
        wantsListening = false
        stopCapture()
        Log.i(TAG, "Wake word detected: $text")
        onStatus("detected", text)
        onWakeWord(text)
        mainHandler.postDelayed(
            { detectionLatched = false },
            DETECTION_COOLDOWN_MS,
        )
    }

    private fun isWakeWord(raw: String): Boolean {
        val normalized = raw
            .replace(Regex("[\\s\\p{Punct}]+"), "")
            .lowercase()
        if (WAKE_FORMS.any { normalized.contains(it) }) return true

        // 조어인 "펭트"는 일반 한국어 모델에서 한 음절 정도 다르게
        // 전사되기 쉽다. 문장 시작부의 3~4음절만 한 글자 오차까지 허용해
        // 일반 대화에서의 과도한 오작동은 막으면서 호출 누락을 줄인다.
        val prefix = normalized.take(7)
        val fuzzyTargets = WAKE_FORMS.filter { it.endsWith("야") && it.length >= 3 }
        return fuzzyTargets.any { target ->
            val minLength = (target.length - 1).coerceAtLeast(2)
            val maxLength = target.length + 1
            (0..minOf(2, prefix.lastIndex)).any { start ->
                (minLength..maxLength).any { length ->
                    val end = start + length
                    end <= prefix.length &&
                        levenshtein(prefix.substring(start, end), target) <= 1
                }
            }
        }
    }

    private fun levenshtein(left: String, right: String): Int {
        var previous = IntArray(right.length + 1) { it }
        for (i in left.indices) {
            val current = IntArray(right.length + 1)
            current[0] = i + 1
            for (j in right.indices) {
                val substitution = previous[j] + if (left[i] == right[j]) 0 else 1
                current[j + 1] = minOf(
                    current[j] + 1,
                    previous[j + 1] + 1,
                    substitution,
                )
            }
            previous = current
        }
        return previous[right.length]
    }

    private fun hypothesisText(hypothesis: String?, field: String): String {
        if (hypothesis.isNullOrBlank()) return ""
        return runCatching { JSONObject(hypothesis).optString(field).trim() }
            .getOrDefault("")
    }

    private fun clearPendingWake() {
        pendingWakeText = ""
        mainHandler.removeCallbacks(partialWakeFallback)
        mainHandler.removeCallbacks(hearingStatusReset)
    }
}
