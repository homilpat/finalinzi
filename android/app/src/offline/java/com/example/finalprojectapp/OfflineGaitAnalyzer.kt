package com.example.finalprojectapp

import org.json.JSONObject
import kotlin.math.exp
import kotlin.math.roundToInt
import kotlin.math.sqrt

/**
 * 서버의 gait_axis_aligned_core.py + scikit-learn Pipeline을 Android/Kotlin으로
 * 옮긴 오프라인 보행 추론기. 네트워크나 Python 런타임을 사용하지 않는다.
 */
object OfflineGaitAnalyzer {
    private const val TARGET_FS = 100.0
    private const val WINDOW_SIZE = 1000
    private const val STEP_SIZE = 200
    private const val VERTICAL_ALPHA = 1.9705093832241642
    private const val THRESHOLD = 0.5170902618496815

    private val imputer = doubleArrayOf(
        1.4471702257824859,
        0.09226742561047795,
        0.06263614033261605,
    )
    private val robustCenter = imputer.copyOf()
    private val robustScale = doubleArrayOf(
        0.7188879118977822,
        0.07669117496261557,
        0.03631651253439665,
    )
    private val coefficients = doubleArrayOf(
        -0.9136257862906123,
        -1.4121614361227843,
        0.7958750175751691,
    )
    private const val INTERCEPT = 0.5645617665761614

    // scipy.signal.butter(4, [0.6/50, 3.0/50], btype="bandpass", output="sos")
    private val bandpassSos = arrayOf(
        doubleArrayOf(2.6734904005273981e-05, 5.3469808010547962e-05, 2.6734904005273981e-05, 1.0, -1.7992000481700647, 0.8163245917560644),
        doubleArrayOf(1.0, 2.0, 1.0, 1.0, -1.8775362075039115, 0.9094645372273146),
        doubleArrayOf(1.0, -2.0, 1.0, 1.0, -1.9238264102815568, 0.9264078618972685),
        doubleArrayOf(1.0, -2.0, 1.0, 1.0, -1.9785064372034438, 0.9799974273646499),
    )
    private val bandpassZi = arrayOf(
        doubleArrayOf(0.00621808064299275, -0.00507106159798981),
        doubleArrayOf(0.7761093634654586, -0.7052785658164213),
        doubleArrayOf(-0.7823541790124778, 0.7823541790124763),
        doubleArrayOf(-0.0, 0.0),
    )

    data class InputSample(val timestampNs: Long, val verticalG: Double)

    data class Result(
        val probability: Double,
        val impaired: Boolean,
        val jerkMedian: Double,
        val jerkIqr: Double,
        val symmetryIqr: Double,
        val subWindowCount: Int,
        val durationSec: Double,
    ) {
        fun toJson(): JSONObject = JSONObject()
            .put("probability", probability)
            .put("impaired", impaired)
            .put("label", if (impaired) "운동기능 저하 가능성" else "안정 범위")
            .put("v_jerk_rms_median", jerkMedian)
            .put("v_jerk_rms_iqr", jerkIqr)
            .put("v_harmonic_ratio_iqr", symmetryIqr)
            .put("n_sub_windows", subWindowCount)
            .put("duration_sec", durationSec)
            .put("threshold", THRESHOLD)
            .put("engine", "android-kotlin-offline-v1")
    }

    fun analyze(rawSamples: List<InputSample>): Result {
        require(rawSamples.size >= 80) { "센서 샘플이 부족합니다." }
        val sorted = rawSamples.sortedBy { it.timestampNs }
        val durationSec = (sorted.last().timestampNs - sorted.first().timestampNs) / 1e9
        require(durationSec >= 18.0) { "20초 보행 데이터가 필요합니다." }

        var vertical = resample(sorted)
        if (vertical.size in 1800 until 2000) {
            vertical = vertical.copyOf(2000).also { out ->
                for (i in vertical.size until out.size) out[i] = vertical.last()
            }
        }
        require(vertical.size >= 1800) { "분석 가능한 보행 구간이 부족합니다." }
        for (i in vertical.indices) vertical[i] *= VERTICAL_ALPHA

        val jerks = mutableListOf<Double>()
        val symmetries = mutableListOf<Double>()
        var start = 0
        while (start + WINDOW_SIZE <= minOf(vertical.size, 2000)) {
            val filtered = bandpassZeroPhase(vertical.copyOfRange(start, start + WINDOW_SIZE))
            jerks += jerkRms(filtered)
            harmonicRatio(filtered)?.let { symmetries += it }
            start += STEP_SIZE
        }
        require(jerks.size >= 2 && symmetries.size >= 2) { "유효한 걸음 패턴을 찾지 못했습니다." }

        val features = doubleArrayOf(
            quantile(jerks, 0.50),
            quantile(jerks, 0.75) - quantile(jerks, 0.25),
            quantile(symmetries, 0.75) - quantile(symmetries, 0.25),
        )
        var logit = INTERCEPT
        for (i in features.indices) {
            val value = if (features[i].isFinite()) features[i] else imputer[i]
            logit += coefficients[i] * ((value - robustCenter[i]) / robustScale[i])
        }
        val probability = 1.0 / (1.0 + exp(-logit))
        return Result(
            probability = probability,
            impaired = probability >= THRESHOLD,
            jerkMedian = features[0],
            jerkIqr = features[1],
            symmetryIqr = features[2],
            subWindowCount = minOf(jerks.size, symmetries.size),
            durationSec = durationSec,
        )
    }

    private fun resample(samples: List<InputSample>): DoubleArray {
        val startNs = samples.first().timestampNs
        val endNs = samples.last().timestampNs
        val count = (((endNs - startNs) / 1e9) * TARGET_FS).roundToInt().coerceAtLeast(1)
        val output = DoubleArray(count)
        var sourceIndex = 0
        for (i in 0 until count) {
            val targetNs = startNs + (i * 1e9 / TARGET_FS).toLong()
            while (sourceIndex + 1 < samples.size && samples[sourceIndex + 1].timestampNs < targetNs) sourceIndex++
            val left = samples[sourceIndex]
            val right = samples[minOf(sourceIndex + 1, samples.lastIndex)]
            val span = (right.timestampNs - left.timestampNs).toDouble()
            val ratio = if (span <= 0.0) 0.0 else ((targetNs - left.timestampNs) / span).coerceIn(0.0, 1.0)
            output[i] = left.verticalG + (right.verticalG - left.verticalG) * ratio
        }
        return output
    }

    private fun bandpassZeroPhase(input: DoubleArray): DoubleArray {
        val centered = input.copyOf()
        val median = quantile(centered.asList(), 0.50)
        for (i in centered.indices) centered[i] -= median

        // scipy.signal.sosfiltfilt 기본값과 같은 27-sample odd extension + zi 초기화.
        val pad = minOf(27, centered.size - 1)
        val padded = DoubleArray(centered.size + pad * 2)
        for (i in 0 until pad) padded[i] = 2.0 * centered.first() - centered[pad - i]
        centered.copyInto(padded, pad)
        for (i in 0 until pad) {
            padded[pad + centered.size + i] =
                2.0 * centered.last() - centered[centered.lastIndex - 1 - i]
        }
        var filtered = sosFilter(padded, padded.first())
        filtered.reverse()
        filtered = sosFilter(filtered, filtered.first())
        filtered.reverse()
        return filtered.copyOfRange(pad, pad + centered.size)
    }

    private fun sosFilter(input: DoubleArray, initialScale: Double): DoubleArray {
        var current = input.copyOf()
        for (sectionIndex in bandpassSos.indices) {
            val section = bandpassSos[sectionIndex]
            val next = DoubleArray(current.size)
            var z1 = bandpassZi[sectionIndex][0] * initialScale
            var z2 = bandpassZi[sectionIndex][1] * initialScale
            for (i in current.indices) {
                val x = current[i]
                val y = section[0] * x + z1
                z1 = section[1] * x - section[4] * y + z2
                z2 = section[2] * x - section[5] * y
                next[i] = y
            }
            current = next
        }
        return current
    }

    private fun jerkRms(values: DoubleArray): Double {
        var sum = 0.0
        for (i in 1 until values.size) {
            val diff = values[i] - values[i - 1]
            sum += diff * diff
        }
        return sqrt(sum / (values.size - 1)) * TARGET_FS
    }

    private fun harmonicRatio(values: DoubleArray): Double? {
        val mean = values.average()
        val centered = DoubleArray(values.size) { values[it] - mean }
        var variance = 0.0
        for (value in centered) variance += value * value
        if (variance <= 1e-12) return null
        val acf = DoubleArray(values.size)
        for (lag in acf.indices) {
            var sum = 0.0
            for (i in 0 until values.size - lag) sum += centered[i] * centered[i + lag]
            acf[lag] = (sum / (values.size - lag)) / (variance / values.size)
        }
        val strideIndex = peakIndex(acf, 80, minOf(170, acf.lastIndex)) ?: return null
        val stridePeak = acf[strideIndex]
        if (!stridePeak.isFinite() || stridePeak <= 1e-6) return null
        val half = strideIndex / 2.0
        val stepIndex = peakIndex(
            acf,
            (half * 0.6).roundToInt().coerceAtLeast(1),
            (half * 1.4).roundToInt().coerceAtMost(acf.lastIndex),
        ) ?: return null
        return acf[stepIndex] / stridePeak
    }

    private fun peakIndex(values: DoubleArray, low: Int, high: Int): Int? {
        if (high <= low) return null
        val localPeaks = (low + 1 until high).filter { values[it] > values[it - 1] && values[it] >= values[it + 1] }
        val prominent = localPeaks.map { index -> index to peakProminence(values, index, low, high) }
            .filter { it.second >= 0.03 }
        return if (prominent.isNotEmpty()) {
            prominent.maxByOrNull { it.second }?.first
        } else {
            (low..high).maxByOrNull { values[it] }
        }
    }

    private fun peakProminence(values: DoubleArray, index: Int, low: Int, high: Int): Double {
        val height = values[index]
        var leftMin = height
        var i = index - 1
        while (i >= low && values[i] <= height) {
            leftMin = minOf(leftMin, values[i])
            i--
        }
        var rightMin = height
        i = index + 1
        while (i <= high && values[i] <= height) {
            rightMin = minOf(rightMin, values[i])
            i++
        }
        return height - maxOf(leftMin, rightMin)
    }

    private fun quantile(source: List<Double>, q: Double): Double {
        val sorted = source.filter { it.isFinite() }.sorted()
        require(sorted.isNotEmpty()) { "값이 없습니다." }
        val position = (sorted.size - 1) * q
        val lower = position.toInt()
        val upper = minOf(lower + 1, sorted.lastIndex)
        val fraction = position - lower
        return sorted[lower] + (sorted[upper] - sorted[lower]) * fraction
    }
}
