package com.gesture.recognition

/**
 * Result of gesture recognition with performance timing data
 * ENHANCED: Now includes backend detection and diagnostic info
 */
data class GestureResult(
    val gesture: String,
    val confidence: Float,
    val allProbabilities: FloatArray,
    val handDetected: Boolean = true,
    val bufferProgress: Float = 1f,
    val isStable: Boolean = false,

    // ═══════════════════════════════════════════════════════════
    // Performance timing breakdown (in milliseconds)
    // ═══════════════════════════════════════════════════════════
    val handDetectorTimeMs: Double = 0.0,   // Stage 1: Palm/hand detection
    val landmarksTimeMs: Double = 0.0,      // Stage 2: Landmark extraction
    val gestureTimeMs: Double = 0.0,        // Stage 3: Gesture classification
    val totalTimeMs: Double = 0.0,          // Total pipeline time
    val wasTracking: Boolean = false,       // Was in tracking mode (vs detection)

    // ═══════════════════════════════════════════════════════════
    // NEW: Backend detection (CPU/GPU/NPU)
    // ═══════════════════════════════════════════════════════════
    val detectorBackend: String = "UNKNOWN",   // HandDetector backend
    val landmarksBackend: String = "UNKNOWN",  // Landmarks backend
    val gestureBackend: String = "UNKNOWN",    // Gesture backend

    // ═══════════════════════════════════════════════════════════
    // NEW: Frame info for verification
    // ═══════════════════════════════════════════════════════════
    val frameWidth: Int = 0,    // Actual frame width processed
    val frameHeight: Int = 0,   // Actual frame height processed

    // Keep old names for backward compatibility (deprecated)
    @Deprecated("Use handDetectorTimeMs and landmarksTimeMs instead")
    val mediaPipeTimeMs: Double = 0.0,
    @Deprecated("Use gestureTimeMs instead")
    val onnxTimeMs: Double = 0.0
) {
    /**
     * Check if prediction meets confidence threshold
     */
    fun meetsThreshold(): Boolean {
        return confidence >= Config.CONFIDENCE_THRESHOLD
    }

    /**
     * Get formatted gesture name (replace underscores with spaces)
     */
    fun getFormattedGesture(): String {
        return gesture.replace('_', ' ')
    }

    /**
     * Check if all models are using CPU (performance warning)
     */
    fun isAllCPU(): Boolean {
        return detectorBackend.contains("CPU", ignoreCase = true) &&
               landmarksBackend.contains("CPU", ignoreCase = true) &&
               gestureBackend.contains("CPU", ignoreCase = true)
    }

    /**
     * Check if any model is using NPU/NNAPI
     */
    fun hasNPU(): Boolean {
        return detectorBackend.contains("Nnapi", ignoreCase = true) ||
               landmarksBackend.contains("Nnapi", ignoreCase = true) ||
               gestureBackend.contains("Nnapi", ignoreCase = true)
    }

    /**
     * Get performance warning message
     */
    fun getPerformanceWarning(): String? {
        return when {
            isAllCPU() && totalTimeMs > 100 -> "⚠️ All models on CPU - very slow!"
            isAllCPU() -> "⚠️ Running on CPU - NNAPI unavailable"
            totalTimeMs > 50 -> "⚠️ Slow performance detected"
            else -> null
        }
    }

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as GestureResult

        if (gesture != other.gesture) return false
        if (confidence != other.confidence) return false
        if (!allProbabilities.contentEquals(other.allProbabilities)) return false
        if (handDetected != other.handDetected) return false
        if (bufferProgress != other.bufferProgress) return false
        if (isStable != other.isStable) return false

        return true
    }

    override fun hashCode(): Int {
        var result = gesture.hashCode()
        result = 31 * result + confidence.hashCode()
        result = 31 * result + allProbabilities.contentHashCode()
        result = 31 * result + handDetected.hashCode()
        result = 31 * result + bufferProgress.hashCode()
        result = 31 * result + isStable.hashCode()
        return result
    }
}