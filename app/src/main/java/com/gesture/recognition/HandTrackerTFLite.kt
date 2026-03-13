package com.gesture.recognition

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import kotlin.math.*

/**
 * Complete Hand Tracking Pipeline using TFLite models
 *
 * Two-stage pipeline with detection/tracking state machine:
 *
 * DETECTION MODE (first frame or after loss):
 *   1. Run HandDetector on full frame
 *   2. Extract landmarks from detected ROI
 *   3. Switch to tracking mode
 *
 * TRACKING MODE (most frames):
 *   1. Build ROI from previous landmarks (no detector!)
 *   2. Extract landmarks from ROI
 *   3. Continue tracking or revert to detection if lost
 *
 * Performance:
 *   - Detection mode: ~8-18ms (detector + landmarks)
 *   - Tracking mode: ~3-12ms (landmarks only)
 *   - Average: ~5-15ms (mostly tracking)
 */
class HandTrackerTFLite(private val context: Context) {

    companion object {
        private const val TAG = "HandTrackerTFLite"
        private const val LANDMARK_SMOOTHING_ALPHA = 0.5f  // EMA: 0=frozen, 1=raw, 0.5=balanced

        // ROI transformation constants (for tracking mode)
        private const val SCALE_X = 2.9f
        private const val SCALE_Y = 2.9f
        private const val SHIFT_X = 0.0f
        private const val SHIFT_Y = -0.5f
    }

    private val detector: HandDetectorTFLite
    private val landmarker: HandLandmarkDetectorTFLite

    // State machine
    private var isTracking = false
    private var smoothedLandmarks: Array<FloatArray>? = null
    private var currentROI: HandROI? = null

    // Performance monitoring
    private var lastDetectorTime = 0f
    private var lastLandmarkTime = 0f

    init {
        Log.d(TAG, "Initializing Hand Tracker TFLite...")
        detector = HandDetectorTFLite(context)
        landmarker = HandLandmarkDetectorTFLite(context)
        Log.d(TAG, "✓ Hand Tracker TFLite ready")
    }

    /**
     * Process frame and extract hand landmarks
     *
     * @param frame Input frame (any size)
     * @return HandTrackingResult or null if no hand
     */
    fun processFrame(frame: Bitmap): HandTrackingResult? {
        val frameWidth = frame.width
        val frameHeight = frame.height

        var roi: HandROI?
        var detectorMs = 0f
        var landmarkMs = 0f

        // ── STAGE 1: Get ROI (detection or tracking) ──
        if (!isTracking) {
            // DETECTION MODE: Run full detector
            val t0 = System.nanoTime()
            val detection = detector.detectHand(frame)
            detectorMs = (System.nanoTime() - t0) / 1_000_000f

            if (detection == null) {
                Log.d(TAG, "[DETECT] No hand found")
                return null
            }

            roi = detection.roi
            Log.d(TAG, "[DETECT] Hand found: conf=${detection.confidence}")

        } else {
            // TRACKING MODE: Use previous landmarks for ROI
            val prevLandmarks = smoothedLandmarks
            if (prevLandmarks != null) {
                roi = buildTrackingROI(prevLandmarks, frameWidth, frameHeight)
                detectorMs = 0f  // Detector not used!
                Log.d(TAG, "[TRACK] Using previous landmarks for ROI")
            } else {
                // Shouldn't happen, but fallback to detection
                isTracking = false
                return processFrame(frame)
            }
        }

        currentROI = roi

        // ── STAGE 2: Extract landmarks from ROI ──
        val t0 = System.nanoTime()
        val result = landmarker.detectLandmarks(frame, roi)
        landmarkMs = (System.nanoTime() - t0) / 1_000_000f

        if (result == null) {
            // Hand lost → reset to detection mode
            Log.d(TAG, "[LOST] Hand presence too low → back to detector")
            isTracking = false
            smoothedLandmarks = null
            currentROI = null
            return null
        }

        // Apply EMA smoothing
        val rawLandmarks = result.landmarks
        if (smoothedLandmarks == null) {
            // First frame: initialize
            smoothedLandmarks = rawLandmarks.map { it.copyOf() }.toTypedArray()
        } else {
            // Smooth only x, y (keep z raw)
            for (i in rawLandmarks.indices) {
                smoothedLandmarks!![i][0] =
                    LANDMARK_SMOOTHING_ALPHA * rawLandmarks[i][0] +
                    (1 - LANDMARK_SMOOTHING_ALPHA) * smoothedLandmarks!![i][0]

                smoothedLandmarks!![i][1] =
                    LANDMARK_SMOOTHING_ALPHA * rawLandmarks[i][1] +
                    (1 - LANDMARK_SMOOTHING_ALPHA) * smoothedLandmarks!![i][1]

                smoothedLandmarks!![i][2] = rawLandmarks[i][2]  // z unchanged
            }
        }

        // Enable tracking for next frame
        isTracking = true

        // Store timing
        lastDetectorTime = detectorMs
        lastLandmarkTime = landmarkMs

        return HandTrackingResult(
            landmarks = smoothedLandmarks!!,
            presence = result.presence,
            handedness = result.handedness,
            roi = roi,
            detectorTimeMs = detectorMs,
            landmarkTimeMs = landmarkMs,
            totalTimeMs = detectorMs + landmarkMs,
            wasTracking = isTracking
        )
    }

    /**
     * Build tracking ROI from previous landmarks
     * Much faster than running detector!
     */
    private fun buildTrackingROI(
        landmarks: Array<FloatArray>,
        frameWidth: Int,
        frameHeight: Int
    ): HandROI {

        // Use subset of landmarks for stability
        val subsetIndices = intArrayOf(0, 1, 2, 3, 5, 6, 9, 10, 13, 14, 17, 18)

        // Calculate rotation from wrist and palm center
        val wristX = landmarks[0][0]
        val wristY = landmarks[0][1]
        val palmCenterX = 0.25f * (landmarks[5][0] + landmarks[13][0]) + 0.5f * landmarks[9][0]
        val palmCenterY = 0.25f * (landmarks[5][1] + landmarks[13][1]) + 0.5f * landmarks[9][1]
        val rotation = normalizeRadians(
            0.5f * PI.toFloat() - atan2(wristY - palmCenterY, palmCenterX - wristX)
        )

        // Find bounding box of subset landmarks
        val subset = subsetIndices.map { landmarks[it] }
        val minX = subset.minOf { it[0] }
        val maxX = subset.maxOf { it[0] }
        val minY = subset.minOf { it[1] }
        val maxY = subset.maxOf { it[1] }

        val axisCenterX = (minX + maxX) / 2
        val axisCenterY = (minY + maxY) / 2

        // Rotate subset points
        val c = cos(rotation)
        val s = sin(rotation)

        val rotatedX = mutableListOf<Float>()
        val rotatedY = mutableListOf<Float>()

        for (lm in subset) {
            val dx = lm[0] - axisCenterX
            val dy = lm[1] - axisCenterY
            rotatedX.add(c * dx - s * dy)
            rotatedY.add(s * dx + c * dy)
        }

        val minRotX = rotatedX.minOrNull() ?: 0f
        val maxRotX = rotatedX.maxOrNull() ?: 0f
        val minRotY = rotatedY.minOrNull() ?: 0f
        val maxRotY = rotatedY.maxOrNull() ?: 0f

        val width = maxRotX - minRotX
        val height = maxRotY - minRotY
        val size = 2 * maxOf(width, height)

        // Transform center back
        val centerRotX = (minRotX + maxRotX) / 2
        val centerRotY = (minRotY + maxRotY) / 2

        var centerX = c * centerRotX + s * centerRotY + axisCenterX
        var centerY = -s * centerRotX + c * centerRotY + axisCenterY

        // Apply small shift
        centerX += 0.1f * height * s
        centerY -= 0.1f * height * c

        // Calculate 4 corner points
        val rectPoints = rotatedRectToPoints(centerX, centerY, size, size, rotation)

        return HandROI(
            rotation = rotation,
            centerX = centerX,
            centerY = centerY,
            width = size,
            height = size,
            rectPoints = rectPoints,
            frameWidth = frameWidth,
            frameHeight = frameHeight
        )
    }

    /**
     * Calculate 4 corners of rotated rectangle
     */
    private fun rotatedRectToPoints(
        cx: Float, cy: Float,
        w: Float, h: Float,
        rotation: Float
    ): Array<FloatArray> {
        val b = cos(rotation) * 0.5f
        val a = sin(rotation) * 0.5f

        val p0x = cx - a * h - b * w
        val p0y = cy + b * h - a * w
        val p1x = cx + a * h - b * w
        val p1y = cy - b * h - a * w
        val p2x = 2 * cx - p0x
        val p2y = 2 * cy - p0y
        val p3x = 2 * cx - p1x
        val p3y = 2 * cy - p1y

        return arrayOf(
            floatArrayOf(p0x, p0y),  // bottom-left
            floatArrayOf(p1x, p1y),  // top-left
            floatArrayOf(p2x, p2y),  // top-right
            floatArrayOf(p3x, p3y)   // bottom-right
        )
    }

    /**
     * Normalize angle to [-π, π]
     */
    private fun normalizeRadians(angle: Float): Float {
        val twoPi = 2 * PI.toFloat()
        return angle - twoPi * floor((angle + PI.toFloat()) / twoPi)
    }

    /**
     * Get last smoothed landmarks (for display or gesture recognition)
     */
    fun getLastLandmarks(): Array<FloatArray>? = smoothedLandmarks

    /**
     * Get current ROI (for visualization)
     */
    fun getCurrentROI(): HandROI? = currentROI

    /**
     * Reset tracking state
     */
    fun reset() {
        isTracking = false
        smoothedLandmarks = null
        currentROI = null
        Log.d(TAG, "Tracking state reset")
    }

    /**
     * Get detector backend info
     */
    fun getDetectorBackend(): String = detector.getBackend()

    /**
     * Get landmark backend info
     */
    fun getLandmarkBackend(): String = landmarker.getBackend()

    /**
     * Release resources
     */
    fun close() {
        detector.close()
        landmarker.close()
        Log.d(TAG, "✓ Hand Tracker closed")
    }
}

/**
 * Hand tracking result
 */
data class HandTrackingResult(
    val landmarks: Array<FloatArray>,  // [21, 3] - x, y, z
    val presence: Float,
    val handedness: String,
    val roi: HandROI,
    val detectorTimeMs: Float,
    val landmarkTimeMs: Float,
    val totalTimeMs: Float,
    val wasTracking: Boolean
)