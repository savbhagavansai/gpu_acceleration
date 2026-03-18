package com.gesture.recognition

import android.content.Context
import android.graphics.Bitmap
import com.google.android.gms.tasks.Task
import com.google.android.gms.tasks.Tasks

/**
 * Complete Gesture Recognition Pipeline with Hand Tracking
 *
 * Pipeline:
 * 1. Hand Detection (first frame or after tracking loss)
 * 2. Hand Tracking (subsequent frames - builds ROI from previous landmarks)
 * 3. Landmark Detection (on ROI)
 * 4. Landmark Normalization
 * 5. Sequence Buffering (15 frames)
 * 6. Gesture Classification (ONNX TCN model)
 * 7. Prediction Smoothing
 */
class GestureRecognizerGPU(private val context: Context) {

    companion object {
        private const val TAG = "GestureRecognizer"
    }

    // Components
    private val handDetector: HandDetectorGPU
    private val landmarkDetector: HandLandmarkDetectorGPU
    private val onnxInference: ONNXInference
    private val sequenceBuffer: SequenceBuffer
    private val predictionSmoother: PredictionSmoother

    // State
    var latestLandmarks: Array<FloatArray>? = null
        private set

    // ✅ TRACKING STATE
    private var isTracking = false
    private var consecutiveTrackingFailures = 0
    private val MAX_TRACKING_FAILURES = 2

    init {
        FileLogger.d(TAG, "Initializing Gesture Recognizer with Tracking...")

        handDetector = HandDetectorGPU(context)
        landmarkDetector = HandLandmarkDetectorGPU(context)
        onnxInference = ONNXInference(context)
        sequenceBuffer = SequenceBuffer(Config.SEQUENCE_LENGTH)
        predictionSmoother = PredictionSmoother(windowSize = 5)  // ✅ FIXED: windowSize parameter

        FileLogger.d(TAG, "✓ Gesture Recognizer ready")
    }

    /**
     * Initialize async (GPU delegate takes time)
     */
    fun initialize(): Task<Boolean> {
        return Tasks.whenAll(
            handDetector.initialize(),
            landmarkDetector.initialize()
        ).continueWith { task ->
            if (task.isSuccessful) {
                FileLogger.d(TAG, "✓ All models initialized")
                true
            } else {
                FileLogger.e(TAG, "✗ Initialization failed: ${task.exception?.message}")
                false
            }
        }
    }

    /**
     * Build tracking ROI from previous landmarks
     * Much faster than running hand detection!
     */
    private fun createTrackingROI(
        landmarks: Array<FloatArray>,
        frameWidth: Int,
        frameHeight: Int
    ): HandTrackingROI {

        // Find bounding box of all landmarks
        var xMin = Float.MAX_VALUE
        var yMin = Float.MAX_VALUE
        var xMax = Float.MIN_VALUE
        var yMax = Float.MIN_VALUE

        for (lm in landmarks) {
            val x = lm[0]
            val y = lm[1]
            if (x < xMin) xMin = x
            if (x > xMax) xMax = x
            if (y < yMin) yMin = y
            if (y > yMax) yMax = y
        }

        // Box dimensions
        val boxW = xMax - xMin
        val boxH = yMax - yMin
        val boxCx = xMin + boxW / 2
        val boxCy = yMin + boxH / 2

        // Square ROI (2.2x gives good margin for tracking)
        var size = maxOf(boxW, boxH) * 2.2f

        // Clamp to image bounds to prevent out-of-bounds errors
        val left = boxCx - size / 2
        val top = boxCy - size / 2
        val right = boxCx + size / 2
        val bottom = boxCy + size / 2

        if (left < 0 || top < 0 || right > frameWidth || bottom > frameHeight) {
            val maxDist = minOf(boxCx, boxCy, frameWidth - boxCx, frameHeight - boxCy)
            size = maxDist * 2
        }

        return HandTrackingROI(
            centerX = boxCx,
            centerY = boxCy,
            roiWidth = size,
            roiHeight = size,
            rotation = 0f
        )
    }

    /**
     * Create ROI from detection
     * Uses MediaPipe's standard 2.9x expansion and -0.5 Y-shift
     */
    private fun createROIFromDetection(
        detection: DetectionResult,
        imageWidth: Int,
        imageHeight: Int
    ): HandTrackingROI {
        val box = detection.box

        // MediaPipe palm detection constants
        val SCALE_X = 2.9f
        val SCALE_Y = 2.9f
        val SHIFT_X = 0.0f
        val SHIFT_Y = -0.5f

        // Box dimensions
        val bx = box[0]
        val by = box[1]
        val bw = box[2] - box[0]
        val bh = box[3] - box[1]

        // Center of box
        val rx = bx + bw / 2
        val ry = by + bh / 2

        // Apply shift to center
        val cx_a = rx + bw * SHIFT_X
        val cy_a = ry + bh * SHIFT_Y

        // ROI size: use larger dimension and scale
        val ls = maxOf(bw, bh)
        var w_a = ls * SCALE_X
        var h_a = ls * SCALE_Y

        // ✅ CLAMP ROI to stay within image bounds
        val left = cx_a - w_a / 2
        val top = cy_a - h_a / 2
        val right = cx_a + w_a / 2
        val bottom = cy_a + h_a / 2

        if (left < 0 || top < 0 || right > imageWidth || bottom > imageHeight) {
            // Calculate maximum size that fits in all directions
            val maxLeft = cx_a
            val maxTop = cy_a
            val maxRight = imageWidth - cx_a
            val maxBottom = imageHeight - cy_a

            val maxSize = minOf(maxLeft, maxTop, maxRight, maxBottom) * 2

            if (maxSize < ls * SCALE_X) {
                // ROI too large, scale down to fit
                w_a = maxSize
                h_a = maxSize
                FileLogger.d(TAG, "⚠️ ROI clamped: %.1fx%.1f -> %.1fx%.1f"
                    .format(ls * SCALE_X, ls * SCALE_Y, w_a, h_a))
            }
        }

        return HandTrackingROI(
            centerX = cx_a,
            centerY = cy_a,
            roiWidth = w_a,
            roiHeight = h_a,
            rotation = 0f
        )
    }

    /**
     * Recognize gesture from bitmap
     * Uses tracking for stability (only runs hand detection when needed)
     */
    fun recognize(bitmap: Bitmap): GestureResult? {
        try {
            val startTime = System.nanoTime()

            var roi: HandTrackingROI?
            var detectorTime = 0.0
            var usedTracking = false

            // ── STAGE 1: Get ROI (detection or tracking) ──
            if (!isTracking || latestLandmarks == null) {
                // DETECTION MODE: Run full hand detector
                val detectorStart = System.nanoTime()
                val detection = handDetector.detectHand(bitmap)
                detectorTime = (System.nanoTime() - detectorStart) / 1_000_000.0

                if (detection == null) {
                    latestLandmarks = null
                    isTracking = false
                    return GestureResult(
                        gesture = "no_hand",
                        confidence = 0.0f,
                        allProbabilities = FloatArray(11) { 0f },
                        handDetected = false,
                        bufferProgress = 0f,
                        isStable = false,
                        handDetectorTimeMs = detectorTime,
                        landmarksTimeMs = 0.0,
                        gestureTimeMs = 0.0,
                        totalTimeMs = detectorTime,
                        wasTracking = false
                    )
                }

                roi = createROIFromDetection(detection, bitmap.width, bitmap.height)
                FileLogger.d(TAG, "[DETECT] Hand found, starting tracking")

            } else {
                // TRACKING MODE: Build ROI from previous landmarks (skip detector!)
                roi = createTrackingROI(latestLandmarks!!, bitmap.width, bitmap.height)
                usedTracking = true
                detectorTime = 0.0  // Detector not used!
            }

            // ── STAGE 2: Extract landmarks from ROI ──
            val landmarkStart = System.nanoTime()
            val landmarkResult = landmarkDetector.detectLandmarks(bitmap, roi)
            val landmarkTime = (System.nanoTime() - landmarkStart) / 1_000_000.0

            if (landmarkResult == null) {
                // Tracking failed
                consecutiveTrackingFailures++

                if (consecutiveTrackingFailures >= MAX_TRACKING_FAILURES) {
                    // Too many failures → reset to detection mode
                    FileLogger.d(TAG, "[LOST] Tracking failed → back to detector")
                    isTracking = false
                    consecutiveTrackingFailures = 0
                }

                latestLandmarks = null
                return GestureResult(
                    gesture = "no_landmarks",
                    confidence = 0.0f,
                    allProbabilities = FloatArray(11) { 0f },
                    handDetected = true,
                    bufferProgress = 0f,
                    isStable = false,
                    handDetectorTimeMs = detectorTime,
                    landmarksTimeMs = landmarkTime,
                    gestureTimeMs = 0.0,
                    totalTimeMs = detectorTime + landmarkTime,
                    wasTracking = usedTracking
                )
            }

            // ✅ Success! Enable tracking for next frame
            isTracking = true
            consecutiveTrackingFailures = 0
            latestLandmarks = landmarkResult.landmarks

            // ── STAGE 3: Gesture recognition ──

            // Normalize landmarks (flatten to FloatArray first)
            val normalized = normalizeLandmarks(landmarkResult.landmarks)

            // Add to sequence buffer
            sequenceBuffer.add(normalized)
            val bufferProgress = sequenceBuffer.size().toFloat() / Config.SEQUENCE_LENGTH

            // Check if buffer is full
            if (!sequenceBuffer.isFull()) {
                return GestureResult(
                    gesture = "buffering",
                    confidence = 0.0f,
                    allProbabilities = FloatArray(11) { 0f },
                    handDetected = true,
                    bufferProgress = bufferProgress,
                    isStable = false,
                    handDetectorTimeMs = detectorTime,
                    landmarksTimeMs = landmarkTime,
                    gestureTimeMs = 0.0,
                    totalTimeMs = (System.nanoTime() - startTime) / 1_000_000.0,
                    wasTracking = usedTracking
                )
            }

            // Run gesture classification
            val gestureStart = System.nanoTime()
            val sequence = sequenceBuffer.getSequence()

            // Handle null sequence
            if (sequence == null) {
                return GestureResult(
                    gesture = "buffer_error",
                    confidence = 0.0f,
                    allProbabilities = FloatArray(11) { 0f },
                    handDetected = true,
                    bufferProgress = 1.0f,
                    isStable = false,
                    handDetectorTimeMs = detectorTime,
                    landmarksTimeMs = landmarkTime,
                    gestureTimeMs = 0.0,
                    totalTimeMs = (System.nanoTime() - startTime) / 1_000_000.0,
                    wasTracking = usedTracking
                )
            }

            val prediction = onnxInference.predict(sequence)  // ✅ FIXED: Returns Pair<Int, FloatArray>?
            val gestureTime = (System.nanoTime() - gestureStart) / 1_000_000.0

            if (prediction == null) {
                return GestureResult(
                    gesture = "unknown",
                    confidence = 0.0f,
                    allProbabilities = FloatArray(11) { 0f },
                    handDetected = true,
                    bufferProgress = 1.0f,
                    isStable = false,
                    handDetectorTimeMs = detectorTime,
                    landmarksTimeMs = landmarkTime,
                    gestureTimeMs = gestureTime,
                    totalTimeMs = (System.nanoTime() - startTime) / 1_000_000.0,
                    wasTracking = usedTracking
                )
            }

            val (gestureIdx, allProbs) = prediction  // ✅ FIXED: Destructure Pair correctly
            val confidence = allProbs[gestureIdx]
            val gestureLabel = Config.IDX_TO_LABEL[gestureIdx] ?: "unknown"

            // Smooth prediction
            predictionSmoother.addPrediction(gestureIdx)
            val smoothedIdx = predictionSmoother.getSmoothedPrediction()
            val smoothedLabel = Config.IDX_TO_LABEL[smoothedIdx] ?: "unknown"

            val totalTime = (System.nanoTime() - startTime) / 1_000_000.0

            return GestureResult(
                gesture = smoothedLabel,  // ✅ FIXED: Return label string
                confidence = confidence,
                allProbabilities = allProbs,
                handDetected = true,
                bufferProgress = 1.0f,
                isStable = predictionSmoother.isStable(),
                handDetectorTimeMs = detectorTime,
                landmarksTimeMs = landmarkTime,
                gestureTimeMs = gestureTime,
                totalTimeMs = totalTime,
                wasTracking = usedTracking
            )

        } catch (e: Exception) {
            FileLogger.e(TAG, "Recognition failed: ${e.message}")
            isTracking = false
            return null
        }
    }

    /**
     * Normalize landmarks for gesture recognition
     * Flatten Array<FloatArray> to FloatArray first
     */
    private fun normalizeLandmarks(landmarks: Array<FloatArray>): FloatArray {
        // Flatten landmarks from [21, 3] to [63]
        val flattened = FloatArray(Config.NUM_FEATURES)
        var idx = 0
        for (lm in landmarks) {
            flattened[idx++] = lm[0]
            flattened[idx++] = lm[1]
            flattened[idx++] = lm[2]
        }

        // Normalize using LandmarkNormalizer
        return LandmarkNormalizer.normalize(flattened)  // ✅ FIXED: Pass FloatArray
    }

    /**
     * Get detector backend info
     */
    fun getDetectorBackend(): String = handDetector.getBackend()

    /**
     * Get landmark detector backend info
     */
    fun getLandmarkBackend(): String = landmarkDetector.getBackend()

    /**
     * Close and release resources
     */
    fun close() {
        handDetector.close()
        landmarkDetector.close()
        onnxInference.close()
        FileLogger.d(TAG, "✓ Gesture Recognizer closed")
    }
}