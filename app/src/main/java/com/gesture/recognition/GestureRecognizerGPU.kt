package com.gesture.recognition

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.google.android.gms.tasks.Task
import com.google.android.gms.tasks.Tasks

/**
 * Complete Gesture Recognition Pipeline with GPU Acceleration
 *
 * Components:
 * - Hand Detector: Google Play Services GPU
 * - Landmark Detector: Google Play Services GPU
 * - Gesture Classifier: ONNX Runtime NPU (already working!)
 */
class GestureRecognizerGPU(private val context: Context) {

    companion object {
        private const val TAG = "GestureRecognizerGPU"
    }

    // Components
    private val handDetector = HandDetectorGPU(context)
    private val landmarkDetector = HandLandmarkDetectorGPU(context)
    private val gestureClassifier = ONNXInference(context)

    // State
    private val sequenceBuffer = SequenceBuffer(Config.SEQUENCE_LENGTH)
    private val landmarkNormalizer = LandmarkNormalizer

    // Latest landmarks (for drawing)
    var latestLandmarks: FloatArray? = null
        private set

    /**
     * Initialize all components
     * Returns Task<Boolean> because GPU initialization is async
     */
    fun initialize(): Task<Boolean> {
        Log.i(TAG, "Initializing Gesture Recognizer (GPU Pipeline)")

        // Initialize hand detector (async)
        return handDetector.initialize().continueWithTask { detectorTask ->

            if (!detectorTask.isSuccessful || detectorTask.result != true) {
                Log.e(TAG, "Hand detector initialization failed")
                return@continueWithTask Tasks.forResult(false)
            }

            // Initialize landmark detector (async)
            landmarkDetector.initialize().continueWith { landmarkTask ->

                if (!landmarkTask.isSuccessful || landmarkTask.result != true) {
                    Log.e(TAG, "Landmark detector initialization failed")
                    return@continueWith false
                }

                // Initialize gesture classifier (sync - ONNX)
                try {
                    Log.i(TAG, "✓ Gesture classifier ready (ONNX NPU)")
                    Log.i(TAG, "✓ Complete pipeline initialized")
                    true
                } catch (e: Exception) {
                    Log.e(TAG, "Gesture classifier failed", e)
                    false
                }
            }
        }
    }

    /**
     * Recognize gesture from bitmap
     */
    fun recognize(bitmap: Bitmap): GestureResult? {
        try {
            val startTime = System.nanoTime()

            // Step 1: Detect hand
            val detectorStart = System.nanoTime()
            val detection = handDetector.detectHand(bitmap)
            val detectorTime = (System.nanoTime() - detectorStart) / 1_000_000.0

            if (detection == null) {
                latestLandmarks = null
                return GestureResult(
                    gesture = "no_hand",
                    confidence = 0.0f,
                    allProbabilities = FloatArray(8) { 0f },
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

            // Step 2: Extract landmarks
            val landmarkStart = System.nanoTime()
            val roi = createROIFromDetection(detection, bitmap.width, bitmap.height)
            val landmarkResult = landmarkDetector.detectLandmarks(bitmap, roi)
            val landmarkTime = (System.nanoTime() - landmarkStart) / 1_000_000.0

            if (landmarkResult == null) {
                latestLandmarks = null
                return GestureResult(
                    gesture = "no_landmarks",
                    confidence = 0.0f,
                    allProbabilities = FloatArray(8) { 0f },
                    handDetected = true,
                    bufferProgress = 0f,
                    isStable = false,
                    handDetectorTimeMs = detectorTime,
                    landmarksTimeMs = landmarkTime,
                    gestureTimeMs = 0.0,
                    totalTimeMs = detectorTime + landmarkTime,
                    wasTracking = true
                )
            }

            // Step 3: Normalize landmarks - force non-null with !!
            val landmarksArray = landmarkResult.landmarks!!
            val landmarksFlat = flattenLandmarks(landmarksArray)
            val normalized = LandmarkNormalizer.normalize(landmarksFlat)

            // Store for drawing
            latestLandmarks = landmarksFlat

            // Step 4: Add to sequence buffer
            sequenceBuffer.add(normalized)

            // Step 5: Classify gesture (if buffer is full)
            val gestureStart = System.nanoTime()
            var gestureName = "buffering"
            var confidence = 0.0f
            var allProbabilities = FloatArray(8) { 0f }

            if (sequenceBuffer.isFull()) {
                val sequence = sequenceBuffer.getSequence()!!
                val result = gestureClassifier.predict(sequence)
                if (result != null) {
                    val (gestureId, probabilities) = result
                    gestureName = getGestureLabel(gestureId)
                    confidence = probabilities[gestureId]
                    allProbabilities = probabilities
                }
            }

            val gestureTime = (System.nanoTime() - gestureStart) / 1_000_000.0
            val totalTime = (System.nanoTime() - startTime) / 1_000_000.0

            return GestureResult(
                gesture = gestureName,
                confidence = confidence,
                allProbabilities = allProbabilities,
                handDetected = true,
                bufferProgress = sequenceBuffer.size().toFloat() / Config.SEQUENCE_LENGTH,
                isStable = sequenceBuffer.isFull(),
                handDetectorTimeMs = detectorTime,
                landmarksTimeMs = landmarkTime,
                gestureTimeMs = gestureTime,
                totalTimeMs = totalTime,
                wasTracking = true
            )

        } catch (e: Exception) {
            Log.e(TAG, "Recognition failed", e)
            return null
        }
    }

    /**
     * Get gesture label by ID
     */
    private fun getGestureLabel(gestureId: Int): String {
        val labels = arrayOf(
            "no_gesture",
            "swipe_left",
            "swipe_right",
            "swipe_up",
            "swipe_down",
            "stop_sign",
            "thumbs_up",
            "thumbs_down"
        )
        return if (gestureId in labels.indices) labels[gestureId] else "unknown"
    }

    /**
     * Create ROI from detection
     */
        /**
     * Create ROI from detection - MATCHING Python predict_tflite.py logic
     */
    /**
     * Create ROI from detection - MATCHING Python predict_tflite.py logic
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
        // Calculate boundaries
        val left = cx_a - w_a / 2
        val top = cy_a - h_a / 2
        val right = cx_a + w_a / 2
        val bottom = cy_a + h_a / 2

        // If ROI exceeds bounds, reduce size to fit
        if (left < 0 || top < 0 || right > imageWidth || bottom > imageHeight) {
            // Calculate how much we can expand while staying in bounds
            val maxLeft = cx_a
            val maxTop = cy_a
            val maxRight = imageWidth - cx_a
            val maxBottom = imageHeight - cy_a

            // Maximum size that fits in all directions
            val maxSize = minOf(maxLeft, maxTop, maxRight, maxBottom) * 2

            if (maxSize < ls * SCALE_X) {
                // ROI too large, scale down to fit
                w_a = maxSize
                h_a = maxSize
                FileLogger.d("GestureRecognizer", "⚠️ ROI clamped: %.1fx%.1f -> %.1fx%.1f"
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
     * Flatten landmarks array
     */
    private fun flattenLandmarks(landmarks: Array<FloatArray>): FloatArray {
        val result = FloatArray(landmarks.size * 3)
        for (i in landmarks.indices) {
            result[i * 3] = landmarks[i][0]
            result[i * 3 + 1] = landmarks[i][1]
            result[i * 3 + 2] = landmarks[i][2]
        }
        return result
    }


    /**
     * Get backend info
     */
    fun getDetectorBackend(): String = handDetector.getBackend()
    fun getLandmarkBackend(): String = landmarkDetector.getBackend()

    /**
     * Release resources
     */
    fun close() {
        handDetector.close()
        landmarkDetector.close()
        gestureClassifier.close()
        Log.i(TAG, "✓ Gesture Recognizer closed")
    }
}