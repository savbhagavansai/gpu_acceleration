package com.gesture.recognition

import android.content.Context
import android.graphics.Bitmap
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
    private val gestureClassifier = ONNXInference(context)  // Keep existing ONNX (working!)

    // State
    private val sequenceBuffer = SequenceBuffer(Config.SEQUENCE_LENGTH)
    private val landmarkNormalizer = LandmarkNormalizer()

    /**
     * Initialize all components
     * Returns Task<Boolean> because GPU initialization is async
     */
    fun initialize(): Task<Boolean> {
        FileLogger.section("Initializing Gesture Recognizer (GPU Pipeline)")

        // Initialize hand detector (async)
        return handDetector.initialize().continueWithTask { detectorTask ->

            if (!detectorTask.isSuccessful || detectorTask.result != true) {
                FileLogger.e(TAG, "Hand detector initialization failed")
                return@continueWithTask Tasks.forResult(false)
            }

            // Initialize landmark detector (async)
            landmarkDetector.initialize().continueWith { landmarkTask ->

                if (!landmarkTask.isSuccessful || landmarkTask.result != true) {
                    FileLogger.e(TAG, "Landmark detector initialization failed")
                    return@continueWith false
                }

                // Initialize gesture classifier (sync - ONNX)
                try {
                    // Gesture classifier uses existing ONNX Runtime (already working!)
                    FileLogger.i(TAG, "✓ Gesture classifier ready (ONNX NPU)")
                    FileLogger.i(TAG, "✓ Complete pipeline initialized")
                    true
                } catch (e: Exception) {
                    FileLogger.e(TAG, "Gesture classifier failed", e)
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
                return GestureResult(
                    gesture = "no_hand",
                    confidence = 0.0f,
                    landmarks = null,
                    detectorTimeMs = detectorTime,
                    landmarkTimeMs = 0.0,
                    gestureTimeMs = 0.0,
                    totalTimeMs = detectorTime,
                    bufferFilled = sequenceBuffer.size(),
                    wasTracking = false
                )
            }

            // Step 2: Extract landmarks
            val landmarkStart = System.nanoTime()
            val roi = createROIFromDetection(detection, bitmap.width, bitmap.height)
            val landmarkResult = landmarkDetector.detectLandmarks(bitmap, roi)
            val landmarkTime = (System.nanoTime() - landmarkStart) / 1_000_000.0

            if (landmarkResult == null) {
                return GestureResult(
                    gesture = "no_landmarks",
                    confidence = 0.0f,
                    landmarks = null,
                    detectorTimeMs = detectorTime,
                    landmarkTimeMs = landmarkTime,
                    gestureTimeMs = 0.0,
                    totalTimeMs = detectorTime + landmarkTime,
                    bufferFilled = sequenceBuffer.size(),
                    wasTracking = true
                )
            }

            // Step 3: Normalize landmarks
            val landmarksFlat = flattenLandmarks(landmarkResult.landmarks)
            val normalized = landmarkNormalizer.normalize(landmarksFlat)

            // Step 4: Add to sequence buffer
            sequenceBuffer.add(normalized)

            // Step 5: Classify gesture (if buffer is full)
            val gestureStart = System.nanoTime()
            var gestureName = "buffering"
            var confidence = 0.0f

            if (sequenceBuffer.isFull()) {
                val sequence = sequenceBuffer.getSequence()
                val (gestureId, probabilities) = gestureClassifier.predict(sequence)
                gestureName = Config.GESTURE_LABELS[gestureId]
                confidence = probabilities[gestureId]
            }

            val gestureTime = (System.nanoTime() - gestureStart) / 1_000_000.0
            val totalTime = (System.nanoTime() - startTime) / 1_000_000.0

            return GestureResult(
                gesture = gestureName,
                confidence = confidence,
                landmarks = landmarksFlat,
                detectorTimeMs = detectorTime,
                landmarkTimeMs = landmarkTime,
                gestureTimeMs = gestureTime,
                totalTimeMs = totalTime,
                bufferFilled = sequenceBuffer.size(),
                wasTracking = true
            )

        } catch (e: Exception) {
            FileLogger.e(TAG, "Recognition failed", e)
            return null
        }
    }

    /**
     * Create ROI from detection
     */
    private fun createROIFromDetection(
        detection: DetectionResult,
        imageWidth: Int,
        imageHeight: Int
    ): HandTrackingROI {
        val box = detection.box

        val width = box[2] - box[0]
        val height = box[3] - box[1]
        val xCenter = box[0] + width / 2
        val yCenter = box[1] + height / 2

        // Expand ROI by 50% for landmarks
        val expandedWidth = width * 1.5f
        val expandedHeight = height * 1.5f

        return HandTrackingROI(
            xCenter = xCenter,
            yCenter = yCenter,
            width = expandedWidth,
            height = expandedHeight,
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
        FileLogger.i(TAG, "✓ Gesture Recognizer closed")
    }
}

/**
 * Gesture recognition result
 */
data class GestureResult(
    val gesture: String,
    val confidence: Float,
    val landmarks: FloatArray?,
    val detectorTimeMs: Double,
    val landmarkTimeMs: Double,
    val gestureTimeMs: Double,
    val totalTimeMs: Double,
    val bufferFilled: Int,
    val wasTracking: Boolean
)