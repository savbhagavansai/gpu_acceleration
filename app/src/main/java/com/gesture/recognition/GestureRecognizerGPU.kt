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
    private val gestureClassifier = ONNXInference(context)  // Keep existing ONNX (working!)

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
        Log.i(TAG,("Initializing Gesture Recognizer (GPU Pipeline)")

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
                    // Gesture classifier uses existing ONNX Runtime (already working!)
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
                latestLandmarks = null  // Clear landmarks
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
                latestLandmarks = null  // Clear landmarks
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

            // Step 3: Normalize landmarks (landmarkResult is guaranteed non-null here)
            val landmarksFlat = flattenLandmarks(landmarkResult!!.landmarks)
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
                val sequence = sequenceBuffer.getSequence()
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
            centerX = xCenter,
            centerY = yCenter,
            roiWidth = expandedWidth,
            roiHeight = expandedHeight,
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