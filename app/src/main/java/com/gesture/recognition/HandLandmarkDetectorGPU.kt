package com.gesture.recognition

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import com.google.android.gms.tasks.Task
import com.google.android.gms.tasks.Tasks
import com.google.android.gms.tflite.java.TfLite
import org.tensorflow.lite.InterpreterApi
import org.tensorflow.lite.InterpreterApi.Options.TfLiteRuntime
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.gpu.GpuDelegateFactory

/**
 * Hand Landmark Detector using Google Play Services TFLite with GPU
 *
 * Input: Cropped hand ROI (256×256)
 * Output: 21 landmarks (x, y, z), presence score, handedness
 */
class HandLandmarkDetectorGPU(private val context: Context) {

    companion object {
        private const val TAG = "HandLandmarkGPU"
        private const val MODEL_NAME = "mediapipe_hand-handlandmarkdetector.tflite"
        private const val INPUT_SIZE = 256
        private const val NUM_LANDMARKS = 21
    }

    private var interpreter: InterpreterApi? = null
    private var isGpuAvailable = false
    private var actualBackend = "UNKNOWN"

    // Image processor - normalize to [0, 1]
    private val imageProcessor = ImageProcessor.Builder()
        .add(ResizeOp(INPUT_SIZE, INPUT_SIZE, ResizeOp.ResizeMethod.BILINEAR))
        .build()

    /**
     * Initialize with GPU support
     */
    fun initialize(): Task<Boolean> {
        FileLogger.section("Initializing Landmark Detector (Play Services GPU)")

        return TfLite.initialize(context).continueWith { task ->
            if (!task.isSuccessful) {
                FileLogger.e(TAG, "✗ Play Services TFLite init failed")
                return@continueWith false
            }

            FileLogger.i(TAG, "✓ Play Services TFLite initialized")

            try {
                val gpuDelegateFactory = GpuDelegateFactory()
                isGpuAvailable = gpuDelegateFactory != null
                FileLogger.i(TAG, "✓ GPU delegate available")
            } catch (e: Exception) {
                isGpuAvailable = false
                FileLogger.e(TAG, "GPU delegate not available: ${e.message}")
            }

            // Load model as ByteBuffer
            val modelBuffer = context.assets.openFd(MODEL_NAME).use { fileDescriptor ->
                val inputStream = fileDescriptor.createInputStream()
                val modelBytes = inputStream.readBytes()
                inputStream.close()
                java.nio.ByteBuffer.allocateDirect(modelBytes.size).apply {
                    put(modelBytes)
                    rewind()
                }
            }

            // Try GPU first
            if (isGpuAvailable) {
                FileLogger.d(TAG, "Loading model with GPU delegate...")
                try {
                    val options = InterpreterApi.Options()
                        .setRuntime(TfLiteRuntime.FROM_SYSTEM_ONLY)
                        .addDelegateFactory(GpuDelegateFactory())

                    interpreter = InterpreterApi.create(modelBuffer, options)
                    actualBackend = "GPU"
                    FileLogger.i(TAG, "✓ Landmark Detector ready on GPU")
                    return@continueWith true
                } catch (e: Exception) {
                    FileLogger.e(TAG, "GPU loading failed, falling back to CPU")
                    FileLogger.e(TAG, "  Exception: ${e.javaClass.simpleName}: ${e.message}")
                }
            }

            // Fallback to CPU
            FileLogger.d(TAG, "Loading model on CPU...")
            try {
                val options = InterpreterApi.Options()
                    .setRuntime(TfLiteRuntime.FROM_SYSTEM_ONLY)

                interpreter = InterpreterApi.create(modelBuffer, options)
                actualBackend = "CPU"
                FileLogger.i(TAG, "✓ Landmark Detector ready on CPU")
                true
            } catch (e: Exception) {
                FileLogger.e(TAG, "✗ Landmark Detector init failed: ${e.message}")
                false
            }
        }
    }

    /**
     * Detect landmarks from hand ROI
     */
    fun detectLandmarks(
        bitmap: Bitmap,
        roi: HandTrackingROI
    ): LandmarkResult? {
        try {
            // Crop and warp ROI to 256×256
            val warpedBitmap = warpAffineROI(bitmap, roi, INPUT_SIZE, INPUT_SIZE)

            // Preprocess - convert to FLOAT32 and normalize to [0, 1]
            val tensorImage = TensorImage(org.tensorflow.lite.DataType.FLOAT32)
            tensorImage.load(warpedBitmap)
            val processedImage = imageProcessor.process(tensorImage)

            // Prepare output buffers - VERIFIED from Python
            val outputLandmarks = Array(1) { Array(NUM_LANDMARKS) { FloatArray(3) } }  // [1, 21, 3]
            val outputScores = FloatArray(1)       // [1] - presence score
            val outputHandedness = FloatArray(1)   // [1] - left/right score

            // Output mapping - VERIFIED from Python code (line 208)
            val outputs = mapOf(
                0 to outputScores,      // scores (presence) [1]
                1 to outputHandedness,  // lr (handedness) [1]
                2 to outputLandmarks    // landmarks [1, 21, 3]
            )

            // Run inference
            interpreter?.runForMultipleInputsOutputs(
                arrayOf(processedImage.buffer),
                outputs
            )

            val presence = outputScores[0]
            val handedness = if (outputHandedness[0] > 0.5f) "Right" else "Left"

            // Check presence threshold
            if (presence < 0.5f) {
                FileLogger.d(TAG, "Hand presence too low: $presence")
                return null
            }

            FileLogger.d(TAG, "✓ Landmarks detected! Handedness: $handedness, Presence: $presence")

            // Unproject landmarks from ROI to image space
            val unprojectedLandmarks = unprojectLandmarks(
                outputLandmarks[0],  // Extract [21, 3] from [1, 21, 3]
                roi,
                bitmap.width,
                bitmap.height
            )

            return LandmarkResult(
                landmarks = unprojectedLandmarks,
                presence = presence,
                handedness = handedness
            )

        } catch (e: Exception) {
            FileLogger.e(TAG, "Landmark detection failed: ${e.message}")
            return null
        }
    }

    /**
     * Warp and crop ROI from bitmap
     * FIXED: Properly clamp ROI to image boundaries to prevent crashes
     */
    private fun warpAffineROI(
        bitmap: Bitmap,
        roi: HandTrackingROI,
        targetWidth: Int,
        targetHeight: Int
    ): Bitmap {
        // Calculate ROI boundaries
        val roiLeft = roi.centerX - roi.roiWidth / 2
        val roiTop = roi.centerY - roi.roiHeight / 2
        val roiRight = roi.centerX + roi.roiWidth / 2
        val roiBottom = roi.centerY + roi.roiHeight / 2

        // ✅ CLAMP to image boundaries
        val clampedLeft = maxOf(0f, roiLeft)
        val clampedTop = maxOf(0f, roiTop)
        val clampedRight = minOf(bitmap.width.toFloat(), roiRight)
        val clampedBottom = minOf(bitmap.height.toFloat(), roiBottom)

        // Calculate clamped dimensions
        val clampedWidth = clampedRight - clampedLeft
        val clampedHeight = clampedBottom - clampedTop

        // Safety check
        if (clampedWidth <= 0 || clampedHeight <= 0) {
            FileLogger.e(TAG, "Invalid ROI dimensions after clamping!")
            // Return a small region from center as fallback
            return Bitmap.createScaledBitmap(
                Bitmap.createBitmap(
                    bitmap,
                    maxOf(0, bitmap.width / 2 - 50),
                    maxOf(0, bitmap.height / 2 - 50),
                    minOf(100, bitmap.width),
                    minOf(100, bitmap.height)
                ),
                targetWidth,
                targetHeight,
                true
            )
        }

        // Create transformation matrix for the CLAMPED region
        val matrix = Matrix()

        val srcPoints = floatArrayOf(
            clampedLeft, clampedTop,      // Top-left
            clampedRight, clampedTop,     // Top-right
            clampedLeft, clampedBottom    // Bottom-left
        )

        val dstPoints = floatArrayOf(
            0f, 0f,
            targetWidth.toFloat(), 0f,
            0f, targetHeight.toFloat()
        )

        matrix.setPolyToPoly(srcPoints, 0, dstPoints, 0, 3)

        // Crop the CLAMPED region (guaranteed to be within bounds)
        return Bitmap.createBitmap(
            bitmap,
            clampedLeft.toInt(),
            clampedTop.toInt(),
            clampedWidth.toInt(),
            clampedHeight.toInt(),
            matrix,
            true
        )
    }

    /**
     * Unproject landmarks from ROI space to image space
     * FIXED: Landmarks are already [0,1] from model, no division needed
     */
    private fun unprojectLandmarks(
        landmarks: Array<FloatArray>,  // [21, 3] from model
        roi: HandTrackingROI,
        imageWidth: Int,
        imageHeight: Int
    ): Array<FloatArray> {
        val result = Array(NUM_LANDMARKS) { FloatArray(3) }

        // Log ROI info for debugging (only once per detection)
        FileLogger.d(TAG, "ROI Info: center=(%.1f, %.1f), size=(%.1f, %.1f), rotation=%.2f"
            .format(roi.centerX, roi.centerY, roi.roiWidth, roi.roiHeight, roi.rotation))

        for (i in 0 until NUM_LANDMARKS) {
            // Extract landmark - landmarks are already in [0,1] normalized space from model
            val xNorm = landmarks[i][0]  // Already [0,1]
            val yNorm = landmarks[i][1]  // Already [0,1]
            val z = landmarks[i][2]

            // Log first landmark (wrist) raw values
            if (i == 0) {
                FileLogger.d(TAG, "Wrist raw from model: x=%.4f, y=%.4f, z=%.4f"
                    .format(xNorm, yNorm, z))
            }

            // Simple transformation (no rotation for now)
            // Map from [0,1] ROI space to image pixel coordinates
            val roiLeft = roi.centerX - roi.roiWidth / 2
            val roiTop = roi.centerY - roi.roiHeight / 2

            val xPixel = roiLeft + xNorm * roi.roiWidth
            val yPixel = roiTop + yNorm * roi.roiHeight

            result[i][0] = xPixel
            result[i][1] = yPixel
            result[i][2] = z * roi.roiWidth  // Scale z relative to ROI size

            // Log first landmark (wrist) final values
            if (i == 0) {
                FileLogger.d(TAG, "Wrist unprojected: x=%.1f, y=%.1f (image: %dx%d)"
                    .format(xPixel, yPixel, imageWidth, imageHeight))
            }
        }

        return result
    }

    /**
     * Get backend info
     */
    fun getBackend(): String = actualBackend

    /**
     * Release resources
     */
    fun close() {
        interpreter?.close()
        interpreter = null
        FileLogger.i(TAG, "✓ Landmark Detector closed")
    }
}

/**
 * Landmark detection result
 */
data class LandmarkResult(
    val landmarks: Array<FloatArray>,  // [21, 3] - x, y, z in frame coords
    val presence: Float,
    val handedness: String  // "Left" or "Right"
)