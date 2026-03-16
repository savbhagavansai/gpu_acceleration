package com.gesture.recognition

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.util.Log
import com.google.android.gms.tasks.Task
import com.google.android.gms.tasks.Tasks
import com.google.android.gms.tflite.java.TfLite
import org.tensorflow.lite.InterpreterApi
import org.tensorflow.lite.InterpreterApi.Options.TfLiteRuntime
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.gpu.GpuDelegateFactory

/**
 * Hand Landmark Detector using Google Play Services TFLite with GPU acceleration
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

    // Image processor
    private val imageProcessor = ImageProcessor.Builder()
        .add(ResizeOp(INPUT_SIZE, INPUT_SIZE, ResizeOp.ResizeMethod.BILINEAR))
        .add(NormalizeOp(127.5f, 127.5f))  // Normalize to [-1, 1]
        .build()

    /**
     * Initialize with GPU support
     */
    fun initialize(): Task<Boolean> {
        FileLogger.section("Initializing Landmark Detector (Play Services GPU)")

        return TfLite.initialize(context).continueWithTask { initTask ->
            if (!initTask.isSuccessful) {
                FileLogger.e(TAG, "TfLite initialization failed", initTask.exception)
                return@continueWithTask Tasks.forResult(false)
            }

            FileLogger.i(TAG, "✓ Play Services TFLite initialized")

            // Check GPU availability
            checkGpuAvailability().continueWith { gpuTask ->
                isGpuAvailable = gpuTask.result ?: false

                if (isGpuAvailable) {
                    FileLogger.i(TAG, "✓ GPU delegate available")
                    loadModelWithGpu()
                } else {
                    FileLogger.w(TAG, "GPU not available, using CPU")
                    loadModelWithCpu()
                }

                true
            }
        }
    }

    /**
     * Check if GPU delegate is available
     */
    private fun checkGpuAvailability(): Task<Boolean> {
        return try {
            com.google.android.gms.tflite.gpu.support.TfLiteGpu.isGpuDelegateAvailable(context)
        } catch (e: Exception) {
            FileLogger.w(TAG, "GPU check failed: ${e.message}")
            Tasks.forResult(false)
        }
    }

    /**
     * Load model with GPU acceleration
     */
    private fun loadModelWithGpu() {
        try {
            FileLogger.d(TAG, "Loading model with GPU delegate...")

            val modelBuffer = loadModelFile()

            val options = InterpreterApi.Options()
                .setRuntime(TfLiteRuntime.FROM_APPLICATION_ONLY)
                .addDelegateFactory(GpuDelegateFactory())

            interpreter = InterpreterApi.create(modelBuffer, options)
            actualBackend = "GPU (Mali-G68 MP5)"

            FileLogger.i(TAG, "✓ Landmark Detector ready on GPU")
            Log.d(TAG, "✓ Model loaded on GPU")

        } catch (e: Exception) {
            FileLogger.e(TAG, "GPU loading failed, falling back to CPU", e)
            loadModelWithCpu()
        }
    }

    /**
     * Load model with CPU fallback
     */
    private fun loadModelWithCpu() {
        try {
            FileLogger.d(TAG, "Loading model on CPU...")

            val modelBuffer = loadModelFile()

            val options = InterpreterApi.Options()
                .setRuntime(TfLiteRuntime.FROM_APPLICATION_ONLY)
                .setNumThreads(4)

            interpreter = InterpreterApi.create(modelBuffer, options)
            actualBackend = "CPU (4 threads)"

            FileLogger.i(TAG, "✓ Landmark Detector ready on CPU")
            Log.d(TAG, "✓ Model loaded on CPU")

        } catch (e: Exception) {
            FileLogger.e(TAG, "Model loading failed!", e)
            throw e
        }
    }

    /**
     * Load model file from assets
     */
    private fun loadModelFile(): java.nio.ByteBuffer {
        val fileDescriptor = context.assets.openFd(MODEL_NAME)
        val inputStream = java.io.FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(
            java.nio.channels.FileChannel.MapMode.READ_ONLY,
            startOffset,
            declaredLength
        )
    }

    /**
     * Detect hand landmarks in ROI
     */
    fun detectLandmarks(bitmap: Bitmap, roi: HandTrackingROI): LandmarkResult? {
        val interp = interpreter ?: run {
            FileLogger.e(TAG, "Interpreter not initialized!")
            return null
        }

        try {
            // Warp ROI to square
            val warpedBitmap = warpAffineROI(bitmap, roi, INPUT_SIZE, INPUT_SIZE)

            // Create TensorImage with explicit FLOAT32 type
            var tensorImage = TensorImage(org.tensorflow.lite.DataType.FLOAT32)
            tensorImage.load(warpedBitmap)

            // Apply preprocessing
            tensorImage = imageProcessor.process(tensorImage)

            // Prepare output buffers
            val outputLandmarks = Array(1) { FloatArray(NUM_LANDMARKS * 3) }  // [1, 63]
            val outputScores = FloatArray(1)  // [1] - presence score
            val outputHandedness = FloatArray(1)  // [1] - left/right score

            val outputs = mapOf(
                0 to outputLandmarks,
                1 to outputScores,
                2 to outputHandedness
            )

            // Run inference
            interp.runForMultipleInputsOutputs(
                arrayOf(tensorImage.buffer),
                outputs
            )

            val presenceScore = outputScores[0]

            if (presenceScore < 0.5f) {
                FileLogger.d(TAG, "Hand presence too low: $presenceScore")
                return null
            }

            // Unproject landmarks back to original image coordinates
            val landmarksInFrame = unprojectLandmarks(
                outputLandmarks[0],
                roi,
                bitmap.width,
                bitmap.height
            )

            val handedness = if (outputHandedness[0] > 0.5f) "Right" else "Left"

            FileLogger.d(TAG, "✓ Landmarks detected! Handedness: $handedness, Presence: $presenceScore")

            return LandmarkResult(
                landmarksInFrame,
                presenceScore,
                handedness
            )

        } catch (e: Exception) {
            FileLogger.e(TAG, "Landmark detection failed", e)
            return null
        }
    }

    /**
     * Warp ROI using affine transformation
     */
    private fun warpAffineROI(
        bitmap: Bitmap,
        roi: HandTrackingROI,
        targetWidth: Int,
        targetHeight: Int
    ): Bitmap {
        val matrix = Matrix()

        // Calculate transformation matrix
        val srcPoints = floatArrayOf(
            roi.centerX - roi.roiWidth / 2, roi.centerY - roi.roiHeight / 2,  // Top-left
            roi.centerX + roi.roiWidth / 2, roi.centerY - roi.roiHeight / 2,  // Top-right
            roi.centerX - roi.roiWidth / 2, roi.centerY + roi.roiHeight / 2   // Bottom-left
        )

        val dstPoints = floatArrayOf(
            0f, 0f,
            targetWidth.toFloat(), 0f,
            0f, targetHeight.toFloat()
        )

        matrix.setPolyToPoly(srcPoints, 0, dstPoints, 0, 3)

        return Bitmap.createBitmap(
            bitmap,
            maxOf(0, (roi.centerX - roi.roiWidth / 2).toInt()),
            maxOf(0, (roi.centerY - roi.roiHeight / 2).toInt()),
            minOf(bitmap.width, roi.roiWidth.toInt()),
            minOf(bitmap.height, roi.roiHeight.toInt()),
            matrix,
            true
        )
    }

    /**
     * Unproject landmarks from ROI back to original frame
     */
    private fun unprojectLandmarks(
        landmarks: FloatArray,
        roi: HandTrackingROI,
        imageWidth: Int,
        imageHeight: Int
    ): Array<FloatArray> {
        val result = Array(NUM_LANDMARKS) { FloatArray(3) }

        for (i in 0 until NUM_LANDMARKS) {
            // Landmarks are in [0, 256] pixel coordinates
            val x = landmarks[i * 3]
            val y = landmarks[i * 3 + 1]
            val z = landmarks[i * 3 + 2]

            // Map back to ROI coordinates
            val xNorm = x / INPUT_SIZE
            val yNorm = y / INPUT_SIZE

            // Map to original image coordinates
            result[i][0] = (roi.centerX - roi.roiWidth / 2 + xNorm * roi.roiWidth)
            result[i][1] = (roi.centerY - roi.roiHeight / 2 + yNorm * roi.roiHeight)
            result[i][2] = z / INPUT_SIZE * roi.roiWidth  // Scale z relative to ROI size
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